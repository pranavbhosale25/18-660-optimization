"""Portable numeric NPZ checkpoints without executable or optimizer state."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np

from .model import MODEL_LAYER_COUNT, validate_params
from .training import TrainConfig, TrainResult

_FORMAT_VERSION = 1
_MAX_METADATA_BYTES = 4 * 1024 * 1024


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def save_checkpoint(result, problem, path):
    """Atomically save the model, fixed problem data, and training metadata.

    The file is a single NPZ archive of numeric arrays plus UTF-8 JSON
    encoded as uint8. It contains enough information for inference and
    reevaluation. It does not claim to resume Adam because optimizer state
    and minibatch keys are absent.
    """
    result.config.validate()
    validate_params(problem, result.params, result.config.width)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    problem_fields = {}
    for index, (name, value) in enumerate(problem.state_dict().items()):
        array = np.asarray(value)
        if array.dtype.kind not in "biufc" or not np.isfinite(array).all():
            raise ValueError(
                f"Problem-state field {name!r} must be a finite numeric array"
            )
        key = f"problem_{index}"
        arrays[key] = array
        problem_fields[name] = key
    layer_fields = []
    for index, layer in enumerate(result.params):
        fields = {}
        for field in ("w", "b"):
            key = f"layer_{index}_{field}"
            arrays[key] = np.asarray(jax.device_get(layer[field]))
            fields[field] = key
        layer_fields.append(fields)
    metadata = _json_value(
        {
            "format": "l2ws_jax",
            "format_version": _FORMAT_VERSION,
            "problem_name": problem.name,
            "problem_config": problem.config,
            "dtype": problem.dtype.name,
            "runtime": problem.runtime.info(),
            "problem_fields": problem_fields,
            "layer_fields": layer_fields,
            "config": asdict(result.config),
            "history": result.history,
            "elapsed_seconds": result.elapsed_seconds,
            "best_step": result.best_step,
            "reference_info": result.reference_info,
            "compilation_seconds": result.compilation_seconds,
            "execution_seconds": result.execution_seconds,
        }
    )
    encoded = json.dumps(
        metadata, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    arrays["metadata"] = np.frombuffer(encoded, dtype=np.uint8)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _nonnegative_time(metadata, field):
    value = metadata.get(field, 0.0)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"Invalid checkpoint {field}")
    return float(value)


def _history(metadata, config):
    history = metadata.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("Checkpoint history must be a nonempty list")
    prior = -1
    for row in history:
        if not isinstance(row, dict) or set(row) != {
            "step",
            "train_loss",
            "validation_loss",
            "validation_objective",
        }:
            raise ValueError("Invalid checkpoint history row")
        step = row["step"]
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or not prior < step <= config.steps
        ):
            raise ValueError(
                "Checkpoint history steps must be increasing and within "
                "training steps"
            )
        prior = step
        for key in ("train_loss", "validation_loss", "validation_objective"):
            value = row[key]
            if key == "train_loss" and value is None and step == 0:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Invalid checkpoint history {key}")
    if history[0]["step"] != 0:
        raise ValueError("Checkpoint history must begin at step zero")
    return history


def load_checkpoint(path, *, dtype=None, device="auto"):
    """Restore an immutable problem and model using ``allow_pickle=False``.

    By default precision is preserved and JAX selects an available device. An
    explicit dtype/device override casts/moves both fixed data and model arrays.
    This is an inference checkpoint conversion, not optimizer-state resumption.
    """
    from .problems import make_problem

    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            if "metadata" not in archive:
                raise ValueError("Checkpoint has no metadata")
            encoded = archive["metadata"]
            if (
                encoded.dtype != np.uint8
                or encoded.ndim != 1
                or encoded.size > _MAX_METADATA_BYTES
            ):
                raise ValueError(
                    "Checkpoint metadata must be a bounded uint8 vector"
                )
            metadata = json.loads(encoded.tobytes().decode("utf-8"))
            if (
                not isinstance(metadata, dict)
                or metadata.get("format") != "l2ws_jax"
                or metadata.get("format_version") != _FORMAT_VERSION
            ):
                raise ValueError("Unsupported checkpoint format")
            saved_dtype = np.dtype(metadata["dtype"])
            if saved_dtype not in (np.dtype("float32"), np.dtype("float64")):
                raise ValueError("Unsupported checkpoint precision")
            config = TrainConfig(**metadata["config"])
            config.validate()
            history = _history(metadata, config)
            best_step = metadata["best_step"]
            if (
                isinstance(best_step, bool)
                or not isinstance(best_step, int)
                or best_step not in {row["step"] for row in history}
            ):
                raise ValueError(
                    "Checkpoint best_step must identify a validation row"
                )
            state_fields = metadata["problem_fields"]
            layer_fields = metadata["layer_fields"]
            if not isinstance(state_fields, dict) or not all(
                isinstance(name, str) and isinstance(key, str)
                for name, key in state_fields.items()
            ):
                raise ValueError("Invalid checkpoint problem-state mapping")
            if (
                not isinstance(layer_fields, list)
                or len(layer_fields) != MODEL_LAYER_COUNT
                or not all(
                    isinstance(layer, dict)
                    and set(layer) == {"w", "b"}
                    and all(isinstance(key, str) for key in layer.values())
                    for layer in layer_fields
                )
            ):
                raise ValueError("Invalid checkpoint model mapping")
            keys = list(state_fields.values()) + [
                key for layer in layer_fields for key in layer.values()
            ]
            if len(set(keys)) != len(keys) or set(archive.files) != set(
                keys
            ) | {"metadata"}:
                raise ValueError("Checkpoint arrays do not match its metadata")
            state = {
                name: np.array(archive[key], copy=True)
                for name, key in state_fields.items()
            }
            for name, array in state.items():
                if (
                    array.dtype.kind not in "biufc"
                    or not np.isfinite(array).all()
                ):
                    raise ValueError(f"Invalid problem-state field {name!r}")
            host_layers = tuple(
                {
                    field: np.array(archive[key], copy=True)
                    for field, key in layer.items()
                }
                for layer in layer_fields
            )
            if any(
                array.dtype != saved_dtype
                for layer in host_layers
                for array in layer.values()
            ):
                raise ValueError(
                    "Model arrays disagree with checkpoint precision"
                )
    except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid checkpoint metadata: {error}") from error
    target_dtype = saved_dtype if dtype is None else dtype
    problem = make_problem(
        metadata["problem_name"],
        dtype=target_dtype,
        device=device,
        **metadata["problem_config"],
    )
    # with_state creates a new immutable object; it also verifies every fixed
    # field against the requested constructor's shape before casting/moving it.
    problem = problem.with_state(state)
    params = tuple(
        {field: problem.array(value) for field, value in layer.items()}
        for layer in host_layers
    )
    validate_params(problem, params, config.width)
    reference_info = metadata.get("reference_info")
    if reference_info is not None and not isinstance(reference_info, dict):
        raise ValueError(
            "Checkpoint reference_info must be a dictionary or null"
        )
    result = TrainResult(
        params=params,
        config=config,
        history=history,
        elapsed_seconds=_nonnegative_time(metadata, "elapsed_seconds"),
        best_step=best_step,
        reference_info=reference_info,
        compilation_seconds=_nonnegative_time(metadata, "compilation_seconds"),
        execution_seconds=_nonnegative_time(metadata, "execution_seconds"),
    )
    return problem, result

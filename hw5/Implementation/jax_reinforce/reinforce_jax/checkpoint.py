"""Portable, non-pickle checkpoints for the supplied MLP actor."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_FORMAT_VERSION = 1


def _validated_layers(params):
    """Return host weight/bias pairs after checking the default MLP layout."""
    layers = tuple(params)
    if not layers:
        raise ValueError("actor parameters must contain at least one layer")
    validated = []
    previous_width = None
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict) or set(layer) != {"w", "b"}:
            raise ValueError(f"actor layer {index} must contain exactly 'w' and 'b'")
        weight, bias = jax.device_get((layer["w"], layer["b"]))
        weight = np.asarray(weight)
        bias = np.asarray(bias)
        if weight.ndim != 2 or bias.ndim != 1:
            raise ValueError(
                f"actor layer {index} requires a matrix weight and vector bias"
            )
        if weight.shape[1] != bias.shape[0]:
            raise ValueError(
                f"actor layer {index} weight output and bias sizes disagree"
            )
        if previous_width is not None and weight.shape[0] != previous_width:
            raise ValueError(
                f"actor layer {index} input width does not match prior output"
            )
        if weight.dtype.kind != "f" or bias.dtype != weight.dtype:
            raise ValueError(f"actor layer {index} must use one floating-point dtype")
        validated.append((weight, bias))
        previous_width = weight.shape[1]
    return validated


def save_mlp_checkpoint(path, params):
    """Save the supplied tuple-of-layer-dictionaries MLP without pickle.

    The checkpoint contains actor parameters only. It is suitable for policy
    evaluation and visualization, but it does not contain optimizer or critic
    state and therefore cannot resume training.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("MLP checkpoint path must end in .npz")
    layers = _validated_layers(params)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": np.asarray(_FORMAT_VERSION, dtype=np.int64),
        "layer_count": np.asarray(len(layers), dtype=np.int64),
    }
    for index, (weight, bias) in enumerate(layers):
        payload[f"layer_{index}_w"] = weight
        payload[f"layer_{index}_b"] = bias
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_mlp_checkpoint(path, *, dtype=None):
    """Load and validate an MLP checkpoint as JAX arrays.

    Args:
        path: ``.npz`` file written by :func:`save_mlp_checkpoint`.
        dtype: Optional expected NumPy/JAX floating dtype. A mismatch raises
            an error instead of silently changing the saved policy.

    Returns:
        A tuple of ``{"w": weight, "b": bias}`` dictionaries accepted by
        ``mlp_apply``.
    """
    path = Path(path)
    try:
        archive_context = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read actor checkpoint {path}") from exc
    if not hasattr(archive_context, "files"):
        raise ValueError(f"actor checkpoint {path} is not an NPZ archive")
    with archive_context as archive:
        try:
            version = int(np.asarray(archive["format_version"]).item())
            layer_count = int(np.asarray(archive["layer_count"]).item())
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("actor checkpoint metadata is invalid") from exc
        if version != _FORMAT_VERSION:
            raise ValueError(f"unsupported actor checkpoint format version {version}")
        if layer_count < 1:
            raise ValueError("actor checkpoint must contain at least one layer")
        expected = {"format_version", "layer_count"}
        for index in range(layer_count):
            expected.update({f"layer_{index}_w", f"layer_{index}_b"})
        if set(archive.files) != expected:
            raise ValueError("actor checkpoint layer fields are incomplete")
        raw_layers = []
        try:
            for index in range(layer_count):
                raw_layers.append(
                    {
                        "w": np.asarray(archive[f"layer_{index}_w"]),
                        "b": np.asarray(archive[f"layer_{index}_b"]),
                    }
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("actor checkpoint layer arrays are invalid") from exc
    validated = _validated_layers(raw_layers)
    if dtype is not None:
        expected_dtype = np.dtype(dtype)
        for index, (weight, _) in enumerate(validated):
            if weight.dtype != expected_dtype:
                raise ValueError(
                    f"actor layer {index} has dtype {weight.dtype}, "
                    f"expected {expected_dtype}"
                )
    return tuple(
        {"w": jnp.asarray(weight), "b": jnp.asarray(bias)} for weight, bias in validated
    )

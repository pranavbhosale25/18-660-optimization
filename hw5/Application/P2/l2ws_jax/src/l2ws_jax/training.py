"""Reproducible training with compiled update chunks and honest timing."""

from __future__ import annotations

import math
import numbers
import time
import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .model import DEFAULT_MODEL_WIDTH, init_params, initialize
from .solver import batch_loss, rollout

RANDOM_SEED_UPPER_BOUND = 2**32
_TRAINING_RANDOM_STREAM_COUNT = 4
_BATCHED_DATA_RANK = 2
SUPPORTED_LOSSES = frozenset(("objective", "fixed_point", "regression"))


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 300
    unroll: int = 5
    loss: str = "objective"
    width: int = DEFAULT_MODEL_WIDTH
    learning_rate: float = 0.002
    batch_size: int = 64
    train_size: int = 1024
    validation_size: int = 128
    validate_every: int = 25
    grad_clip: float = 10.0
    seed: int = 7
    jit: bool = True

    def validate(self):
        positive = (
            "steps",
            "width",
            "batch_size",
            "train_size",
            "validation_size",
            "validate_every",
        )
        for name in positive + ("unroll", "seed"):
            value = getattr(self, name)
            minimum = 1 if name in positive else 0
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or value < minimum
            ):
                adjective = "positive" if minimum else "nonnegative"
                raise ValueError(f"{name} must be a {adjective} integer")
        if self.seed >= RANDOM_SEED_UPPER_BOUND:
            raise ValueError(
                f"seed must be smaller than {RANDOM_SEED_UPPER_BOUND}"
            )
        for name in ("learning_rate", "grad_clip"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.loss not in SUPPORTED_LOSSES:
            raise ValueError(
                "loss must be objective, fixed_point, or regression"
            )
        if not isinstance(self.jit, bool):
            raise ValueError("jit must be a boolean")


DEFAULT_TRAIN_CONFIG = TrainConfig()


@dataclass
class TrainResult:
    params: tuple
    config: TrainConfig
    history: list[dict]
    elapsed_seconds: float
    best_step: int
    reference_info: dict | None = None
    compilation_seconds: float = 0.0
    execution_seconds: float = 0.0


def _check_data(problem, data, columns, label):
    """Accept selected-dtype NumPy/JAX arrays and place them explicitly."""
    if not isinstance(data, (np.ndarray, jax.Array)):
        raise ValueError(
            f"{label} must be a NumPy or JAX array; use problem.array(values)"
        )
    host = np.asarray(jax.device_get(data))
    if (
        host.ndim != _BATCHED_DATA_RANK
        or host.shape[1] != columns
        or len(host) == 0
    ):
        raise ValueError(f"{label} must have nonempty shape [batch, {columns}]")
    if host.dtype != problem.dtype:
        raise ValueError(
            f"{label} must use problem.dtype={problem.dtype.name}; "
            "cast explicitly with problem.array(values)"
        )
    if not np.isfinite(host).all():
        raise ValueError(f"{label} must be finite")
    return jax.device_put(data, problem.device)


def _tree_finite(tree):
    checks = [
        jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(tree)
    ]
    return jnp.all(jnp.stack(checks)) if checks else jnp.asarray(True)


def _reference_labels(problem, data, label, information):
    from .reference import solve_reference

    reference = solve_reference(problem, data)
    certificates = np.asarray(reference.certificates, dtype=np.float64)
    finite = certificates[np.isfinite(certificates)]
    certified = np.asarray(reference.globally_certified, dtype=np.bool_)
    successes = np.asarray(reference.solver_success, dtype=np.bool_)
    stationarity = np.asarray(reference.stationarity, dtype=np.float64)
    information[label] = {
        "source": "generated",
        "kind": reference.label,
        "count": len(data),
        "elapsed_seconds": float(reference.elapsed_seconds),
        "compilation_seconds": float(reference.compilation_seconds),
        "certificate_tolerance": float(reference.tolerance),
        "training_label_dtype": problem.dtype.name,
        "certification_applies_before_label_cast": True,
        "solver_success_fraction": float(np.mean(successes)),
        "certified_fraction_at_1e-7": float(np.mean(certified)),
        "certificate_dtype": "float64",
        "certificate_device": "cpu",
        "max_certificate": float(np.max(finite)) if len(finite) else None,
        "max_gradient_mapping": float(np.max(stationarity))
        if np.isfinite(stationarity).all()
        else None,
    }
    if problem.convex and not bool(np.all(certified)):
        count = int(np.count_nonzero(~certified))
        warnings.warn(
            f"{count} {label} reference labels missed the CPU float64 1e-7 "
            "certificate tolerance; inspect result.reference_info before "
            "treating these labels as accurate optimizers.",
            RuntimeWarning,
            stacklevel=3,
        )
    # References intentionally use CPU float64. Label precision for training is
    # an explicit conversion, never an accidental promotion of a float32 run.
    return problem.array(reference.solution)


def train(
    problem,
    config=None,
    *,
    contexts=None,
    targets=None,
    validation=None,
    validation_targets=None,
    progress=None,
):
    """Return parameters selected by an independent validation set.

    Four keys split from ``seed`` control training data, validation data, model
    initialization, and minibatches. Supplied arrays must already have the
    selected floating dtype; they are moved to ``problem.device`` here.

    With JIT enabled, one compiled ``lax.scan`` performs a chunk of Adam
    updates. Its fixed size is ``min(validate_every, steps)``; a dynamic
    active count masks the last partial chunk, avoiding a second compilation.
    Validation is also a compiled whole function. The host only inspects
    losses/finite flags and logs at chunk boundaries. With ``jit=False``, JIT
    and scan staging are disabled during updates and validation to support
    line-by-line debugging.

    ``compilation_seconds`` measures explicit lowering/compilation of those two
    whole functions. ``execution_seconds`` measures synchronized update and
    validation calls. ``elapsed_seconds`` includes data, initialization,
    reference generation, compilation, execution, host checks, and progress
    callbacks; primitive setup costs are included only in that total. Timings
    are wall time, and the best parameters are selected by the configured
    validation loss.
    """
    started = time.perf_counter()
    config = config or TrainConfig()
    config.validate()
    with jax.default_device(problem.device):
        data_key, validation_key, model_key, batch_key = jax.random.split(
            problem.key(config.seed), _TRAINING_RANDOM_STREAM_COUNT
        )
        if contexts is None:
            contexts = problem.sample(config.train_size, data_key)
        if validation is None:
            validation = problem.sample(config.validation_size, validation_key)
        contexts = _check_data(
            problem, contexts, problem.context_dim, "contexts"
        )
        validation = _check_data(
            problem, validation, problem.context_dim, "validation"
        )
        reference_info = None
        if config.loss == "regression":
            reference_info = {}
            if targets is None:
                targets = _reference_labels(
                    problem, contexts, "training", reference_info
                )
            else:
                reference_info["training"] = {
                    "source": "provided",
                    "count": len(contexts),
                }
            if validation_targets is None:
                validation_targets = _reference_labels(
                    problem, validation, "validation", reference_info
                )
            else:
                reference_info["validation"] = {
                    "source": "provided",
                    "count": len(validation),
                }
        for data, count, name in (
            (targets, len(contexts), "targets"),
            (validation_targets, len(validation), "validation_targets"),
        ):
            if data is not None:
                checked = _check_data(problem, data, problem.decision_dim, name)
                if len(checked) != count:
                    raise ValueError(
                        f"{name} must have the same batch size as its contexts"
                    )
                if name == "targets":
                    targets = checked
                else:
                    validation_targets = checked
        if config.loss != "regression":
            # Do not carry unused label arrays through a compiled objective.
            targets = validation_targets = None
        params = init_params(problem, model_key, config.width)
        optimizer = optax.chain(
            optax.clip_by_global_norm(
                jnp.asarray(config.grad_clip, dtype=problem.dtype)
            ),
            optax.adam(jnp.asarray(config.learning_rate, dtype=problem.dtype)),
        )
        state = (params, optimizer.init(params), batch_key)
        chunk_size = min(config.validate_every, config.steps)
        active = problem.runtime.array(chunk_size, dtype=jnp.int32)
        indices_in_chunk = jnp.arange(chunk_size, dtype=jnp.int32)

        def train_loss(parameters, batch, labels):
            initial = initialize(problem, parameters, batch)
            final = rollout(problem, initial, batch, int(config.unroll))
            return batch_loss(problem, final, batch, config.loss, labels)

        loss_and_grad = jax.value_and_grad(train_loss)

        def run_chunk(current_state, data, labels, active_count):
            def body(carry, index):
                def update(current):
                    parameters, opt_state, key = current
                    key, draw_key = jax.random.split(key)
                    selected = jax.random.randint(
                        draw_key,
                        (config.batch_size,),
                        0,
                        len(data),
                        dtype=jnp.int32,
                    )
                    batch = data[selected]
                    batch_labels = (
                        labels[selected] if labels is not None else None
                    )
                    loss, gradient = loss_and_grad(
                        parameters, batch, batch_labels
                    )
                    changes, next_opt_state = optimizer.update(
                        gradient, opt_state, parameters
                    )
                    next_parameters = optax.apply_updates(parameters, changes)
                    finite = (
                        jnp.isfinite(loss)
                        & _tree_finite(gradient)
                        & _tree_finite(next_parameters)
                        & _tree_finite(next_opt_state)
                    )
                    return (next_parameters, next_opt_state, key), (
                        loss,
                        finite,
                    )

                def skip(current):
                    return current, (
                        jnp.asarray(0.0, dtype=problem.dtype),
                        jnp.asarray(True),
                    )

                return jax.lax.cond(index < active_count, update, skip, carry)

            return jax.lax.scan(body, current_state, indices_in_chunk)

        def validate_parameters(parameters, data, labels):
            final = rollout(
                problem,
                initialize(problem, parameters, data),
                data,
                int(config.unroll),
            )
            selected_loss = batch_loss(
                problem, final, data, config.loss, labels
            )
            objective = jnp.mean(problem.objective(final, data))
            return selected_loss, objective

        compilation_seconds = 0.0
        if config.jit:
            compile_started = time.perf_counter()
            update_function = (
                jax.jit(run_chunk)
                .lower(state, contexts, targets, active)
                .compile()
            )
            validation_function = (
                jax.jit(validate_parameters)
                .lower(params, validation, validation_targets)
                .compile()
            )
            compilation_seconds = time.perf_counter() - compile_started
        else:
            update_function, validation_function = (
                run_chunk,
                validate_parameters,
            )
        execution_seconds = 0.0
        history = []
        best_value, best_step, best_params = math.inf, 0, params

        def validate(step, train_value):
            nonlocal execution_seconds, best_value, best_step, best_params
            evaluation_started = time.perf_counter()
            values = validation_function(
                state[0], validation, validation_targets
            )
            jax.block_until_ready(values)
            execution_seconds += time.perf_counter() - evaluation_started
            loss, objective = (float(value) for value in values)
            if not math.isfinite(loss) or not math.isfinite(objective):
                raise FloatingPointError(
                    f"Nonfinite validation value at update {step}"
                )
            row = {
                "step": int(step),
                "train_loss": train_value,
                "validation_loss": loss,
                "validation_objective": objective,
            }
            history.append(row)
            if loss < best_value:
                best_value, best_step, best_params = loss, int(step), state[0]
            if progress is not None:
                progress(dict(row))

        with jax.disable_jit(not config.jit):
            validate(0, None)
            completed = 0
            while completed < config.steps:
                count = min(chunk_size, config.steps - completed)
                active = problem.runtime.array(count, dtype=jnp.int32)
                execution_started = time.perf_counter()
                state, (losses, finite) = update_function(
                    state, contexts, targets, active
                )
                jax.block_until_ready((state, losses, finite))
                execution_seconds += time.perf_counter() - execution_started
                finite_host = np.asarray(jax.device_get(finite))[:count]
                if not bool(np.all(finite_host)):
                    offset = int(np.flatnonzero(~finite_host)[0])
                    raise FloatingPointError(
                        f"Nonfinite training loss, gradient, or optimizer "
                        f"state at update {completed + offset + 1}"
                    )
                last_loss = float(losses[count - 1])
                completed += count
                validate(completed, last_loss)
        return TrainResult(
            best_params,
            config,
            history,
            time.perf_counter() - started,
            best_step,
            reference_info,
            compilation_seconds,
            execution_seconds,
        )

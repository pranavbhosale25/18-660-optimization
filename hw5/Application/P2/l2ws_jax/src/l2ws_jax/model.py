"""A small functional context-to-initialization network without a framework."""

from __future__ import annotations

import numbers

import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_MODEL_WIDTH = 64
HIDDEN_LAYER_COUNT = 2
MODEL_LAYER_COUNT = HIDDEN_LAYER_COUNT + 1
_OUTPUT_LAYER_INDEX = MODEL_LAYER_COUNT - 1


def _layer_sizes(problem, width):
    return (
        problem.context_dim,
        *((width,) * HIDDEN_LAYER_COUNT),
        problem.decision_dim,
    )


def init_params(problem, key, width=DEFAULT_MODEL_WIDTH):
    """Create tanh hidden layers and a zero output layer on the device.

    ``features_one`` should preserve ``context_dim``. The zero final layer makes
    the initial decoded prediction exactly the problem's classical baseline.
    Parameters are an ordinary tuple of ``{'w': array, 'b': array}``
    dictionaries. This JAX PyTree needs no mutable module or global random
    state.
    """
    if (
        isinstance(width, bool)
        or not isinstance(width, numbers.Integral)
        or width < 1
    ):
        raise ValueError("width must be a positive integer")
    width = int(width)
    with jax.default_device(problem.device):
        key = jax.device_put(key, problem.device)
        keys = jax.random.split(key, HIDDEN_LAYER_COUNT)
        sizes = _layer_sizes(problem, width)
        layers = []
        for index, (input_size, output_size) in enumerate(
            zip(sizes[:-1], sizes[1:])
        ):
            if index == _OUTPUT_LAYER_INDEX:
                weights = jnp.zeros(
                    (input_size, output_size), dtype=problem.dtype
                )
            else:
                # Glorot initialization, with all floating arithmetic in the
                # selected precision even when JAX's x64 capability is enabled.
                limit = jnp.sqrt(
                    jnp.asarray(
                        6.0 / (input_size + output_size), dtype=problem.dtype
                    )
                )
                weights = jax.random.uniform(
                    keys[index],
                    (input_size, output_size),
                    dtype=problem.dtype,
                    minval=-limit,
                    maxval=limit,
                )
            bias = jnp.zeros(output_size, dtype=problem.dtype)
            layers.append({"w": weights, "b": bias})
        return tuple(layers)


def apply_model(params, features):
    """Apply the pure network to one feature vector or a batch of them."""
    output = features
    for layer in params[:-1]:
        output = jnp.tanh(output @ layer["w"] + layer["b"])
    final = params[-1]
    return output @ final["w"] + final["b"]


def initialize(problem, params, contexts):
    """Decode raw outputs into a feasible batch of solver initializations."""
    return problem.decode(
        apply_model(params, problem.features(contexts)), contexts
    )


def validate_params(problem, params, width):
    """Validate numeric checkpoint structure on the host, outside JIT."""
    if (
        not isinstance(params, (tuple, list))
        or len(params) != MODEL_LAYER_COUNT
    ):
        raise ValueError(
            f"Model parameters must contain exactly {MODEL_LAYER_COUNT} layers"
        )
    sizes = _layer_sizes(problem, width)
    for index, (layer, input_size, output_size) in enumerate(
        zip(params, sizes[:-1], sizes[1:])
    ):
        if not isinstance(layer, dict) or set(layer) != {"w", "b"}:
            raise ValueError(f"Layer {index} must contain exactly w and b")
        for field, shape in (
            ("w", (input_size, output_size)),
            ("b", (output_size,)),
        ):
            host = np.asarray(jax.device_get(layer[field]))
            if host.shape != shape or host.dtype != problem.dtype:
                raise ValueError(
                    f"Layer {index} {field} must have shape {shape} "
                    f"and dtype {problem.dtype.name}"
                )
            if not np.isfinite(host).all():
                raise ValueError(f"Layer {index} {field} must be finite")

"""Small default models, not a mandatory neural-network framework.

The rest of the package consumes apply(params, observation) callables. Use a
Flax apply function, an Equinox partition/combine adapter, a tabular model,
a CNN, or another pure-JAX function instead.
"""
from collections.abc import Sequence
import jax
import jax.numpy as jnp
from .types import PyTree, Array


def init_mlp(key: Array, input_dim: int, output_dim: int,
             hidden: Sequence[int] = (64, 64), *, dtype=jnp.float32,
             output_scale: float = 0.01) -> PyTree:
    sizes = (input_dim, *hidden, output_dim)
    if any(n <= 0 for n in sizes):
        raise ValueError("All layer sizes must be positive")
    keys = jax.random.split(key, len(sizes) - 1)
    layers = []
    for i, (fan_in, fan_out, k) in enumerate(zip(sizes[:-1], sizes[1:], keys)):
        gain = output_scale if i == len(keys) - 1 else 1.0
        scale = jnp.asarray(gain * (2.0 / (fan_in + fan_out)) ** 0.5, dtype)
        layers.append({"w": scale * jax.random.normal(k, (fan_in, fan_out), dtype),
                       "b": jnp.zeros((fan_out,), dtype)})
    return tuple(layers)


def mlp_apply(params: PyTree, obs: Array) -> Array:
    x = obs
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


def value_apply(params: PyTree, obs: Array, action: Array | None = None) -> Array:
    del action
    return mlp_apply(params, obs)[0]


def discrete_q_apply(params: PyTree, obs: Array, action: Array) -> Array:
    return mlp_apply(params, obs)[action.astype(jnp.int32)]


def continuous_q_apply(params: PyTree, obs: Array, action: Array) -> Array:
    return mlp_apply(params, jnp.concatenate((obs, jnp.ravel(action))))[0]

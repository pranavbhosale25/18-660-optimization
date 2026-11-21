"""Native-JAX optimizer fallbacks, using the Optax init/update protocol."""
from typing import Callable, NamedTuple
import jax
import jax.numpy as jnp
from .types import PyTree, Optimizer


class GradientTransformation(NamedTuple):
    init: Callable
    update: Callable


class AdamState(NamedTuple):
    count: jax.Array
    m: PyTree
    v: PyTree


def apply_updates(params: PyTree, updates: PyTree) -> PyTree:
    return jax.tree.map(lambda p, u: (p + u).astype(p.dtype), params, updates)


def global_norm(tree: PyTree):
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(0., jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


def sgd(learning_rate: float = 1e-3) -> GradientTransformation:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    def init(params):
        del params
        return ()

    def update(grads, state, params=None):
        del params
        return jax.tree.map(lambda g: -jnp.asarray(learning_rate, g.dtype) * g, grads), state

    return GradientTransformation(init, update)


def adam(learning_rate: float = 3e-4, b1: float = 0.9, b2: float = 0.999,
         eps: float = 1e-8) -> GradientTransformation:
    if learning_rate <= 0 or not (0 <= b1 < 1 and 0 <= b2 < 1) or eps <= 0:
        raise ValueError("Invalid Adam hyperparameters")

    def init(params):
        zeros = jax.tree.map(jnp.zeros_like, params)
        return AdamState(jnp.asarray(0, jnp.int32), zeros, zeros)

    def update(grads, state, params=None):
        del params
        count = state.count + 1
        m = jax.tree.map(lambda old, g: b1 * old + (1 - b1) * g, state.m, grads)
        v = jax.tree.map(lambda old, g: b2 * old + (1 - b2) * g**2, state.v, grads)

        def delta(mm, vv):
            bc1 = 1 - jnp.asarray(b1, mm.dtype) ** count
            bc2 = 1 - jnp.asarray(b2, mm.dtype) ** count
            return -learning_rate * (mm / bc1) / (jnp.sqrt(vv / bc2) + eps)

        return jax.tree.map(delta, m, v), AdamState(count, m, v)

    return GradientTransformation(init, update)


def clip_by_global_norm(optimizer: Optimizer, max_norm: float = 1.0
                        ) -> GradientTransformation:
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")

    def update(grads, state, params=None):
        norm = global_norm(grads)
        factor = jnp.minimum(1., max_norm / jnp.maximum(norm, jnp.asarray(1e-12, norm.dtype)))
        return optimizer.update(jax.tree.map(lambda g: g * factor, grads), state, params)

    return GradientTransformation(optimizer.init, update)

"""Small PyTree operations shared by the solver and public schedule helpers."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

PyTree = Any


@jax.custom_jvp
def safe_sqrt(x):
    """Square root on nonnegative inputs, choosing derivative zero at zero."""
    return jnp.sqrt(x)


@safe_sqrt.defjvp
def _safe_sqrt_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    y = safe_sqrt(x)
    denominator = jnp.where(x > 0, 2 * y, jnp.ones_like(y))
    return y, jnp.where(x > 0, dx / denominator, jnp.zeros_like(dx))


def tree_zeros_like(tree: PyTree) -> PyTree:
    return jax.tree.map(jnp.zeros_like, tree)


def tree_add(a: PyTree, b: PyTree) -> PyTree:
    return jax.tree.map(jnp.add, a, b)


def tree_sub(a: PyTree, b: PyTree) -> PyTree:
    return jax.tree.map(jnp.subtract, a, b)


def tree_scale(scale, tree: PyTree) -> PyTree:
    return jax.tree.map(lambda x: jnp.asarray(scale, dtype=x.dtype) * x, tree)


def tree_dot(a: PyTree, b: PyTree):
    """Real Euclidean inner product over every corresponding array leaf."""
    products = jax.tree.leaves(jax.tree.map(lambda x, y: jnp.vdot(x, y).real, a, b))
    return sum(products[1:], products[0]) if products else jnp.asarray(0.0)


def tree_norm(tree: PyTree):
    """Global Euclidean norm, with the zero subgradient chosen at the origin."""
    return safe_sqrt(tree_dot(tree, tree))


def masked_mean(values: PyTree, mask) -> PyTree:
    """Mean over active entries on axis zero, independently for each leaf.

    ``mask`` is a one-dimensional boolean array. Integer inputs are converted
    to floating point; floating inputs retain their dtype. At least one entry
    should be active (an all-false mask returns zeros). This masks the reduction,
    not arbitrary invalid computations that happened before this function.
    """
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    if mask.ndim != 1:
        raise ValueError("mask must be a one-dimensional boolean array")

    def reduce(x):
        x = jnp.asarray(x)
        if x.ndim < 1 or x.shape[0] != mask.shape[0]:
            raise ValueError("every values leaf needs a leading batch axis matching mask")
        dtype = x.dtype if jnp.issubdtype(x.dtype, jnp.inexact) else jnp.result_type(float)
        x = x.astype(dtype)
        weights = mask.reshape((mask.shape[0],) + (1,) * (x.ndim - 1))
        count = jnp.maximum(jnp.sum(mask, dtype=jnp.int32), 1).astype(dtype)
        return jnp.sum(jnp.where(weights, x, jnp.zeros_like(x)), axis=0) / count

    return jax.tree.map(reduce, values)


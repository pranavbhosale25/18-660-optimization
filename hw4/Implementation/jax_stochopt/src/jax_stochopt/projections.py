"""Euclidean projections implemented with native JAX operations.

Every factory returns ``projection(params, context=None)``. The optional
context is ignored by these fixed-set projections and forwarded by
``sequential``. A user-defined projection may inspect the solver's context.
Projections are differentiable almost everywhere; active-set boundaries and
rank changes are generally nonsmooth. No straight-through gradient is used.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np


Projection = Callable[[Any, Any], Any]


def _is_scalar(value: Any) -> bool:
    return isinstance(value, numbers.Number) or getattr(value, "ndim", -1) == 0


def _concrete(value: Any) -> np.ndarray | None:
    """Return a host value only for constructor-time input validation."""

    if isinstance(value, jax.core.Tracer):
        return None
    try:
        return np.asarray(value)
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
        return None


def _validate_total(value: Any, name: str) -> None:
    array = jnp.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real.")
    concrete = _concrete(value)
    if concrete is not None:
        scalar = float(concrete)
        if not math.isfinite(scalar) or scalar < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")


def _float_array(value: Any) -> Any:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"Projection parameters must be real floating-point arrays, got {array.dtype}.")
    return array


def _arguments_for_leaves(value: Any, treedef: Any, count: int, name: str) -> list[Any]:
    if value is None or _is_scalar(value):
        return [value] * count
    # A root array can have array-like bounds or a center without their list
    # syntax being interpreted as a parameter PyTree.
    if treedef == jax.tree_util.tree_structure(0.0):
        return [value]
    leaves, other_treedef = jax.tree_util.tree_flatten(
        value, is_leaf=lambda x: x is None
    )
    if other_treedef != treedef:
        raise ValueError(f"{name} must be scalar or have the same PyTree structure as params.")
    return leaves


def _cast_broadcast(value: Any, array: Any, name: str) -> Any:
    result = jnp.asarray(value, dtype=array.dtype)
    if jnp.broadcast_shapes(result.shape, array.shape) != array.shape:
        raise ValueError(
            f"{name} shape {result.shape} would change parameter shape {array.shape}."
        )
    return result


def _validate_box_pair(lower: Any, upper: Any) -> None:
    for bound in (lower, upper):
        if bound is None:
            continue
        value = _concrete(bound)
        if value is not None:
            if np.iscomplexobj(value):
                raise TypeError("Box bounds must be real.")
            if np.any(np.isnan(value)):
                raise ValueError("Box bounds must not contain NaN.")
    if lower is not None and upper is not None:
        lo, hi = _concrete(lower), _concrete(upper)
        if lo is not None and hi is not None and np.any(lo > hi):
            raise ValueError("Each lower box bound must be <= its upper bound.")


def box(lower: Any = None, upper: Any = None) -> Projection:
    """Project onto coordinate bounds, preserving each leaf's dtype and shape.

    Each bound may be a scalar, ``None`` (unbounded), or a matching PyTree.
    Matching leaves may be scalars, broadcastable arrays, or ``None``. With a
    single array parameter, bounds can also be array-like sequences. Bounds
    must satisfy ``lower <= upper`` and contain no NaNs. Concrete constructor
    inputs are validated where their structures can be paired; constraints
    on traced bounds remain the caller's responsibility.
    """

    lo_leaves, lo_tree = jax.tree_util.tree_flatten(lower, is_leaf=lambda x: x is None)
    hi_leaves, hi_tree = jax.tree_util.tree_flatten(upper, is_leaf=lambda x: x is None)
    if lower is None or _is_scalar(lower):
        for hi in hi_leaves:
            _validate_box_pair(lower, hi)
    elif upper is None or _is_scalar(upper):
        for lo in lo_leaves:
            _validate_box_pair(lo, upper)
    elif lo_tree == hi_tree:
        for lo, hi in zip(lo_leaves, hi_leaves):
            _validate_box_pair(lo, hi)

    def project(params: Any, context: Any = None) -> Any:
        leaves, treedef = jax.tree_util.tree_flatten(params)
        lows = _arguments_for_leaves(lower, treedef, len(leaves), "lower")
        highs = _arguments_for_leaves(upper, treedef, len(leaves), "upper")
        projected = []
        for value, lo, hi in zip(leaves, lows, highs):
            array = _float_array(value)
            low = None if lo is None else _cast_broadcast(lo, array, "lower")
            high = None if hi is None else _cast_broadcast(hi, array, "upper")
            # clip permits either bound to be absent, but an entirely unbounded
            # set should also be a valid identity projection.
            projected.append(array if low is None and high is None else jnp.clip(array, low, high))
        return jax.tree_util.tree_unflatten(treedef, projected)

    return project


def l2_ball(radius: Any, center: Any = None) -> Projection:
    """Project onto one Euclidean ball across an entire parameter PyTree.

    The constraint is ``sum(||leaf - center_leaf||**2) <= radius**2``. ``center``
    may be a scalar, a matching PyTree, or ``None`` for zero. This is one global
    ball, not one ball per leaf. ``radius`` must be finite and nonnegative;
    traced radii are accepted with that same caller-enforced precondition.
    Radius zero is supported. The derivative at the center is finite, including
    for the zero-radius singleton set.
    """

    _validate_total(radius, "radius")

    def project(params: Any, context: Any = None) -> Any:
        leaves, treedef = jax.tree_util.tree_flatten(params)
        if not leaves:
            raise ValueError("l2_ball requires at least one parameter leaf.")
        center_leaves = _arguments_for_leaves(center, treedef, len(leaves), "center")
        centers, deltas = [], []
        for value, center_value in zip(leaves, center_leaves):
            array = _float_array(value)
            origin = jnp.zeros_like(array) if center_value is None else _cast_broadcast(center_value, array, "center")
            centers.append(origin)
            deltas.append(array - origin)
        squared_norm = sum(jnp.sum(delta * delta) for delta in deltas)
        # sqrt(0) has an infinite derivative even in an unselected where branch.
        # Replace its input *before* sqrt so reverse-mode derivatives stay finite.
        nonzero = squared_norm > 0
        safe_norm = jnp.sqrt(jnp.where(nonzero, squared_norm, jnp.ones_like(squared_norm)))
        total = jnp.asarray(radius, dtype=squared_norm.dtype)
        ordinary_scale = jnp.minimum(jnp.ones_like(total), total / safe_norm)
        center_scale = jnp.where(total > 0, jnp.ones_like(total), jnp.zeros_like(total))
        scale = jnp.where(nonzero, ordinary_scale, center_scale)
        projected = [
            origin + delta * scale.astype(delta.dtype)
            for origin, delta in zip(centers, deltas)
        ]
        return jax.tree_util.tree_unflatten(treedef, projected)

    return project


def simplex(total: Any = 1.0, axis: int = -1) -> Projection:
    """Project onto ``x >= 0, sum(x, axis) == total`` by sorting.

    Each vector along ``axis`` is projected independently. For a parameter
    PyTree this is applied independently to every array leaf, giving a Cartesian
    product of simplices, not a single simplex across all leaves. Each leaf
    must have at least one dimension and a nonempty selected axis. ``total``
    is a finite nonnegative scalar; traced totals must satisfy that precondition.
    The sorting/active-set projection has its usual piecewise derivative.
    """

    _validate_total(total, "total")
    if not isinstance(axis, numbers.Integral):
        raise TypeError("simplex axis must be a static integer.")

    def project_leaf(value: Any) -> Any:
        array = _float_array(value)
        if array.ndim == 0 or not -array.ndim <= axis < array.ndim:
            raise ValueError(f"simplex axis {axis} is invalid for shape {array.shape}.")
        moved = jnp.moveaxis(array, axis, -1)
        size = moved.shape[-1]
        if size == 0:
            raise ValueError("The simplex projection axis must be nonempty.")
        mass = jnp.asarray(total, dtype=array.dtype)
        sorted_values = jnp.flip(jnp.sort(moved, axis=-1), axis=-1)
        indices = jnp.arange(1, size + 1, dtype=array.dtype)
        cumulative = jnp.cumsum(sorted_values, axis=-1) - mass
        active = sorted_values - cumulative / indices > 0
        rho = jnp.maximum(jnp.sum(active, axis=-1, keepdims=True) - 1, 0)
        threshold = jnp.take_along_axis(cumulative, rho, axis=-1) / (rho + 1).astype(array.dtype)
        result = jnp.maximum(moved - threshold, jnp.zeros_like(moved))
        # A zero-mass simplex is the singleton zero even in the presence of
        # rounding in the threshold computation.
        result = jnp.where(mass > 0, result, jnp.zeros_like(result))
        return jnp.moveaxis(result, -1, axis)

    def project(params: Any, context: Any = None) -> Any:
        return jax.tree_util.tree_map(project_leaf, params)

    return project


def affine(matrix: Any, target: Any, *, rtol: Any = None) -> Projection:
    """Project a one-dimensional array onto ``matrix @ x == target``.

    The matrix has shape ``(m, n)`` and target has shape ``(m,)``. A JAX
    pseudoinverse is computed once when constructing the projection, and the
    returned function evaluates ``x - pinv(matrix) @ (matrix @ x - target)``.
    Consistent rank-deficient systems are supported. Clearly inconsistent
    concrete systems are rejected; traced systems must be consistent. An
    inconsistent system does not define a nonempty affine constraint set.

    ``rtol`` is passed to ``jax.numpy.linalg.pinv``. Supply a floating matrix in
    the desired precision, and enable JAX x64 before construction for float64
    accuracy. Outputs preserve the parameter dtype. Differentiation with
    respect to matrix/target is possible when constructed inside a transformed
    function, subject to pseudoinverse smoothness at a fixed matrix rank.
    """

    a = jnp.asarray(matrix)
    b = jnp.asarray(target)
    if a.ndim != 2 or b.ndim != 1 or a.shape[0] != b.shape[0]:
        raise ValueError("affine requires matrix shape (m, n) and target shape (m,).")
    if jnp.issubdtype(a.dtype, jnp.complexfloating) or jnp.issubdtype(b.dtype, jnp.complexfloating):
        raise TypeError("affine matrix and target must be real.")
    if not jnp.issubdtype(a.dtype, jnp.floating):
        a = a.astype(jnp.result_type(a, 1.0))
    b = b.astype(a.dtype)
    a_pinv = jnp.linalg.pinv(a, rtol=rtol)

    concrete_a, concrete_b, concrete_pinv = _concrete(a), _concrete(b), _concrete(a_pinv)
    if concrete_a is not None and concrete_b is not None and concrete_pinv is not None:
        if not np.all(np.isfinite(concrete_a)) or not np.all(np.isfinite(concrete_b)):
            raise ValueError("affine matrix and target must be finite.")
        residual = concrete_a @ (concrete_pinv @ concrete_b) - concrete_b
        tolerance = 32 * np.finfo(concrete_a.dtype).eps * max(1, *a.shape) * (1 + np.linalg.norm(concrete_b))
        if np.linalg.norm(residual) > tolerance:
            raise ValueError("The affine system is inconsistent at the chosen pseudoinverse tolerance.")

    def project(params: Any, context: Any = None) -> Any:
        x = _float_array(params)
        if x.ndim != 1 or x.shape[0] != a.shape[1]:
            raise ValueError(f"affine expects a parameter array of shape ({a.shape[1]},), got {x.shape}.")
        local_a = a.astype(x.dtype)
        local_b = b.astype(x.dtype)
        local_pinv = a_pinv.astype(x.dtype)
        return x - local_pinv @ (local_a @ x - local_b)

    return project


def sequential(*projections: Projection) -> Projection:
    """Apply projections in order, forwarding the same context to each.

    This is a composition of maps, **not** the Euclidean projection onto an
    intersection. A later map can violate an earlier constraint. Use an exact
    joint projector, Dykstra's algorithm, or a suitably converged alternating
    method when intersection feasibility is required. With no inputs this
    returns the identity map.
    """

    if not all(callable(projection) for projection in projections):
        raise TypeError("Every argument to sequential must be callable.")

    def project(params: Any, context: Any = None) -> Any:
        result = params
        for projection in projections:
            result = projection(result, context)
        return result

    return project


__all__ = ["Projection", "affine", "box", "l2_ball", "sequential", "simplex"]

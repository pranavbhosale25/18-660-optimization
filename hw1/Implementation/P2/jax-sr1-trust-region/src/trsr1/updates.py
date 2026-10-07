"""Safeguarded symmetric-rank-one Hessian updates."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import SR1UpdateResult

Array = jax.Array


def safeguarded_sr1_update(
    matrix: Array,
    step: Array,
    gradient_difference: Array,
    *,
    skip_tolerance: float,
    symmetrize: bool = True,
) -> SR1UpdateResult:
    """Apply the Nocedal--Wright safeguarded direct SR1 update.

    The update

    ``B+ = B + u u.T / (u.T s)``, where ``u = y - B s``,

    is applied only when

    ``abs(u.T s) >= r * ||s|| * ||u||``.

    The function is pure and compatible with ``jit`` and ``vmap``.
    """

    matrix_step = matrix @ step
    residual = gradient_difference - matrix_step
    denominator = jnp.dot(residual, step)
    step_norm = jnp.linalg.norm(step)
    residual_norm = jnp.linalg.norm(residual)
    threshold = (
        jnp.asarray(skip_tolerance, dtype=matrix.dtype)
        * step_norm
        * residual_norm
    )
    eps = jnp.asarray(jnp.finfo(matrix.dtype).eps, dtype=matrix.dtype)
    numerical_residual_floor = 32.0 * eps * jnp.maximum(
        jnp.asarray(1.0, dtype=matrix.dtype),
        jnp.maximum(jnp.linalg.norm(gradient_difference), jnp.linalg.norm(matrix_step)),
    )

    finite_inputs = (
        jnp.all(jnp.isfinite(matrix))
        & jnp.all(jnp.isfinite(step))
        & jnp.all(jnp.isfinite(gradient_difference))
        & jnp.isfinite(denominator)
        & jnp.isfinite(step_norm)
        & jnp.isfinite(residual_norm)
    )
    nondegenerate = (
        (step_norm > 0.0)
        & (residual_norm > numerical_residual_floor)
        & (jnp.abs(denominator) >= threshold)
        & (denominator != 0.0)
    )
    should_apply = finite_inputs & nondegenerate

    safe_denominator = jnp.where(should_apply, denominator, 1.0)
    candidate = matrix + jnp.outer(residual, residual) / safe_denominator
    if symmetrize:
        candidate = 0.5 * (candidate + jnp.swapaxes(candidate, -1, -2))

    candidate_is_finite = jnp.all(jnp.isfinite(candidate))
    applied = should_apply & candidate_is_finite
    updated_matrix = jnp.where(applied, candidate, matrix)
    secant_residual_norm = jnp.linalg.norm(
        updated_matrix @ step - gradient_difference
    )

    return SR1UpdateResult(
        matrix=updated_matrix,
        applied=applied,
        denominator=denominator,
        threshold=threshold,
        residual_norm=residual_norm,
        secant_residual_norm=secant_residual_norm,
    )

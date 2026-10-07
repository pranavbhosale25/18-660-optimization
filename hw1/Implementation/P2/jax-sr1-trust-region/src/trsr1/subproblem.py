"""Simple textbook step for the quadratic trust-region model.

The routine uses the dogleg construction when the current SR1 matrix is
positive definite.  Because an SR1 approximation may be indefinite, the
closed-form Cauchy point is used as a safe fallback.  The implementation uses
one matrix-vector product, one Cholesky factorization, and scalar algebra; it
contains no eigendecomposition or iterative inner solve.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from .types import SubproblemResult

Array = jax.Array


def _cauchy_point(gradient: Array, matrix: Array, radius: Array) -> Array:
    """Minimize the model along the negative-gradient direction."""

    dtype = gradient.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)

    gradient_norm = jnp.linalg.norm(gradient)
    gradient_norm_squared = jnp.dot(gradient, gradient)
    safe_gradient_norm = jnp.where(gradient_norm > zero, gradient_norm, one)

    boundary_scale = radius / safe_gradient_norm
    directional_curvature = jnp.dot(gradient, matrix @ gradient)
    safe_curvature = jnp.where(
        directional_curvature > zero,
        directional_curvature,
        one,
    )
    line_minimizer_scale = gradient_norm_squared / safe_curvature

    scale = jnp.where(
        directional_curvature > zero,
        jnp.minimum(line_minimizer_scale, boundary_scale),
        boundary_scale,
    )
    scale = jnp.where(gradient_norm > zero, scale, zero)
    return -scale * gradient


def solve_quadratic_subproblem(
    gradient: Array,
    matrix: Array,
    radius: Array,
) -> SubproblemResult:
    """Return a dogleg step, with a Cauchy fallback for indefinite models.

    The quadratic model is

    ``q(s) = gradient.T @ s + 0.5 * s.T @ matrix @ s``

    subject to ``||s|| <= radius``.  The returned step is an approximate
    solution suitable for the outer trust-region iteration.

    Hint: a Cholesky factorization can be used to determine whether the
    symmetric model matrix is positive definite. Under JAX transformations,
    inspect whether the factor is finite instead of relying on a Python
    exception.
    """

    dtype = gradient.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    two = jnp.asarray(2.0, dtype=dtype)

    # Construct a harmless step even when the supplied radius is invalid.  The
    # original radius is checked below when the result is marked valid.
    safe_radius = jnp.maximum(radius, zero)
    cauchy_step = _cauchy_point(gradient, matrix, safe_radius)

    # Under JAX, Cholesky returns non-finite entries for an indefinite matrix
    # instead of raising an exception that can be caught in Python.
    factor = jnp.linalg.cholesky(matrix)
    is_positive_definite = jnp.all(jnp.isfinite(factor))

    def positive_definite_step(_: None) -> Array:
        # Solve matrix @ step = -gradient using the Cholesky factor.
        intermediate = solve_triangular(factor, -gradient, lower=True)
        full_step = solve_triangular(
            factor.T,
            intermediate,
            lower=False,
        )

        def boundary_step(_: None) -> Array:
            cauchy_norm = jnp.linalg.norm(cauchy_step)

            def dogleg_intersection(_: None) -> Array:
                # Find the point on the segment from the Cauchy point to the
                # full step whose norm equals the trust-region radius.
                direction = full_step - cauchy_step
                quadratic = jnp.dot(direction, direction)
                linear = two * jnp.dot(cauchy_step, direction)
                constant = jnp.dot(cauchy_step, cauchy_step) - safe_radius**2

                discriminant = jnp.maximum(
                    linear**2 - 4.0 * quadratic * constant,
                    zero,
                )
                safe_quadratic = jnp.where(
                    quadratic > zero,
                    quadratic,
                    one,
                )
                fraction = (
                    -linear + jnp.sqrt(discriminant)
                ) / (two * safe_quadratic)
                fraction = jnp.clip(fraction, zero, one)
                return cauchy_step + fraction * direction

            # The Cauchy point may already lie on the boundary.  Otherwise,
            # continue along the second segment of the dogleg path.
            return jax.lax.cond(
                cauchy_norm >= safe_radius,
                lambda _: cauchy_step,
                dogleg_intersection,
                operand=None,
            )

        return jax.lax.cond(
            jnp.linalg.norm(full_step) <= safe_radius,
            lambda _: full_step,
            boundary_step,
            operand=None,
        )

    # Indefinite SR1 matrices are expected.  Their safe fallback is the Cauchy
    # point, rather than an invalid subproblem result.
    step = jax.lax.cond(
        is_positive_definite,
        positive_definite_step,
        lambda _: cauchy_step,
        operand=None,
    )

    step_norm = jnp.linalg.norm(step)
    predicted_reduction = -(
        jnp.dot(gradient, step)
        + 0.5 * jnp.dot(step, matrix @ step)
    )

    epsilon = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
    boundary_tolerance = 32.0 * epsilon * jnp.maximum(one, safe_radius)
    finite_inputs = (
        jnp.all(jnp.isfinite(gradient))
        & jnp.all(jnp.isfinite(matrix))
        & jnp.isfinite(radius)
    )
    valid = (
        finite_inputs
        & (radius >= zero)
        & jnp.all(jnp.isfinite(step))
        & jnp.isfinite(predicted_reduction)
        & (step_norm <= safe_radius + boundary_tolerance)
    )
    hit_boundary = (
        valid
        & (safe_radius > zero)
        & (jnp.abs(step_norm - safe_radius) <= boundary_tolerance)
    )

    return SubproblemResult(
        step=step,
        predicted_reduction=predicted_reduction,
        step_norm=step_norm,
        hit_boundary=hit_boundary,
        valid=valid,
    )

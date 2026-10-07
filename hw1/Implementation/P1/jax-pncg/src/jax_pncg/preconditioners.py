from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


Preconditioner = Callable[..., jax.Array]


def identity_preconditioner(
    x: jax.Array, gradient: jax.Array, *args: Any
) -> jax.Array:
    """Return ``gradient`` unchanged."""

    # STUDENT TODO: Implement the identity preconditioner.
    # raise NotImplementedError("STUDENT TODO: implement identity_preconditioner")
    return gradient


def diagonal_preconditioner(
    diagonal: Any,
    *,
    damping: float = 0.0,
    absolute: bool = False,
) -> Preconditioner:
    """Create ``z = M^{-1} g`` for a fixed diagonal ``M``.

    Parameters
    ----------
    diagonal:
        Diagonal entries of ``M``.  The value is converted to the gradient's
        dtype inside the traced function, so one factory works in both 32- and
        64-bit modes.
    damping:
        Positive floor added before division.
    absolute:
        Use ``abs(diagonal)``.  This is useful for a Jacobi-like curvature
        scale when an objective's Hessian diagonal may be indefinite, but a
        fixed positive diagonal is preferable when available.
    """

    if damping < 0.0:
        raise ValueError("damping must be nonnegative")

    def apply(x: jax.Array, gradient: jax.Array, *args: Any) -> jax.Array:
        # STUDENT TODO: Apply the fixed diagonal preconditioner, including
        # the documented dtype, absolute-value, and damping behavior.
        # raise NotImplementedError("STUDENT TODO: implement diagonal_preconditioner")
        diag = jnp.asarray(diagonal, dtype=gradient.dtype)
        if absolute:
            diag = jnp.abs(diag)
        denominator = jnp.maximum(diag, jnp.asarray(damping, dtype=gradient.dtype))
        return gradient / denominator

    return apply


def jacobi_preconditioner(
    hessian_diagonal: Callable[..., jax.Array],
    *,
    damping: float = 1.0e-8,
    absolute: bool = True,
    max_inverse: float | None = None,
) -> Preconditioner:
    """Create an x-dependent Jacobi preconditioner.

    ``hessian_diagonal(x, *args)`` should return a vector.  Because this varies
    with ``x``, the method is a flexible/preconditioned PR heuristic rather
    than the fixed-SPD change-of-variables case used in the classical theory.
    """

    if damping <= 0.0:
        raise ValueError("damping must be positive")
    if max_inverse is not None and max_inverse <= 0.0:
        raise ValueError("max_inverse must be positive when supplied")

    def apply(x: jax.Array, gradient: jax.Array, *args: Any) -> jax.Array:
        # STUDENT TODO: Apply the position-dependent Jacobi preconditioner,
        # including the documented safeguards.
        # raise NotImplementedError("STUDENT TODO: implement jacobi_preconditioner")
        diag = hessian_diagonal(x, *args)
        diag = jnp.asarray(diag, dtype=gradient.dtype)
        if absolute:
            diag = jnp.abs(diag)
        denominator = jnp.maximum(diag, jnp.asarray(damping, dtype=gradient.dtype))
        inverse = 1.0 / denominator

        if max_inverse is not None:
            inverse = jnp.minimum(inverse, jnp.asarray(max_inverse, dtype=gradient.dtype))

        return gradient * inverse


    return apply

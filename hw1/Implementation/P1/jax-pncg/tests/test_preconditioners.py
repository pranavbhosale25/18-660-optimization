from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_pncg import (
    diagonal_preconditioner,
    identity_preconditioner,
    jacobi_preconditioner,
)


def test_identity_preconditioner_returns_the_gradient() -> None:
    x = jnp.asarray([3.0, -4.0], dtype=jnp.float64)
    gradient = jnp.asarray([2.0, -5.0], dtype=jnp.float64)

    result = identity_preconditioner(x, gradient, jnp.asarray(7.0))

    np.testing.assert_array_equal(result, gradient)


def test_fixed_diagonal_preconditioner_applies_options() -> None:
    x = jnp.zeros((3,), dtype=jnp.float64)
    gradient = jnp.asarray([8.0, -3.0, 2.0], dtype=jnp.float64)
    preconditioner = diagonal_preconditioner(
        [-2.0, 0.25, 4.0],
        damping=0.5,
        absolute=True,
    )

    result = preconditioner(x, gradient)

    np.testing.assert_allclose(
        result,
        np.asarray([4.0, -6.0, 0.5]),
        rtol=0.0,
        atol=0.0,
    )


def test_jacobi_preconditioner_uses_current_point_and_options() -> None:
    def hessian_diagonal(x, shift):
        return x + shift

    x = jnp.asarray([-3.0, 0.0, 4.0], dtype=jnp.float64)
    gradient = jnp.asarray([6.0, 2.0, -8.0], dtype=jnp.float64)
    shift = jnp.asarray([1.0, 0.25, 0.0], dtype=jnp.float64)
    preconditioner = jacobi_preconditioner(
        hessian_diagonal,
        damping=0.5,
        absolute=True,
        max_inverse=1.0,
    )

    result = preconditioner(x, gradient, shift)

    np.testing.assert_allclose(
        result,
        np.asarray([3.0, 2.0, -2.0]),
        rtol=0.0,
        atol=0.0,
    )

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import trsr1
from trsr1.problems import rosenbrock

jax.config.update("jax_enable_x64", True)


def test_vmap_solver_handles_multiple_rosenbrock_starts() -> None:
    starts = jnp.asarray(
        [[-1.2, 1.0], [2.0, 2.0], [-1.0, 2.0]],
        dtype=jnp.float64,
    )
    solver = trsr1.make_batched_solver(
        rosenbrock,
        trsr1.SR1Options(
            max_iterations=350,
            gradient_tolerance=1.0e-7,
            store_history=False,
        ),
    )
    result = solver(starts)

    assert result.x.shape == starts.shape
    assert np.all(np.asarray(result.success))
    np.testing.assert_allclose(result.x, np.ones_like(starts), atol=2.0e-5)


def test_vmap_solver_maps_parameterized_quadratics() -> None:
    def objective(x, matrix, linear):
        return 0.5 * x @ matrix @ x + linear @ x

    matrices = jnp.asarray(
        [
            [[2.0, 0.0], [0.0, 5.0]],
            [[4.0, 1.0], [1.0, 3.0]],
        ],
        dtype=jnp.float64,
    )
    linear = jnp.asarray(
        [[1.0, -2.0], [-3.0, 1.0]], dtype=jnp.float64
    )
    starts = jnp.zeros_like(linear)
    solver = trsr1.make_batched_solver(
        objective,
        trsr1.SR1Options(
            max_iterations=30,
            gradient_tolerance=1.0e-10,
            initial_radius=10.0,
            store_history=False,
        ),
        arg_in_axes=(0, 0),
    )
    result = solver(starts, matrices, linear)
    expected = -jax.vmap(jnp.linalg.solve)(matrices, linear)

    np.testing.assert_allclose(result.x, expected, rtol=1.0e-8, atol=1.0e-8)
    assert np.all(np.asarray(result.success))


def test_batched_explicit_initial_matrices() -> None:
    def objective(x, matrix):
        return 0.5 * x @ matrix @ x - jnp.sum(x)

    matrices = jnp.asarray(
        [jnp.diag(jnp.asarray([2.0, 3.0])), jnp.diag(jnp.asarray([4.0, 5.0]))],
        dtype=jnp.float64,
    )
    starts = jnp.zeros((2, 2), dtype=jnp.float64)
    solver = trsr1.make_batched_solver_with_matrix(
        objective,
        trsr1.SR1Options(
            max_iterations=5,
            initial_radius=10.0,
            gradient_tolerance=1.0e-12,
            store_history=False,
        ),
        arg_in_axes=(0,),
        matrix_in_axes=0,
    )
    result = solver(starts, matrices, matrices)
    expected = jax.vmap(jnp.linalg.solve)(matrices, jnp.ones_like(starts))

    np.testing.assert_allclose(result.x, expected, atol=1.0e-12)
    assert np.all(np.asarray(result.success))

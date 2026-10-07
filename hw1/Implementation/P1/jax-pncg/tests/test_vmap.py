from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_pncg import PNCGOptions, make_batched_solver
from jax_pncg.test_problems import diagonal_quadratic, rosenbrock


def test_vmap_multiple_rosenbrock_starts() -> None:
    starts = jnp.asarray(
        [
            [-1.2, 1.0],
            [-1.0, 2.0],
            [2.0, 2.0],
            [0.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    solver = make_batched_solver(
        rosenbrock,
        options=PNCGOptions(maxiter=1000, gtol=1.0e-7),
    )
    result = solver(starts)

    assert result.x.shape == starts.shape
    assert np.all(np.asarray(result.success))
    np.testing.assert_allclose(result.x, np.ones_like(starts), atol=3.0e-6, rtol=0.0)


def test_vmap_batches_objective_arguments_and_preconditioner() -> None:
    starts = jnp.asarray([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=jnp.float64)
    diagonals = jnp.asarray(
        [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]], dtype=jnp.float64
    )

    def preconditioner(x, gradient, diagonal):
        del x
        return gradient / diagonal

    solver = make_batched_solver(
        diagonal_quadratic,
        preconditioner=preconditioner,
        options=PNCGOptions(maxiter=5, gtol=1.0e-10),
        arg_in_axes=(0,),
    )
    result = solver(starts, diagonals)

    assert np.all(np.asarray(result.success))
    np.testing.assert_allclose(result.x, np.zeros_like(starts), atol=1.0e-13, rtol=0.0)
    np.testing.assert_array_equal(np.asarray(result.nit), np.ones(2, dtype=np.int32))

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import trsr1
from trsr1.problems import rosenbrock

jax.config.update("jax_enable_x64", True)


def test_exact_initial_matrix_solves_spd_quadratic_in_one_iteration() -> None:
    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    linear = jnp.asarray([-1.0, 2.0], dtype=jnp.float64)

    def objective(x):
        return 0.5 * x @ matrix @ x + linear @ x

    solver = trsr1.make_solver_with_matrix(
        objective,
        trsr1.SR1Options(
            max_iterations=10,
            gradient_tolerance=1.0e-12,
            initial_radius=10.0,
            store_history=True,
        ),
    )
    result = solver(jnp.zeros(2, dtype=jnp.float64), matrix)
    expected = -jnp.linalg.solve(matrix, linear)

    np.testing.assert_allclose(result.x, expected, atol=1.0e-12)
    assert bool(result.success)
    assert int(result.iterations) == 1
    assert int(result.accepted_steps) == 1
    assert int(result.sr1_updates) == 0
    assert int(result.sr1_skips) == 1
    assert int(jnp.sum(result.history.valid)) == 1


def test_rejected_trial_still_updates_sr1_matrix() -> None:
    start = jnp.asarray([-1.2, 1.0], dtype=jnp.float64)
    initial_matrix = jnp.eye(2, dtype=jnp.float64)
    solver = trsr1.make_solver_with_matrix(
        rosenbrock,
        trsr1.SR1Options(
            max_iterations=1,
            gradient_tolerance=1.0e-12,
            store_history=True,
        ),
    )
    result = solver(start, initial_matrix)

    assert not bool(result.history.accepted[0])
    assert bool(result.history.sr1_updated[0])
    np.testing.assert_array_equal(result.x, start)
    assert not np.allclose(result.hessian_approximation, initial_matrix)
    assert int(result.sr1_updates) == 1
    assert int(result.rejected_steps) == 1


def test_history_mask_and_no_history_path_agree() -> None:
    start = jnp.asarray([-1.2, 1.0], dtype=jnp.float64)
    common = dict(
        max_iterations=300,
        gradient_tolerance=1.0e-8,
    )
    with_history = trsr1.make_solver(
        rosenbrock, trsr1.SR1Options(**common, store_history=True)
    )(start)
    without_history = trsr1.make_solver(
        rosenbrock, trsr1.SR1Options(**common, store_history=False)
    )(start)

    np.testing.assert_allclose(with_history.x, without_history.x, atol=1.0e-10)
    np.testing.assert_allclose(with_history.fun, without_history.fun, atol=1.0e-16)
    assert int(jnp.sum(with_history.history.valid)) == int(
        with_history.iterations
    )
    assert without_history.history.objective.shape == (0,)


def test_solver_factory_returns_lowerable_jitted_callable() -> None:
    solver = trsr1.make_solver(
        rosenbrock,
        trsr1.SR1Options(max_iterations=5, store_history=False),
    )
    assert hasattr(solver, "lower")
    lowered = solver.lower(jnp.asarray([-1.2, 1.0], dtype=jnp.float64))
    assert lowered is not None

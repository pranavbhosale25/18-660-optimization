from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import trsr1

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("problem", trsr1.standard_problems(), ids=lambda p: p.name)
def test_standard_problem_suite(problem) -> None:
    solver = trsr1.make_solver(
        problem.fun,
        trsr1.SR1Options(
            max_iterations=problem.max_iterations,
            gradient_tolerance=problem.gradient_tolerance,
            store_history=False,
        ),
    )
    result = solver(problem.initial_point(jnp.float64))

    assert bool(result.success), trsr1.status_message(result.status)
    assert float(result.grad_norm) <= 1.05 * problem.gradient_tolerance
    assert np.isclose(
        float(result.fun),
        problem.known_minimum,
        rtol=1.0e-8,
        atol=1.0e-8,
    )

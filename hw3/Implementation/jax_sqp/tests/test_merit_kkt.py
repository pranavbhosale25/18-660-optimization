"""Independent checks of the assignment's nonsmooth penalties and KKT signs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_sqp.merit import violation
from jax_sqp.solver import (
    kkt_residual,
    merit_directional_derivative,
    merit_eval,
)


@pytest.mark.parametrize("neq", [0, 3, 6])
def test_one_sided_merit_slope_against_finite_differences(neq):
    # Includes both signs at zero: autodiff's arbitrary subgradient there
    # would not, in general, equal the required right directional derivative.
    c = jnp.array([-2.0, 0.0, 3.0, -1.0, 0.0, 0.0])
    ap = jnp.array([0.4, -0.7, -0.2, 0.8, 0.6, -0.9])
    g, p = jnp.array([1.5, -0.2]), jnp.array([-0.3, 0.4])
    penalty = 2.5
    slope = jax.jit(
        lambda c, ap: merit_directional_derivative(g, p, c, ap, neq, penalty)
    )(c, ap)
    step = 1e-6
    numerical = (
        merit_eval(1.0 + step * jnp.dot(g, p), c + step * ap, neq, penalty)
        - merit_eval(1.0, c, neq, penalty)
    ) / step
    np.testing.assert_allclose(slope, numerical, atol=5e-9, rtol=5e-9)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("neq,expected", [(0, 3.0), (2, 5.0), (4, 9.0)])
def test_total_violation_order_and_dtype(dtype, neq, expected):
    c = jnp.array([-2.0, 0.0, 3.0, -4.0], dtype=dtype)
    actual = jax.jit(lambda values: violation(values, neq))(c)
    assert actual.shape == ()
    assert actual.dtype == dtype
    np.testing.assert_allclose(actual, expected)
    empty = violation(jnp.zeros(0, dtype=dtype), 0)
    assert empty.dtype == dtype
    assert float(empty) == 0.0


def test_kkt_residuals_use_correct_multiplier_signs_and_norms():
    matrix = jnp.array([[1.0, -1.0], [0.0, 1.0], [1.0, 0.0]])
    gradient = jnp.array([1.0, 2.0])
    constraints = jnp.array([2.0, 3.0, -4.0])
    multipliers = jnp.array([-5.0, -2.0, 0.5])
    residuals = jax.jit(
        lambda g, c, lam: kkt_residual(g, c, lambda v: matrix.T @ v, lam, 1)
    )(gradient, constraints, multipliers)
    # Equality multiplier -5 is allowed; its product with c is excluded
    # from complementarity. The inequality multiplier -2 violates duality.
    np.testing.assert_allclose(residuals, [3.0, 5.0, 2.0, 6.0, 6.0])


def test_unconstrained_penalties_and_kkt_under_vmap():
    empty = jnp.zeros(0)

    def measures(g):
        residuals = kkt_residual(g, empty, lambda _: jnp.zeros_like(g), empty, 0)
        slope = merit_directional_derivative(g, -g, empty, empty, 0, 10.0)
        return residuals, slope

    residuals, slopes = jax.jit(jax.vmap(measures))(jnp.array([[0.0, 0.0], [3.0, -4.0]]))
    np.testing.assert_allclose(residuals.error, [0.0, 4.0])
    np.testing.assert_allclose(residuals.primal, 0.0)
    np.testing.assert_allclose(residuals.dual, 0.0)
    np.testing.assert_allclose(residuals.complementarity, 0.0)
    np.testing.assert_allclose(slopes, [0.0, -25.0])


def test_saved_merit_and_violation_describe_actual_nonlinear_iterates():
    from jax_sqp import HISTORY_COLUMNS
    from jax_sqp.problems import hs71

    problem = hs71()
    result = problem.solver().solve(problem.x0)
    assert result.success
    history = np.asarray(result.history)[:int(result.iterations) + 1]
    columns = {name: i for i, name in enumerate(HISTORY_COLUMNS)}
    values = history[:, columns["constraint_violation_l1"]]
    np.testing.assert_allclose(values[0], 12.0)
    np.testing.assert_allclose(
        history[:, columns["merit"]],
        history[:, columns["objective"]] + history[:, columns["penalty"]] * values,
    )
    final_constraints = jnp.concatenate((
        problem.eq(result.x, None), problem.ineq(result.x, None),
        1.0 - result.x, result.x - 5.0,
    ))
    np.testing.assert_allclose(values[-1], violation(final_constraints, 1), atol=1e-12)

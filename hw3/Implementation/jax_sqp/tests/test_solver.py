import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from jax_sqp import SQP, Options, Status
from jax_sqp.problems import (
    broyden_tridiagonal,
    circle,
    conflicting_linearizations,
    constrained_rosenbrock,
    hs06,
    hs07,
    hs26,
    hs71,
    hs100,
    nw_ex18_3,
    optimal_control,
)


@pytest.mark.parametrize(
    "factory",
    [hs06, hs07, hs26, hs71, hs100, nw_ex18_3, circle, conflicting_linearizations],
)
def test_standard_small_problems(factory):
    p = factory()
    r = p.solver().solve(p.x0, p.params)
    assert bool(r.success), (p.name, int(r.status), float(r.kkt.error))
    assert float(r.kkt.error) <= 1e-6
    if p.reference_objective is not None:
        assert abs(float(r.objective) - p.reference_objective) < 3e-6
    if factory == conflicting_linearizations:
        assert int(r.elastic_steps) > 0
        assert (
            float(r.history[1, 7]) > 0.5
        )  # genuinely positive elastic slack at first QP


@pytest.mark.parametrize("factory", [hs71, lambda: optimal_control(12)])
def test_against_independent_scipy_slsqp(factory):
    p = factory()
    r = p.solver().solve(p.x0, p.params)
    cs = []
    if p.eq is not None:
        cs.append(
            {
                "type": "eq",
                "fun": lambda x: np.asarray(p.eq(jnp.array(x), p.params)),
                "jac": lambda x: np.asarray(jax.jacfwd(p.eq)(jnp.array(x), p.params)),
            }
        )
    if p.ineq is not None:
        cs.append(
            {
                "type": "ineq",
                "fun": lambda x: -np.asarray(p.ineq(jnp.array(x), p.params)),
                "jac": lambda x: -np.asarray(
                    jax.jacfwd(p.ineq)(jnp.array(x), p.params)
                ),
            }
        )
    n = p.x0.size
    lo = np.broadcast_to(-np.inf if p.lower is None else p.lower, (n,))
    up = np.broadcast_to(np.inf if p.upper is None else p.upper, (n,))
    ref = minimize(
        lambda x: float(p.fun(jnp.array(x), p.params)),
        np.array(p.x0),
        jac=lambda x: np.asarray(jax.grad(p.fun)(jnp.array(x), p.params)),
        constraints=cs,
        bounds=list(zip(lo, up)),
        method="SLSQP",
        options={"ftol": 1e-11, "maxiter": 1000},
    )
    assert bool(r.success)
    assert ref.success, ref.message
    np.testing.assert_allclose(r.objective, ref.fun, atol=2e-6)
    np.testing.assert_allclose(r.x, ref.x, atol=2e-5)


def test_vmap_and_single_solve_agree():
    p = hs71()
    s = p.solver()
    xs = jnp.array([[1.0, 5.0, 5.0, 1.0], [1.2, 4.5, 4.0, 1.4], [1.0, 4.7, 3.8, 1.4]])
    batch = s.solve_batch(xs, None)
    assert np.all(batch.success)
    for i in range(3):
        single = s.solve(xs[i], None)
        np.testing.assert_allclose(batch.x[i], single.x, atol=2e-6)


def test_vmap_over_parameter_pytrees():
    s = SQP(
        lambda x, p: 0.5 * jnp.sum((x - p["target"]) ** 2),
        eq=lambda x, p: jnp.array([x.sum() - p["sum"]]),
    )
    params = {
        "target": jnp.array([[1.0, 2.0], [2.0, 4.0]]),
        "sum": jnp.array([1.0, 2.0]),
    }
    r = jax.jit(jax.vmap(s.run, in_axes=(0, 0)))(jnp.zeros((2, 2)), params)
    assert np.all(r.success)
    np.testing.assert_allclose(r.x, [[0.0, 1.0], [0.0, 2.0]], atol=1e-7)


@pytest.mark.parametrize("factory", [hs71, lambda dtype: optimal_control(30, dtype)])
def test_float32(factory):
    p = factory(jnp.float32)
    r = p.solver(Options.for_dtype(32)).solve(p.x0, p.params)
    assert r.x.dtype == jnp.float32
    assert bool(r.success), (r.status, r.kkt)
    assert float(r.kkt.error) <= 3e-4


def test_unconstrained_and_zero_sized_constraints():
    s = SQP(lambda x, _: 0.5 * jnp.dot(x, x))
    r = s.solve(jnp.array([1.0, -2.0]))
    assert bool(r.success)
    assert r.multipliers.size == 0
    np.testing.assert_allclose(r.x, 0, atol=1e-10)


def test_active_bound_multiplier_sign_and_fixed_bound():
    for lo, up, expected in [(None, 1.0, 1.0), (0.0, 0.0, 0.0)]:
        s = SQP(lambda x, _: 0.5 * (x[0] - 2) ** 2, lower=lo, upper=up)
        r = s.solve(jnp.array([0.5]))
        assert bool(r.success)
        assert np.all(r.multipliers >= 0)
        np.testing.assert_allclose(r.x, [expected], atol=1e-8)


def test_infeasible_problem_never_reports_success():
    s = SQP(lambda x, _: jnp.dot(x, x), eq=lambda x, _: jnp.array([x[0], x[0] - 1]))
    r = s.solve(jnp.array([0.5]))
    assert not bool(r.success)
    assert float(r.kkt.primal) >= 0.5 - 1e-7
    assert int(r.status) in (
        Status.STALLED_INFEASIBLE,
        Status.LINE_SEARCH_FAILED,
        Status.QP_FAILED,
    )


def test_nonfinite_and_iteration_limit_status():
    r = SQP(lambda x, _: jnp.log(x[0])).solve(jnp.array([-1.0]))
    assert int(r.status) == Status.NONFINITE
    p = hs71()
    r = p.solver(Options(max_iter=1)).solve(p.x0)
    assert int(r.status) == Status.MAX_ITER


def test_elastic_only_mode():
    p = hs71()
    r = p.solver(Options(qp_mode="elastic")).solve(p.x0)
    assert bool(r.success)
    assert int(r.elastic_steps) > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lower": 2.0, "upper": 1.0},
        {"lower": np.inf},
        {"upper": -np.inf},
        {"lower": np.nan},
    ],
)
def test_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        SQP(lambda x, _: jnp.sum(x * x), **kwargs).solve(jnp.zeros(2))


def test_bad_shapes_and_dtypes():
    with pytest.raises(ValueError):
        SQP(lambda x, _: jnp.sum(x * x)).solve(jnp.zeros((2, 2)))
    with pytest.raises(TypeError):
        SQP(lambda x, _: jnp.sum(x * x)).solve(jnp.array([1, 2]))
    with pytest.raises(ValueError):
        SQP(lambda x, _: jnp.sum(x * x), eq=lambda x, _: x[0]).solve(jnp.zeros(2))


def test_exact_merit_slope_mode():
    problem = hs71()
    options = Options(merit_slope="exact")
    result = problem.solver(options).solve(problem.x0, problem.params)
    assert bool(result.success), (result.status, result.kkt)
    assert float(result.kkt.error) <= options.tolerance


@pytest.mark.slow
@pytest.mark.parametrize(
    "p", [broyden_tridiagonal(1000), optimal_control(100), constrained_rosenbrock(1000)]
)
def test_large_matrix_free_problems(p):
    r = p.solver().solve(p.x0, p.params)
    assert bool(r.success), (p.name, r.status, r.kkt)
    assert float(r.kkt.error) <= 1e-6

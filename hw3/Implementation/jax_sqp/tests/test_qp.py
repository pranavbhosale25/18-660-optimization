import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from jax_sqp.linalg import pcg
from jax_sqp.qp import QPOptions, solve_qp


@pytest.mark.parametrize("dense_threshold", [0, 64])
def test_hard_qp_primal_and_multipliers(dense_threshold):
    a = jnp.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    g = jnp.array([0.0, -0.5])
    c = jnp.array([-1.0, 0.0, 0.0])
    opts = QPOptions(dense_threshold=dense_threshold)
    r = jax.jit(
        lambda: solve_qp(
            g,
            c,
            lambda p: a @ p,
            lambda z: a.T @ z,
            lambda v: v,
            1,
            jnp.array(jnp.inf),
            jnp.zeros(3),
            opts,
        )
    )()
    assert bool(r.success)
    np.testing.assert_allclose(r.step, [0.25, 0.75], atol=1e-8)
    np.testing.assert_allclose(r.multipliers, [-0.25, 0, 0], atol=1e-8)
    assert float(r.linear_violation_inf) < 1e-8


def test_elastic_qp_conflicting_equalities():
    a = jnp.ones((2, 1))
    c = jnp.array([-1.0, -2.0])
    g = jnp.zeros(1)
    r = jax.jit(
        lambda: solve_qp(
            g,
            c,
            lambda p: a @ p,
            lambda z: a.T @ z,
            lambda v: v,
            2,
            jnp.array(2.0),
            jnp.zeros(2),
            QPOptions(),
        )
    )()
    assert bool(r.success)
    np.testing.assert_allclose(r.step, [1.0], atol=1e-8)
    np.testing.assert_allclose(r.multipliers, [1.0, -2.0], atol=1e-8)
    assert abs(float(r.linear_violation) - 1) < 1e-8


@pytest.mark.parametrize("threshold,probes", [(64, 0), (0, 0), (0, 8)])
def test_random_psd_dual_against_scipy(threshold, probes):
    rng = np.random.default_rng(4)
    m, n, neq = 15, 10, 3
    a = jnp.array(rng.normal(size=(m, n)))
    g = jnp.array(rng.normal(size=n))
    c = jnp.array(rng.normal(size=m))
    opts = QPOptions(
        dense_threshold=threshold, jacobi_probes=probes, tolerance=1e-8, max_iter=500
    )
    r = jax.jit(
        lambda: solve_qp(
            g,
            c,
            lambda p: a @ p,
            lambda z: a.T @ z,
            lambda v: v,
            neq,
            jnp.array(10.0),
            jnp.zeros(m),
            opts,
        )
    )()
    k = np.array(a @ a.T)
    d = np.array(a @ g - c)
    ref = minimize(
        lambda z: 0.5 * z @ k @ z + d @ z,
        np.zeros(m),
        jac=lambda z: k @ z + d,
        bounds=[(-10, 10)] * neq + [(0, 10)] * (m - neq),
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 3000},
    )
    assert bool(r.success)
    np.testing.assert_allclose(r.step, -g - a.T @ ref.x, atol=3e-6, rtol=3e-6)
    assert np.all(np.array(r.multipliers[neq:]) >= 0)


def test_cg_zero_and_spd_and_breakdown():
    a = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    for rhs in [jnp.zeros(2), jnp.array([1.0, 2.0])]:
        r = jax.jit(lambda b: pcg(lambda x: a @ x, b, lambda x: x, 20))(rhs)
        assert bool(r.success)
        np.testing.assert_allclose(a @ r.x, rhs, atol=1e-10)
    bad = pcg(lambda x: -x, jnp.ones(2), lambda x: x, 10)
    assert not bool(bad.success)

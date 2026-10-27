import jax
import jax.numpy as jnp
import numpy as np

from jax_sqp import lbfgs


def test_operators_match_explicit_dense_bfgs_and_fifo():
    rng = np.random.default_rng(7)
    n, history = 13, 4
    mem = lbfgs.initialize(n, history, jnp.float64)
    q = rng.normal(size=(n, n))
    q = q.T @ q + np.eye(n)
    update = jax.jit(lbfgs.update)
    for k in range(9):
        b = lbfgs.hessian_operator(mem)
        s = jnp.array(rng.normal(size=n))
        y = jnp.array(q @ np.array(s))
        mem, ok = update(mem, b, s, y)
        assert bool(ok)
        explicit = np.eye(n) * float(mem.scale)
        for ss, yy in zip(
            np.array(mem.s)[: int(mem.count)], np.array(mem.y)[: int(mem.count)]
        ):
            bs = explicit @ ss
            explicit -= np.outer(bs, bs) / (ss @ bs)
            explicit += np.outer(yy, yy) / (ss @ yy)
        actual = jax.vmap(lambda v: lbfgs.product(lbfgs.hessian_operator(mem), v))(
            jnp.eye(n)
        ).T
        inv = jax.vmap(lambda v: lbfgs.inverse_product(mem, v))(jnp.eye(n)).T
        np.testing.assert_allclose(actual, explicit, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(inv @ actual, np.eye(n), rtol=1e-9, atol=1e-9)
        assert np.linalg.eigvalsh(actual).min() > 0
    assert int(mem.count) == history


def test_negative_zero_and_nonfinite_pairs_are_skipped():
    mem = lbfgs.initialize(3, 3, jnp.float64)
    b = lbfgs.hessian_operator(mem)
    for s, y in [
        (jnp.ones(3), -jnp.ones(3)),
        (jnp.zeros(3), jnp.zeros(3)),
        (jnp.ones(3), jnp.full(3, jnp.nan)),
    ]:
        new, ok = jax.jit(lbfgs.update)(mem, b, s, y)
        assert not bool(ok)
        np.testing.assert_array_equal(new.s, mem.s)


def test_powell_damping_positive_small_curvature():
    mem = lbfgs.initialize(3, 3, jnp.float64)
    b = lbfgs.hessian_operator(mem)
    s = jnp.ones(3)
    y = 0.01 * s
    new, ok = lbfgs.update(mem, b, s, y)
    assert bool(ok)
    assert float(jnp.dot(new.s[0], new.y[0])) >= 0.2 * float(jnp.dot(s, s)) - 1e-14

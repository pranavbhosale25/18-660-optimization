"""Contract tests for masking, PyTrees, momentum, and Langevin behavior."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_stochopt import SGD, make_solver
from jax_stochopt.core import _evaluate_objective_minibatch
from jax_stochopt.extensions.sources import ScoreGradient
from jax_stochopt.extensions.langevin import LangevinNoise
from jax_stochopt.types import ScheduleContext, UpdateContext

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("individual", [False, True])
def test_masked_nested_minibatch(dtype, individual):
    params = {"x": jnp.array([1., 2.], dtype), "y": (jnp.array(3., dtype),)}
    samples = {"target": jnp.array([2., 4., 100.], dtype)}
    mask = jnp.array([True, True, False])

    def loss(p, sample, args):
        return args * jnp.sum(p["x"]**2) + (p["y"][0] - sample["target"])**2

    evaluate = jax.jit(lambda p: _evaluate_objective_minibatch(
        loss, p, samples, jnp.array(2., dtype), mask, dtype, individual))
    value, losses, grads, per_lane = evaluate(params)
    np.testing.assert_allclose(value, 11.)
    np.testing.assert_allclose(losses, [11., 11., 9419.])
    np.testing.assert_allclose(grads["x"], [4., 8.])
    np.testing.assert_allclose(grads["y"][0], 0.)
    assert grads["x"].dtype == dtype
    if individual:
        np.testing.assert_allclose(per_lane["y"][0], [2., -2., -194.])
    else:
        assert per_lane is None
    # The averaged gradient remains differentiable (a Hessian-vector check).
    derivative = jax.jvp(lambda p: evaluate(p)[2], (params,),
                         (jax.tree.map(jnp.ones_like, params),))[1]
    np.testing.assert_allclose(derivative["x"], [4., 4.])


@pytest.mark.parametrize("individual", [False, True])
def test_empty_sample_tree(individual):
    value, losses, grads, _ = _evaluate_objective_minibatch(
        lambda p, s, a: p**2, jnp.array(2.), (), None,
        jnp.array([True, False, True]), jnp.float64, individual)
    np.testing.assert_allclose([value, grads], [4., 4.])
    assert losses.shape == (3,)


def test_classical_momentum_and_zero_momentum():
    params, grads = {"x": jnp.array([1., 2.])}, {"x": jnp.array([3., 4.])}
    rule = SGD(0.6)
    direction, state = rule.update(params, grads, rule.init(params), value_fn=None, context=None)
    np.testing.assert_allclose(direction["x"], [3., 4.])
    direction, _ = rule.update(params, grads, state, value_fn=None, context=None)
    np.testing.assert_allclose(direction["x"], [4.8, 6.4])
    direction, _ = SGD().update(params, grads, state, value_fn=None, context=None)
    np.testing.assert_allclose(direction["x"], grads["x"])


def context(params, rate=0.1):
    before = ScheduleContext(jnp.array(0), params, (), None, None)
    return UpdateContext(before, before, params, params, jnp.array(rate), jax.random.key(8))


def test_score_conversion_temperature_derivatives_and_invalid_values():
    params = {"x": jnp.array([1., 2.])}
    ctx = context(params)
    score = lambda p, s, a, t: jax.tree.map(lambda x: -a * x / t, p)
    gradient = ScoreGradient(score, temperature=lambda c: jnp.array(0.7))
    np.testing.assert_allclose(jax.jit(gradient)(params, (), 3., ctx.before)["x"], [3., 6.])
    np.testing.assert_allclose(jax.grad(lambda a: jnp.sum(
        gradient(params, (), a, ctx.before)["x"]))(3.), 3.)
    for invalid in (0., -1., np.inf, np.nan):
        bad = ScoreGradient(score, temperature=lambda c: jnp.asarray(invalid))
        assert np.all(np.isnan(bad(params, (), 3., ctx.before)["x"]))
    with pytest.raises(ValueError):
        ScoreGradient(score, temperature=0.)
    with pytest.raises(ValueError):
        ScoreGradient(score, temperature=lambda c: jnp.ones(2))(params, (), 3., ctx.before)
    with pytest.raises(TypeError):
        ScoreGradient(score, temperature=lambda c: jnp.array(1j))(params, (), 3., ctx.before)


def test_langevin_independent_noise_and_diagnostics():
    params = {"a": jnp.zeros(50000), "b": (jnp.zeros(50000),)}
    updated, info = jax.jit(LangevinNoise(0.7))(params, context(params))
    noise = np.concatenate(jax.tree.leaves(updated))
    np.testing.assert_allclose(info.noise_scale, np.sqrt(0.14))
    np.testing.assert_allclose(info.noise_norm, np.linalg.norm(noise))
    assert abs(noise.mean()) < 0.005
    assert abs(noise.var() - 0.14) < 0.004
    assert abs(np.corrcoef(updated["a"], updated["b"][0])[0, 1]) < 0.02
    assert bool(info.valid_temperature)


def test_langevin_zero_temperature_and_invalid_dynamic_temperature():
    params = jnp.array([1., 2.])
    updated, info = LangevinNoise(0.)(params, context(params))
    np.testing.assert_array_equal(updated, params)
    assert float(info.noise_norm) == 0.
    np.testing.assert_array_equal(jax.jacfwd(
        lambda p: LangevinNoise(0.)(p, context(p))[0])(params), np.eye(2))
    for value in (-1., np.inf, np.nan):
        updated, info = LangevinNoise(lambda c: jnp.asarray(value))(params, context(params))
        assert np.all(np.isnan(updated))
        assert np.isnan(info.noise_norm) and np.isnan(info.noise_scale)
        assert not bool(info.valid_temperature)


def test_noise_precedes_projection_and_resume_retains_momentum():
    solver = make_solver(lambda p, s, a: jnp.sum(p**2), lambda k, a: (),
                         optimizer=SGD(0.6), steps=3, learning_rate=0.1,
                         update_transform=LangevinNoise(0.7),
                         projection=lambda p, c: jnp.clip(p, -0.1, 0.1))
    result = solver.run(jnp.array([1., 2.]), jax.random.key(4))
    resumed = solver.resume(result.state)
    assert np.max(np.abs(resumed.params)) <= 0.1
    assert int(resumed.state.step) == 6

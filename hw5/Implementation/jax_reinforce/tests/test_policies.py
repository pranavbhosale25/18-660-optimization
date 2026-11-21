import numpy as np
import pytest
import jax
import jax.numpy as jnp
from reinforce_jax import CategoricalPolicy, GaussianPolicy, make_bootstrap


def apply(params, obs):
    del obs
    return params


def test_categorical_logprob_entropy_and_vmap():
    policy = CategoricalPolicy(apply, 3)
    params = jnp.log(jnp.asarray([0.2, 0.3, 0.5]))
    obs, key = jnp.zeros(1), jax.random.key(1)
    samples = jax.jit(jax.vmap(lambda k: policy.sample(params, obs, k)))(jax.random.split(key, 1000))
    np.testing.assert_allclose(samples.log_prob, params[samples.action])
    np.testing.assert_allclose(policy.entropy(params, obs, key), -jnp.sum(jnp.exp(params) * params))
    assert samples.action.dtype == jnp.int32


def test_discrete_q_bootstrap_is_policy_expectation_not_max():
    policy = CategoricalPolicy(apply, 2)
    fn = make_bootstrap(policy, "q", critic_apply=lambda q, o, a: q[a])
    value = jax.jit(fn)(jnp.log(jnp.asarray([0.75, 0.25])), jnp.asarray([1., 5.]),
                        jnp.zeros(1), jax.random.key(0))
    np.testing.assert_allclose(value, 2.)


def test_q_sample_mode_and_continuous_expectation():
    policy = GaussianPolicy(apply, 1)
    fn = make_bootstrap(policy, "q", critic_apply=lambda q, o, a: a[0], q_samples=8192)
    value = jax.jit(fn)(jnp.asarray([0.7, -0.5]), {}, jnp.zeros(1), jax.random.key(5))
    np.testing.assert_allclose(value, 0.7, atol=0.025)


def test_tanh_density_matches_change_of_variables():
    policy = GaussianPolicy(apply, 2, (-2., -1.), (4., 3.))
    params, raw = jnp.asarray([0.1, -0.2, -0.5, 0.3]), jnp.asarray([0.4, -0.7])
    z = (raw - params[:2]) / jnp.exp(params[2:])
    expected = jnp.sum(-0.5 * z**2 - params[2:] - 0.5 * jnp.log(2 * jnp.pi)
                       - jnp.log(jnp.asarray([3., 2.])) - jnp.log(1 - jnp.tanh(raw)**2))
    np.testing.assert_allclose(policy.log_prob(params, jnp.zeros(1), raw), expected)
    saturated = policy.log_prob(params, jnp.zeros(1), jnp.asarray([100., -100.]))
    assert bool(jnp.isfinite(saturated))


def test_gaussian_bounds_and_reparameterized_gradients():
    policy = GaussianPolicy(apply, 1, (-2.,), (3.,))
    params, obs, key = jnp.asarray([0.2, -0.4]), jnp.zeros(1), jax.random.key(6)
    samples = jax.jit(jax.vmap(lambda k: policy.sample(params, obs, k)))(jax.random.split(key, 1024))
    assert bool(((samples.action >= -2) & (samples.action <= 3)).all())
    gradient = jax.grad(lambda p: policy.sample(p, obs, key).action.sum())(params)
    assert bool(jnp.isfinite(gradient).all()) and float(jnp.linalg.norm(gradient)) > 0
    entropy_grad = jax.grad(lambda p: policy.entropy(p, obs, key))(params)
    assert bool(jnp.isfinite(entropy_grad).all())


@pytest.mark.parametrize("kwargs", [{"low": (-1.,)}, {"low": (1.,), "high": (-1.,)},
                                    {"low": (-np.inf,), "high": (1.,)}])
def test_invalid_gaussian_bounds(kwargs):
    with pytest.raises(ValueError):
        GaussianPolicy(apply, 1, **kwargs)

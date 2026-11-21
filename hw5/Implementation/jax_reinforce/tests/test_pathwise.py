import numpy as np
import pytest
import jax
import jax.numpy as jnp
from reinforce_jax import (GaussianPolicy, CategoricalPolicy, make_bootstrap, make_pathwise_objective,
                           ReturnConfig, PathwiseContext, make_direct_learner, ObjectiveResult, sgd)
from reinforce_jax.toy_envs import LinearQuadraticEnv, LQState
from examples.analytic_gradient import expected_return, analytic_gradient
from .helpers import assert_trees_equal


def test_pathwise_gradient_matches_central_difference_including_value_tail():
    dtype = jnp.float64
    env = LinearQuadraticEnv(episode_length=8, noise=0.03, dtype=dtype)
    policy = GaussianPolicy(lambda p, o: p, 1, (-2.,), (2.,))
    bootstrap = make_bootstrap(policy, "v", critic_apply=lambda p, o, a: -p * o[0]**2)
    objective = make_pathwise_objective(env=env, policy=policy, bootstrap=bootstrap,
                                        config=ReturnConfig(0.9, 3), rollout_steps=5)
    states = LQState(jnp.asarray([0.5, -0.7], dtype), jnp.zeros(2, jnp.int32))
    context = PathwiseContext(states, jnp.asarray(1.2, dtype))
    key, params = jax.random.key(3), jnp.asarray([0.2, -0.4], dtype)
    f = jax.jit(lambda p: objective(p, context, key).value)
    grad = jax.jit(jax.grad(f))(params)
    h = 1e-5
    finite_diff = np.asarray([(f(params + h * e) - f(params - h * e)) / (2 * h)
                              for e in jnp.eye(2, dtype=dtype)])
    np.testing.assert_allclose(grad, finite_diff, rtol=2e-6, atol=2e-7)
    assert float(jnp.linalg.norm(grad)) > 0


def test_q_tail_preserves_policy_and_state_derivatives():
    policy = GaussianPolicy(lambda p, obs: jnp.asarray([p, -1.]), 1)
    fn = make_bootstrap(policy, "q", critic_apply=lambda phi, obs, action: phi * (obs[0] + action[0]),
                        q_samples=4)
    key = jax.random.key(0)
    derivative = jax.grad(lambda p: fn(p, jnp.asarray(2.), jnp.asarray([3.]), key))(jnp.asarray(0.1))
    np.testing.assert_allclose(derivative, 2.)


def test_analytic_gradient_matches_autodiff_and_update():
    params = {"mean": jnp.asarray([0.1, -0.2]), "log_std": jnp.asarray([-0.3, -0.4])}
    target, key = jnp.asarray([0.7, 0.2]), jax.random.key(0)
    exact = analytic_gradient(params, target, key)
    autodiff = jax.grad(lambda p: expected_return(p, target, key).value)(params)
    assert_trees_equal(exact, autodiff)
    manual = make_direct_learner(expected_return, sgd(0.01), analytic_grad=analytic_gradient)
    automatic = make_direct_learner(expected_return, sgd(0.01))
    m, _ = manual.update(manual.init(params), target, key)
    a, _ = automatic.update(automatic.init(params), target, key)
    assert_trees_equal(m, a)


def test_direct_optimizer_step_is_itself_differentiable():
    def objective(p, target, key):
        return ObjectiveResult(-(p - target)**2, jnp.asarray(1, jnp.int32))
    learner = make_direct_learner(objective, sgd(0.1))
    def updated_parameter(p):
        state, _ = learner.update(learner.init(p), jnp.asarray(1.), jax.random.key(0))
        return state.params
    np.testing.assert_allclose(jax.grad(updated_parameter)(jnp.asarray(0.3)), 0.8)


def test_pathwise_rejects_sampled_categorical_policy():
    env = LinearQuadraticEnv()
    policy = CategoricalPolicy(lambda p, obs: p, 2)
    with pytest.raises(ValueError, match="categorical"):
        make_pathwise_objective(env=env, policy=policy, bootstrap=make_bootstrap(policy),
                                config=ReturnConfig(), rollout_steps=3)


def test_direct_drop_all_skips_update():
    policy = GaussianPolicy(lambda p, obs: p, 1)
    env = LinearQuadraticEnv(episode_length=8, dtype=jnp.float64)
    objective = make_pathwise_objective(env=env, policy=policy, bootstrap=make_bootstrap(policy),
                                        config=ReturnConfig(0.9, 4, "drop"), rollout_steps=2)
    learner = make_direct_learner(objective, sgd(0.1))
    state = learner.init(jnp.asarray([0., -1.]))
    context = PathwiseContext(LQState(jnp.ones(3), jnp.zeros(3, jnp.int32)), {})
    new, metrics = learner.update(state, context, jax.random.key(0))
    assert_trees_equal(state, new)
    assert int(metrics["valid_count"]) == 0


def test_exact_discrete_expectation_uses_direct_api_without_relaxation():
    def exact_return(logits, rewards, key):
        return ObjectiveResult(jnp.sum(jax.nn.softmax(logits) * rewards), jnp.asarray(1))
    learner = make_direct_learner(exact_return, sgd(0.1))
    state = learner.init(jnp.zeros(2))
    state, _ = learner.update(state, jnp.asarray([0., 1.]), jax.random.key(0))
    np.testing.assert_allclose(state.params, [-0.025, 0.025])

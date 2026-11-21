import pytest
import jax
import jax.numpy as jnp
from reinforce_jax import (init_mlp, mlp_apply, CategoricalPolicy, GaussianPolicy,
                           make_rollout, make_bootstrap, make_learner, ReturnConfig,
                           NStepEstimator, ReinforceObjective, adam)
from reinforce_jax.toy_envs import ContextBandit


@pytest.mark.parametrize("ds,da", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_four_space_combinations_jitted_update(ds, da, dtype):
    env = ContextBandit(ds, da, dtype)
    policy = CategoricalPolicy(mlp_apply, 2) if da else GaussianPolicy(mlp_apply, 1, (-1.,), (1.,))
    key = jax.random.key(12)
    ik, rk, tk, pk, uk = jax.random.split(key, 5)
    params = init_mlp(ik, 3 if ds else 1, 2, (8,), dtype=dtype)
    states = jax.vmap(env.reset)(jax.random.split(rk, 16))
    data = make_rollout(env, policy, 3)(params, states, tk)
    assert data.rewards.dtype == dtype
    assert int(data.valid.sum()) == 16  # One-step bandit, then two padding rows.
    learner = make_learner(policy=policy, bootstrap=make_bootstrap(policy),
                          estimator=NStepEstimator(ReturnConfig()),
                          objective=ReinforceObjective(), actor_optimizer=adam(0.01))
    state = learner.init(params)
    prepared = learner.prepare(state.actor_params, state.critic_params, data, pk)
    new, metrics = learner.update(state, prepared, uk)
    assert int(new.updates) == 1
    assert bool(jnp.isfinite(metrics["actor_loss"]))
    assert all(p.dtype == dtype for p in jax.tree.leaves(new.actor_params))


@pytest.mark.parametrize("discrete_actions", [True, False])
@pytest.mark.parametrize("kind", ["v", "q"])
@pytest.mark.parametrize("algorithm", ["a2c", "gae", "ppo"])
def test_actor_critic_components_swap_without_changing_collector(discrete_actions, kind, algorithm):
    from reinforce_jax import (value_apply, discrete_q_apply, continuous_q_apply,
                               GAEEstimator, PPOObjective)
    env = ContextBandit(False, discrete_actions, jnp.float64)
    policy = CategoricalPolicy(mlp_apply, 2) if discrete_actions else GaussianPolicy(mlp_apply, 1)
    ik, ck, rk, tk, pk, uk = jax.random.split(jax.random.key(42), 6)
    actor = init_mlp(ik, 1, 2, (8,), dtype=jnp.float64)
    if kind == "v":
        critic_apply = value_apply
        critic = init_mlp(ck, 1, 1, (8,), dtype=jnp.float64)
    elif discrete_actions:
        critic_apply = discrete_q_apply
        critic = init_mlp(ck, 1, 2, (8,), dtype=jnp.float64)
    else:
        critic_apply = continuous_q_apply
        critic = init_mlp(ck, 2, 1, (8,), dtype=jnp.float64)
    estimator = NStepEstimator(ReturnConfig(0.99, 1)) if algorithm == "a2c" else GAEEstimator()
    objective = PPOObjective(normalize_advantages=False) if algorithm == "ppo" else ReinforceObjective()
    learner = make_learner(policy=policy, bootstrap=make_bootstrap(
        policy, kind, critic_apply=critic_apply, q_samples=3), estimator=estimator,
        objective=objective, actor_optimizer=adam(0.01), critic_apply=critic_apply,
        critic_optimizer=adam(0.01))
    state = learner.init(actor, critic)
    states = jax.vmap(env.reset)(jax.random.split(rk, 16))
    data = make_rollout(env, policy, 2)(actor, states, tk)
    prepared = learner.prepare(actor, critic, data, pk)
    for _ in range(3 if algorithm == "ppo" else 1):
        state, metrics = learner.update(state, prepared, uk)
    assert int(state.updates) == (3 if algorithm == "ppo" else 1)
    assert bool(jnp.isfinite(metrics["critic_loss"]))
    assert bool(jnp.isfinite(metrics["actor_loss"]))

"""Complete minimal composition example from README.md."""
import jax
from reinforce_jax import (
    configure, init_mlp, mlp_apply, value_apply, GaussianPolicy,
    ReturnConfig, NStepEstimator, ReinforceObjective,
    make_bootstrap, make_learner, make_rollout, adam,
)
from reinforce_jax.toy_envs import LinearQuadraticEnv

dtype = configure(dtype="float64", platform="cpu")
env = LinearQuadraticEnv(episode_length=24, dtype=dtype)
policy = GaussianPolicy(mlp_apply, action_dim=1, low=(-2.0,), high=(2.0,))
k_actor, k_critic, k_reset, k_roll, k_prep, k_update = jax.random.split(
    jax.random.key(0), 6
)
actor = init_mlp(k_actor, 2, 2, dtype=dtype)
critic = init_mlp(k_critic, 2, 1, dtype=dtype)

learner = make_learner(
    policy=policy,
    bootstrap=make_bootstrap(policy, "v", critic_apply=value_apply),
    estimator=NStepEstimator(
        ReturnConfig(gamma=0.99, horizon=8, missing="bootstrap"),
        use_baseline=True,
    ),
    objective=ReinforceObjective(gamma=0.99),
    actor_optimizer=adam(3e-4),
    critic_apply=value_apply,
    critic_optimizer=adam(1e-3),
)
state = learner.init(actor, critic)
initial_states = jax.vmap(env.reset)(jax.random.split(k_reset, 32))
trajectory = make_rollout(env, policy, steps=24)(
    state.actor_params, initial_states, k_roll
)
prepared = learner.prepare(
    state.actor_params, state.critic_params, trajectory, k_prep
)
state, metrics = learner.update(state, prepared, k_update)
print(metrics)

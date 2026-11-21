"""Entirely JAX-native rollout and pathwise return objectives (scan + vmap)."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .types import Trajectory, ObjectiveResult, PyTree, FunctionalEnv, Policy, Bootstrap
from .returns import ReturnConfig, n_step_returns
from .objectives import masked_mean


class PathwiseContext(NamedTuple):
    initial_states: PyTree  # Every leaf has a leading batch axis.
    critic_params: PyTree


def make_rollout(env: FunctionalEnv, policy: Policy, steps: int):
    """One episode per batch lane; no automatic reset. Pads after either boundary.

    Returned samples remain differentiable: the score-function learner detaches
    them, whereas the pathwise learner differentiates through their generation.
    Stochastic dynamics must reparameterize their own exogenous randomness.
    """
    if steps < 1:
        raise ValueError("steps must be positive")

    def single(params, initial_state, key):
        def step(carry, t):
            state, live, rng = carry
            rng, action_key, env_key = jax.random.split(rng, 3)
            obs = env.observe(state)
            sample = policy.sample(params, obs, action_key)

            def advance(_):
                return env.step(state, sample.action, env_key)

            def pad(_):
                return state, jnp.zeros((), obs.dtype), jnp.asarray(False), jnp.asarray(False)

            next_state, reward, terminal, truncated = jax.lax.cond(live, advance, pad, operand=None)
            transition = Trajectory(obs, sample.action, sample.raw_action, reward,
                                    env.observe(next_state), terminal, truncated, live,
                                    t, sample.log_prob)
            return (next_state, live & ~terminal & ~truncated, rng), transition

        _, batch = jax.lax.scan(step, (initial_state, jnp.asarray(True), key),
                                jnp.arange(steps, dtype=jnp.int32))
        return batch

    def batched(params, initial_states, key):
        batch_size = jax.tree.leaves(initial_states)[0].shape[0]
        return jax.vmap(single, in_axes=(None, 0, 0), out_axes=1)(
            params, initial_states, jax.random.split(key, batch_size))

    return jax.jit(batched)


def make_pathwise_objective(*, env: FunctionalEnv, policy: Policy, bootstrap: Bootstrap,
                            config: ReturnConfig, rollout_steps: int):
    """Differentiate the return FROM EACH INITIAL STATE, not all overlapping returns.

    Gradients pass through actions, rewards, dynamics and the tail's input state
    (and policy-dependent Q expectation). Critic parameters are in context,
    independent of actor params: this freezes their weights, NOT their inputs.
    Boolean/event boundaries do not have ordinary smooth pathwise derivatives.
    """
    if not policy.reparameterizable:
        raise ValueError("Sampled categorical actions are not pathwise differentiable; "
                         "use REINFORCE or an exact-enumeration return_fn instead")
    rollout = make_rollout(env, policy, rollout_steps)

    def objective(params, context: PathwiseContext, key):
        rollout_key, tail_key = jax.random.split(key)
        trajectory = rollout(params, context.initial_states, rollout_key)
        keys = jax.random.split(tail_key, trajectory.rewards.shape)
        next_values = jax.vmap(jax.vmap(lambda obs, k: bootstrap(
            params, context.critic_params, obs, k)))(trajectory.next_observations, keys)
        result = n_step_returns(trajectory, next_values, config)
        mask = result.mask[0]
        return ObjectiveResult(masked_mean(result.returns[0], mask), jnp.sum(mask))

    return objective

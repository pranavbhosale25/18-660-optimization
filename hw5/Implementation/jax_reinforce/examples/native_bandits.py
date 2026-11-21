"""REINFORCE on all four space combinations, entirely in JAX.

These native tasks complement the four real Gymnasium examples. They also
provide an executable smoke test when Gymnasium is not installed.
"""
import argparse
import jax
from reinforce_jax import (configure, init_mlp, mlp_apply, CategoricalPolicy, GaussianPolicy,
                           ReturnConfig, NStepEstimator, ReinforceObjective, make_bootstrap,
                           make_learner, make_rollout, adam)
from reinforce_jax.toy_envs import ContextBandit


def run(case="dd", iterations=150, batch_size=128, dtype="float32", platform="cpu", seed=0):
    dtype = configure(dtype=dtype, platform=platform)
    ds, da = case[0] == "d", case[1] == "d"
    env = ContextBandit(ds, da, dtype)
    policy = CategoricalPolicy(mlp_apply, 2) if da else GaussianPolicy(mlp_apply, 1, (-1.,), (1.,))
    key = jax.random.key(seed)
    key, init_key, eval_key = jax.random.split(key, 3)
    params = init_mlp(init_key, 3 if ds else 1, 2, (32,), dtype=dtype)
    learner = make_learner(policy=policy, bootstrap=make_bootstrap(policy, "zero"),
                          estimator=NStepEstimator(ReturnConfig(gamma=1.), use_baseline=False),
                          objective=ReinforceObjective(gamma=1.), actor_optimizer=adam(1e-2))
    state = learner.init(params)
    reset = jax.jit(jax.vmap(env.reset))
    rollout = make_rollout(env, policy, 1)
    eval_reset_key, eval_rollout_key = jax.random.split(eval_key)
    eval_states = reset(jax.random.split(eval_reset_key, 2048))
    before = float(rollout(params, eval_states, eval_rollout_key).rewards.mean())
    for _ in range(iterations):
        key, reset_key, rollout_key, prepare_key, update_key = jax.random.split(key, 5)
        initial_states = reset(jax.random.split(reset_key, batch_size))
        batch = rollout(state.actor_params, initial_states, rollout_key)
        data = learner.prepare(state.actor_params, state.critic_params, batch, prepare_key)
        state, metrics = learner.update(state, data, update_key)
    after = float(rollout(state.actor_params, eval_states, eval_rollout_key).rewards.mean())
    print(f"case={case} fixed-key evaluation: before={before:.6f}, after={after:.6f}")
    return before, after


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["dd", "dc", "cd", "cc", "all"], default="all")
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--platform", choices=["cpu", "gpu", "tpu"], default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = vars(p.parse_args())
    choice = args.pop("case")
    for case in (["dd", "dc", "cd", "cc"] if choice == "all" else [choice]):
        run(case=case, **args)

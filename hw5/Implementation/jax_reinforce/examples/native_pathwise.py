"""Fully jitted/vmapped differentiable control. No Gymnasium installation needed.

python -m examples.native_pathwise --dtype float64 --platform cpu
"""
import argparse
import jax
from reinforce_jax import (configure, GaussianPolicy, init_mlp, mlp_apply, make_bootstrap,
                           ReturnConfig, PathwiseContext, make_pathwise_objective,
                           make_direct_learner, adam, clip_by_global_norm)
from reinforce_jax.toy_envs import LinearQuadraticEnv


def run(iterations=150, batch_size=128, dtype="float32", platform="cpu", seed=0):
    dtype = configure(dtype=dtype, platform=platform)
    env = LinearQuadraticEnv(episode_length=24, dtype=dtype)
    policy = GaussianPolicy(mlp_apply, 1, (-2.,), (2.,))
    key = jax.random.key(seed)
    key, init_key, eval_key = jax.random.split(key, 3)
    params = init_mlp(init_key, 2, 2, (32, 32), dtype=dtype)
    objective = make_pathwise_objective(
        env=env, policy=policy, bootstrap=make_bootstrap(policy, "zero"),
        config=ReturnConfig(gamma=0.99, horizon=None), rollout_steps=24)
    learner = make_direct_learner(objective, clip_by_global_norm(adam(3e-3), 10.))
    state = learner.init(params)
    reset = jax.jit(jax.vmap(env.reset))
    eval_reset_key, eval_rollout_key = jax.random.split(eval_key)
    eval_context = PathwiseContext(reset(jax.random.split(eval_reset_key, batch_size)), {})
    evaluate = jax.jit(objective)
    before = float(evaluate(state.params, eval_context, eval_rollout_key).value)
    for i in range(iterations):
        key, reset_key, update_key = jax.random.split(key, 3)
        context = PathwiseContext(reset(jax.random.split(reset_key, batch_size)), {})
        state, metrics = learner.update(state, context, update_key)
        if i % 25 == 0 or i == iterations - 1:
            print(f"iter={i:4d} training_return={float(metrics['return']):9.4f}")
    after = float(evaluate(state.params, eval_context, eval_rollout_key).value)
    print(f"fixed-key evaluation: before={before:.6f}, after={after:.6f}")
    return before, after


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--platform", choices=["cpu", "gpu", "tpu"], default="cpu")
    p.add_argument("--seed", type=int, default=0)
    run(**vars(p.parse_args()))

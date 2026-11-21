"""Train on four Gymnasium space combinations and optionally record a CSV.

The two case letters describe observation and action spaces, respectively:
``d`` means ``Discrete`` and ``c`` means continuous ``Box``. For example,
``cd`` selects continuous observations and discrete actions.

See ``README.md`` for paired REINFORCE/A2C commands and output definitions.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import ClassVar

import jax
import numpy as np

from reinforce_jax import (
    GAEEstimator,
    NStepEstimator,
    PPOObjective,
    ReinforceObjective,
    ReturnConfig,
    adam,
    clip_by_global_norm,
    configure,
    continuous_q_apply,
    discrete_q_apply,
    init_mlp,
    make_bootstrap,
    make_learner,
    save_mlp_checkpoint,
    sgd,
    value_apply,
)
from reinforce_jax.gymnasium_adapter import (
    GymnasiumCollector,
    from_gym_spaces,
)

HISTORY_FIELDS = (
    "case",
    "algorithm",
    "seed",
    "iteration",
    "training_step",
    "rollout_step",
    "batch_rollout_steps",
    "completed_episodes",
    "cumulative_episodes",
    "average_reward",
    "average_return",
    "actor_loss",
    "critic_loss",
    "valid_training_records",
    "tail",
    "horizon",
    "collection",
    "gamma",
    "actor_learning_rate",
    "critic_learning_rate",
    "entropy_coefficient",
    "num_envs",
    "rollout_length",
)


def _write_run_outputs(
    output_dir, case, algorithm, seed, history, config, actor_params
):
    """Write one run without combining or summarizing separate trials."""
    run_dir = Path(output_dir) / case / algorithm / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
        stream.write("\n")
    save_mlp_checkpoint(run_dir / "actor.npz", actor_params)
    return history_path


def _summarize_rollout(batch, completed_returns):
    """Return raw rollout statistics while excluding padded transitions."""
    host_valid, host_rewards = jax.device_get((batch.valid, batch.rewards))
    valid = np.asarray(host_valid, dtype=bool)
    batch_rollout_steps = int(valid.sum())
    average_reward = (
        float(np.asarray(host_rewards)[valid].mean())
        if batch_rollout_steps
        else float("nan")
    )
    completed_returns = np.asarray(completed_returns, dtype=np.float64)
    average_return = (
        float(completed_returns.mean()) if completed_returns.size else float("nan")
    )
    return {
        "batch_rollout_steps": batch_rollout_steps,
        "completed_episodes": int(completed_returns.size),
        "average_reward": average_reward,
        "average_return": average_return,
    }


def _positive_int(value):
    """Argparse type for counts that must be at least one."""
    integer = int(value)
    if integer < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return integer


def context_bandit_target(context):
    """Return the optimal scalar action for one discrete bandit context."""
    context = int(context)
    if context < 0 or context >= 5:
        raise ValueError("bandit context must be an integer from zero to four")
    return (context - 2) * 0.4


def make_discrete_context_env(dtype=np.float32):
    """Discrete observation / continuous action, one-step contextual bandit."""
    import gymnasium as gym

    class DiscreteContextBandit(gym.Env):
        metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

        def __init__(self):
            self.observation_space = gym.spaces.Discrete(5)
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=dtype)
            self.context = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.context = int(self.np_random.integers(5))
            return self.context, {}

        def step(self, action):
            target = context_bandit_target(self.context)
            reward = 1.0 - (float(action[0]) - target) ** 2
            return self.context, reward, True, False, {}

    return DiscreteContextBandit()


def make_env(case, dtype=np.float32, *, render_mode=None):
    import gymnasium as gym

    if case == "dd":
        return gym.make(
            "FrozenLake-v1",
            is_slippery=False,
            render_mode=render_mode,
        )
    if case == "cd":
        return gym.make("CartPole-v1", render_mode=render_mode)
    if case == "cc":
        return gym.make("Pendulum-v1", render_mode=render_mode)
    if case == "dc":
        if render_mode is not None:
            raise ValueError(
                "the contextual bandit uses a diagnostic plot, not rendering"
            )
        return make_discrete_context_env(dtype)
    raise ValueError(f"unknown case: {case}")


def run(args, case):
    dtype = configure(dtype=args.dtype, platform=args.platform)
    envs = [make_env(case, np.dtype(dtype)) for _ in range(args.num_envs)]
    adapter = from_gym_spaces(
        envs[0].observation_space,
        envs[0].action_space,
        dtype=dtype,
    )
    policy = adapter.policy
    key = jax.random.key(args.seed)
    key, actor_key, critic_key = jax.random.split(key, 3)
    actor = init_mlp(
        actor_key,
        adapter.input_dim,
        adapter.actor_output_dim,
        dtype=dtype,
    )
    tail = args.tail or ("zero" if args.algorithm == "reinforce" else "v")
    critic, critic_apply = {}, None
    if tail == "v":
        critic_apply = value_apply
        critic = init_mlp(
            critic_key,
            adapter.input_dim,
            1,
            dtype=dtype,
            output_scale=1.0,
        )
    elif tail == "q":
        if policy.is_discrete:
            critic_apply = discrete_q_apply
            critic = init_mlp(
                critic_key,
                adapter.input_dim,
                policy.n_actions,
                dtype=dtype,
                output_scale=1.0,
            )
        else:
            critic_apply = continuous_q_apply
            critic = init_mlp(
                critic_key,
                adapter.input_dim + policy.action_dim,
                1,
                dtype=dtype,
                output_scale=1.0,
            )
    bootstrap = make_bootstrap(
        policy,
        tail,
        critic_apply=critic_apply,
        q_samples=4,
    )
    horizon = None if args.horizon == "mc" else int(args.horizon)
    if args.algorithm in ("gae", "ppo"):
        if args.horizon != "mc":
            raise ValueError(
                "--horizon applies to REINFORCE/A2C; GAE uses lambda instead"
            )
        estimator = GAEEstimator(args.gamma, args.lam, args.missing)
    else:
        estimator = NStepEstimator(
            ReturnConfig(args.gamma, horizon, args.missing),
            use_baseline=tail != "zero",
        )
    objective = (
        PPOObjective(gamma=args.gamma, entropy_coef=args.entropy)
        if args.algorithm == "ppo"
        else ReinforceObjective(gamma=args.gamma, entropy_coef=args.entropy)
    )
    if args.optimizer == "optax-adam":
        try:
            import optax
        except ImportError as exc:
            raise ImportError(
                'Install optional Optax: pip install -e ".[optax]"'
            ) from exc
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(args.lr),
        )
    else:
        optimizer_factory = adam if args.optimizer == "adam" else sgd
        optimizer = clip_by_global_norm(optimizer_factory(args.lr), 1.0)
    learner = make_learner(
        policy=policy,
        bootstrap=bootstrap,
        estimator=estimator,
        objective=objective,
        actor_optimizer=optimizer,
        critic_apply=critic_apply,
        critic_optimizer=adam(args.critic_lr) if critic_apply else None,
    )
    state = learner.init(actor, critic)
    episodes = args.collection == "episodes" or (
        args.collection == "auto" and args.algorithm == "reinforce"
    )
    effective_collection = "episodes" if episodes else "chunks"
    print(
        f"case={case} algorithm={args.algorithm} tail={tail} "
        f"dtype={args.dtype} devices={jax.devices()} "
        f"collection={effective_collection}"
    )
    history = []
    rollout_step = 0
    cumulative_episodes = 0
    with GymnasiumCollector(
        envs,
        policy,
        adapter.encode_observation,
        adapter.decode_action,
        seed=args.seed,
        dtype=dtype,
    ) as collector:
        for iteration in range(args.iterations):
            batch, key, completed = collector.collect(
                state.actor_params,
                key,
                steps=args.steps,
                episodes=episodes,
            )
            key, prepare_key = jax.random.split(key)
            prepared = learner.prepare(
                state.actor_params,
                state.critic_params,
                batch,
                prepare_key,
            )
            # PPO targets/advantages remain FIXED for all epochs on this batch.
            for _ in range(args.ppo_epochs if args.algorithm == "ppo" else 1):
                key, update_key = jax.random.split(key)
                state, metrics = learner.update(state, prepared, update_key)
            rollout_summary = _summarize_rollout(batch, completed)
            rollout_step += rollout_summary["batch_rollout_steps"]
            cumulative_episodes += rollout_summary["completed_episodes"]
            host_metrics = jax.device_get(metrics)
            training_step = int(jax.device_get(state.updates))
            history.append(
                {
                    "case": case,
                    "algorithm": args.algorithm,
                    "seed": args.seed,
                    "iteration": iteration + 1,
                    "training_step": training_step,
                    "rollout_step": rollout_step,
                    "batch_rollout_steps": rollout_summary["batch_rollout_steps"],
                    "completed_episodes": rollout_summary["completed_episodes"],
                    "cumulative_episodes": cumulative_episodes,
                    "average_reward": rollout_summary["average_reward"],
                    "average_return": rollout_summary["average_return"],
                    "actor_loss": float(host_metrics["actor_loss"]),
                    "critic_loss": float(host_metrics["critic_loss"]),
                    "valid_training_records": int(host_metrics["valid_count"]),
                    "tail": tail,
                    "horizon": "mc" if horizon is None else horizon,
                    "collection": effective_collection,
                    "gamma": args.gamma,
                    "actor_learning_rate": args.lr,
                    "critic_learning_rate": args.critic_lr,
                    "entropy_coefficient": args.entropy,
                    "num_envs": args.num_envs,
                    "rollout_length": args.steps,
                }
            )
            if iteration % args.log_every == 0 or iteration == args.iterations - 1:
                print(
                    f"iter={iteration + 1:4d} train_step={training_step:4d} "
                    f"rollout_step={rollout_step:8d} "
                    f"completed={rollout_summary['completed_episodes']:3d} "
                    f"average_reward={rollout_summary['average_reward']:9.3f} "
                    f"average_return={rollout_summary['average_return']:9.3f} "
                    f"actor_loss={float(host_metrics['actor_loss']):9.3f} "
                    f"critic_loss={float(host_metrics['critic_loss']):9.3f} "
                    f"valid={int(host_metrics['valid_count'])}"
                )
    if args.output_dir is not None:
        config = {
            "algorithm": args.algorithm,
            "actor_checkpoint": "actor.npz",
            "case": case,
            "collection": effective_collection,
            "dtype": args.dtype,
            "entropy_coefficient": args.entropy,
            "gamma": args.gamma,
            "horizon": "mc" if horizon is None else horizon,
            "iterations": args.iterations,
            "learning_rate": args.lr,
            "critic_learning_rate": args.critic_lr,
            "missing": args.missing,
            "num_envs": args.num_envs,
            "optimizer": args.optimizer,
            "platform": args.platform,
            "rollout_length": args.steps,
            "seed": args.seed,
            "tail": tail,
        }
        history_path = _write_run_outputs(
            args.output_dir,
            case,
            args.algorithm,
            args.seed,
            history,
            config,
            state.actor_params,
        )
        print(f"wrote {history_path}")
    return state


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["dd", "dc", "cd", "cc", "all"], default="cd")
    p.add_argument(
        "--algorithm",
        choices=["reinforce", "a2c", "gae", "ppo"],
        default="reinforce",
    )
    p.add_argument("--tail", choices=["zero", "v", "q"])
    p.add_argument(
        "--horizon",
        default="mc",
        help="Positive integer or mc; a2c commonly uses 1 or 8",
    )
    p.add_argument("--missing", choices=["bootstrap", "drop"], default="bootstrap")
    p.add_argument(
        "--collection",
        choices=["auto", "episodes", "chunks"],
        default="auto",
    )
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--platform", choices=["cpu", "gpu", "tpu"], default="cpu")
    p.add_argument("--optimizer", choices=["adam", "sgd", "optax-adam"], default="adam")
    p.add_argument("--num-envs", type=_positive_int, default=8)
    p.add_argument("--steps", type=_positive_int, default=128)
    p.add_argument("--iterations", type=_positive_int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Run every selected case for each seed; overrides --seed",
    )
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--entropy", type=float, default=0.0)
    p.add_argument("--ppo-epochs", type=_positive_int, default=4)
    p.add_argument("--log-every", type=_positive_int, default=10)
    p.add_argument(
        "--output-dir",
        help=(
            "Write history.csv, config.json, and actor.npz below "
            "<output-dir>/<case>/<algorithm>/seed-<seed>"
        ),
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    cases = ["dd", "dc", "cd", "cc"] if args.case == "all" else [args.case]
    seeds = args.seeds if args.seeds is not None else [args.seed]
    for seed in seeds:
        args.seed = seed
        for case in cases:
            run(args, case)

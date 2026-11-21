"""Replay a saved Gymnasium actor live or record it to video.

FrozenLake, CartPole, and Pendulum use Gymnasium rendering. The supplied
one-step contextual bandit instead produces an action-versus-target diagnostic
because it has no meaningful animation.
"""

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.gymnasium_train import context_bandit_target, make_env
from reinforce_jax import configure, load_mlp_checkpoint
from reinforce_jax.gymnasium_adapter import from_gym_spaces

_CASES = {"dd", "dc", "cd", "cc"}


def _positive_int(value):
    integer = int(value)
    if integer < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return integer


def load_run(run_dir, *, platform="cpu"):
    """Load one training configuration and its final actor checkpoint."""
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    try:
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read run configuration {config_path}") from exc
    if not isinstance(config, dict):
        raise TypeError("run configuration must be a JSON object")
    case = config.get("case")
    dtype_name = config.get("dtype")
    if case not in _CASES:
        raise ValueError(f"run configuration has unsupported case {case!r}")
    if dtype_name not in ("float32", "float64"):
        raise ValueError(f"run configuration has unsupported dtype {dtype_name!r}")
    checkpoint_name = config.get("actor_checkpoint", "actor.npz")
    if not isinstance(checkpoint_name, str):
        raise TypeError("actor_checkpoint must be one .npz filename")
    checkpoint_part = Path(checkpoint_name)
    if checkpoint_part.name != checkpoint_name or checkpoint_part.suffix != ".npz":
        raise ValueError("actor_checkpoint must be one .npz filename")
    dtype = configure(dtype=dtype_name, platform=platform)
    actor_params = load_mlp_checkpoint(
        run_dir / checkpoint_name,
        dtype=np.dtype(dtype),
    )
    return config, dtype, actor_params


def _policy_action(
    adapter,
    actor_params,
    observation,
    key,
    *,
    sample_actions=False,
):
    encoded = jnp.asarray(adapter.encode_observation(observation))
    if sample_actions:
        key, action_key = jax.random.split(key)
        action = adapter.policy.sample(
            actor_params,
            encoded,
            action_key,
        ).action
    else:
        action = adapter.policy.mode(actor_params, encoded)
    host_action = np.asarray(jax.device_get(action))
    return adapter.decode_action(host_action), key


def rollout_episodes(
    env,
    adapter,
    actor_params,
    *,
    episodes,
    seed,
    max_steps,
    sample_actions=False,
):
    """Run complete or capped episodes and return raw episode summaries."""
    key = jax.random.key(seed)
    summaries = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total_return = 0.0
        terminated = truncated = False
        length = 0
        for length in range(1, max_steps + 1):
            action, key = _policy_action(
                adapter,
                actor_params,
                observation,
                key,
                sample_actions=sample_actions,
            )
            observation, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
            if terminated or truncated:
                break
        summaries.append(
            {
                "episode": episode,
                "return": total_return,
                "length": length,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "capped": not bool(terminated or truncated),
            }
        )
    return summaries


def bandit_diagnostic_rows(
    adapter,
    actor_params,
    *,
    seed,
    sample_actions=False,
):
    """Evaluate the policy action and reward at every bandit context."""
    key = jax.random.key(seed)
    rows = []
    for context in range(5):
        action, key = _policy_action(
            adapter,
            actor_params,
            context,
            key,
            sample_actions=sample_actions,
        )
        action_value = float(np.asarray(action).reshape(-1)[0])
        target = context_bandit_target(context)
        rows.append(
            {
                "context": context,
                "target_action": target,
                "policy_action": action_value,
                "reward": 1.0 - (action_value - target) ** 2,
            }
        )
    return rows


def write_bandit_diagnostic(output_dir, rows):
    """Write the contextual-bandit table and action comparison plot."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise ImportError(
            'Install visualization dependencies with: pip install -e ".[visualize]"'
        ) from exc
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "policy_diagnostic.csv"
    fields = ("context", "target_action", "policy_action", "reward")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    contexts = [row["context"] for row in rows]
    targets = [row["target_action"] for row in rows]
    actions = [row["policy_action"] for row in rows]
    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.plot(contexts, targets, "o-", label="target action")
    axis.plot(contexts, actions, "s--", label="policy action")
    axis.set_xlabel("discrete context")
    axis.set_ylabel("continuous action")
    axis.set_xticks(contexts)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    plot_path = output_dir / "policy_diagnostic.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return csv_path, plot_path


def _make_playback_env(case, dtype, *, video_dir, name_prefix):
    render_mode = "rgb_array" if video_dir is not None else "human"
    env = make_env(case, np.dtype(dtype), render_mode=render_mode)
    if video_dir is None:
        return env
    import gymnasium as gym

    return gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_dir),
        episode_trigger=lambda _: True,
        name_prefix=name_prefix,
        disable_logger=True,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing config.json and actor.npz",
    )
    parser.add_argument("--episodes", type=_positive_int, default=3)
    parser.add_argument(
        "--seed",
        type=int,
        help="Evaluation seed; default is training seed plus 10000",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        help="Per-episode cap; default is the saved rollout length",
    )
    parser.add_argument(
        "--platform",
        choices=["cpu", "gpu", "tpu"],
        default="cpu",
        help="JAX platform for policy inference",
    )
    parser.add_argument(
        "--sample-actions",
        action="store_true",
        help="Sample from the policy instead of using its deterministic mode",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        help=(
            "Record MP4 files instead of opening a live window. For the "
            "contextual bandit, this selects the diagnostic output directory."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config, dtype, actor_params = load_run(
        args.run_dir,
        platform=args.platform,
    )
    case = config["case"]
    seed = args.seed
    if seed is None:
        seed = int(config.get("seed", 0)) + 10_000
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = int(config.get("rollout_length", 500))
    if max_steps < 1:
        raise ValueError("saved rollout length must be positive")

    if case == "dc":
        env = make_env(case, np.dtype(dtype))
        try:
            adapter = from_gym_spaces(
                env.observation_space,
                env.action_space,
                dtype=dtype,
            )
            rows = bandit_diagnostic_rows(
                adapter,
                actor_params,
                seed=seed,
                sample_actions=args.sample_actions,
            )
        finally:
            env.close()
        output_dir = args.video_dir or Path(args.run_dir) / "visualization"
        csv_path, plot_path = write_bandit_diagnostic(output_dir, rows)
        for row in rows:
            print(
                f"context={row['context']} "
                f"target={row['target_action']:+.3f} "
                f"action={row['policy_action']:+.3f} "
                f"reward={row['reward']:+.3f}"
            )
        print(f"wrote {csv_path}")
        print(f"wrote {plot_path}")
        return rows

    prefix = (
        f"{case}-{config.get('algorithm', 'policy')}-"
        f"seed-{config.get('seed', 'unknown')}"
    )
    env = _make_playback_env(
        case,
        dtype,
        video_dir=args.video_dir,
        name_prefix=prefix,
    )
    try:
        adapter = from_gym_spaces(
            env.observation_space,
            env.action_space,
            dtype=dtype,
        )
        summaries = rollout_episodes(
            env,
            adapter,
            actor_params,
            episodes=args.episodes,
            seed=seed,
            max_steps=max_steps,
            sample_actions=args.sample_actions,
        )
    finally:
        env.close()
    for summary in summaries:
        print(
            f"episode={summary['episode']} "
            f"return={summary['return']:.3f} "
            f"length={summary['length']} "
            f"terminated={summary['terminated']} "
            f"truncated={summary['truncated']} "
            f"capped={summary['capped']}"
        )
    if args.video_dir is not None:
        print(f"wrote videos under {args.video_dir}")
    return summaries


if __name__ == "__main__":
    main()

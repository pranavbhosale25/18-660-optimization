"""Headless checks for saved-policy playback helpers."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from examples.visualize_policy import (
    bandit_diagnostic_rows,
    parse_args,
    rollout_episodes,
)


class _DiscretePolicy:
    def mode(self, params, observation):
        del params
        return jnp.argmax(observation)

    def sample(self, params, observation, key):
        del key
        return SimpleNamespace(action=self.mode(params, observation))


class _DiscreteAdapter:
    policy = _DiscretePolicy()

    @staticmethod
    def encode_observation(observation):
        return np.eye(3, dtype=np.float32)[observation]

    @staticmethod
    def decode_action(action):
        return int(np.asarray(action))


class _TwoStepEnvironment:
    def reset(self, *, seed):
        self.step_count = 0
        return seed % 3, {}

    def step(self, action):
        self.step_count += 1
        terminated = self.step_count == 2
        return action % 3, float(action), terminated, False, {}


class _BanditPolicy:
    def mode(self, params, observation):
        del params
        context = jnp.argmax(observation)
        return jnp.asarray([(context - 2) * 0.4], dtype=jnp.float32)

    def sample(self, params, observation, key):
        del key
        return SimpleNamespace(action=self.mode(params, observation))


class _BanditAdapter:
    policy = _BanditPolicy()

    @staticmethod
    def encode_observation(observation):
        return np.eye(5, dtype=np.float32)[observation]

    @staticmethod
    def decode_action(action):
        return np.asarray(action, dtype=np.float32)


def test_rollout_episodes_uses_policy_mode_and_reports_boundaries():
    summaries = rollout_episodes(
        _TwoStepEnvironment(),
        _DiscreteAdapter(),
        actor_params=(),
        episodes=2,
        seed=4,
        max_steps=5,
    )

    assert [summary["length"] for summary in summaries] == [2, 2]
    assert all(summary["terminated"] for summary in summaries)
    assert all(not summary["capped"] for summary in summaries)
    assert [summary["return"] for summary in summaries] == [2.0, 4.0]


def test_bandit_diagnostic_covers_every_context_and_exact_policy():
    rows = bandit_diagnostic_rows(
        _BanditAdapter(),
        actor_params=(),
        seed=0,
    )

    assert [row["context"] for row in rows] == list(range(5))
    assert [row["reward"] for row in rows] == pytest.approx([1.0] * 5)
    assert [row["policy_action"] for row in rows] == pytest.approx(
        [row["target_action"] for row in rows]
    )


def test_visualizer_parser_requires_run_directory_and_positive_counts():
    args = parse_args(["--run-dir", "saved-run", "--episodes", "2"])
    assert args.run_dir.name == "saved-run"
    assert args.episodes == 2
    assert args.platform == "cpu"

    with pytest.raises(SystemExit):
        parse_args(["--run-dir", "saved-run", "--episodes", "0"])

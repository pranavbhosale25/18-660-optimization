"""Deterministic checks for experiment histories; Gymnasium is not required."""

import csv
import json
import math
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from examples.gymnasium_train import (
    HISTORY_FIELDS,
    _summarize_rollout,
    _write_run_outputs,
    parse_args,
)
from reinforce_jax import load_mlp_checkpoint


def test_rollout_summary_excludes_padding_and_averages_completed_returns():
    batch = SimpleNamespace(
        valid=jnp.asarray([[True, True], [True, False]]),
        rewards=jnp.asarray([[1.0, 3.0], [5.0, 1000.0]]),
    )

    summary = _summarize_rollout(batch, [7.0, 11.0])

    assert summary == {
        "batch_rollout_steps": 3,
        "completed_episodes": 2,
        "average_reward": 3.0,
        "average_return": 9.0,
    }


def test_rollout_summary_marks_missing_episode_return_as_nan():
    batch = SimpleNamespace(
        valid=jnp.asarray([[False]]),
        rewards=jnp.asarray([[1000.0]]),
    )

    summary = _summarize_rollout(batch, [])

    assert summary["batch_rollout_steps"] == 0
    assert summary["completed_episodes"] == 0
    assert math.isnan(summary["average_reward"])
    assert math.isnan(summary["average_return"])


def test_run_outputs_use_separate_case_algorithm_and_seed_directories(tmp_path):
    row = {field: "" for field in HISTORY_FIELDS}
    row.update(
        {
            "case": "dd",
            "algorithm": "reinforce",
            "seed": 4,
            "iteration": 1,
            "training_step": 1,
            "rollout_step": 12,
            "average_reward": 0.25,
            "average_return": 1.0,
        }
    )
    config = {"case": "dd", "algorithm": "reinforce", "seed": 4}
    actor_params = (
        {
            "w": jnp.asarray([[1.0], [-2.0]], dtype=jnp.float32),
            "b": jnp.asarray([0.5], dtype=jnp.float32),
        },
    )

    history_path = _write_run_outputs(
        tmp_path,
        "dd",
        "reinforce",
        4,
        [row],
        config,
        actor_params,
    )

    expected_dir = tmp_path / "dd" / "reinforce" / "seed-4"
    assert history_path == expected_dir / "history.csv"
    with history_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames) == HISTORY_FIELDS
        saved_rows = list(reader)
    assert len(saved_rows) == 1
    assert saved_rows[0]["rollout_step"] == "12"
    with (expected_dir / "config.json").open(encoding="utf-8") as stream:
        assert json.load(stream) == config
    restored = load_mlp_checkpoint(expected_dir / "actor.npz")
    np.testing.assert_array_equal(restored[0]["w"], actor_params[0]["w"])
    np.testing.assert_array_equal(restored[0]["b"], actor_params[0]["b"])


def test_parser_accepts_multiple_seeds_and_rejects_nonpositive_counts():
    args = parse_args(
        [
            "--case",
            "all",
            "--seeds",
            "2",
            "5",
            "--output-dir",
            "results",
        ]
    )
    assert args.seeds == [2, 5]
    assert args.output_dir == "results"

    with pytest.raises(SystemExit):
        parse_args(["--log-every", "0"])

"""Select warm-start settings using validation data, then run matched tests."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from l2ws_jax.benchmark import benchmark_problem, plot_benchmarks
from l2ws_jax.problems import make_problem
from l2ws_jax.training import TrainConfig, train


def select_settings(problem_name, destination):
    """Keep final-test data out of the hyperparameter selection process."""
    problem = make_problem(problem_name, dtype="float64", device="cpu")
    candidates = [
        ("baseline", {"steps": 300, "width": 64, "learning_rate": 0.002}),
        ("wide_slow", {"steps": 1000, "width": 128, "learning_rate": 0.001}),
        ("wide_fast", {"steps": 1000, "width": 128, "learning_rate": 0.003}),
    ]
    trials = []
    for name, settings in candidates:
        scores = []
        for unroll in (0, 5):
            config = TrainConfig(**settings, seed=7, unroll=unroll, validate_every=50)
            result = train(problem, config)
            selected = next(
                row for row in result.history if row["step"] == result.best_step
            )
            scores.append(selected["validation_objective"])
            trials.append(
                {
                    "candidate": name,
                    "config": asdict(config),
                    "best_step": result.best_step,
                    "validation_objective": scores[-1],
                    "history": result.history,
                }
            )
        print(problem_name, name, "validation mean:", sum(scores) / 2, flush=True)
    means = {
        name: sum(t["validation_objective"] for t in trials if t["candidate"] == name)
        / 2
        for name, _ in candidates
    }
    selected_name = min(means, key=means.get)
    settings = dict(candidates)[selected_name]
    record = {
        "problem": problem_name,
        "selection": "mean validation objective over K=0,5",
        "selected": selected_name,
        "validation_means": means,
        "trials": trials,
    }
    (destination / f"{problem_name}_tuning.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results/warmstart_tuned"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    payloads = []
    for name in ("kinematics", "phase"):
        settings = select_settings(name, args.output)
        payloads.append(
            benchmark_problem(
                name,
                args.output,
                dtype="float64",
                device="cpu",
                train_steps=settings["steps"],
                width=settings["width"],
                learning_rate=settings["learning_rate"],
                validate_every=50,
                train_size=1024,
                test_size=64,
                seed=7,
                evaluation_seed_base=107,
                horizons=(0, 5, 20, 50),
            )
        )
    plot_benchmarks(payloads, args.output / "benchmark.png")


if __name__ == "__main__":
    main()

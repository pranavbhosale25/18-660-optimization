"""Create the submitted figures and summary CSVs from saved raw results."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hw5-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#187b8d", "#dd7a28", "#7356ad"]
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
    }
)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_bayesian_optimization():
    destination = ROOT / "Application/figures"
    destination.mkdir(exist_ok=True)
    summaries = []
    for task in ("coating", "pipeline", "production", "materials", "controller"):
        runs = ROOT / f"Application/results/bayesopt/{task}/instance-0/seed-0"
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), layout="constrained")
        variants = sorted(runs.iterdir(), key=lambda p: (p.name != "primary", p.name))
        for color, folder in zip(COLORS, variants):
            result = json.loads((folder / "result.json").read_text())
            observations = json.loads((folder / "observations.json").read_text())
            trace = json.loads((folder / "recommendations.json").read_text())
            score = result["score"]
            x = [record["cumulative_cost"] for record in observations]
            best, running = [], -np.inf
            for record in observations:
                if record["constraint"] is None or record["constraint"] <= 0:
                    running = max(running, record["y"])
                best.append(running if np.isfinite(running) else np.nan)
            label = folder.name.replace("_", " ")
            axes[0].step(x, best, where="post", color=color, label=label)
            axes[1].plot(
                [r["spent"] for r in trace],
                [r["true_mean"] if r["feasible"] else np.nan for r in trace],
                marker="o",
                ms=3,
                color=color,
                label=label,
            )
            summaries.append(
                {
                    "problem": task,
                    "variant": folder.name,
                    "quality": score["true_mean"],
                    "validation_mean": score["validation_mean"],
                    "validation_standard_error": score["validation_standard_error"],
                    "target": score["target"],
                    "passed": score["passed"],
                    "feasible": score["feasible"],
                    "observations": score["observations"],
                    "completed_runs": score["completed_runs"],
                    "spent": score["spent"],
                    "charged_cost": sum(r["cost"] for r in observations),
                    "canceled_reservation": score["canceled_reservation"],
                    "infeasible_observations": sum(
                        r["constraint"] is not None and r["constraint"] > 0
                        for r in observations
                    ),
                }
            )
        for ax in axes:
            ax.set_xlabel("Consumed " + score["budget_unit"].replace("_", " "))
            ax.set_xlim(0, score["budget"])
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Best observed feasible quality")
        axes[1].set_ylabel("Recommended point: post-run latent quality")
        axes[1].axhline(score["target"], color="0.4", ls=":", label="target")
        axes[0].set_title("Charged observations (includes measurement noise)")
        axes[1].set_title("Post-run diagnostics; unavailable to the optimizer")
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
        fig.suptitle(task.capitalize() + ": instance 0, algorithm seed 0")
        fig.savefig(destination / f"bo_{task}.png")
        plt.close(fig)
    write_csv(ROOT / "Application/results/bayesopt_summary.csv", summaries)


def plot_reinforcement_learning():
    destination = ROOT / "Implementation/figures"
    destination.mkdir(exist_ok=True)
    names = {
        "dd": "FrozenLake",
        "dc": "Contextual bandit",
        "cd": "CartPole",
        "cc": "Pendulum",
    }
    summaries = []
    for case, name in names.items():
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.7), layout="constrained")
        for algorithm, color in zip(("reinforce", "a2c"), COLORS):
            trials = []
            for path in sorted(
                (
                    ROOT / f"Implementation/results/gym-comparison/{case}/{algorithm}"
                ).glob("seed-*/history.csv")
            ):
                with path.open() as stream:
                    trials.append(list(csv.DictReader(stream)))
            if len(trials) != 3:
                raise ValueError(
                    f"Expected three complete trials for {case}/{algorithm}"
                )
            last_returns = [
                np.mean([float(row["average_return"]) for row in trial[-10:]])
                for trial in trials
            ]
            last_rewards = [
                np.mean([float(row["average_reward"]) for row in trial[-10:]])
                for trial in trials
            ]
            summaries.append(
                {
                    "case": case,
                    "environment": name,
                    "algorithm": algorithm,
                    "return_last10_mean": np.mean(last_returns),
                    "return_last10_seed_sd": np.std(last_returns, ddof=1),
                    "reward_last10_mean": np.mean(last_rewards),
                    "transitions_min": min(int(t[-1]["rollout_step"]) for t in trials),
                    "transitions_max": max(int(t[-1]["rollout_step"]) for t in trials),
                }
            )
            for row_index, metric in enumerate(("average_reward", "average_return")):
                for col, axis in enumerate(
                    ("iteration", "training_step", "rollout_step")
                ):
                    ax = axes[row_index, col]
                    xs = [np.array([float(row[axis]) for row in t]) for t in trials]
                    ys = [np.array([float(row[metric]) for row in t]) for t in trials]
                    # Interpolate only over the range observed in every seed.
                    grid = np.linspace(
                        max(x[0] for x in xs), min(x[-1] for x in xs), 100
                    )
                    curves = np.stack([np.interp(grid, x, y) for x, y in zip(xs, ys)])
                    mean, sd = curves.mean(axis=0), curves.std(axis=0, ddof=1)
                    for x, y in zip(xs, ys):
                        ax.plot(x, y, color=color, lw=0.5, alpha=0.13)
                    ax.plot(grid, mean, color=color, label=algorithm.upper())
                    ax.fill_between(grid, mean - sd, mean + sd, color=color, alpha=0.16)
                    ax.set_xlabel(axis.replace("_", " ").capitalize())
                    ax.set_ylabel(metric.replace("_", " ").capitalize())
                    ax.grid(alpha=0.2)
        axes[0, 0].legend()
        fig.suptitle(
            f"{name}: mean and one sample SD across seeds 0, 1, 2\n"
            "Thin lines: individual runs. Raw batch statistics; no smoothing."
        )
        fig.savefig(destination / f"{case}_learning_curves.png")
        plt.close(fig)
    write_csv(ROOT / "Implementation/results/training_summary.csv", summaries)


def plot_warmstarts():
    destination = ROOT / "Application/figures"
    tuning = []
    for problem in ("kinematics", "phase"):
        data = json.loads(
            (
                ROOT / f"Application/results/warmstart_tuned/{problem}_tuning.json"
            ).read_text()
        )
        for trial in data["trials"]:
            tuning.append(
                {
                    "problem": problem,
                    "candidate": trial["candidate"],
                    "unroll": trial["config"]["unroll"],
                    "steps": trial["config"]["steps"],
                    "width": trial["config"]["width"],
                    "learning_rate": trial["config"]["learning_rate"],
                    "best_step": trial["best_step"],
                    "validation_objective": trial["validation_objective"],
                }
            )
        payload = json.loads(
            (
                ROOT / f"Application/results/warmstart_tuned/{problem}/benchmark.json"
            ).read_text()
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
        for ax, distribution in zip(axes, ("test", "shift")):
            for color, method in zip(
                COLORS, ("baseline", "learned_K0_objective", "learned_K5_objective")
            ):
                rows = [
                    row
                    for row in payload["rows"]
                    if row["distribution"] == distribution and row["method"] == method
                ]
                ax.plot(
                    [r["refinement_steps"] for r in rows],
                    [r["objective_mean"] for r in rows],
                    marker="o",
                    color=color,
                    label=method.replace("_objective", "").replace("_", " "),
                )
            ax.set_yscale("log")
            ax.set_xticks([0, 5, 20, 50])
            ax.set_xlabel("Refinement steps")
            ax.set_ylabel("Mean objective (lower is better)")
            ax.set_title("Nominal test" if distribution == "test" else "Shifted test")
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(
            problem.capitalize() + ": 64 matched test contexts per distribution"
        )
        fig.savefig(destination / f"warmstart_{problem}.png")
        plt.close(fig)
    write_csv(ROOT / "Application/results/warmstart_tuning_summary.csv", tuning)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Root containing Application and Implementation results",
    )
    ROOT = parser.parse_args().root.resolve()
    plot_bayesian_optimization()
    plot_reinforcement_learning()
    plot_warmstarts()

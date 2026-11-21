"""Reproduce tests, experiments, videos, and plots from the submission root."""

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT
BO = ROOT / "Application/P1/jaxbo_review"
RL = ROOT / "Implementation/jax_reinforce"
WARM = ROOT / "Application/P2/l2ws_jax"


def python(environment):
    suffix = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return str(ROOT / environment / suffix)


def run(arguments, directory=ROOT, environment=".venv"):
    command = [python(environment), *map(str, arguments)]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    subprocess.run(command, cwd=directory, env=env, check=True)


def tests():
    for directory, environment in [(BO, ".venv"), (RL, ".venv"), (WARM, ".venv-l2ws")]:
        run(["-m", "pytest", "-q"], directory, environment)
    for problem in ("kinematics", "phase"):
        run(
            [
                "scripts/check_problem.py",
                "--problem",
                problem,
                "--dtype",
                "float64",
                "--device",
                "cpu",
            ],
            WARM,
            ".venv-l2ws",
        )


def bo():
    for problem in ("coating", "pipeline", "production", "materials", "controller"):
        run(
            [
                "-m",
                "examples.run",
                "--problem",
                problem,
                "--instance",
                "0",
                "--seed",
                "0",
                "--variant",
                "all",
                "--output-dir",
                OUTPUT_ROOT / "Application/results/bayesopt",
            ],
            BO,
        )


def rl():
    for algorithm, tail, horizon in [("reinforce", "zero", "mc"), ("a2c", "v", "8")]:
        run(
            [
                "-m",
                "examples.gymnasium_train",
                "--case",
                "all",
                "--algorithm",
                algorithm,
                "--tail",
                tail,
                "--horizon",
                horizon,
                "--collection",
                "episodes",
                "--steps",
                "500",
                "--iterations",
                "100",
                "--seeds",
                "0",
                "1",
                "2",
                "--lr",
                "0.003",
                "--entropy",
                "0.01",
                "--output-dir",
                OUTPUT_ROOT / "Implementation/results/gym-comparison",
                "--platform",
                "cpu",
            ],
            RL,
        )


def warmstart():
    baseline = OUTPUT_ROOT / "Application/results/warmstart_baseline"
    tuned = OUTPUT_ROOT / "Application/results/warmstart_tuned"
    if any(
        (p / problem).exists()
        for p in (baseline, tuned)
        for problem in ("kinematics", "phase")
    ):
        raise SystemExit(
            "Move the existing warmstart_baseline and warmstart_tuned "
            "folders aside before rerunning; the supplied benchmark "
            "requires new output directories."
        )
    run(
        [
            "-m",
            "l2ws_jax",
            "benchmark",
            "--problems",
            "kinematics",
            "phase",
            "--dtype",
            "float64",
            "--device",
            "cpu",
            "--steps",
            "300",
            "--train-size",
            "1024",
            "--test-size",
            "64",
            "--seed",
            "7",
            "--evaluation-seed",
            "107",
            "--output",
            baseline,
        ],
        WARM,
        ".venv-l2ws",
    )
    run(["Application/tune_warmstarts.py", "--output", tuned], environment=".venv-l2ws")


def videos():
    for case in ("cd", "cc"):
        for algorithm in ("reinforce", "a2c"):
            run(
                [
                    "-m",
                    "examples.visualize_policy",
                    "--run-dir",
                    OUTPUT_ROOT
                    / f"Implementation/results/gym-comparison/{case}/{algorithm}/seed-0",
                    "--episodes",
                    "2",
                    "--seed",
                    "1000",
                    "--video-dir",
                    OUTPUT_ROOT
                    / f"Implementation/results/policy-videos/{case}-{algorithm}",
                ],
                RL,
            )


def plots():
    run(["scripts/plot_results.py", "--root", OUTPUT_ROOT])


if __name__ == "__main__":
    actions = {
        "tests": tests,
        "bo": bo,
        "rl": rl,
        "warmstart": warmstart,
        "videos": videos,
        "plots": plots,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=(*actions, "all"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="Fresh artifact root for a separate reproduction",
    )
    args = parser.parse_args()
    OUTPUT_ROOT = args.output_dir.resolve()
    for action in actions.values() if args.part == "all" else [actions[args.part]]:
        action()

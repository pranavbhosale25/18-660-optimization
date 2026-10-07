"""Command-line benchmark runner with early JAX runtime configuration."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from .runtime import configure_runtime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trsr1-bench",
        description="Run standard problems with the JAX trust-region SR1 solver.",
    )
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "gpu", "tpu"),
        default="auto",
    )
    parser.add_argument("--precision", type=int, choices=(32, 64), default=64)
    parser.add_argument(
        "--problem",
        action="append",
        default=[],
        help="Problem name; repeat to select several. The default is all.",
    )
    parser.add_argument(
        "--gradient-tolerance",
        type=float,
        default=None,
        help="Override each problem's default gradient tolerance.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override each problem's default outer-iteration limit.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Use the dynamic while-loop outer path instead of fixed history.",
    )
    parser.add_argument(
        "--plot-directory",
        type=Path,
        default=None,
        help=(
            "Save objective-versus-iteration plots and CSV histories in this "
            "directory. History must remain enabled."
        ),
    )
    return parser


def _safe_filename(name: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "_" for character in name
    ).strip("_")


def _save_convergence_history(
    *,
    problem_name: str,
    known_minimum: float,
    precision: int,
    result: object,
    output_directory: Path,
) -> None:
    """Save the valid objective history as CSV and a noninteractive PNG."""

    import matplotlib.pyplot as plt
    import numpy as np

    plt.switch_backend("Agg")

    valid = np.asarray(result.history.valid, dtype=bool)
    objective = np.asarray(result.history.objective)[valid]
    gradient_norm = np.asarray(result.history.gradient_norm)[valid]

    # Each history record stores the state at the beginning of an iteration.
    # Append the returned state so the plot includes the final accepted point.
    objective = np.append(objective, float(result.fun))
    gradient_norm = np.append(gradient_norm, float(result.grad_norm))
    iteration = np.arange(objective.size)
    objective_gap = objective - known_minimum

    output_directory.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename(problem_name)
    np.savetxt(
        output_directory / f"{filename}.csv",
        np.column_stack((iteration, objective, objective_gap, gradient_norm)),
        delimiter=",",
        header="iteration,objective,objective_gap,gradient_norm",
        comments="",
    )

    figure, axis = plt.subplots()
    axis.plot(iteration, objective, label="SR1 objective")
    axis.axhline(
        known_minimum,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="known minimum",
    )
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Objective value")
    axis.set_title(f"{problem_name}, float{precision}")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / f"{filename}.png", dpi=200)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.no_history and args.plot_directory is not None:
        raise SystemExit("--plot-directory cannot be used with --no-history")

    runtime = configure_runtime(args.platform, args.precision)

    import jax
    import jax.numpy as jnp

    from .options import SR1Options
    from .problems import problem_by_name, standard_problems
    from .solver import make_solver
    from .types import status_message

    selected = (
        tuple(problem_by_name(name) for name in args.problem)
        if args.problem
        else standard_problems()
    )
    dtype = jnp.float64 if args.precision == 64 else jnp.float32

    print(
        f"backend={runtime.default_backend} precision={args.precision} "
        f"jax={jax.__version__}"
    )
    print(
        f"{'problem':24s} {'status':22s} {'iter':>6s} "
        f"{'f(x)':>15s} {'||g||':>12s} {'updates':>8s} {'seconds':>9s}"
    )
    print("-" * 104)

    all_successful = True
    for problem in selected:
        options = SR1Options(
            max_iterations=(
                problem.max_iterations
                if args.max_iterations is None
                else args.max_iterations
            ),
            gradient_tolerance=(
                problem.gradient_tolerance
                if args.gradient_tolerance is None
                else args.gradient_tolerance
            ),
            store_history=not args.no_history,
        )
        solver = make_solver(problem.fun, options)
        start = problem.initial_point(dtype)
        begin = time.perf_counter()
        result = solver(start)
        result.x.block_until_ready()
        elapsed = time.perf_counter() - begin
        if args.plot_directory is not None:
            _save_convergence_history(
                problem_name=problem.name,
                known_minimum=problem.known_minimum,
                precision=args.precision,
                result=result,
                output_directory=args.plot_directory,
            )
        message = status_message(int(result.status))
        all_successful &= bool(result.success)
        print(
            f"{problem.name:24s} {message:22.22s} "
            f"{int(result.iterations):6d} {float(result.fun):15.7e} "
            f"{float(result.grad_norm):12.4e} "
            f"{int(result.sr1_updates):8d} {elapsed:9.3f}"
        )

    return 0 if all_successful else 1

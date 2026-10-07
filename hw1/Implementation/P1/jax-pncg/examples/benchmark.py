from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jax_pncg import (
    PNCGOptions,
    PNCGResult,
    Preconditioner,
    configure_runtime,
    diagonal_preconditioner,
    identity_preconditioner,
    jacobi_preconditioner,
    make_solver,
    status_message,
)
from jax_pncg.test_problems import (
    STANDARD_PROBLEMS,
    TestProblem,
    diagonal_quadratic,
    ill_conditioned_quadratic_diagonal,
)

plt.switch_backend("Agg")


@dataclass(frozen=True)
class Row:
    name: str
    preconditioner: str
    status: str
    nit: int
    nfev: int
    gtol: float
    value: float
    grad_norm: float
    milliseconds: float


def make_hessian_diagonal(
    objective: Callable[..., jax.Array],
) -> Callable[..., jax.Array]:
    """Return an exact, automatic-differentiation Hessian-diagonal function."""

    hessian = jax.hessian(objective)

    def hessian_diagonal(x: jax.Array, *args: object) -> jax.Array:
        return jnp.diag(hessian(x, *args))

    return hessian_diagonal


def choose_preconditioner(
    problem: TestProblem,
    x0: jax.Array,
) -> tuple[str, Preconditioner]:
    """Return a display name and configured preconditioner for ``problem``."""

    if problem.name in {
        "Rosenbrock (2D)",
        "Extended Rosenbrock (10D)",
    }:
        hessian_diagonal = make_hessian_diagonal(problem.objective)
        diagonal_at_start = hessian_diagonal(x0)

        return (
            "fixed initial Hessian diagonal",
            diagonal_preconditioner(
                diagonal_at_start,
                damping=1.0e-6,
                absolute=True,
            ),
        )

    if problem.name in {
        "Powell singular (4D)",
        "Wood (4D)",
    }:
        return "identity", identity_preconditioner

    hessian_diagonal = make_hessian_diagonal(problem.objective)

    return (
        "Jacobi",
        jacobi_preconditioner(
            hessian_diagonal,
            damping=1.0e-6,
            absolute=True,
            max_inverse=1.0e6,
        ),
    )


def timed_run(solver: Callable, *solver_args, repeat: int) -> tuple[PNCGResult, float]:
    # First call includes tracing/compilation; keep it out of steady-state timing.
    result = solver(*solver_args)
    result.x.block_until_ready()

    start = time.perf_counter()
    for _ in range(repeat):
        result = solver(*solver_args)
    result.x.block_until_ready()
    elapsed = (time.perf_counter() - start) * 1000.0 / repeat
    return result, elapsed


def save_convergence_history(
    *,
    problem_name: str,
    minimum: float,
    preconditioner_name: str,
    precision: str,
    result: PNCGResult,
) -> None:
    """Save convergence values and an objective-versus-iteration plot."""

    history_size = int(result.history.size)
    if history_size == 0:
        return

    iterations = np.arange(history_size)
    objective_values = np.asarray(result.history.value[:history_size])
    gradient_norms = np.asarray(result.history.grad_norm[:history_size])
    objective_gaps = objective_values - minimum

    output_directory = Path("benchmark-results") / precision / "convergence"
    output_directory.mkdir(parents=True, exist_ok=True)

    combined_name = f"{problem_name}-{preconditioner_name}"
    safe_name = "".join(
        character.lower() if character.isalnum() else "_" for character in combined_name
    ).strip("_")

    np.savetxt(
        output_directory / f"{safe_name}.csv",
        np.column_stack((iterations, objective_values, objective_gaps, gradient_norms)),
        delimiter=",",
        header="iteration,objective,objective_gap,gradient_norm",
        comments="",
    )

    figure, axis = plt.subplots()
    axis.plot(iterations, objective_values, label="PNCG objective")
    axis.axhline(
        minimum,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="known minimum",
    )
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Objective value")
    axis.set_title(f"{problem_name}\n{preconditioner_name}, {precision}")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / f"{safe_name}.png", dpi=200)
    plt.close(figure)


def format_table(rows: list[Row]) -> str:
    headers = (
        "problem",
        "preconditioner",
        "status",
        "nit",
        "nfev",
        "gtol",
        "f",
        "||g||inf",
        "ms",
    )
    body = [
        (
            row.name,
            row.preconditioner,
            row.status,
            str(row.nit),
            str(row.nfev),
            f"{row.gtol:.1e}",
            f"{row.value:.3e}",
            f"{row.grad_norm:.3e}",
            f"{row.milliseconds:.3f}",
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in body))
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(header.ljust(width) for header, width in zip(headers, widths)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.ljust(width) for value, width in zip(row, widths))
        for row in body
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark JAX preconditioned Polak--Ribiere nonlinear CG"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=3000)
    parser.add_argument(
        "--gtol",
        type=float,
        default=None,
        help="override every problem's precision-specific default tolerance",
    )
    parser.add_argument("--quadratic-size", type=int, default=128)
    parser.add_argument("--quadratic-condition", type=float, default=1.0e8)
    args = parser.parse_args()

    if args.repeat <= 0:
        parser.error("--repeat must be positive")

    runtime = configure_runtime(platform=args.device, precision=args.precision)
    default_gtol = 5.0e-4 if args.precision == "float32" else 1.0e-5

    rows: list[Row] = []
    with runtime.default_device():
        for problem in STANDARD_PROBLEMS:
            problem_gtol = args.gtol
            if problem_gtol is None:
                problem_gtol = (
                    problem.gtol32 if args.precision == "float32" else problem.gtol64
                )
            options = PNCGOptions(
                maxiter=args.maxiter,
                gtol=problem_gtol,
                record_history=True,
            )
            x0 = runtime.asarray(problem.start)
            preconditioner_name, preconditioner = choose_preconditioner(
                problem,
                x0,
            )
            solver = make_solver(
                problem.objective,
                preconditioner=preconditioner,
                options=options,
            )
            result, milliseconds = timed_run(solver, x0, repeat=args.repeat)
            save_convergence_history(
                problem_name=problem.name,
                minimum=problem.minimum,
                preconditioner_name=preconditioner_name,
                precision=args.precision,
                result=result,
            )
            rows.append(
                Row(
                    name=problem.name,
                    preconditioner=preconditioner_name,
                    status=status_message(result.status),
                    nit=int(result.nit),
                    nfev=int(result.nfev),
                    gtol=problem_gtol,
                    value=float(result.fun),
                    grad_norm=float(result.grad_norm),
                    milliseconds=milliseconds,
                )
            )

        diagonal_np = ill_conditioned_quadratic_diagonal(
            args.quadratic_size, args.quadratic_condition
        )
        diagonal = runtime.asarray(diagonal_np)
        x0 = runtime.asarray(np.ones(args.quadratic_size))
        quadratic_gtol = args.gtol if args.gtol is not None else default_gtol
        quadratic_options = PNCGOptions(
            maxiter=args.maxiter,
            gtol=quadratic_gtol,
        )

        for label, preconditioner_name, preconditioner in (
            (
                "Ill-conditioned quadratic",
                "identity",
                identity_preconditioner,
            ),
            (
                "Ill-conditioned quadratic",
                "fixed diagonal",
                diagonal_preconditioner(diagonal_np),
            ),
        ):
            solver = make_solver(
                diagonal_quadratic,
                preconditioner=preconditioner,
                options=quadratic_options,
            )
            result, milliseconds = timed_run(solver, x0, diagonal, repeat=args.repeat)
            rows.append(
                Row(
                    name=label,
                    preconditioner=preconditioner_name,
                    status=status_message(result.status),
                    nit=int(result.nit),
                    nfev=int(result.nfev),
                    gtol=quadratic_gtol,
                    value=float(result.fun),
                    grad_norm=float(result.grad_norm),
                    milliseconds=milliseconds,
                )
            )

    print(f"device={runtime.device}; precision={runtime.precision}")
    print(format_table(rows))


if __name__ == "__main__":
    main()

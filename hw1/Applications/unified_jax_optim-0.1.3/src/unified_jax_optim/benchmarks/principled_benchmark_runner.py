"""Launch each selected benchmark case in a fresh Python process.

This module deliberately imports only the Python standard library until a
backend-specific child process has started. In particular, do not add a JAX
or ``unified_jax_optim`` import at module scope: JAX chooses its backend when it
is imported, before command-line backend settings could otherwise take effect.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, cast

CaseKey: TypeAlias = Literal[
    "batched_relaxation",
    "implicit_dynamics",
    "occupancy_inversion",
    "implicit_diffusion",
    "shallow_arch",
]
CaseSize: TypeAlias = Literal["small", "full"]
BenchmarkMode: TypeAlias = Literal["prepared", "one-shot"]
Backend: TypeAlias = Literal["cpu", "gpu"]
DTypeName: TypeAlias = Literal["float32", "float64"]

CASE_KEYS: Final[tuple[CaseKey, ...]] = (
    "batched_relaxation",
    "implicit_dynamics",
    "occupancy_inversion",
    "implicit_diffusion",
    "shallow_arch",
)
METHODS: Final[tuple[str, ...]] = (
    "adam",
    "newton",
    "lbfgs",
    "cg",
    "trust-region",
)


@dataclass(frozen=True)
class OptimizerChoice:
    """A method name and plain option mapping selected by the driver."""

    method: str
    options: Mapping[str, Any]


def choose_optimizer(
    case_key: CaseKey,
    size: CaseSize,
    backend: Backend,
    dtype: DTypeName,
) -> OptimizerChoice:
    """Choose one optimizer and its options for one benchmark case."""

    if case_key == "batched_relaxation":
        return OptimizerChoice(
            method="adam",
            options={
                "learning_rate": 0.01,
                "max_iter": 2_000,
                "gtol": 1.0e-6,
                "store_history": True,
            },
        )
    if case_key == "implicit_dynamics":
        return OptimizerChoice(
            method="newton",
            options={
                "max_iter": 50,
                "gtol": 1.0e-8,
                "damping": 1.0e-8,
                "store_history": True,
            },
        )
    if case_key == "occupancy_inversion":
        return OptimizerChoice(
            method="lbfgs",
            options={
                "max_iter": 100,
                "gtol": 1.0e-7,
                "history_size": 10,
                "store_history": True,
            },
        )
    if case_key == "implicit_diffusion":
        return OptimizerChoice(
            method="cg",
            options={
                "max_iter": 200,
                "gtol": 1.0e-6,
                "beta_method": "polak-ribiere+",
                "line_search": "strong-wolfe",
                "store_history": True,
            },
        )
    if case_key == "shallow_arch":
        return OptimizerChoice(
            method="trust-region",
            options={
                "max_iter": 100,
                "gtol": 1.0e-8,
                "initial_radius": 1.0,
                "store_history": True,
            },
        )
    raise ValueError(f"Unknown case key: {case_key}")


def _validate_choice(case_key: CaseKey, choice: OptimizerChoice) -> dict[str, Any]:
    if not isinstance(choice, OptimizerChoice):
        raise TypeError(
            f"choose_optimizer({case_key!r}, ...) must return OptimizerChoice."
        )
    if choice.method not in METHODS:
        raise ValueError(
            f"Unknown optimizer {choice.method!r} for {case_key!r}; "
            f"choose from {METHODS}."
        )
    if not isinstance(choice.options, Mapping):
        raise TypeError(f"Options for {case_key!r} must be a mapping.")
    options = dict(choice.options)
    if not all(isinstance(key, str) for key in options):
        raise TypeError(f"Option names for {case_key!r} must be strings.")
    try:
        json.dumps(options, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"Options for {case_key!r} must contain JSON-serializable values."
        ) from error
    return options


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run selected physics benchmark cases in isolated processes."
    )
    parser.add_argument(
        "--size",
        choices=("small", "full"),
        default="full",
        help="problem size (default: full)",
    )
    parser.add_argument(
        "--case",
        choices=CASE_KEYS,
        action="append",
        dest="cases",
        help="case to run; repeat this option to select multiple cases",
    )
    parser.add_argument(
        "--mode",
        choices=("prepared", "one-shot"),
        default="prepared",
        help="whether setup is excluded from or included in each timed run",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="measured runs per starting condition (default: 5)",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="warmup runs per starting condition (default: 1)",
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        default="cpu",
        help="required JAX backend (default: cpu)",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="floating-point dtype (default: float64)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark-results",
        help="directory made available to benchmark recording code",
    )

    # These arguments are an implementation detail of the isolated child.
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_method", help=argparse.SUPPRESS)
    parser.add_argument("--_options-json", help=argparse.SUPPRESS)
    return parser


def _validate_counts(repeats: int, warmups: int) -> None:
    if repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if warmups < 0:
        raise ValueError("--warmups must be non-negative.")


def _selected_cases(values: Sequence[str] | None) -> tuple[CaseKey, ...]:
    if values is None:
        return CASE_KEYS
    if len(set(values)) != len(values):
        raise ValueError("A case may be selected only once.")
    return cast(tuple[CaseKey, ...], tuple(values))


def _child_main(args: argparse.Namespace) -> int:
    """Import JAX-dependent code only after the child environment is active."""

    _validate_counts(args.repeats, args.warmups)
    cases = _selected_cases(args.cases)
    if len(cases) != 1:
        raise ValueError("An isolated benchmark child must receive exactly one case.")
    if args._method is None or args._options_json is None:
        raise ValueError("The isolated benchmark child is missing optimizer data.")

    decoded = json.loads(args._options_json)
    if not isinstance(decoded, dict):
        raise TypeError("The isolated benchmark child expected an option mapping.")

    # This is intentionally the first import from the package in this process.
    from unified_jax_optim.benchmarks.principled_benchmark import run_case

    run_case(
        case_key=cases[0],
        size=args.size,
        method=args._method,
        options=decoded,
        mode=args.mode,
        repeats=args.repeats,
        warmups=args.warmups,
        backend=args.backend,
        dtype=args.dtype,
        output_dir=args.output_dir,
    )
    return 0


def _child_environment(backend: Backend, dtype: DTypeName) -> dict[str, str]:
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu" if backend == "cpu" else "cuda"
    environment["JAX_ENABLE_X64"] = "true" if dtype == "float64" else "false"
    if backend == "gpu":
        # Reserving most device memory at import time is fragile on shared
        # workstation and laptop GPUs. Allocate benchmark buffers on demand.
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment.pop("JAX_PLATFORM_NAME", None)

    # Prefer the source tree containing this runner over another installed copy.
    source_root = str(Path(__file__).resolve().parents[2])
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not old_pythonpath else source_root + os.pathsep + old_pythonpath
    )
    return environment


def _parent_main(args: argparse.Namespace) -> int:
    _validate_counts(args.repeats, args.warmups)
    selected = _selected_cases(args.cases)
    runner_path = Path(__file__).resolve()
    output_dir = Path(args.output_dir).resolve()

    for case_key in selected:
        choice = choose_optimizer(
            case_key,
            cast(CaseSize, args.size),
            cast(Backend, args.backend),
            cast(DTypeName, args.dtype),
        )
        options = _validate_choice(case_key, choice)
        command = [
            sys.executable,
            str(runner_path),
            "--_child",
            "--case",
            case_key,
            "--size",
            args.size,
            "--mode",
            args.mode,
            "--repeats",
            str(args.repeats),
            "--warmups",
            str(args.warmups),
            "--backend",
            args.backend,
            "--dtype",
            args.dtype,
            "--output-dir",
            str(output_dir),
            "--_method",
            choice.method,
            "--_options-json",
            json.dumps(options, separators=(",", ":"), allow_nan=False),
        ]
        print(
            f"Running {case_key} with {choice.method} "
            f"({args.backend}, {args.dtype})...",
            flush=True,
        )
        subprocess.run(
            command,
            check=True,
            env=_child_environment(
                cast(Backend, args.backend), cast(DTypeName, args.dtype)
            ),
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args._child:
            return _child_main(args)
        return _parent_main(args)
    except subprocess.CalledProcessError as error:
        print(
            f"Benchmark child exited with status {error.returncode}.",
            file=sys.stderr,
        )
        return error.returncode or 1
    except (json.JSONDecodeError, NotImplementedError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 2  # pragma: no cover - argparse.error raises SystemExit


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())

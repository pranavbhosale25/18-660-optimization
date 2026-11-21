"""Command-line training, inference, and benchmarks; no notebook runtime."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax

from .benchmark import (
    DEFAULT_EVALUATION_HORIZONS,
    DEFAULT_TEST_SIZE,
    DEFAULT_TIMING_REPETITIONS,
    evaluation_seed,
    run_benchmarks,
    write_json,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .evaluation import evaluate_points, reference_summary, summarize_metrics
from .problems import PROBLEM_NAMES, make_problem
from .reference import solve_reference
from .solver import predict
from .training import DEFAULT_TRAIN_CONFIG, TrainConfig, train

_DEFAULT_REFINEMENT_STEPS = 20


def _runtime_options(parser, *, checkpoint=False, jit=True):
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default=None if checkpoint else "float64",
        help=(
            "Array/computation precision; evaluation preserves checkpoint "
            "precision by default"
        ),
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, gpu, cpu:N, or gpu:N"
    )
    if jit:
        parser.add_argument(
            "--no-jit",
            action="store_true",
            help="Disable whole-function JIT compilation",
        )


def _training_options(
    parser, *, validate_every=DEFAULT_TRAIN_CONFIG.validate_every
):
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_TRAIN_CONFIG.steps,
        help="Outer parameter updates",
    )
    parser.add_argument(
        "--train-size", type=int, default=DEFAULT_TRAIN_CONFIG.train_size
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=DEFAULT_TRAIN_CONFIG.validation_size,
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_TRAIN_CONFIG.batch_size
    )
    parser.add_argument("--width", type=int, default=DEFAULT_TRAIN_CONFIG.width)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_TRAIN_CONFIG.learning_rate,
    )
    parser.add_argument(
        "--grad-clip", type=float, default=DEFAULT_TRAIN_CONFIG.grad_clip
    )
    parser.add_argument(
        "--validate-every",
        type=int,
        default=validate_every,
        help=(
            "Updates between validation checks; the benchmark defaults to "
            "six update intervals plus the initial check"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TRAIN_CONFIG.seed,
        help=(
            "Training/data/model seed; use a different value for each "
            "independent training trial"
        ),
    )


def _evaluation_seed_option(parser):
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=None,
        help=(
            "Optional fixed base seed for final nominal data; shifted data "
            "use the next seed"
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="JAX learning-to-warm-start teaching experiments"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "devices", help="Show JAX backends/devices visible to this installation"
    )
    listing = sub.add_parser(
        "list", help="List the two applications and completed diagnostic"
    )
    _runtime_options(listing, jit=False)
    fitting = sub.add_parser(
        "train", help="Fit an initializer with a fixed refinement horizon"
    )
    fitting.add_argument(
        "--problem", choices=("toy",) + PROBLEM_NAMES, default="toy"
    )
    _runtime_options(fitting)
    _training_options(fitting)
    fitting.add_argument(
        "--unroll", type=int, default=DEFAULT_TRAIN_CONFIG.unroll
    )
    fitting.add_argument(
        "--loss",
        choices=("objective", "fixed_point", "regression"),
        default="objective",
    )
    fitting.add_argument(
        "--output", type=Path, default=Path("results/model.npz")
    )
    evaluation = sub.add_parser(
        "evaluate",
        help="Assess an inference checkpoint on independent instances",
    )
    evaluation.add_argument("checkpoint", type=Path)
    _runtime_options(evaluation, checkpoint=True)
    evaluation.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    evaluation.add_argument(
        "--refine", type=int, default=_DEFAULT_REFINEMENT_STEPS
    )
    evaluation.add_argument("--shift", action="store_true")
    _evaluation_seed_option(evaluation)
    evaluation.add_argument(
        "--output", type=Path, default=Path("results/evaluation.json")
    )
    comparison = sub.add_parser(
        "benchmark",
        help="Compare classical, direct, and refined learned starts",
    )
    comparison.add_argument(
        "--problems",
        nargs="+",
        choices=("toy",) + PROBLEM_NAMES,
        default=list(PROBLEM_NAMES),
    )
    _runtime_options(comparison)
    _training_options(comparison, validate_every=None)
    comparison.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    _evaluation_seed_option(comparison)
    comparison.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_EVALUATION_HORIZONS),
    )
    comparison.add_argument("--include-fixed-point", action="store_true")
    comparison.add_argument(
        "--timing-repetitions",
        type=int,
        default=DEFAULT_TIMING_REPETITIONS,
    )
    comparison.add_argument("--no-timing", action="store_true")
    comparison.add_argument("--no-plots", action="store_true")
    comparison.add_argument(
        "--output", type=Path, default=Path("results/benchmark")
    )
    args = parser.parse_args(argv)
    if hasattr(args, "test_size") and args.test_size < 1:
        parser.error("--test-size must be positive")
    if hasattr(args, "refine") and args.refine < 0:
        parser.error("--refine must be nonnegative")
    if hasattr(args, "timing_repetitions") and args.timing_repetitions < 1:
        parser.error("--timing-repetitions must be positive")
    if args.command == "devices":
        devices = [
            {
                "platform": d.platform,
                "id": d.id,
                "kind": d.device_kind,
                "name": str(d),
            }
            for d in jax.devices()
        ]
        print(
            json.dumps({"jax": jax.__version__, "devices": devices}, indent=2)
        )
    elif args.command == "list":
        for name in ("toy",) + PROBLEM_NAMES:
            problem = make_problem(name, dtype=args.dtype, device=args.device)
            print(
                f"{name:12s} context={problem.context_dim:2d} "
                f"decision={problem.decision_dim:2d} "
                f"convex={problem.convex} constraint={problem.constraint} "
                f"dtype={problem.dtype.name} device={problem.device}"
            )
    elif args.command == "train":
        problem = make_problem(
            args.problem, dtype=args.dtype, device=args.device
        )
        config = TrainConfig(
            steps=args.steps,
            unroll=args.unroll,
            loss=args.loss,
            width=args.width,
            batch_size=args.batch_size,
            train_size=args.train_size,
            validation_size=args.validation_size,
            learning_rate=args.learning_rate,
            grad_clip=args.grad_clip,
            validate_every=args.validate_every,
            seed=args.seed,
            jit=not args.no_jit,
        )
        print(json.dumps(problem.runtime.info()))
        result = train(
            problem,
            config,
            progress=lambda row: print(
                f"step {row['step']:4d} validation "
                f"loss={row['validation_loss']:.7g} "
                f"objective={row['validation_objective']:.7g}",
                flush=True,
            ),
        )
        save_checkpoint(result, problem, args.output)
        write_json(
            args.output.with_suffix(".json"),
            {
                "runtime": problem.runtime.info(),
                "config": asdict(config),
                "history": result.history,
                "best_step": result.best_step,
                "elapsed_seconds": result.elapsed_seconds,
                "compilation_seconds": result.compilation_seconds,
                "execution_seconds": result.execution_seconds,
                "reference_info": result.reference_info,
            },
        )
        print(
            f"Saved {args.output}; selected update {result.best_step}. "
            f"Compile {result.compilation_seconds:.3f}s; synchronized "
            f"execution {result.execution_seconds:.3f}s; total "
            f"{result.elapsed_seconds:.3f}s."
        )
    elif args.command == "evaluate":
        problem, result = load_checkpoint(
            args.checkpoint, dtype=args.dtype, device=args.device
        )
        selected_evaluation_seed = evaluation_seed(
            result.config.seed,
            args.shift,
            base_seed=args.evaluation_seed,
        )
        contexts = problem.sample(
            args.test_size,
            problem.key(selected_evaluation_seed),
            shift=args.shift,
        )
        points = predict(
            problem, result.params, contexts, args.refine, jit=not args.no_jit
        )
        reference = solve_reference(problem, contexts)
        report = {
            "problem": problem.name,
            "runtime": problem.runtime.info(),
            "refinement_steps": args.refine,
            "test_size": args.test_size,
            "distribution": "shift" if args.shift else "test",
            "training_seed": result.config.seed,
            "evaluation_seed": selected_evaluation_seed,
            "quality_precision": "CPU float64; original decisions unchanged",
            "reference": reference_summary(reference),
            **summarize_metrics(
                evaluate_points(problem, points, contexts, reference)
            ),
        }
        write_json(args.output, report)
        print(args.output.read_text(encoding="utf-8"))
    else:
        run_benchmarks(
            args.output,
            args.problems,
            dtype=args.dtype,
            device=args.device,
            jit=not args.no_jit,
            train_steps=args.steps,
            train_size=args.train_size,
            validation_size=args.validation_size,
            batch_size=args.batch_size,
            width=args.width,
            learning_rate=args.learning_rate,
            grad_clip=args.grad_clip,
            validate_every=args.validate_every,
            test_size=args.test_size,
            seed=args.seed,
            evaluation_seed_base=args.evaluation_seed,
            horizons=tuple(args.horizons),
            include_fixed_point=args.include_fixed_point,
            timing_repetitions=args.timing_repetitions,
            timing=not args.no_timing,
            make_plot=not args.no_plots,
        )
        print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()

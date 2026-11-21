"""Train two warm starts, assess independent instances, and save a run.

Example (from the project root):
    python examples/walkthrough.py --problem toy --dtype float64 --device cpu \
        --steps 100 --output results/toy_walkthrough

The completed toy is the default, including in the student export.
Application choices run after their mathematical TODOs are completed.
Matplotlib is optional; all numerical tables and checkpoints are saved even
without it.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Prefer the adjacent source tree, including when this is a student export.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if (PROJECT_ROOT / "src").is_dir():
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import jax
import numpy as np

from l2ws_jax.checkpoint import load_checkpoint, save_checkpoint
from l2ws_jax.problems import PROBLEM_NAMES, make_problem
from l2ws_jax.reference import solve_reference
from l2ws_jax.solver import (
    fixed_point_residual,
    gradient_mapping_norm,
    make_baseline_solver,
    make_predictor,
    predict,
)
from l2ws_jax.training import TrainConfig, train

LABELS = {
    "baseline": "Classical start",
    "objective_k0": "Objective training, K=0",
    "objective_k5": "Objective training, K=5",
}
COLORS = {
    "baseline": "#566573",
    "objective_k0": "#1178AA",
    "objective_k5": "#8739A5",
}
_UPPER_TAIL_QUANTILE = 0.90
_TIMING_REPETITIONS = 5
_DEFAULT_TRAINING_STEPS = 100
_DEFAULT_TEST_SIZE = 16
_DEFAULT_REFINEMENT_STEPS = 20
_DEFAULT_SEED = 7
_MAX_RANDOM_SEED = 2**32
_TRAINING_HORIZONS = (0, 5)
_MODEL_WIDTH = 32
_TRAINING_SIZE = 512
_VALIDATION_SIZE = 64
_BATCH_SIZE = 64
_MAX_VALIDATION_INTERVAL = 25
_EVALUATION_SEED_OFFSET = 100_003
_CANDIDATE_REFINEMENT_HORIZONS = (1, 2, 5, 10, 20, 50)
_CHECKPOINT_TOLERANCE_ULPS = 32


def _statistics(values):
    """Keep absent diagnostics explicit rather than ignoring failures."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    complete = bool(finite.all()) and values.size > 0
    return {
        "mean": float(np.mean(values)) if complete else None,
        "median": float(np.median(values)) if complete else None,
        "p90": float(np.quantile(values, _UPPER_TAIL_QUANTILE))
        if complete
        else None,
        "finite_fraction": float(np.mean(finite)) if values.size else 0.0,
    }


def _reference_description(reference):
    finite = reference.certificates[np.isfinite(reference.certificates)]
    return {
        "label": reference.label,
        "elapsed_seconds": reference.elapsed_seconds,
        "compilation_seconds": reference.compilation_seconds,
        "solver_success_fraction": float(np.mean(reference.solver_success)),
        "globally_certified_fraction": float(
            np.mean(reference.globally_certified)
        ),
        "max_certificate": float(np.max(finite)) if finite.size else None,
        "tolerance": reference.tolerance,
    }


def _make_quality_evaluator(problem64):
    """Evaluate actual decisions in CPU float64 without repairing them first."""

    def evaluate(decisions, context, reference):
        values = problem64.objective(decisions, context)
        metrics = {
            "objective": values,
            "objective_difference": values
            - problem64.objective(reference, context),
            "solution_distance": problem64.solution_distance(
                decisions, reference
            ),
            "fixed_point_residual": fixed_point_residual(
                problem64, decisions, context
            ),
            "gradient_mapping_norm": gradient_mapping_norm(
                problem64, decisions, context
            ),
            "feasibility": problem64.feasibility(decisions, context),
            "certificate": problem64.certificate(decisions, context),
        }
        metrics.update(problem64.extra_metrics(decisions, context))
        lower_bound = getattr(problem64, "objective_lower_bound", None)
        if lower_bound is not None:
            metrics["objective_gap_upper_bound"] = values - lower_bound
        return metrics

    return jax.jit(evaluate)


def _compile_solvers(problem, models, example_context, horizons):
    """Compile shared shapes once; time compilation and warmup separately."""
    executables, setup = {}, []
    for horizon in horizons:
        for kind in ("baseline", "learned"):
            if kind == "baseline":
                function = make_baseline_solver(problem, horizon, jit=True)
                arguments = (example_context,)
            else:
                function = make_predictor(problem, horizon, jit=True)
                arguments = (models["objective_k0"].params, example_context)
            started = time.perf_counter()
            compiled = function.lower(*arguments).compile()
            compilation = time.perf_counter() - started
            started = time.perf_counter()
            jax.block_until_ready(compiled(*arguments))
            warmup = time.perf_counter() - started
            executables[kind, horizon] = compiled
            setup.append(
                {
                    "solver_kind": kind,
                    "refinement_steps": horizon,
                    "lower_and_compile_seconds": compilation,
                    "warmup_seconds": warmup,
                }
            )
    return executables, setup


def _time_execution(function, arguments, *, repeats=_TIMING_REPETITIONS):
    durations, result = [], None
    # An untimed call gives every distribution/model the same warmup policy.
    jax.block_until_ready(function(*arguments))
    for _ in range(repeats):
        started = time.perf_counter()
        result = function(*arguments)
        jax.block_until_ready(result)
        durations.append(time.perf_counter() - started)
    return result, float(np.median(durations))


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_plots(output, rows, models):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Plots skipped: install the optional plots extra to enable "
            "Matplotlib.",
            flush=True,
        )
        return []
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    paths = []
    gaps = np.asarray(
        [row["objective_difference_mean"] for row in rows], dtype=float
    )
    signed = bool(np.any(gaps <= 0.0))
    scale = "symlog" if signed else "log"
    positive_magnitudes = np.abs(gaps[np.nonzero(gaps)])
    linear_threshold = (
        max(1e-12, float(positive_magnitudes.min()) / 2)
        if positive_magnitudes.size
        else 1e-12
    )
    markers = {"baseline": "o", "objective_k0": "s", "objective_k5": "^"}
    styles = {"baseline": "--", "objective_k0": "-.", "objective_k5": "-"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for axis, distribution in zip(axes, ("nominal", "shifted")):
        for method, label in LABELS.items():
            selected = [
                row
                for row in rows
                if row["distribution"] == distribution
                and row["method"] == method
            ]
            axis.plot(
                [row["refinement_steps"] for row in selected],
                [row["objective_difference_mean"] for row in selected],
                color=COLORS[method],
                marker=markers[method],
                linestyle=styles[method],
                linewidth=2,
                label=label,
            )
        if signed:
            axis.axhline(0.0, color="#ADB5BD", linewidth=0.8)
            axis.set_yscale("symlog", linthresh=linear_threshold)
        else:
            axis.set_yscale("log")
        axis.set_xticks(sorted({row["refinement_steps"] for row in rows}))
        axis.set_title("Independent " + distribution + " contexts")
        axis.set_xlabel("Classical refinement steps after initialization")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(f"Mean objective minus numerical reference ({scale})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Comparing initialization and classical refinement")
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    for extension in ("png", "svg"):
        path = output / ("refinement." + extension)
        fig.savefig(path, dpi=170, bbox_inches="tight")
        paths.append(path.name)
    plt.close(fig)

    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for axis, (method, result) in zip(axes, models.items()):
        axis.plot(
            [row["step"] for row in result.history],
            [row["validation_objective"] for row in result.history],
            marker=markers[method],
            color=COLORS[method],
            linewidth=2,
        )
        axis.set_xlabel("Outer training updates")
        axis.set_ylabel("Validation objective after the training horizon")
        axis.set_title(LABELS[method])
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        axis.grid(alpha=0.2)
    fig.suptitle("Training progress at each model's own refinement horizon")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for extension in ("png", "svg"):
        path = output / ("training." + extension)
        fig.savefig(path, dpi=170, bbox_inches="tight")
        paths.append(path.name)
    plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem", choices=("toy", *PROBLEM_NAMES), default="toy"
    )
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, gpu, or an indexed backend"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=_DEFAULT_TRAINING_STEPS,
        help="outer updates per model",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=_DEFAULT_TEST_SIZE,
        help="instances per evaluation distribution",
    )
    parser.add_argument(
        "--refine",
        type=int,
        default=_DEFAULT_REFINEMENT_STEPS,
        help="largest plotted refinement horizon",
    )
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument(
        "--output", type=Path, default=Path("results/walkthrough")
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.steps < 1
        or args.test_size < 1
        or args.refine < 0
        or not 0 <= args.seed < _MAX_RANDOM_SEED
    ):
        parser.error(
            "steps/test-size must be positive, refine nonnegative, and seed "
            "in [0, 2**32)"
        )
    output = args.output.resolve()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            parser.error(
                f"--output must name a new or empty directory: {output}"
            )
    else:
        output.mkdir(parents=True)
    problem = make_problem(args.problem, dtype=args.dtype, device=args.device)
    print(
        "Runtime:",
        json.dumps(problem.runtime.info(), sort_keys=True),
        flush=True,
    )
    print(
        f"One context has {problem.context_dim} observed coordinates; "
        f"one decision has {problem.decision_dim} coordinates.",
        flush=True,
    )

    models, histories = {}, []
    for horizon in _TRAINING_HORIZONS:
        method = f"objective_k{horizon}"
        config = TrainConfig(
            steps=args.steps,
            unroll=horizon,
            loss="objective",
            width=_MODEL_WIDTH,
            train_size=_TRAINING_SIZE,
            validation_size=_VALIDATION_SIZE,
            batch_size=_BATCH_SIZE,
            validate_every=min(_MAX_VALIDATION_INTERVAL, args.steps),
            seed=args.seed,
            jit=True,
        )

        def progress(row, name=method):
            print(
                f"{name}: update {row['step']:4d}, "
                f"validation objective {row['validation_objective']:.6g}",
                flush=True,
            )

        result = train(problem, config, progress=progress)
        models[method] = result
        save_checkpoint(result, problem, output / (method + ".npz"))
        histories.extend({"method": method, **row} for row in result.history)

    # Independent from both models' identical training/validation streams.
    test_seed = (args.seed + _EVALUATION_SEED_OFFSET) % _MAX_RANDOM_SEED
    nominal_key, shifted_key = jax.random.split(problem.key(test_seed))
    contexts = {
        "nominal": problem.sample(args.test_size, nominal_key),
        "shifted": problem.sample(args.test_size, shifted_key, shift=True),
    }
    print("Evaluation batch shape:", contexts["nominal"].shape, flush=True)
    horizons = sorted(
        {
            0,
            args.refine,
            *(h for h in _CANDIDATE_REFINEMENT_HORIZONS if h <= args.refine),
        }
    )
    executables, inference_setup = _compile_solvers(
        problem, models, contexts["nominal"], horizons
    )

    # Preserve the actual fixed arrays and observations when changing precision.
    problem64 = problem.to(dtype="float64", device="cpu")
    quality = _make_quality_evaluator(problem64)
    rows, references, arrays, last_predictions = [], {}, {}, {}
    for distribution, x in contexts.items():
        reference = solve_reference(problem, x)
        references[distribution] = _reference_description(reference)
        x64 = problem64.array(jax.device_get(x))
        reference64 = problem64.array(reference.solution)
        arrays[f"{distribution}_contexts"] = np.asarray(jax.device_get(x))
        arrays[f"{distribution}_reference"] = reference.solution
        print(
            f"{distribution}: {reference.label}; "
            f"solver success fraction {np.mean(reference.solver_success):.3f}",
            flush=True,
        )
        for method in LABELS:
            kind = "baseline" if method == "baseline" else "learned"
            for horizon in horizons:
                arguments = (
                    (x,) if kind == "baseline" else (models[method].params, x)
                )
                decisions, batch_seconds = _time_execution(
                    executables[kind, horizon], arguments
                )
                host_decisions = np.asarray(jax.device_get(decisions))
                arrays[f"{distribution}_{method}_k{horizon}"] = host_decisions
                assessed = jax.device_get(
                    quality(problem64.array(host_decisions), x64, reference64)
                )
                if not np.isfinite(assessed["objective"]).all():
                    raise FloatingPointError(
                        "A computed decision has a nonfinite assessed objective"
                    )
                row = {
                    "problem": problem.name,
                    "distribution": distribution,
                    "method": method,
                    "refinement_steps": horizon,
                    "test_size": args.test_size,
                    "inference_dtype": problem.dtype.name,
                    "quality_dtype": "float64",
                    "quality_device": "cpu",
                    "batch_seconds_median": batch_seconds,
                    "seconds_per_instance_at_batch_size": batch_seconds
                    / args.test_size,
                }
                for metric, values in assessed.items():
                    row.update(
                        {
                            f"{metric}_{statistic}": value
                            for statistic, value in _statistics(values).items()
                        }
                    )
                rows.append(row)
                if horizon == args.refine:
                    last_predictions[distribution, method] = host_decisions
                    print(
                        f"  {method}: after {horizon} steps, mean reference "
                        "difference "
                        f"{row['objective_difference_mean']:.6g}; "
                        f"p90 {row['objective_difference_p90']:.6g}",
                        flush=True,
                    )

    # Loading exercises the portable inference contract without retraining.
    epsilon = np.finfo(problem.dtype).eps
    for method in models:
        restored_problem, restored_result = load_checkpoint(
            output / (method + ".npz"), device=args.device
        )
        restored = predict(
            restored_problem,
            restored_result.params,
            contexts["nominal"],
            steps=args.refine,
            jit=True,
        )
        np.testing.assert_allclose(
            np.asarray(jax.device_get(restored)),
            last_predictions["nominal", method],
            rtol=_CHECKPOINT_TOLERANCE_ULPS * epsilon,
            atol=_CHECKPOINT_TOLERANCE_ULPS * epsilon,
        )

    _write_csv(output / "metrics.csv", rows)
    _write_csv(output / "training_history.csv", histories)
    np.savez_compressed(output / "evaluation_arrays.npz", **arrays)
    plots = [] if args.no_plots else _save_plots(output, rows, models)
    summary = {
        "problem": problem.name,
        "problem_config": problem.config,
        "runtime": problem.runtime.info(),
        "python": sys.version.split()[0],
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "optax", "numpy", "scipy")
        },
        "test_seed": test_seed,
        "test_size_per_distribution": args.test_size,
        "horizons": horizons,
        "checkpoint_roundtrip_verified": True,
        "training": {
            name: {
                "config": asdict(result.config),
                "best_step": result.best_step,
                "elapsed_seconds": result.elapsed_seconds,
                "compilation_seconds": result.compilation_seconds,
                "execution_seconds": result.execution_seconds,
                "history": result.history,
            }
            for name, result in models.items()
        },
        "reference": references,
        "inference_setup": inference_setup,
        "metrics": rows,
        "plots": plots,
        "interpretation": [
            "Equal refinement budgets are compared; these timings do not "
            "establish equal-accuracy speedup.",
            f"Batch medians use {_TIMING_REPETITIONS} synchronized calls on "
            "resident data; batch time divided by count is throughput, not "
            "single-query latency.",
            "Inference lowering/compilation and warmup are reported "
            "separately; model and data loading are outside repeated "
            "execution timing.",
            "Quality is assessed on CPU float64 using unchanged computed "
            "decisions and the same stored fixed data and contexts.",
            "Negative differences from a numerical reference are preserved; "
            "nonconvex references are best-found solutions.",
            "A fixed-point residual diagnoses stationarity at the scale of "
            "the step; inspect the gradient-mapping norm and available "
            "certificates.",
            "Phase objective nonnegativity gives a global gap upper bound; "
            "exact equality requires noiseless consistency, with a rounding "
            "floor possible for finite-precision observations.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        "Saved tables, two verified checkpoints, evaluation arrays, and "
        f"{len(plots)} plot files to {output}",
        flush=True,
    )
    print(
        "Use the full benchmark and assigned test counts/seeds for graded "
        "conclusions.",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    main()

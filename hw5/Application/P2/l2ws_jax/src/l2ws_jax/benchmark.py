"""Matched-instance comparisons with synchronized JAX timings and saved data."""

from __future__ import annotations

import csv
import json
import math
import numbers
import platform
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import jax
import numpy as np

from .evaluation import evaluate_points, reference_summary, summarize_metrics
from .problems import PROBLEM_NAMES, make_problem
from .runtime import synchronize
from .solver import make_baseline_solver, make_predictor
from .training import (
    DEFAULT_TRAIN_CONFIG,
    RANDOM_SEED_UPPER_BOUND,
    TrainConfig,
    train,
)

DEFAULT_TEST_SIZE = 32
DEFAULT_EVALUATION_HORIZONS = (0, 5, 20, 50)
DEFAULT_TIMING_REPETITIONS = 7
EVALUATION_SEED_OFFSET = 100
_EVALUATION_SEEDS_PER_TRAINING_SEED = 2
_VALIDATION_INTERVAL_COUNT = 6
_OBJECTIVE_TRAINING_HORIZONS = (0, 5)
_FIXED_POINT_TRAINING_HORIZON = 5
_MILLISECONDS_PER_SECOND = 1000.0


def write_json(path, data):
    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if math.isfinite(float(value)) else None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean(data), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def measure(
    function,
    args,
    *,
    jit=True,
    repetitions=DEFAULT_TIMING_REPETITIONS,
    return_output=False,
):
    """Inputs already reside on device. Synchronize every timed invocation."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    synchronize(args)
    compile_seconds = 0.0
    if jit:
        begun = time.perf_counter()
        compiled = function.lower(*args).compile()
        compile_seconds = time.perf_counter() - begun
    else:
        compiled = function
    begun = time.perf_counter()
    output = synchronize(
        compiled(*args)
    )  # also warm caches before repeated timings
    first_execution_seconds = time.perf_counter() - begun
    samples = []
    for _ in range(repetitions):
        begun = time.perf_counter()
        output = synchronize(compiled(*args))
        samples.append(time.perf_counter() - begun)
    result = {
        "lower_and_compile_seconds": compile_seconds,
        "first_execution_seconds": first_execution_seconds,
        "latency_ms": _MILLISECONDS_PER_SECOND * statistics.median(samples),
        "repetitions": repetitions,
    }
    return (result, output) if return_output else result


def benchmark_problem(
    name,
    output,
    *,
    dtype="float64",
    device="auto",
    jit=True,
    train_steps=DEFAULT_TRAIN_CONFIG.steps,
    train_size=DEFAULT_TRAIN_CONFIG.train_size,
    validation_size=DEFAULT_TRAIN_CONFIG.validation_size,
    batch_size=DEFAULT_TRAIN_CONFIG.batch_size,
    width=DEFAULT_TRAIN_CONFIG.width,
    learning_rate=DEFAULT_TRAIN_CONFIG.learning_rate,
    grad_clip=DEFAULT_TRAIN_CONFIG.grad_clip,
    validate_every=None,
    test_size=DEFAULT_TEST_SIZE,
    seed=DEFAULT_TRAIN_CONFIG.seed,
    evaluation_seed_base=None,
    horizons=DEFAULT_EVALUATION_HORIZONS,
    include_fixed_point=False,
    timing_repetitions=DEFAULT_TIMING_REPETITIONS,
    timing=True,
    progress=print,
):
    from .checkpoint import save_checkpoint
    from .reference import solve_reference

    horizons = tuple(horizons)
    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, numbers.Integral)
        or test_size < 1
        or not horizons
        or any(
            isinstance(horizon, bool)
            or not isinstance(horizon, numbers.Integral)
            or horizon < 0
            for horizon in horizons
        )
    ):
        raise ValueError(
            "test_size must be positive and horizons nonempty/"
            "nonnegative integers"
        )
    if (
        isinstance(timing_repetitions, bool)
        or not isinstance(timing_repetitions, numbers.Integral)
        or timing_repetitions < 1
    ):
        raise ValueError("timing_repetitions must be a positive integer")
    evaluation_seeds = {
        distribution: evaluation_seed(
            seed,
            shifted,
            base_seed=evaluation_seed_base,
        )
        for distribution, shifted in (("test", False), ("shift", True))
    }
    horizons = tuple(sorted(set(horizons)))
    validation_interval = (
        max(1, train_steps // _VALIDATION_INTERVAL_COUNT)
        if validate_every is None
        else validate_every
    )
    config = TrainConfig(
        steps=train_steps,
        train_size=train_size,
        validation_size=validation_size,
        batch_size=batch_size,
        width=width,
        learning_rate=learning_rate,
        grad_clip=grad_clip,
        seed=seed,
        jit=jit,
        validate_every=validation_interval,
    )
    config.validate()
    problem = make_problem(name, dtype=dtype, device=device)
    output = Path(output) / name
    output.mkdir(parents=True, exist_ok=False)
    specifications = [
        (f"learned_K{horizon}_objective", horizon, "objective")
        for horizon in _OBJECTIVE_TRAINING_HORIZONS
    ]
    if include_fixed_point:
        specifications.append(
            (
                f"learned_K{_FIXED_POINT_TRAINING_HORIZON}_fixed_point",
                _FIXED_POINT_TRAINING_HORIZON,
                "fixed_point",
            )
        )
    trained, training_metadata = {}, {}
    for label, horizon, loss in specifications:
        if progress:
            progress(
                f"[{name}/{problem.dtype.name}] training {label}, "
                f"{train_steps} updates"
            )
        setting = replace(config, unroll=horizon, loss=loss)
        result = train(problem, setting)
        trained[label] = result
        save_checkpoint(result, problem, output / f"{label}.npz")
        record = {
            "config": asdict(setting),
            "history": result.history,
            "elapsed_seconds": result.elapsed_seconds,
            "compilation_seconds": result.compilation_seconds,
            "execution_seconds": result.execution_seconds,
            "best_step": result.best_step,
            "reference_info": result.reference_info,
        }
        training_metadata[label] = record
        write_json(output / f"{label}_training.json", record)

    rows, references = [], {}
    for distribution, shifted in (("test", False), ("shift", True)):
        current_evaluation_seed = evaluation_seeds[distribution]
        contexts = problem.sample(
            test_size,
            problem.key(current_evaluation_seed),
            shift=shifted,
        )
        synchronize(contexts)
        if progress:
            progress(
                f"[{name}/{problem.dtype.name}] {distribution}: "
                f"{test_size} CPU64 references"
            )
        reference = solve_reference(problem, contexts)
        references[distribution] = reference_summary(reference)
        np.savez_compressed(
            output / f"{distribution}_data.npz",
            contexts=np.asarray(jax.device_get(contexts)),
            reference=reference.solution,
            reference_objective=reference.objective,
            reference_certificate=reference.certificates,
        )
        methods = [("baseline", None)] + [
            (label, run.params) for label, run in trained.items()
        ]
        for label, params in methods:
            for horizon in horizons:
                function = (
                    make_baseline_solver(problem, horizon, jit=jit)
                    if params is None
                    else make_predictor(problem, horizon, jit=jit)
                )
                arguments = (
                    (contexts,) if params is None else (params, contexts)
                )
                timing_metrics = {}
                if timing:
                    measured_batch, points = measure(
                        function,
                        arguments,
                        jit=jit,
                        repetitions=timing_repetitions,
                        return_output=True,
                    )
                    single_arguments = (
                        (contexts[:1],)
                        if params is None
                        else (params, contexts[:1])
                    )
                    measured_single = measure(
                        function,
                        single_arguments,
                        jit=jit,
                        repetitions=timing_repetitions,
                    )
                    for suffix, measured in (
                        ("single", measured_single),
                        ("batch", measured_batch),
                    ):
                        timing_metrics[f"latency_{suffix}_ms"] = measured[
                            "latency_ms"
                        ]
                        timing_metrics[
                            f"lower_and_compile_{suffix}_seconds"
                        ] = measured["lower_and_compile_seconds"]
                        timing_metrics[f"first_execution_{suffix}_seconds"] = (
                            measured["first_execution_seconds"]
                        )
                    timing_metrics["timing_batch_size"] = test_size
                    timing_metrics["timing_repetitions"] = measured_batch[
                        "repetitions"
                    ]
                else:
                    points = synchronize(function(*arguments))
                metrics = evaluate_points(problem, points, contexts, reference)
                row = {
                    "problem": name,
                    "dtype": problem.dtype.name,
                    "backend": problem.device.platform,
                    "jit": jit,
                    "distribution": distribution,
                    "training_seed": seed,
                    "evaluation_seed": current_evaluation_seed,
                    "method": label,
                    "refinement_steps": horizon,
                    **summarize_metrics(metrics),
                    "reference_objective_mean": float(
                        np.mean(reference.objective)
                    ),
                    "reference_kind": reference.label,
                    **timing_metrics,
                }
                rows.append(row)
                np.savez_compressed(
                    output / f"{distribution}_{label}_K{horizon}.npz",
                    points=np.asarray(jax.device_get(points)),
                    **metrics,
                )
    with (output / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "problem": name,
        "convex": problem.convex,
        "runtime": problem.runtime.info(),
        "jit": jit,
        "problem_config": problem.config,
        "train_config": asdict(config),
        "evaluation_seeds": evaluation_seeds,
        "objective_lower_bound": getattr(
            problem, "objective_lower_bound", None
        ),
        "quality_evaluation": {
            "dtype": "float64",
            "backend": "cpu",
            "decisions_modified": False,
            "fixed_data": "preserved before casting",
        },
        "references": references,
        "training": training_metadata,
        "rows": rows,
        "timing_scope": (
            "Device-resident inputs and parameters. Initialization and "
            "refinement included; every call synchronized. Compilation/cache "
            "lookup and first execution reported separately. Warm latency "
            "excludes transfers and CPU64 quality/reference work. "
            "Equal iteration budgets do not establish equal-accuracy speedups."
        ),
        "interpretation": (
            "Signed differences from numerical references are preserved. "
            "Nonconvex references are best-found. Phase's nonnegative "
            "objective upper-bounds the gap; zero is the ideal noiseless "
            "optimum, with possible tiny consistency floors after rounding. "
            "Reusing a seed across different dtypes does not guarantee "
            "identical random draws."
        ),
    }
    write_json(output / "benchmark.json", payload)
    return payload


def plot_benchmarks(payloads, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "baseline": "#475569",
        "learned_K0_objective": "#ea580c",
        "learned_K5_objective": "#2563eb",
        "learned_K5_fixed_point": "#7c3aed",
    }
    labels = {
        "baseline": "Classical start",
        "learned_K0_objective": "Learned (K=0, objective)",
        "learned_K5_objective": "Learned (K=5, objective)",
        "learned_K5_fixed_point": "Learned (K=5, residual)",
    }
    fig, axes = plt.subplots(
        len(payloads), 2, figsize=(11, 2.9 * len(payloads)), squeeze=False
    )
    handles = {}
    for index, payload in enumerate(payloads):
        metric = (
            "objective_gap_upper_bound_mean"
            if payload.get("objective_lower_bound") is not None
            else "objective_difference_mean"
        )
        for column, distribution in enumerate(("test", "shift")):
            ax = axes[index, column]
            all_values = []
            for method in dict.fromkeys(
                row["method"] for row in payload["rows"]
            ):
                selected = [
                    row
                    for row in payload["rows"]
                    if row["distribution"] == distribution
                    and row["method"] == method
                ]
                x = [row["refinement_steps"] for row in selected]
                y = [row[metric] for row in selected]
                (line,) = ax.plot(
                    x, y, "o-", color=colors[method], lw=1.8, ms=4
                )
                handles[method] = line
                all_values.extend(y)
            ax.set_yscale("symlog", linthresh=1e-8)
            ax.set_ylim(
                min(0.0, min(all_values) * 1.3),
                max(1e-8, max(all_values) * 1.5),
            )
            ax.set_title(
                f"{payload['problem']} - "
                f"{'held-out' if column == 0 else 'shifted'}",
                loc="left",
            )
            ax.set_xlabel("Refinement steps")
            ax.set_ylabel(
                "Objective gap upper bound"
                if metric.startswith("objective_gap")
                else "Objective - numerical reference"
            )
            ax.grid(alpha=0.2)
    runtime = payloads[0]["runtime"]
    fig.suptitle(
        f"JAX learned warm starts | {runtime['dtype']} | {runtime['backend']}",
        fontsize=15,
    )
    fig.legend(
        handles.values(),
        [labels[name] for name in handles],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.97))
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluation_seed(training_seed, shifted, *, base_seed=None):
    """Return a bounded seed for nominal or shifted final-test data.

    Without a base seed, each training seed receives a disjoint adjacent
    nominal/shifted pair for ordinary consecutive trials. Supplying a base
    seed holds final-test instances fixed while training seeds vary.
    """
    for label, value in (
        ("training_seed", training_seed),
        ("base_seed", base_seed),
    ):
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            or not 0 <= value < RANDOM_SEED_UPPER_BOUND
        ):
            raise ValueError(
                f"{label} must be an integer in [0, {RANDOM_SEED_UPPER_BOUND})"
            )
    if not isinstance(shifted, (bool, np.bool_)):
        raise ValueError("shifted must be a boolean")
    if base_seed is None:
        seed = (
            _EVALUATION_SEEDS_PER_TRAINING_SEED * int(training_seed)
            + EVALUATION_SEED_OFFSET
        )
    else:
        seed = int(base_seed)
    return (seed + int(shifted)) % RANDOM_SEED_UPPER_BOUND


def run_benchmarks(output, names=PROBLEM_NAMES, *, make_plot=True, **kwargs):
    output = Path(output)
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(
                f"Benchmark output directory must be new or empty: {output}"
            )
    names = tuple(names)
    if not names:
        raise ValueError("At least one benchmark problem is required")
    if len(set(names)) != len(names):
        raise ValueError("Benchmark problem names must be unique")
    allowed = {"toy", *PROBLEM_NAMES}
    unknown = set(names) - allowed
    if unknown:
        raise ValueError(f"Unknown benchmark problems: {sorted(unknown)}")
    if not output.exists():
        output.mkdir(parents=True)
    payloads = [benchmark_problem(name, output, **kwargs) for name in names]
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "numpy": np.__version__,
    }
    write_json(
        output / "summary.json", {"environment": metadata, "problems": payloads}
    )
    if make_plot:
        try:
            plot_benchmarks(payloads, output / "benchmark.png")
        except ImportError:
            progress = kwargs.get("progress", print)
            if progress:
                progress(
                    "Results saved. Install the 'plots' extra to generate "
                    "the optional figure."
                )
    return payloads

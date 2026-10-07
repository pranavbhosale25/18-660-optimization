"""Execution scaffold for the computational-physics benchmark.

This module deliberately separates benchmark *execution* from benchmark
*policy*. The supplied code prepares optimizers, performs synchronized
warmups and measured runs, and repeats the experiment from robustness starts.
The marked student section decides which results to retain, summarize, print,
or write to disk.

The outer driver configures the JAX backend and floating-point mode before it
imports this module, then calls :func:`run_case` once in the child process.
"""

from __future__ import annotations

import gc
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Literal, TypeAlias

import jax
from hw1_optimizer_runtime import (
    BenchmarkMode,
    OptimizeResult,
    minimize,
    prepare_minimize,
    synchronize_result,
)

from .principled_physics import CaseKey, CaseSize, PhysicsCase, make_case

Backend: TypeAlias = Literal["cpu", "gpu"]
FloatingPoint: TypeAlias = Literal["float32", "float64"]
RunPhase: TypeAlias = Literal["nominal", "robustness"]
RunKind: TypeAlias = Literal["warmup", "measured"]
RunRecord: TypeAlias = dict[str, Any]
RecordingState: TypeAlias = list[RunRecord]


@dataclass(frozen=True)
class BenchmarkContext:
    """Configuration and environment shared by every run of one case."""

    case_key: CaseKey
    case_title: str
    size: CaseSize
    dimension: int
    method: str
    options: Mapping[str, Any]
    mode: BenchmarkMode
    repeats: int
    warmups: int
    backend: Backend
    dtype: FloatingPoint
    output_dir: Path
    devices: tuple[str, ...]
    x64_enabled: bool


@dataclass(frozen=True)
class TimedObjective:
    """One objective evaluation and its time since the solve began."""

    elapsed_seconds: float
    fun: float


class _ObjectiveTimeline:
    """Host-side collector called from the compiled objective."""

    def __init__(self) -> None:
        self._start_ns = 0
        self._samples: list[TimedObjective] = []

    def begin(self, start_ns: int) -> None:
        self._start_ns = start_ns
        self._samples.clear()

    def record(self, value: Any) -> None:
        if self._start_ns == 0:
            return
        self._samples.append(
            TimedObjective(
                elapsed_seconds=(perf_counter_ns() - self._start_ns) * 1.0e-9,
                fun=float(value),
            )
        )

    def finish(self) -> tuple[TimedObjective, ...]:
        samples = tuple(self._samples)
        self._start_ns = 0
        return samples


@dataclass(frozen=True)
class RunSample:
    """Unprocessed information from one synchronized optimizer execution."""

    case_key: CaseKey
    start_id: str
    start_index: int
    phase: RunPhase
    kind: RunKind
    repetition: int
    elapsed_seconds: float
    backend: Backend
    dtype: FloatingPoint
    minimum_value: float
    result: OptimizeResult
    objective_timeline: tuple[TimedObjective, ...]


def _json_float(value: Any) -> float | None:
    """Convert a scalar to a finite JSON number, or ``None`` if non-finite."""

    converted = float(value)
    return converted if math.isfinite(converted) else None


def begin_recording(context: BenchmarkContext) -> RecordingState:
    """Create any state needed while samples are received."""

    print(
        f"Recording {context.case_key}: method={context.method}, "
        f"size={context.size}, backend={context.backend}, dtype={context.dtype}"
    )
    return []


def record_run(state: RecordingState, sample: RunSample) -> None:
    """Receive one raw warmup or measured sample after synchronization."""

    result = sample.result
    objective = _json_float(result.fun)
    minimum = _json_float(sample.minimum_value)
    objective_gap = (
        objective - minimum if objective is not None and minimum is not None else None
    )

    history = [
        {
            "iteration": int(item.iteration),
            "fun": _json_float(item.fun),
            "grad_norm": _json_float(item.grad_norm),
            "step_norm": _json_float(item.step_norm),
            "accepted": bool(item.accepted),
            "step_size": (
                None if item.step_size is None else _json_float(item.step_size)
            ),
            "trust_radius": (
                None if item.trust_radius is None else _json_float(item.trust_radius)
            ),
            "reduction_ratio": (
                None
                if item.reduction_ratio is None
                else _json_float(item.reduction_ratio)
            ),
        }
        for item in result.history
    ]

    work = result.work
    state.append(
        {
            "case_key": sample.case_key,
            "start_id": sample.start_id,
            "start_index": sample.start_index,
            "phase": sample.phase,
            "kind": sample.kind,
            "repetition": sample.repetition,
            "elapsed_seconds": _json_float(sample.elapsed_seconds),
            "backend": sample.backend,
            "dtype": sample.dtype,
            "method": result.method,
            "success": bool(result.success),
            "status": int(result.status),
            "message": result.message,
            "fun": objective,
            "minimum_value": minimum,
            "objective_gap": objective_gap,
            "grad_norm": _json_float(result.grad_norm),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "njev": int(result.njev),
            "nhev": int(result.nhev),
            "nhvp": int(result.nhvp),
            "work": {
                "minor_iterations": int(work.minor_iterations),
                "line_search_calls": int(work.line_search_calls),
                "line_search_iterations": int(work.line_search_iterations),
                "line_search_zoom_iterations": int(work.line_search_zoom_iterations),
                "damping_iterations": int(work.damping_iterations),
                "trust_region_cg_iterations": int(work.trust_region_cg_iterations),
                "accepted_steps": int(work.accepted_steps),
                "rejected_steps": int(work.rejected_steps),
                "gradient_fallbacks": int(work.gradient_fallbacks),
                "direction_restarts": int(work.direction_restarts),
                "cauchy_fallbacks": int(work.cauchy_fallbacks),
            },
            "history": history,
            "objective_timeline": [
                {
                    "elapsed_seconds": point.elapsed_seconds,
                    "fun": _json_float(point.fun),
                }
                for point in sample.objective_timeline
            ],
        }
    )

    objective_text = "non-finite" if objective is None else f"{objective:.6e}"
    print(
        f"  {sample.kind:8s} {sample.start_id:12s} "
        f"rep={sample.repetition} success={bool(result.success)} "
        f"nit={int(result.nit)} f={objective_text} "
        f"time={sample.elapsed_seconds:.6f}s"
    )


def finish_recording(state: RecordingState, context: BenchmarkContext) -> None:
    """Summarize or persist the experiment after all starts have run."""

    measured = [record for record in state if record["kind"] == "measured"]
    measured_times = [
        record["elapsed_seconds"]
        for record in measured
        if record["elapsed_seconds"] is not None
    ]
    successful_runs = sum(bool(record["success"]) for record in measured)
    average_time = sum(measured_times) / len(measured_times) if measured_times else None

    payload = {
        "context": {
            "case_key": context.case_key,
            "case_title": context.case_title,
            "size": context.size,
            "dimension": context.dimension,
            "method": context.method,
            "options": dict(context.options),
            "mode": context.mode,
            "repeats": context.repeats,
            "warmups": context.warmups,
            "backend": context.backend,
            "dtype": context.dtype,
            "devices": list(context.devices),
            "x64_enabled": context.x64_enabled,
        },
        "summary": {
            "measured_runs": len(measured),
            "successful_runs": successful_runs,
            "average_elapsed_seconds": average_time,
        },
        "runs": state,
    }

    filename = (
        f"{context.case_key}-{context.method}-{context.size}-"
        f"{context.backend}-{context.dtype}.json"
    )
    output_path = context.output_dir / filename
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    plot_path = _save_convergence_outputs(measured, context)

    average_text = "n/a" if average_time is None else f"{average_time:.6f}s"
    print(
        f"Finished {context.case_key}: {successful_runs}/{len(measured)} "
        f"measured runs succeeded; average time={average_text}"
    )
    print(f"Results written to {output_path}")
    if plot_path is not None:
        print(f"Convergence plot written to {plot_path}")


def _save_convergence_outputs(
    measured: list[RunRecord],
    context: BenchmarkContext,
) -> Path | None:
    """Plot the first measured nominal run against iteration and real time."""

    nominal = next(
        (
            record
            for record in measured
            if record["start_id"] == "nominal" and record["repetition"] == 1
        ),
        None,
    )
    if nominal is None:
        return None

    history = [item for item in nominal["history"] if item["fun"] is not None]
    timeline = [
        item for item in nominal["objective_timeline"] if item["fun"] is not None
    ]
    if not history or not timeline:
        return None

    import matplotlib.pyplot as plt
    import numpy as np

    plt.switch_backend("Agg")
    minimum = float(nominal["minimum_value"])
    iterations = np.asarray([item["iteration"] for item in history])
    iteration_objective = np.asarray([item["fun"] for item in history])
    matched_times: list[float] = []
    matched_objective: list[float] = []
    timeline_index = 0
    for item in history:
        target = float(item["fun"])
        for index in range(timeline_index, len(timeline)):
            if math.isclose(
                float(timeline[index]["fun"]),
                target,
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            ):
                matched_times.append(float(timeline[index]["elapsed_seconds"]))
                matched_objective.append(target)
                timeline_index = index + 1
                break

    elapsed = np.asarray(matched_times)
    timed_objective = np.asarray(matched_objective)

    stem = (
        f"{context.case_key}-{context.method}-{context.size}-"
        f"{context.backend}-{context.dtype}"
    )
    np.savetxt(
        context.output_dir / f"{stem}-iterations.csv",
        np.column_stack(
            (
                iterations,
                iteration_objective,
                iteration_objective - minimum,
            )
        ),
        delimiter=",",
        header="iteration,objective,objective_gap",
        comments="",
    )
    np.savetxt(
        context.output_dir / f"{stem}-wall-time.csv",
        np.column_stack((elapsed, timed_objective, timed_objective - minimum)),
        delimiter=",",
        header="elapsed_seconds,objective,objective_gap",
        comments="",
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(iterations, iteration_objective)
    axes[0].set_xlabel("Major iteration")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss vs. major iteration")

    axes[1].plot(elapsed, timed_objective, marker=".", markersize=3)
    axes[1].set_xlabel("Wall-clock time (seconds)")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss vs. wall-clock time")

    for axis in axes:
        axis.axhline(
            minimum,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="known minimum",
        )
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle(f"{context.case_title}\n{context.method}, {context.dtype}")
    figure.tight_layout()
    plot_path = context.output_dir / f"{stem}-convergence.png"
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)
    return plot_path


def run_case(
    *,
    case_key: CaseKey,
    size: CaseSize,
    method: str,
    options: Mapping[str, Any] | None,
    mode: BenchmarkMode,
    repeats: int,
    warmups: int,
    backend: Backend,
    dtype: FloatingPoint,
    output_dir: str | Path,
) -> BenchmarkContext:
    """Execute one selected optimizer on one case and all of its starts.

    Warmup executions are synchronized and reported to :func:`record_run`, but
    they are identified separately from measured repetitions. Recording runs
    outside the measured interval, and JAX work retained in the recording state
    is synchronized before the next timed solve.
    """

    _validate_run_arguments(
        method=method,
        options=options,
        mode=mode,
        repeats=repeats,
        warmups=warmups,
        backend=backend,
        dtype=dtype,
    )
    devices = _verify_runtime(backend=backend, dtype=dtype)
    case = make_case(case_key, size=size, dtype=dtype)
    resolved_options = dict(options or {})
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    context = BenchmarkContext(
        case_key=case.key,
        case_title=case.title,
        size=size,
        dimension=case.dimension,
        method=method,
        options=resolved_options,
        mode=mode,
        repeats=repeats,
        warmups=warmups,
        backend=backend,
        dtype=dtype,
        output_dir=resolved_output_dir,
        devices=devices,
        x64_enabled=bool(jax.config.jax_enable_x64),
    )

    recording_state = begin_recording(context)
    _synchronize_recording_work(recording_state)
    starts = (("nominal", "nominal", case.x0),) + tuple(
        ("robustness", f"robustness-{index}", start)
        for index, start in enumerate(case.robustness_starts, start=1)
    )

    for start_index, (phase, start_id, start) in enumerate(starts):
        timeline = _ObjectiveTimeline()
        solve = _make_solver(
            case=case,
            start=start,
            method=method,
            options=resolved_options,
            mode=mode,
            timeline=timeline,
        )
        for repetition in range(1, warmups + 1):
            sample = _run_once(
                solve=solve,
                context=context,
                case=case,
                start_id=start_id,
                start_index=start_index,
                phase=phase,
                kind="warmup",
                repetition=repetition,
                timeline=timeline,
            )
            record_run(recording_state, sample)
            _synchronize_recording_work(recording_state)
        for repetition in range(1, repeats + 1):
            sample = _run_once(
                solve=solve,
                context=context,
                case=case,
                start_id=start_id,
                start_index=start_index,
                phase=phase,
                kind="measured",
                repetition=repetition,
                timeline=timeline,
            )
            record_run(recording_state, sample)
            _synchronize_recording_work(recording_state)

        # Prepared optimizers retain derivative transformations and compiled
        # executables. Drop each one before constructing the next start. A
        # child process handles only one case, so its complete JAX cache is
        # naturally released when the child exits.
        del solve
        gc.collect()

    finish_recording(recording_state, context)
    _synchronize_recording_work(recording_state)
    return context


def _make_solver(
    *,
    case: PhysicsCase,
    start: Any,
    method: str,
    options: Mapping[str, Any],
    mode: BenchmarkMode,
    timeline: _ObjectiveTimeline,
) -> Callable[[], OptimizeResult]:
    def timed_objective(x: Any) -> Any:
        value = case.fun(x)
        jax.debug.callback(timeline.record, value, ordered=True)
        return value

    if mode == "prepared":
        prepared = prepare_minimize(
            timed_objective,
            start,
            method,
            options=options,
        )
        return prepared.run

    def one_shot() -> OptimizeResult:
        return minimize(timed_objective, start, method, options=options)

    return one_shot


def _run_once(
    *,
    solve: Callable[[], OptimizeResult],
    context: BenchmarkContext,
    case: PhysicsCase,
    start_id: str,
    start_index: int,
    phase: RunPhase,
    kind: RunKind,
    repetition: int,
    timeline: _ObjectiveTimeline,
) -> RunSample:
    start_ns = perf_counter_ns()
    timeline.begin(start_ns)
    result = solve()
    synchronize_result(result)
    elapsed_seconds = (perf_counter_ns() - start_ns) * 1.0e-9
    objective_timeline = timeline.finish()
    return RunSample(
        case_key=case.key,
        start_id=start_id,
        start_index=start_index,
        phase=phase,
        kind=kind,
        repetition=repetition,
        elapsed_seconds=elapsed_seconds,
        backend=context.backend,
        dtype=context.dtype,
        minimum_value=case.minimum_value,
        result=result,
        objective_timeline=objective_timeline,
    )


def _synchronize_recording_work(state: Any) -> None:
    """Finish JAX work retained by recording code outside timed intervals."""

    jax.block_until_ready(state)
    jax.effects_barrier()


def _verify_runtime(*, backend: Backend, dtype: FloatingPoint) -> tuple[str, ...]:
    try:
        devices = tuple(jax.devices())
    except Exception as error:
        requested = "CUDA GPU" if backend == "gpu" else "CPU"
        raise RuntimeError(
            f"{requested} backend requested, but JAX could not initialize it: {error}"
        ) from error
    if not devices:
        raise RuntimeError("JAX did not report any usable devices.")

    default_backend = jax.default_backend()
    device_platforms = {device.platform for device in devices}
    if backend == "cpu":
        if default_backend != "cpu" or device_platforms != {"cpu"}:
            raise RuntimeError(
                "CPU mode requested, but JAX initialized non-CPU devices: "
                f"{sorted(device_platforms)}."
            )
    elif default_backend not in {"gpu", "cuda", "rocm"} or not (
        device_platforms & {"gpu", "cuda", "rocm"}
    ):
        raise RuntimeError(
            "GPU mode requested, but JAX did not initialize a GPU device. "
            f"Default backend is {default_backend!r}; devices are "
            f"{sorted(device_platforms)}."
        )

    x64_enabled = bool(jax.config.jax_enable_x64)
    expected_x64 = dtype == "float64"
    if x64_enabled != expected_x64:
        raise RuntimeError(
            f"Requested {dtype}, but jax_enable_x64 is {x64_enabled}. The "
            "driver must configure JAX_ENABLE_X64 before importing JAX."
        )
    return tuple(str(device) for device in devices)


def _validate_run_arguments(
    *,
    method: str,
    options: Mapping[str, Any] | None,
    mode: BenchmarkMode,
    repeats: int,
    warmups: int,
    backend: Backend,
    dtype: FloatingPoint,
) -> None:
    if not isinstance(method, str) or not method.strip():
        raise ValueError("method must be a non-empty optimizer name.")
    if options is not None and not isinstance(options, Mapping):
        raise TypeError("options must be a mapping or None.")
    if mode not in ("prepared", "one-shot"):
        raise ValueError("mode must be 'prepared' or 'one-shot'.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    if warmups < 0:
        raise ValueError("warmups must be non-negative.")
    if backend not in ("cpu", "gpu"):
        raise ValueError("backend must be 'cpu' or 'gpu'.")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'.")

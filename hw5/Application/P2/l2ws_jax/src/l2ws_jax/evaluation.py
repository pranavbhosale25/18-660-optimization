"""Independent float64 quality assessment, outside timed online computation."""

from functools import lru_cache

import jax
import numpy as np

from .solver import fixed_point_residual, gradient_mapping_norm


@lru_cache(maxsize=32)
def make_evaluator(problem):
    """Preserve stored problem data and assess unchanged decisions on CPU."""
    high_precision = problem.to(dtype="float64", device="cpu")

    def metrics(points, contexts, reference_points, reference_values):
        values = high_precision.objective(points, contexts)
        result = {
            "objective": values,
            "objective_difference": values - reference_values,
            "gradient_mapping": gradient_mapping_norm(
                high_precision, points, contexts
            ),
            "fixed_point_residual": fixed_point_residual(
                high_precision, points, contexts
            ),
            "feasibility": high_precision.feasibility(points, contexts),
            "certificate": high_precision.certificate(points, contexts),
            "reference_solution_distance": high_precision.solution_distance(
                points, reference_points
            ),
        }
        result.update(high_precision.extra_metrics(points, contexts))
        lower_bound = getattr(problem, "objective_lower_bound", None)
        if lower_bound is not None:
            result["objective_gap_upper_bound"] = values - lower_bound
        return result

    return high_precision, jax.jit(metrics)


def evaluate_points(problem, points, contexts, reference):
    """Return per-instance host arrays. No projection/repair changes the points.

    In particular, float32 projection roundoff remains visible in feasibility.
    CPU64 evaluation avoids comparing rounded float32 objective values against
    float64 references. It is not part of timed prediction or GPU training.
    """
    p64, function = make_evaluator(problem)
    arguments = [
        p64.array(np.asarray(jax.device_get(value), dtype=np.float64))
        for value in (points, contexts, reference.solution, reference.objective)
    ]
    result = jax.device_get(function(*arguments))
    return {name: np.asarray(values) for name, values in result.items()}


def summarize_metrics(metrics):
    row = {}
    for name, values in metrics.items():
        values = np.asarray(values, dtype=np.float64)
        if name == "feasibility":
            row["feasibility_max"] = float(np.max(values))
        else:
            for suffix, statistic in (
                ("mean", np.mean),
                ("median", np.median),
                ("p90", lambda x: np.quantile(x, 0.9)),
            ):
                row[f"{name}_{suffix}"] = float(statistic(values))
    return row


def reference_summary(reference):
    finite = np.asarray(reference.certificates)[
        np.isfinite(reference.certificates)
    ]
    return {
        "kind": reference.label,
        "precision": "float64",
        "backend": "cpu",
        "elapsed_seconds": float(reference.elapsed_seconds),
        "compilation_seconds": float(reference.compilation_seconds),
        "tolerance": float(reference.tolerance),
        "solver_success_fraction": float(np.mean(reference.solver_success)),
        "certified_fraction": float(np.mean(reference.globally_certified)),
        "max_certificate": float(finite.max()) if finite.size else None,
        "max_gradient_mapping": float(np.max(reference.stationarity)),
    }

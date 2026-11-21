"""Independent SciPy reference algorithms evaluated on the CPU in float64.

The optimizer is independent of the learned projected-gradient loop. Its pure
JAX objective/gradient evaluator is compiled once per call, with the context as
an argument, and reused by every SciPy solve. Returned arrays are NumPy host
arrays; training must explicitly cast labels into its chosen dtype/device.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .numerics import euclidean_norm
from .problems.base import Problem

DEFAULT_REFERENCE_TOLERANCE = 1e-7
DEFAULT_REFERENCE_MAX_ITERATIONS = 1000
DEFAULT_REFERENCE_SEED = 123
_DEFAULT_CONVEX_STARTS = 1
_DEFAULT_NONCONVEX_STARTS = 6
_SCIPY_FUNCTION_TOLERANCE = 1e-15
_SCIPY_GRADIENT_TOLERANCE = 1e-10
_SCIPY_MAX_LINE_SEARCH_STEPS = 50
_CONTEXT_BATCH_RANK = 2


@dataclass
class ReferenceResult:
    solution: np.ndarray
    objective: np.ndarray
    stationarity: np.ndarray
    certificates: np.ndarray
    solver_success: np.ndarray
    globally_certified: np.ndarray
    label: str
    elapsed_seconds: float
    compilation_seconds: float
    iterations: list[int]
    tolerance: float


def solve_reference(
    problem: Problem,
    contexts,
    *,
    tolerance: float = DEFAULT_REFERENCE_TOLERANCE,
    max_iter: int = DEFAULT_REFERENCE_MAX_ITERATIONS,
    starts: int | None = None,
    seed: int = DEFAULT_REFERENCE_SEED,
) -> ReferenceResult:
    """Solve each observed instance independently with CPU float64 arithmetic.

    ``elapsed_seconds`` includes transfers, setup, compilation/warmup, classical
    initialization, and SciPy solves. ``compilation_seconds`` is the included
    subset spent compiling and warming the JAX evaluators. Neither is a GPU
    timing. Solver success and mathematical error-bound certification are
    separate fields. Nonconvex results are best-found points, not global labels.

    Converting a float32 problem preserves its actual rounded fixed arrays and
    observed contexts before casting them to float64; it does not regenerate a
    different, higher-precision problem from the original random seed.
    """
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    default_starts = (
        _DEFAULT_CONVEX_STARTS if problem.convex else _DEFAULT_NONCONVEX_STARTS
    )
    count = default_starts if starts is None else starts
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("starts must be a positive integer")
    begun = time.perf_counter()
    host_contexts = np.asarray(jax.device_get(contexts), dtype=np.float64)
    if (
        host_contexts.ndim != _CONTEXT_BATCH_RANK
        or host_contexts.shape[1] != problem.context_dim
    ):
        raise ValueError("contexts must have shape [batch, context_dim]")
    if len(host_contexts) == 0 or not np.isfinite(host_contexts).all():
        raise ValueError("contexts must contain at least one finite instance")

    p64 = problem.to(dtype="float64", device="cpu")
    context_array = p64.array(host_contexts)
    rng = np.random.default_rng(seed)
    if p64.constraint == "box":
        lower = np.asarray(jax.device_get(p64.lower), dtype=np.float64)
        upper = np.asarray(jax.device_get(p64.upper), dtype=np.float64)
        bounds = list(zip(lower, upper))
    else:
        bounds = None

    # Crucially, the context is a dynamic argument to this ONE transformed
    # evaluator, rather than a closure compiled anew for each SciPy instance.
    value_and_grad_one = jax.value_and_grad(p64.objective_one)

    def diagnostics_one(y, context):
        value, gradient = value_and_grad_one(y, context)
        step = p64.step_size_one(context)
        projected = p64.project_one(y - step * gradient, context)
        mapping_norm = euclidean_norm((y - projected) / step)
        return value, mapping_norm, p64.certificate_one(y, context)

    compilation_begun = time.perf_counter()
    with jax.default_device(p64.device):
        example_context = context_array[0]
        baseline_eval = (
            jax.jit(p64.baseline_one).lower(example_context).compile()
        )
        example_initial = baseline_eval(example_context)
        value_grad_eval = (
            jax.jit(value_and_grad_one)
            .lower(example_initial, example_context)
            .compile()
        )
        project_eval = (
            jax.jit(p64.project_one)
            .lower(example_initial, example_context)
            .compile()
        )
        example_batch = jnp.broadcast_to(
            example_initial, (len(host_contexts), p64.decision_dim)
        )
        diagnostics_eval = (
            jax.jit(jax.vmap(diagnostics_one))
            .lower(example_batch, context_array)
            .compile()
        )
        jax.block_until_ready(
            (
                value_grad_eval(example_initial, example_context),
                project_eval(example_initial, example_context),
                diagnostics_eval(example_batch, context_array),
            )
        )
    compilation_seconds = time.perf_counter() - compilation_begun

    solutions, successes, iterations = [], [], []
    for index in range(len(host_contexts)):
        context = context_array[index]
        initial = np.array(
            jax.device_get(baseline_eval(context)), dtype=np.float64, copy=True
        )

        def value_and_grad(z):
            value, gradient = jax.device_get(
                value_grad_eval(p64.array(z), context)
            )
            return float(value), np.array(gradient, dtype=np.float64, copy=True)

        best_value = value_and_grad(initial)[0]
        if not np.isfinite(best_value):
            raise ValueError(
                f"Nonfinite objective at the baseline for instance {index}"
            )
        best_point = initial
        best_success = False
        best_iterations = 0
        for attempt in range(int(count)):
            if attempt == 0:
                x0 = initial.copy()
            elif p64.constraint == "box":
                x0 = rng.uniform(lower, upper)
            else:
                x0 = initial + rng.normal(size=p64.decision_dim)
            result = minimize(
                value_and_grad,
                x0,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "ftol": _SCIPY_FUNCTION_TOLERANCE,
                    "gtol": _SCIPY_GRADIENT_TOLERANCE,
                    "maxiter": int(max_iter),
                    "maxls": _SCIPY_MAX_LINE_SEARCH_STEPS,
                },
            )
            candidate = project_eval(p64.array(result.x), context)
            value = float(
                jax.device_get(value_grad_eval(candidate, context)[0])
            )
            better = value < best_value
            equally_good_success = (
                value == best_value
                and bool(result.success)
                and not best_success
            )
            if np.isfinite(value) and (better or equally_good_success):
                best_value = value
                best_point = np.array(
                    jax.device_get(candidate), dtype=np.float64, copy=True
                )
                best_success = bool(result.success)
                best_iterations = int(result.nit)
        solutions.append(best_point)
        successes.append(best_success)
        iterations.append(best_iterations)

    solution = np.asarray(np.stack(solutions), dtype=np.float64)
    values, stationarity, certificates = jax.device_get(
        diagnostics_eval(p64.array(solution), context_array)
    )
    objective = np.asarray(values, dtype=np.float64)
    stationarity = np.asarray(stationarity, dtype=np.float64)
    certificates = np.asarray(certificates, dtype=np.float64)
    certified = np.isfinite(certificates) & (certificates <= tolerance)
    return ReferenceResult(
        solution=solution,
        objective=objective,
        stationarity=stationarity,
        certificates=certificates,
        solver_success=np.asarray(successes, dtype=np.bool_),
        globally_certified=np.asarray(certified, dtype=np.bool_),
        label=(
            "convex numerical reference"
            if p64.convex
            else "best-found multistart reference"
        ),
        elapsed_seconds=time.perf_counter() - begun,
        compilation_seconds=compilation_seconds,
        iterations=iterations,
        tolerance=float(tolerance),
    )

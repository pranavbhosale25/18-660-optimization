"""JAX-pytree-compatible public result and diagnostic types."""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax

Array = jax.Array


class SolverStatus(IntEnum):
    CONVERGED = 0
    MAX_ITERATIONS = 1
    RADIUS_TOO_SMALL = 2
    NONFINITE_INITIAL_STATE = 3
    INVALID_INITIAL_MATRIX = 4
    SUBPROBLEM_FAILURE = 5


class SubproblemResult(NamedTuple):
    step: Array
    predicted_reduction: Array
    step_norm: Array
    hit_boundary: Array
    valid: Array


class SR1UpdateResult(NamedTuple):
    matrix: Array
    applied: Array
    denominator: Array
    threshold: Array
    residual_norm: Array
    secant_residual_norm: Array


class IterationHistory(NamedTuple):
    objective: Array
    trial_objective: Array
    gradient_norm: Array
    radius_before: Array
    radius_after: Array
    ratio: Array
    actual_reduction: Array
    predicted_reduction: Array
    step_norm: Array
    accepted: Array
    sr1_updated: Array
    sr1_denominator: Array
    hit_boundary: Array
    valid: Array


class SR1Result(NamedTuple):
    x: Array
    fun: Array
    grad: Array
    grad_norm: Array
    hessian_approximation: Array
    radius: Array
    iterations: Array
    accepted_steps: Array
    rejected_steps: Array
    sr1_updates: Array
    sr1_skips: Array
    function_evaluations: Array
    gradient_evaluations: Array
    status: Array
    success: Array
    history: IterationHistory


_STATUS_MESSAGES = {
    SolverStatus.CONVERGED: "gradient tolerance satisfied",
    SolverStatus.MAX_ITERATIONS: "maximum outer iterations reached",
    SolverStatus.RADIUS_TOO_SMALL: "trust-region radius became too small",
    SolverStatus.NONFINITE_INITIAL_STATE: "initial objective or gradient is non-finite",
    SolverStatus.INVALID_INITIAL_MATRIX: "initial Hessian approximation is invalid",
    SolverStatus.SUBPROBLEM_FAILURE: "trust-region step calculation failed",
}


def status_message(status: int | SolverStatus) -> str:
    """Return a host-side human-readable solver status."""

    try:
        key = SolverStatus(int(status))
    except (TypeError, ValueError):
        return f"unknown status ({status})"
    return _STATUS_MESSAGES[key]

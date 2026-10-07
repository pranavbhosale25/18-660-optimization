"""JAX trust-region SR1 optimization.

Only the runtime configuration module is imported eagerly. Solver-related
symbols are loaded lazily so callers can select CPU/GPU and x32/x64 behavior
before JAX itself is imported.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .runtime import RuntimeInfo, configure_runtime

__version__ = "0.3.0"

_LAZY_EXPORTS = {
    "SR1Options": (".options", "SR1Options"),
    "IterationHistory": (".types", "IterationHistory"),
    "SR1Result": (".types", "SR1Result"),
    "SolverStatus": (".types", "SolverStatus"),
    "status_message": (".types", "status_message"),
    "make_solver": (".solver", "make_solver"),
    "make_solver_with_matrix": (".solver", "make_solver_with_matrix"),
    "make_batched_solver": (".solver", "make_batched_solver"),
    "make_batched_solver_with_matrix": (
        ".solver",
        "make_batched_solver_with_matrix",
    ),
    "safeguarded_sr1_update": (
        ".updates",
        "safeguarded_sr1_update",
    ),
    "TestProblem": (".problems", "TestProblem"),
    "standard_problems": (".problems", "standard_problems"),
    "problem_by_name": (".problems", "problem_by_name"),
}

__all__ = [
    "RuntimeInfo",
    "configure_runtime",
    "SR1Options",
    "IterationHistory",
    "SR1Result",
    "SolverStatus",
    "status_message",
    "make_solver",
    "make_solver_with_matrix",
    "make_batched_solver",
    "make_batched_solver_with_matrix",
    "safeguarded_sr1_update",
    "TestProblem",
    "standard_problems",
    "problem_by_name",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

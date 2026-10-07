"""Preconditioned Polak--Ribiere nonlinear conjugate gradient in JAX."""

from ._types import PNCGHistory, PNCGResult
from .options import PNCGOptions
from .preconditioners import (
    Preconditioner,
    diagonal_preconditioner,
    identity_preconditioner,
    jacobi_preconditioner,
)
from .runtime import Runtime, configure_runtime
from .solver import make_batched_solver, make_solver, minimize
from .status import (
    STATUS_CONVERGED,
    STATUS_LINE_SEARCH_FAILED,
    STATUS_MAXITER,
    STATUS_NONFINITE,
    status_message,
)

__all__ = [
    "PNCGHistory",
    "PNCGOptions",
    "PNCGResult",
    "Preconditioner",
    "Runtime",
    "STATUS_CONVERGED",
    "STATUS_LINE_SEARCH_FAILED",
    "STATUS_MAXITER",
    "STATUS_NONFINITE",
    "configure_runtime",
    "diagonal_preconditioner",
    "identity_preconditioner",
    "jacobi_preconditioner",
    "make_batched_solver",
    "make_solver",
    "minimize",
    "status_message",
]

__version__ = "0.1.0"

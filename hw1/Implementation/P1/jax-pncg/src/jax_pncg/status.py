from __future__ import annotations

from typing import Any

import jax

STATUS_CONVERGED = 0
STATUS_MAXITER = 1
STATUS_LINE_SEARCH_FAILED = 2
STATUS_NONFINITE = 3

_STATUS_MESSAGES = {
    STATUS_CONVERGED: "converged: gradient tolerance satisfied",
    STATUS_MAXITER: "maximum iteration count reached",
    STATUS_LINE_SEARCH_FAILED: "strong-Wolfe line search failed",
    STATUS_NONFINITE: "non-finite objective, gradient, or solver state",
}


def status_message(status: Any) -> str:
    """Convert a scalar JAX status code to a host-side message."""

    code = int(jax.device_get(status))
    return _STATUS_MESSAGES.get(code, f"unknown status code {code}")

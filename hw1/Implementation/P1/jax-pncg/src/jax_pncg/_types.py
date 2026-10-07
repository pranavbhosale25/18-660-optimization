from __future__ import annotations

from typing import NamedTuple

import jax

Array = jax.Array


class PNCGHistory(NamedTuple):
    """Scalar per-iteration diagnostics stored on the JAX device.

    Arrays have length ``options.maxiter + 1`` when history recording is
    enabled and length zero otherwise.  Entry zero is the initial point.
    ``size`` is the number of valid entries.
    """

    value: Array
    grad_norm: Array
    step_size: Array
    beta: Array
    line_search_evaluations: Array
    restarted: Array
    size: Array


class PNCGResult(NamedTuple):
    """Result returned by the preconditioned nonlinear CG solver.

    Every field is a JAX value, so the result is itself compatible with
    ``jax.jit`` and ``jax.vmap``.
    """

    x: Array
    fun: Array
    grad: Array
    preconditioned_grad: Array
    direction: Array
    grad_norm: Array
    initial_grad_norm: Array
    nit: Array
    nfev: Array
    njev: Array
    line_search_iterations: Array
    restarts: Array
    status: Array
    success: Array
    history: PNCGHistory


class LineSearchResult(NamedTuple):
    alpha: Array
    value: Array
    grad: Array
    directional_derivative: Array
    success: Array
    nfev: Array
    nit: Array
    status: Array

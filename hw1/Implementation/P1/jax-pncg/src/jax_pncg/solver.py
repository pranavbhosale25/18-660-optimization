from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from ._types import LineSearchResult, PNCGHistory, PNCGResult
from .line_search import LS_BRACKETING_FAILED, strong_wolfe_line_search
from .options import PNCGOptions
from .preconditioners import Preconditioner, identity_preconditioner
from .status import (
    STATUS_CONVERGED,
    STATUS_LINE_SEARCH_FAILED,
    STATUS_MAXITER,
    STATUS_NONFINITE,
)

_STATUS_RUNNING = -1


class _SolverState(NamedTuple):
    iteration: jax.Array
    x: jax.Array
    value: jax.Array
    grad: jax.Array
    z: jax.Array
    direction: jax.Array
    grad_norm: jax.Array
    alpha_guess: jax.Array
    nfev: jax.Array
    njev: jax.Array
    line_search_iterations: jax.Array
    restarts: jax.Array
    status: jax.Array
    history_value: jax.Array
    history_grad_norm: jax.Array
    history_step_size: jax.Array
    history_beta: jax.Array
    history_line_search_evaluations: jax.Array
    history_restarted: jax.Array


def _gradient_norm(gradient: jax.Array, norm: str) -> jax.Array:
    if norm == "inf":
        return jnp.max(jnp.abs(gradient))
    return jnp.linalg.norm(gradient)


def _safe_precondition(
    preconditioner: Preconditioner,
    x: jax.Array,
    gradient: jax.Array,
    args: tuple[Any, ...],
) -> tuple[jax.Array, jax.Array]:
    """Apply M^{-1}g and fall back to identity if it is not SPD-like."""

    candidate = jnp.asarray(preconditioner(x, gradient, *args), dtype=gradient.dtype)
    if candidate.shape != gradient.shape:
        raise ValueError(
            "preconditioner must return an array with the same shape as the gradient"
        )

    metric = jnp.vdot(gradient, candidate).real
    valid = (
        jnp.all(jnp.isfinite(candidate))
        & jnp.isfinite(metric)
        & (metric > jnp.asarray(0.0, dtype=gradient.dtype))
    )
    stationary = jnp.all(gradient == 0)
    use_candidate = valid & (~stationary)
    z = jnp.where(use_candidate, candidate, gradient)
    # A zero gradient is already converged and should not count as a fallback.
    used_fallback = (~valid) & (~stationary)
    return z, used_fallback


def _empty_retry_result(
    value: jax.Array, gradient: jax.Array, slope: jax.Array
) -> LineSearchResult:
    dtype = value.dtype
    return LineSearchResult(
        alpha=jnp.asarray(0.0, dtype=dtype),
        value=value,
        grad=gradient,
        directional_derivative=slope,
        success=jnp.asarray(False),
        nfev=jnp.asarray(0, dtype=jnp.int32),
        nit=jnp.asarray(0, dtype=jnp.int32),
        status=jnp.asarray(LS_BRACKETING_FAILED, dtype=jnp.int32),
    )


def _solve_impl(
    value_and_grad: Callable[..., tuple[jax.Array, jax.Array]],
    preconditioner: Preconditioner,
    options: PNCGOptions,
    x0: jax.Array,
    args: tuple[Any, ...],
) -> PNCGResult:
    value0, grad0 = value_and_grad(x0, *args)
    z0, preconditioner_fallback0 = _safe_precondition(
        preconditioner, x0, grad0, args
    )
    grad_norm0 = _gradient_norm(grad0, options.norm)
    direction0 = -z0

    dtype = x0.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    eps = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
    tolerance = jnp.asarray(options.gtol, dtype=dtype) + jnp.asarray(
        options.rtol, dtype=dtype
    ) * grad_norm0

    finite_initial = (
        jnp.isfinite(value0)
        & jnp.all(jnp.isfinite(grad0))
        & jnp.all(jnp.isfinite(z0))
        & jnp.isfinite(grad_norm0)
    )
    initial_status = jnp.where(
        finite_initial,
        jnp.asarray(_STATUS_RUNNING, dtype=jnp.int32),
        jnp.asarray(STATUS_NONFINITE, dtype=jnp.int32),
    )

    if options.record_history:
        history_length = options.maxiter + 1
        history_value = jnp.full((history_length,), jnp.nan, dtype=dtype)
        history_grad_norm = jnp.full((history_length,), jnp.nan, dtype=dtype)
        history_step_size = jnp.zeros((history_length,), dtype=dtype)
        history_beta = jnp.zeros((history_length,), dtype=dtype)
        history_ls_evals = jnp.zeros((history_length,), dtype=jnp.int32)
        history_restarted = jnp.zeros((history_length,), dtype=jnp.bool_)
        history_value = history_value.at[0].set(value0)
        history_grad_norm = history_grad_norm.at[0].set(grad_norm0)
        history_restarted = history_restarted.at[0].set(
            preconditioner_fallback0
        )
    else:
        history_value = jnp.empty((0,), dtype=dtype)
        history_grad_norm = jnp.empty((0,), dtype=dtype)
        history_step_size = jnp.empty((0,), dtype=dtype)
        history_beta = jnp.empty((0,), dtype=dtype)
        history_ls_evals = jnp.empty((0,), dtype=jnp.int32)
        history_restarted = jnp.empty((0,), dtype=jnp.bool_)

    initial = _SolverState(
        iteration=jnp.asarray(0, dtype=jnp.int32),
        x=x0,
        value=value0,
        grad=grad0,
        z=z0,
        direction=direction0,
        grad_norm=grad_norm0,
        alpha_guess=jnp.asarray(options.initial_step, dtype=dtype),
        nfev=jnp.asarray(1, dtype=jnp.int32),
        njev=jnp.asarray(1, dtype=jnp.int32),
        line_search_iterations=jnp.asarray(0, dtype=jnp.int32),
        restarts=preconditioner_fallback0.astype(jnp.int32),
        status=initial_status,
        history_value=history_value,
        history_grad_norm=history_grad_norm,
        history_step_size=history_step_size,
        history_beta=history_beta,
        history_line_search_evaluations=history_ls_evals,
        history_restarted=history_restarted,
    )

    def solver_cond(state: _SolverState) -> jax.Array:
        return (
            (state.status == _STATUS_RUNNING)
            & (state.iteration < options.maxiter)
            & (state.grad_norm > tolerance)
        )

    def solver_body(state: _SolverState) -> _SolverState:
        metric = jnp.vdot(state.grad, state.z).real
        slope = jnp.vdot(state.grad, state.direction).real
        descent_bound = -jnp.asarray(options.descent_tolerance, dtype=dtype) * metric
        invalid_direction = (
            (~jnp.isfinite(slope))
            | (~jnp.isfinite(metric))
            | (metric <= zero)
            | (slope >= descent_bound)
        )
        restart_before_search = jnp.asarray(options.descent_restart) & invalid_direction
        search_direction = jnp.where(
            restart_before_search, -state.z, state.direction
        )

        line_search = strong_wolfe_line_search(
            value_and_grad,
            state.x,
            state.value,
            state.grad,
            search_direction,
            state.alpha_guess,
            args,
            options,
        )

        retry_requested = (
            jnp.asarray(options.retry_on_line_search_failure)
            & (~line_search.success)
            & (~restart_before_search)
        )
        steepest_slope = -metric

        retry_result = lax.cond(
            retry_requested,
            lambda _: strong_wolfe_line_search(
                value_and_grad,
                state.x,
                state.value,
                state.grad,
                -state.z,
                jnp.minimum(one, state.alpha_guess),
                args,
                options,
            ),
            lambda _: _empty_retry_result(state.value, state.grad, steepest_slope),
            operand=None,
        )

        use_retry = retry_requested & retry_result.success
        accepted_search = lax.cond(
            use_retry,
            lambda _: retry_result,
            lambda _: line_search,
            operand=None,
        )
        accepted_direction = jnp.where(use_retry, -state.z, search_direction)
        total_ls_nfev = line_search.nfev + retry_result.nfev
        total_ls_nit = line_search.nit + retry_result.nit
        search_succeeded = line_search.success | use_retry

        def fail_step(current: _SolverState) -> _SolverState:
            return current._replace(
                nfev=current.nfev + total_ls_nfev,
                njev=current.njev + total_ls_nfev,
                line_search_iterations=(
                    current.line_search_iterations + total_ls_nit
                ),
                restarts=current.restarts
                + restart_before_search.astype(jnp.int32)
                + retry_requested.astype(jnp.int32),
                status=jnp.asarray(STATUS_LINE_SEARCH_FAILED, dtype=jnp.int32),
            )

        def accept_step(current: _SolverState) -> _SolverState:
            alpha = accepted_search.alpha
            x_new = current.x + alpha * accepted_direction
            value_new = accepted_search.value
            grad_new = accepted_search.grad
            grad_norm_new = _gradient_norm(grad_new, options.norm)
            z_new, preconditioner_fallback = _safe_precondition(
                preconditioner, x_new, grad_new, args
            )

            denominator = jnp.vdot(current.grad, current.z).real
            denominator_scale = jnp.maximum(
                one,
                jnp.linalg.norm(current.grad) * jnp.linalg.norm(current.z),
            )
            denominator_valid = (
                jnp.isfinite(denominator)
                & (denominator > 16.0 * eps * denominator_scale)
            )

            # STUDENT TODO: Compute the raw preconditioned Polak--Ribiere
            # coefficient. Use denominator_valid to fall back safely when the
            # denominator is unusable. Replace the exception below with code
            # that assigns beta_raw. The PR+ selection itself is provided.
            # raise NotImplementedError(
            #     "STUDENT TODO: implement the raw Polak--Ribiere beta coefficient"
            # )

            numerator = jnp.vdot(z_new, grad_new - current.grad).real
            safe_denominator = jnp.where(
                denominator_valid,
                denominator,
                jnp.asarray(1.0, dtype=dtype),
            )
            beta_raw = jnp.where(denominator_valid, numerator / safe_denominator, zero)

            beta_raw = jnp.where(jnp.isfinite(beta_raw), beta_raw, zero)
            if options.beta_method == "pr+":
                beta = jnp.maximum(beta_raw, zero)
            else:
                beta = beta_raw

            new_iteration = current.iteration + 1
            if options.restart_interval > 0:
                periodic_restart = (
                    new_iteration % options.restart_interval
                ) == 0
            else:
                periodic_restart = jnp.asarray(False)

            if options.orthogonality_restart > 0.0:
                orthogonality_denominator = jnp.maximum(
                    jnp.vdot(grad_new, grad_new).real, eps
                )
                orthogonality_measure = (
                    jnp.abs(jnp.vdot(grad_new, current.grad).real)
                    / orthogonality_denominator
                )
                orthogonality_restart = orthogonality_measure >= jnp.asarray(
                    options.orthogonality_restart, dtype=dtype
                )
            else:
                orthogonality_restart = jnp.asarray(False)

            beta_restart = (
                (~denominator_valid)
                | periodic_restart
                | orthogonality_restart
                | preconditioner_fallback
            )
            provisional_direction = -z_new + beta * accepted_direction
            provisional_slope = jnp.vdot(grad_new, provisional_direction).real
            new_metric = jnp.vdot(grad_new, z_new).real
            sufficient_descent_bound = -jnp.asarray(
                options.descent_tolerance, dtype=dtype
            ) * new_metric
            loses_descent = (
                (~jnp.isfinite(provisional_slope))
                | (~jnp.isfinite(new_metric))
                | (new_metric <= zero)
                | (provisional_slope >= sufficient_descent_bound)
            )
            descent_restart = jnp.asarray(options.descent_restart) & loses_descent
            converged_new = grad_norm_new <= tolerance
            restart_after_step = (beta_restart | descent_restart) & (~converged_new)
            reset_direction = restart_after_step | converged_new
            beta_used = jnp.where(reset_direction, zero, beta)
            direction_new = -z_new + beta_used * accepted_direction

            finite_new = (
                jnp.isfinite(value_new)
                & jnp.all(jnp.isfinite(grad_new))
                & jnp.all(jnp.isfinite(z_new))
                & jnp.isfinite(grad_norm_new)
            )
            status_new = jnp.where(
                finite_new,
                jnp.asarray(_STATUS_RUNNING, dtype=jnp.int32),
                jnp.asarray(STATUS_NONFINITE, dtype=jnp.int32),
            )

            restarted_this_step = (
                restart_before_search
                | use_retry
                | restart_after_step
                | preconditioner_fallback
            )
            restart_increment = (
                restart_before_search.astype(jnp.int32)
                + use_retry.astype(jnp.int32)
                + restart_after_step.astype(jnp.int32)
            )

            next_alpha_guess = jnp.minimum(
                jnp.asarray(options.max_step, dtype=dtype),
                jnp.maximum(eps, 2.0 * alpha),
            )

            history_value = current.history_value
            history_grad_norm = current.history_grad_norm
            history_step_size = current.history_step_size
            history_beta = current.history_beta
            history_ls_evals = current.history_line_search_evaluations
            history_restarted = current.history_restarted
            if options.record_history:
                history_value = history_value.at[new_iteration].set(value_new)
                history_grad_norm = history_grad_norm.at[new_iteration].set(
                    grad_norm_new
                )
                history_step_size = history_step_size.at[new_iteration].set(alpha)
                history_beta = history_beta.at[new_iteration].set(beta_used)
                history_ls_evals = history_ls_evals.at[new_iteration].set(
                    total_ls_nfev
                )
                history_restarted = history_restarted.at[new_iteration].set(
                    restarted_this_step
                )

            return current._replace(
                iteration=new_iteration,
                x=x_new,
                value=value_new,
                grad=grad_new,
                z=z_new,
                direction=direction_new,
                grad_norm=grad_norm_new,
                alpha_guess=next_alpha_guess,
                nfev=current.nfev + total_ls_nfev,
                njev=current.njev + total_ls_nfev,
                line_search_iterations=(
                    current.line_search_iterations + total_ls_nit
                ),
                restarts=current.restarts + restart_increment,
                status=status_new,
                history_value=history_value,
                history_grad_norm=history_grad_norm,
                history_step_size=history_step_size,
                history_beta=history_beta,
                history_line_search_evaluations=history_ls_evals,
                history_restarted=history_restarted,
            )

        return lax.cond(search_succeeded, accept_step, fail_step, state)

    state = lax.while_loop(solver_cond, solver_body, initial)

    final_status = lax.cond(
        state.status == _STATUS_RUNNING,
        lambda _: jnp.where(
            state.grad_norm <= tolerance,
            jnp.asarray(STATUS_CONVERGED, dtype=jnp.int32),
            jnp.asarray(STATUS_MAXITER, dtype=jnp.int32),
        ),
        lambda _: state.status,
        operand=None,
    )
    success = final_status == STATUS_CONVERGED

    history_size = jnp.where(
        jnp.asarray(options.record_history),
        state.iteration + 1,
        jnp.asarray(0, dtype=jnp.int32),
    )
    history = PNCGHistory(
        value=state.history_value,
        grad_norm=state.history_grad_norm,
        step_size=state.history_step_size,
        beta=state.history_beta,
        line_search_evaluations=state.history_line_search_evaluations,
        restarted=state.history_restarted,
        size=history_size,
    )

    return PNCGResult(
        x=state.x,
        fun=state.value,
        grad=state.grad,
        preconditioned_grad=state.z,
        direction=state.direction,
        grad_norm=state.grad_norm,
        initial_grad_norm=grad_norm0,
        nit=state.iteration,
        nfev=state.nfev,
        njev=state.njev,
        line_search_iterations=state.line_search_iterations,
        restarts=state.restarts,
        status=final_status,
        success=success,
        history=history,
    )


def make_solver(
    fun: Callable[..., jax.Array],
    *,
    preconditioner: Preconditioner | None = None,
    options: PNCGOptions | None = None,
    jit_compile: bool = True,
) -> Callable[..., PNCGResult]:
    """Build a reusable Polak--Ribiere nonlinear-CG solver.

    ``fun`` must have signature ``fun(x, *args) -> scalar``.  The optional
    preconditioner has signature ``preconditioner(x, gradient, *args) -> z``
    and represents the action of ``M^{-1}`` on the gradient.

    Build once and call repeatedly to reuse JAX's compiled executable.  The
    returned function accepts ``(x0, *args)`` and is directly compatible with
    ``jax.vmap``.
    """

    chosen_options = options or PNCGOptions()
    chosen_preconditioner = preconditioner or identity_preconditioner
    value_and_grad = jax.value_and_grad(fun)

    def solve(x0: Any, *args: Any) -> PNCGResult:
        x0_array = jnp.asarray(x0)
        if x0_array.ndim != 1 or x0_array.size == 0:
            raise ValueError("x0 must be a nonempty one-dimensional array")
        if not jnp.issubdtype(x0_array.dtype, jnp.inexact):
            x0_array = x0_array.astype(jnp.result_type(x0_array, 0.0))
        if jnp.issubdtype(x0_array.dtype, jnp.complexfloating):
            raise TypeError("jax_pncg currently supports real-valued variables only")
        return _solve_impl(
            value_and_grad,
            chosen_preconditioner,
            chosen_options,
            x0_array,
            args,
        )

    return jax.jit(solve) if jit_compile else solve


def make_batched_solver(
    fun: Callable[..., jax.Array],
    *,
    preconditioner: Preconditioner | None = None,
    options: PNCGOptions | None = None,
    arg_in_axes: Sequence[int | None] = (),
    jit_compile: bool = True,
) -> Callable[..., PNCGResult]:
    """Build a solver vectorized over a leading batch of starting points.

    ``arg_in_axes`` controls batching of any additional objective arguments.
    For example, ``arg_in_axes=(0, None)`` batches the first extra argument and
    shares the second one across all solves.
    """

    single = make_solver(
        fun,
        preconditioner=preconditioner,
        options=options,
        jit_compile=False,
    )
    batched = jax.vmap(single, in_axes=(0, *tuple(arg_in_axes)))
    return jax.jit(batched) if jit_compile else batched


def minimize(
    fun: Callable[..., jax.Array],
    x0: Any,
    *args: Any,
    preconditioner: Preconditioner | None = None,
    options: PNCGOptions | None = None,
    jit_compile: bool = True,
) -> PNCGResult:
    """One-shot convenience wrapper around :func:`make_solver`."""

    solver = make_solver(
        fun,
        preconditioner=preconditioner,
        options=options,
        jit_compile=jit_compile,
    )
    return solver(x0, *args)

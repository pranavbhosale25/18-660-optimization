from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from ._types import LineSearchResult
from .options import PNCGOptions

LS_SUCCESS = 0
LS_NOT_DESCENT = 1
LS_BRACKETING_FAILED = 2
LS_ZOOM_FAILED = 3

_MODE_RUNNING = 0
_MODE_SUCCESS = 1
_MODE_ZOOM = 2
_MODE_FAILED = 3


class _SearchState(NamedTuple):
    iteration: jax.Array
    alpha_prev: jax.Array
    value_prev: jax.Array
    grad_prev: jax.Array
    slope_prev: jax.Array
    alpha: jax.Array
    accepted_alpha: jax.Array
    accepted_value: jax.Array
    accepted_grad: jax.Array
    accepted_slope: jax.Array
    lo: jax.Array
    value_lo: jax.Array
    grad_lo: jax.Array
    slope_lo: jax.Array
    hi: jax.Array
    value_hi: jax.Array
    grad_hi: jax.Array
    slope_hi: jax.Array
    mode: jax.Array
    nfev: jax.Array


class _ZoomState(NamedTuple):
    iteration: jax.Array
    lo: jax.Array
    value_lo: jax.Array
    grad_lo: jax.Array
    slope_lo: jax.Array
    hi: jax.Array
    value_hi: jax.Array
    grad_hi: jax.Array
    slope_hi: jax.Array
    accepted_alpha: jax.Array
    accepted_value: jax.Array
    accepted_grad: jax.Array
    accepted_slope: jax.Array
    mode: jax.Array
    nfev: jax.Array


def _all_finite(value: jax.Array, grad: jax.Array) -> jax.Array:
    return jnp.isfinite(value) & jnp.all(jnp.isfinite(grad))


def strong_wolfe_line_search(
    value_and_grad: Any,
    x: jax.Array,
    value0: jax.Array,
    grad0: jax.Array,
    direction: jax.Array,
    alpha0: jax.Array,
    args: tuple[Any, ...],
    options: PNCGOptions,
) -> LineSearchResult:
    """JIT-compatible Nocedal--Wright Algorithms 3.5/3.6.

    The zoom interpolation step uses safeguarded bisection, one of the choices
    explicitly allowed by Algorithm 3.6.  Objective and gradient are always
    evaluated together via ``jax.value_and_grad``.
    """

    dtype = x.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    c1 = jnp.asarray(options.c1, dtype=dtype)
    c2 = jnp.asarray(options.c2, dtype=dtype)
    max_step = jnp.asarray(options.max_step, dtype=dtype)
    growth = jnp.asarray(options.step_growth, dtype=dtype)
    eps = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)

    slope0 = jnp.vdot(grad0, direction).real
    descent = jnp.isfinite(slope0) & (slope0 < zero)
    first_alpha = jnp.minimum(jnp.maximum(alpha0, eps), max_step)

    initial = _SearchState(
        iteration=jnp.asarray(0, dtype=jnp.int32),
        alpha_prev=zero,
        value_prev=value0,
        grad_prev=grad0,
        slope_prev=slope0,
        alpha=first_alpha,
        accepted_alpha=zero,
        accepted_value=value0,
        accepted_grad=grad0,
        accepted_slope=slope0,
        lo=zero,
        value_lo=value0,
        grad_lo=grad0,
        slope_lo=slope0,
        hi=first_alpha,
        value_hi=value0,
        grad_hi=grad0,
        slope_hi=slope0,
        mode=jnp.where(
            descent,
            jnp.asarray(_MODE_RUNNING, dtype=jnp.int32),
            jnp.asarray(_MODE_FAILED, dtype=jnp.int32),
        ),
        nfev=jnp.asarray(0, dtype=jnp.int32),
    )

    def search_cond(state: _SearchState) -> jax.Array:
        return (state.mode == _MODE_RUNNING) & (
            state.iteration < options.max_line_search_iterations
        )

    def search_body(state: _SearchState) -> _SearchState:
        alpha = state.alpha
        value, grad = value_and_grad(x + alpha * direction, *args)
        slope = jnp.vdot(grad, direction).real
        finite = _all_finite(value, grad) & jnp.isfinite(slope)

        armijo_rhs = value0 + c1 * alpha * slope0
        violates_armijo = (~finite) | (value > armijo_rhs)
        not_improved = (state.iteration > 0) & (value >= state.value_prev)
        needs_zoom = violates_armijo | not_improved
        strong_wolfe = finite & (jnp.abs(slope) <= -c2 * slope0)
        derivative_turned = finite & (slope >= zero)

        def bracket_from_previous(_: None) -> _SearchState:
            return state._replace(
                iteration=state.iteration + 1,
                lo=state.alpha_prev,
                value_lo=state.value_prev,
                grad_lo=state.grad_prev,
                slope_lo=state.slope_prev,
                hi=alpha,
                value_hi=value,
                grad_hi=grad,
                slope_hi=slope,
                mode=jnp.asarray(_MODE_ZOOM, dtype=jnp.int32),
                nfev=state.nfev + 1,
            )

        def accept(_: None) -> _SearchState:
            return state._replace(
                iteration=state.iteration + 1,
                accepted_alpha=alpha,
                accepted_value=value,
                accepted_grad=grad,
                accepted_slope=slope,
                mode=jnp.asarray(_MODE_SUCCESS, dtype=jnp.int32),
                nfev=state.nfev + 1,
            )

        def bracket_from_current(_: None) -> _SearchState:
            # This is zoom(alpha_i, alpha_{i-1}) in Algorithm 3.5.
            return state._replace(
                iteration=state.iteration + 1,
                lo=alpha,
                value_lo=value,
                grad_lo=grad,
                slope_lo=slope,
                hi=state.alpha_prev,
                value_hi=state.value_prev,
                grad_hi=state.grad_prev,
                slope_hi=state.slope_prev,
                mode=jnp.asarray(_MODE_ZOOM, dtype=jnp.int32),
                nfev=state.nfev + 1,
            )

        def continue_or_fail(_: None) -> _SearchState:
            next_alpha = jnp.minimum(growth * alpha, max_step)
            stuck = next_alpha <= alpha * (one + 4.0 * eps)

            def fail_at_bound(_: None) -> _SearchState:
                return state._replace(
                    iteration=state.iteration + 1,
                    accepted_alpha=alpha,
                    accepted_value=value,
                    accepted_grad=grad,
                    accepted_slope=slope,
                    mode=jnp.asarray(_MODE_FAILED, dtype=jnp.int32),
                    nfev=state.nfev + 1,
                )

            def keep_searching(_: None) -> _SearchState:
                return state._replace(
                    iteration=state.iteration + 1,
                    alpha_prev=alpha,
                    value_prev=value,
                    grad_prev=grad,
                    slope_prev=slope,
                    alpha=next_alpha,
                    accepted_alpha=alpha,
                    accepted_value=value,
                    accepted_grad=grad,
                    accepted_slope=slope,
                    nfev=state.nfev + 1,
                )

            return lax.cond(stuck, fail_at_bound, keep_searching, operand=None)

        return lax.cond(
            needs_zoom,
            bracket_from_previous,
            lambda _: lax.cond(
                strong_wolfe,
                accept,
                lambda __: lax.cond(
                    derivative_turned,
                    bracket_from_current,
                    continue_or_fail,
                    operand=None,
                ),
                operand=None,
            ),
            operand=None,
        )

    search = lax.while_loop(search_cond, search_body, initial)

    # If the iteration cap was hit while still searching, mark bracketing as
    # failed.  The most recently evaluated point remains available for
    # diagnostics, but the nonlinear-CG iteration will not accept it.
    search = lax.cond(
        search.mode == _MODE_RUNNING,
        lambda s: s._replace(mode=jnp.asarray(_MODE_FAILED, dtype=jnp.int32)),
        lambda s: s,
        search,
    )

    zoom_initial = _ZoomState(
        iteration=jnp.asarray(0, dtype=jnp.int32),
        lo=search.lo,
        value_lo=search.value_lo,
        grad_lo=search.grad_lo,
        slope_lo=search.slope_lo,
        hi=search.hi,
        value_hi=search.value_hi,
        grad_hi=search.grad_hi,
        slope_hi=search.slope_hi,
        accepted_alpha=search.lo,
        accepted_value=search.value_lo,
        accepted_grad=search.grad_lo,
        accepted_slope=search.slope_lo,
        mode=jnp.asarray(_MODE_RUNNING, dtype=jnp.int32),
        nfev=search.nfev,
    )

    zoom_base_tol = jnp.maximum(
        jnp.asarray(options.zoom_tolerance, dtype=dtype), 16.0 * eps
    )

    def zoom_interval_large(state: _ZoomState) -> jax.Array:
        width = jnp.abs(state.hi - state.lo)
        scale = jnp.maximum(one, jnp.abs(state.lo) + jnp.abs(state.hi))
        return width > zoom_base_tol * scale

    def zoom_cond(state: _ZoomState) -> jax.Array:
        return (
            (state.mode == _MODE_RUNNING)
            & (state.iteration < options.max_zoom_iterations)
            & zoom_interval_large(state)
        )

    def zoom_body(state: _ZoomState) -> _ZoomState:
        alpha = 0.5 * (state.lo + state.hi)
        value, grad = value_and_grad(x + alpha * direction, *args)
        slope = jnp.vdot(grad, direction).real
        finite = _all_finite(value, grad) & jnp.isfinite(slope)

        armijo_rhs = value0 + c1 * alpha * slope0
        replace_hi = (~finite) | (value > armijo_rhs) | (value >= state.value_lo)
        strong_wolfe = finite & (jnp.abs(slope) <= -c2 * slope0)

        def set_hi(_: None) -> _ZoomState:
            return state._replace(
                iteration=state.iteration + 1,
                hi=alpha,
                value_hi=value,
                grad_hi=grad,
                slope_hi=slope,
                nfev=state.nfev + 1,
            )

        def accept(_: None) -> _ZoomState:
            return state._replace(
                iteration=state.iteration + 1,
                accepted_alpha=alpha,
                accepted_value=value,
                accepted_grad=grad,
                accepted_slope=slope,
                mode=jnp.asarray(_MODE_SUCCESS, dtype=jnp.int32),
                nfev=state.nfev + 1,
            )

        def move_lo(_: None) -> _ZoomState:
            swap_hi = slope * (state.hi - state.lo) >= zero
            new_hi = jnp.where(swap_hi, state.lo, state.hi)
            new_value_hi = jnp.where(swap_hi, state.value_lo, state.value_hi)
            new_grad_hi = jnp.where(swap_hi, state.grad_lo, state.grad_hi)
            new_slope_hi = jnp.where(swap_hi, state.slope_lo, state.slope_hi)
            return state._replace(
                iteration=state.iteration + 1,
                lo=alpha,
                value_lo=value,
                grad_lo=grad,
                slope_lo=slope,
                hi=new_hi,
                value_hi=new_value_hi,
                grad_hi=new_grad_hi,
                slope_hi=new_slope_hi,
                accepted_alpha=alpha,
                accepted_value=value,
                accepted_grad=grad,
                accepted_slope=slope,
                nfev=state.nfev + 1,
            )

        return lax.cond(
            replace_hi,
            set_hi,
            lambda _: lax.cond(strong_wolfe, accept, move_lo, operand=None),
            operand=None,
        )

    zoom = lax.cond(
        search.mode == _MODE_ZOOM,
        lambda state: lax.while_loop(zoom_cond, zoom_body, state),
        lambda state: state,
        zoom_initial,
    )

    zoom_success = (search.mode == _MODE_ZOOM) & (zoom.mode == _MODE_SUCCESS)
    direct_success = search.mode == _MODE_SUCCESS
    success = descent & (direct_success | zoom_success)

    alpha = jnp.where(zoom_success, zoom.accepted_alpha, search.accepted_alpha)
    value = jnp.where(zoom_success, zoom.accepted_value, search.accepted_value)
    grad = jnp.where(zoom_success, zoom.accepted_grad, search.accepted_grad)
    slope = jnp.where(zoom_success, zoom.accepted_slope, search.accepted_slope)
    nfev = jnp.where(search.mode == _MODE_ZOOM, zoom.nfev, search.nfev)
    nit = jnp.where(
        search.mode == _MODE_ZOOM,
        search.iteration + zoom.iteration,
        search.iteration,
    )

    status = jnp.where(
        success,
        jnp.asarray(LS_SUCCESS, dtype=jnp.int32),
        jnp.where(
            ~descent,
            jnp.asarray(LS_NOT_DESCENT, dtype=jnp.int32),
            jnp.where(
                search.mode == _MODE_ZOOM,
                jnp.asarray(LS_ZOOM_FAILED, dtype=jnp.int32),
                jnp.asarray(LS_BRACKETING_FAILED, dtype=jnp.int32),
            ),
        ),
    )

    return LineSearchResult(
        alpha=alpha,
        value=value,
        grad=grad,
        directional_derivative=slope,
        success=success,
        nfev=nfev,
        nit=nit,
        status=status,
    )

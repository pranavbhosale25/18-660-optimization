"""JIT- and VMAP-oriented implementation of trust-region SR1.

The outer iteration follows Nocedal and Wright, Algorithm 6.2. In particular,
the SR1 matrix is updated from every finite trial step, including rejected
steps. This detail is important: a rejected direction still contains curvature
information showing where the current model is inaccurate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .options import SR1Options
from .subproblem import solve_quadratic_subproblem
from .types import (
    IterationHistory,
    SR1Result,
    SolverStatus,
    SubproblemResult,
)
from .updates import safeguarded_sr1_update

Array = jax.Array
Objective = Callable[..., Array]
_RUNNING = -1


class _State(NamedTuple):
    x: Array
    objective: Array
    gradient: Array
    gradient_norm: Array
    matrix: Array
    radius: Array
    iterations: Array
    accepted_steps: Array
    rejected_steps: Array
    sr1_updates: Array
    sr1_skips: Array
    function_evaluations: Array
    gradient_evaluations: Array
    status: Array
    active: Array


def _finite_value_and_gradient(value: Array, gradient: Array) -> Array:
    return jnp.isfinite(value) & jnp.all(jnp.isfinite(gradient))


def _inactive_record(state: _State) -> IterationHistory:
    dtype = state.x.dtype
    nan = jnp.asarray(jnp.nan, dtype=dtype)
    return IterationHistory(
        objective=state.objective,
        trial_objective=nan,
        gradient_norm=state.gradient_norm,
        radius_before=state.radius,
        radius_after=state.radius,
        ratio=nan,
        actual_reduction=nan,
        predicted_reduction=nan,
        step_norm=jnp.asarray(0.0, dtype=dtype),
        accepted=jnp.asarray(False),
        sr1_updated=jnp.asarray(False),
        sr1_denominator=nan,
        hit_boundary=jnp.asarray(False),
        valid=jnp.asarray(False),
    )


def _empty_history(dtype: jnp.dtype) -> IterationHistory:
    floating = jnp.empty((0,), dtype=dtype)
    boolean = jnp.empty((0,), dtype=jnp.bool_)
    return IterationHistory(
        objective=floating,
        trial_objective=floating,
        gradient_norm=floating,
        radius_before=floating,
        radius_after=floating,
        ratio=floating,
        actual_reduction=floating,
        predicted_reduction=floating,
        step_norm=floating,
        accepted=boolean,
        sr1_updated=boolean,
        sr1_denominator=floating,
        hit_boundary=boolean,
        valid=boolean,
    )


def _build_core(
    fun: Objective,
    options: SR1Options,
) -> Callable[..., SR1Result]:
    value_and_gradient = jax.value_and_grad(fun, argnums=0)

    converged_status = jnp.asarray(SolverStatus.CONVERGED, dtype=jnp.int32)
    max_iterations_status = jnp.asarray(
        SolverStatus.MAX_ITERATIONS, dtype=jnp.int32
    )
    radius_small_status = jnp.asarray(
        SolverStatus.RADIUS_TOO_SMALL, dtype=jnp.int32
    )
    nonfinite_initial_status = jnp.asarray(
        SolverStatus.NONFINITE_INITIAL_STATE, dtype=jnp.int32
    )
    invalid_matrix_status = jnp.asarray(
        SolverStatus.INVALID_INITIAL_MATRIX, dtype=jnp.int32
    )
    subproblem_failure_status = jnp.asarray(
        SolverStatus.SUBPROBLEM_FAILURE, dtype=jnp.int32
    )
    running_status = jnp.asarray(_RUNNING, dtype=jnp.int32)

    def solve_with_matrix(
        x0: Array,
        initial_matrix: Array,
        *args: Any,
    ) -> SR1Result:
        x0 = jnp.asarray(x0)
        if x0.ndim != 1:
            raise ValueError("x0 must be a one-dimensional array")
        if jnp.issubdtype(x0.dtype, jnp.complexfloating):
            raise ValueError("complex-valued decision variables are not supported")
        if not jnp.issubdtype(x0.dtype, jnp.floating):
            target_dtype = (
                jnp.float64 if jax.config.x64_enabled else jnp.float32
            )
            x0 = x0.astype(target_dtype)

        initial_matrix = jnp.asarray(initial_matrix, dtype=x0.dtype)
        if initial_matrix.ndim != 2:
            raise ValueError("initial_matrix must be two-dimensional")
        expected_shape = (x0.shape[0], x0.shape[0])
        if initial_matrix.shape != expected_shape:
            raise ValueError(
                "initial_matrix must have shape (x0.size, x0.size)"
            )
        if options.symmetrize_matrices:
            initial_matrix = 0.5 * (
                initial_matrix
                + jnp.swapaxes(initial_matrix, -1, -2)
            )

        objective0, gradient0 = value_and_gradient(x0, *args)
        gradient_norm0 = jnp.linalg.norm(gradient0)
        finite_initial = (
            _finite_value_and_gradient(objective0, gradient0)
            & jnp.isfinite(gradient_norm0)
        )
        finite_matrix = jnp.all(jnp.isfinite(initial_matrix))
        converged_initial = finite_initial & finite_matrix & (
            gradient_norm0
            <= jnp.asarray(options.gradient_tolerance, dtype=x0.dtype)
        )
        active_initial = finite_initial & finite_matrix & (~converged_initial)
        initial_status = jnp.where(
            ~finite_initial,
            nonfinite_initial_status,
            jnp.where(
                ~finite_matrix,
                invalid_matrix_status,
                jnp.where(
                    converged_initial, converged_status, running_status
                ),
            ),
        )

        state0 = _State(
            x=x0,
            objective=objective0,
            gradient=gradient0,
            gradient_norm=gradient_norm0,
            matrix=initial_matrix,
            radius=jnp.asarray(options.initial_radius, dtype=x0.dtype),
            iterations=jnp.asarray(0, dtype=jnp.int32),
            accepted_steps=jnp.asarray(0, dtype=jnp.int32),
            rejected_steps=jnp.asarray(0, dtype=jnp.int32),
            sr1_updates=jnp.asarray(0, dtype=jnp.int32),
            sr1_skips=jnp.asarray(0, dtype=jnp.int32),
            function_evaluations=jnp.asarray(1, dtype=jnp.int32),
            gradient_evaluations=jnp.asarray(1, dtype=jnp.int32),
            status=initial_status,
            active=active_initial,
        )

        def solve_subproblem(state: _State) -> SubproblemResult:
            return solve_quadratic_subproblem(
                state.gradient,
                state.matrix,
                state.radius,
            )

        def active_iteration(
            state: _State,
        ) -> tuple[_State, IterationHistory]:
            dtype = state.x.dtype
            subproblem = solve_subproblem(state)
            model_is_usable = (
                subproblem.valid
                & jnp.isfinite(subproblem.predicted_reduction)
                & (subproblem.predicted_reduction > 0.0)
            )

            def evaluate_trial(_: None) -> tuple[Array, Array]:
                return value_and_gradient(
                    state.x + subproblem.step, *args
                )

            def invalid_trial(_: None) -> tuple[Array, Array]:
                return (
                    jnp.asarray(jnp.nan, dtype=dtype),
                    jnp.full_like(state.gradient, jnp.nan),
                )

            trial_objective, trial_gradient = lax.cond(
                model_is_usable,
                evaluate_trial,
                invalid_trial,
                operand=None,
            )
            trial_gradient_norm = jnp.linalg.norm(trial_gradient)
            finite_trial = (
                model_is_usable
                & _finite_value_and_gradient(
                    trial_objective, trial_gradient
                )
                & jnp.isfinite(trial_gradient_norm)
            )
            actual_reduction = jnp.where(
                finite_trial,
                state.objective - trial_objective,
                -jnp.inf,
            )
            ratio = jnp.where(
                finite_trial,
                actual_reduction / subproblem.predicted_reduction,
                -jnp.inf,
            )

            accepted = finite_trial & (
                ratio
                > jnp.asarray(
                    options.acceptance_threshold, dtype=dtype
                )
            )

            expanded_radius = jnp.minimum(
                jnp.asarray(options.expansion_factor, dtype=dtype)
                * state.radius,
                jnp.asarray(options.maximum_radius, dtype=dtype),
            )
            shrunken_radius = (
                jnp.asarray(options.shrink_factor, dtype=dtype)
                * state.radius
            )
            close_to_boundary = (
                subproblem.step_norm
                > jnp.asarray(options.boundary_fraction, dtype=dtype)
                * state.radius
            )
            next_radius = jnp.where(
                ratio
                > jnp.asarray(
                    options.good_ratio_threshold, dtype=dtype
                ),
                jnp.where(
                    close_to_boundary, expanded_radius, state.radius
                ),
                jnp.where(
                    ratio
                    >= jnp.asarray(
                        options.poor_ratio_threshold, dtype=dtype
                    ),
                    state.radius,
                    shrunken_radius,
                ),
            )

            gradient_difference = jnp.where(
                finite_trial,
                trial_gradient - state.gradient,
                jnp.zeros_like(state.gradient),
            )
            raw_update = safeguarded_sr1_update(
                state.matrix,
                subproblem.step,
                gradient_difference,
                skip_tolerance=options.sr1_skip_tolerance,
                symmetrize=options.symmetrize_matrices,
            )
            update_applied = finite_trial & raw_update.applied
            next_matrix = jnp.where(
                update_applied, raw_update.matrix, state.matrix
            )
            update_skipped = finite_trial & (~update_applied)

            next_x = jnp.where(
                accepted, state.x + subproblem.step, state.x
            )
            next_objective = jnp.where(
                accepted, trial_objective, state.objective
            )
            next_gradient = jnp.where(
                accepted, trial_gradient, state.gradient
            )
            next_gradient_norm = jnp.linalg.norm(next_gradient)

            converged = (
                jnp.isfinite(next_gradient_norm)
                & (
                    next_gradient_norm
                    <= jnp.asarray(
                        options.gradient_tolerance, dtype=dtype
                    )
                )
            )
            radius_too_small = (
                next_radius
                <= jnp.asarray(options.minimum_radius, dtype=dtype)
            )
            subproblem_failed = ~model_is_usable
            next_status = jnp.where(
                subproblem_failed,
                subproblem_failure_status,
                jnp.where(
                    converged,
                    converged_status,
                    jnp.where(
                        radius_too_small,
                        radius_small_status,
                        running_status,
                    ),
                ),
            )
            next_active = next_status == running_status

            next_state = _State(
                x=next_x,
                objective=next_objective,
                gradient=next_gradient,
                gradient_norm=next_gradient_norm,
                matrix=next_matrix,
                radius=next_radius,
                iterations=state.iterations + 1,
                accepted_steps=state.accepted_steps
                + accepted.astype(jnp.int32),
                rejected_steps=state.rejected_steps
                + (model_is_usable & (~accepted)).astype(jnp.int32),
                sr1_updates=state.sr1_updates
                + update_applied.astype(jnp.int32),
                sr1_skips=state.sr1_skips
                + update_skipped.astype(jnp.int32),
                function_evaluations=state.function_evaluations
                + model_is_usable.astype(jnp.int32),
                gradient_evaluations=state.gradient_evaluations
                + model_is_usable.astype(jnp.int32),
                status=next_status,
                active=next_active,
            )
            record = IterationHistory(
                objective=state.objective,
                trial_objective=trial_objective,
                gradient_norm=state.gradient_norm,
                radius_before=state.radius,
                radius_after=next_radius,
                ratio=ratio,
                actual_reduction=actual_reduction,
                predicted_reduction=subproblem.predicted_reduction,
                step_norm=subproblem.step_norm,
                accepted=accepted,
                sr1_updated=update_applied,
                sr1_denominator=jnp.where(
                    finite_trial,
                    raw_update.denominator,
                    jnp.asarray(jnp.nan, dtype=dtype),
                ),
                hit_boundary=subproblem.hit_boundary,
                valid=jnp.asarray(True),
            )
            return next_state, record

        def step_once(
            state: _State,
        ) -> tuple[_State, IterationHistory]:
            return lax.cond(
                state.active,
                active_iteration,
                lambda current: (current, _inactive_record(current)),
                state,
            )

        if options.store_history:
            final_state, history = lax.scan(
                lambda carry, _: step_once(carry),
                state0,
                xs=None,
                length=options.max_iterations,
            )
        else:
            def loop_condition(state: _State) -> Array:
                return state.active & (
                    state.iterations < options.max_iterations
                )

            def loop_body(state: _State) -> _State:
                next_state, _ = active_iteration(state)
                return next_state

            final_state = lax.while_loop(
                loop_condition, loop_body, state0
            )
            history = _empty_history(x0.dtype)

        exhausted = final_state.active & (
            final_state.iterations >= options.max_iterations
        )
        final_status = jnp.where(
            exhausted, max_iterations_status, final_state.status
        )
        success = final_status == converged_status

        return SR1Result(
            x=final_state.x,
            fun=final_state.objective,
            grad=final_state.gradient,
            grad_norm=final_state.gradient_norm,
            hessian_approximation=final_state.matrix,
            radius=final_state.radius,
            iterations=final_state.iterations,
            accepted_steps=final_state.accepted_steps,
            rejected_steps=final_state.rejected_steps,
            sr1_updates=final_state.sr1_updates,
            sr1_skips=final_state.sr1_skips,
            function_evaluations=final_state.function_evaluations,
            gradient_evaluations=final_state.gradient_evaluations,
            status=final_status,
            success=success,
            history=history,
        )

    return solve_with_matrix


def _initial_matrix_for(
    x0: Array,
    options: SR1Options,
    fixed_initial_matrix: Any | None,
) -> Array:
    x0 = jnp.asarray(x0)
    dtype = x0.dtype
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    dimension = x0.shape[0]
    identity = jnp.eye(dimension, dtype=dtype)
    if fixed_initial_matrix is None:
        return (
            jnp.asarray(options.initial_matrix_scale, dtype=dtype)
            * identity
        )
    fixed = jnp.asarray(fixed_initial_matrix, dtype=dtype)
    if fixed.ndim == 0:
        return fixed * identity
    return fixed


def make_solver_with_matrix(
    fun: Objective,
    options: SR1Options | None = None,
    *,
    jit_compile: bool = True,
) -> Callable[..., SR1Result]:
    """Build a solver whose signature is ``solver(x0, B0, *args)``."""

    selected_options = options or SR1Options()
    core = _build_core(fun, selected_options)
    return jax.jit(core) if jit_compile else core


def make_solver(
    fun: Objective,
    options: SR1Options | None = None,
    *,
    initial_matrix: Any | None = None,
    jit_compile: bool = True,
) -> Callable[..., SR1Result]:
    """Build an objective-specialized solver with an identity-style ``B0``.

    ``initial_matrix`` may be a shared dense matrix or a scalar multiple of the
    identity. When omitted, ``options.initial_matrix_scale * I`` is used.
    """

    selected_options = options or SR1Options()
    core = _build_core(fun, selected_options)

    def solve(x0: Array, *args: Any) -> SR1Result:
        matrix0 = _initial_matrix_for(
            x0, selected_options, initial_matrix
        )
        return core(x0, matrix0, *args)

    return jax.jit(solve) if jit_compile else solve


def make_batched_solver(
    fun: Objective,
    options: SR1Options | None = None,
    *,
    arg_in_axes: Sequence[int | None] = (),
    initial_matrix: Any | None = None,
    jit_compile: bool = True,
) -> Callable[..., SR1Result]:
    """Build a ``vmap``-based solver for batches of starts/parameters."""

    selected_options = options or SR1Options()
    core = _build_core(fun, selected_options)

    def solve_one(x0: Array, *args: Any) -> SR1Result:
        matrix0 = _initial_matrix_for(
            x0, selected_options, initial_matrix
        )
        return core(x0, matrix0, *args)

    mapped = jax.vmap(
        solve_one,
        in_axes=(0, *tuple(arg_in_axes)),
        out_axes=0,
    )
    return jax.jit(mapped) if jit_compile else mapped


def make_batched_solver_with_matrix(
    fun: Objective,
    options: SR1Options | None = None,
    *,
    arg_in_axes: Sequence[int | None] = (),
    matrix_in_axes: int | None = 0,
    jit_compile: bool = True,
) -> Callable[..., SR1Result]:
    """Build a batched solver with explicit per-run or shared initial matrices.

    The returned signature is ``solver(x0_batch, B0, *args)``. Set
    ``matrix_in_axes=0`` for one matrix per start or ``None`` to broadcast a
    shared matrix.
    """

    selected_options = options or SR1Options()
    core = _build_core(fun, selected_options)
    mapped = jax.vmap(
        core,
        in_axes=(0, matrix_in_axes, *tuple(arg_in_axes)),
        out_axes=0,
    )
    return jax.jit(mapped) if jit_compile else mapped

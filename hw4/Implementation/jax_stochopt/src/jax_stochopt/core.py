"""JIT-compatible minibatch stochastic optimization, independent of datasets."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import wraps
from numbers import Integral, Real
from typing import Any
import math

import jax
import jax.numpy as jnp

from ._tree import masked_mean, tree_norm, tree_sub
from .objectives import compile_objective
from .optimizers import SGD, Optimizer
from .types import (
    BatchStats,
    PyTree,
    Result,
    ScheduleContext,
    SolverState,
    StepInfo,
    UpdateContext,
)


@dataclass(frozen=True)
class Solver:
    """Four pure, transformable functions with the same static configuration.

    init(params, key, args=None) -> SolverState
    step(state, args=None) -> (SolverState, StepInfo)
    run(params, key, args=None) -> Result
    resume(state, args=None) -> Result

    run and resume execute the factory's fixed number of steps. The latter
    preserves the absolute step, random-key stream, statistics, and momentum.
    Functions are jitted by default; Result and all dynamic states are PyTrees.
    """

    init: Callable
    step: Callable
    run: Callable
    resume: Callable


def _as_schedule(value):
    return value if callable(value) else lambda context: value


def _scalar(value, dtype, name):
    value = jnp.asarray(value, dtype=dtype)
    if value.shape != ():
        raise ValueError(f"{name} must return a scalar, got shape {value.shape}")
    return value


def _validate_params(params):
    params = jax.tree.map(jnp.asarray, params)
    leaves = jax.tree.leaves(params)
    if not leaves:
        raise ValueError("params must have at least one array leaf")
    dtype = leaves[0].dtype
    if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError("parameter leaves must be real float32 or float64 arrays")
    if any(x.dtype != dtype for x in leaves):
        raise TypeError("all parameter leaves must have the same floating dtype")
    if any(x.size == 0 for x in leaves):
        raise ValueError("parameter leaves must be nonempty")
    return params, dtype


def _validate_like(reference, value, name):
    if jax.tree.structure(reference) != jax.tree.structure(value):
        raise ValueError(f"{name} must preserve the parameter PyTree structure")
    for old, new in zip(jax.tree.leaves(reference), jax.tree.leaves(value)):
        if not hasattr(new, "dtype") or new.shape != old.shape or new.dtype != old.dtype:
            raise ValueError(f"{name} must preserve every parameter leaf's shape and dtype")
    return value


def _summary_arrays(data, dtype):
    def cast(x):
        x = jnp.asarray(x)
        return x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x
    return jax.tree.map(cast, data)


def _validate_key(key):
    # RBG's vmap generates a batch from the first key instead of independently
    # mapping each key. That would invalidate prefix sharing and safe padding.
    # 'fry' is the string representation in some older JAX releases.
    if str(jax.random.key_impl(key)) not in ("threefry2x32", "fry"):
        raise ValueError(
            "Use Threefry keys: jax.random.key(seed, impl='threefry2x32'). "
            "This solver requires independent per-key vmap semantics; RBG keys are unsupported."
        )


def _evaluate_objective_minibatch(
    loss_fn: Callable[[PyTree, PyTree, PyTree | None], jax.Array],
    params: PyTree,
    samples: PyTree,
    args: PyTree | None,
    mask: jax.Array,
    dtype: jax.typing.DTypeLike,
    estimate_gradient_variance: bool,
) -> tuple[jax.Array, jax.Array, PyTree, PyTree | None]:
    """Evaluate one fixed-capacity scalar-objective minibatch.

    The surrounding solver calls this function after generating and safely
    padding the samples for one optimization step. This function returns the
    loss and derivative data consumed by the optimizer and statistics code.

    Parameters
    ----------
    loss_fn : Callable[[PyTree, PyTree, PyTree | None], jax.Array]
        Compiled per-sample objective. It is called as
        ``loss_fn(params, one_sample, args)`` and returns one real,
        floating-point scalar.
    params : PyTree[jax.Array]
        Parameters before the current optimizer update. The tree is nonempty;
        every leaf is a nonempty float32 or float64 array, and all leaves use
        the same dtype. Scalar leaves and leaves of any rank are allowed.
    samples : PyTree[jax.Array]
        Already-generated sample batch. Every leaf has a leading dimension of
        ``capacity = mask.shape[0]``; the remaining dimensions describe one
        sample. The tree may be ``()`` for a deterministic objective.
        Sample leaves may use problem-specific numeric or Boolean dtypes and
        need not match the parameter dtype. Inactive lanes have already been
        padded by repeating an active sample.
    args : PyTree | None
        Run-time problem data supplied to the solver. It is shared by all
        sample lanes and has no required minibatch-leading dimension. Its
        leaves may use problem-specific dtypes.
    mask : jax.Array
        One-dimensional Boolean array with shape ``(capacity,)``. True lanes
        belong to the requested minibatch; false lanes are compiled padding
        and must not affect aggregate values. At least one lane is true.
    dtype : dtype-like
        Common parameter-leaf dtype, either float32 or float64. Floating
        scalar outputs from this function use this dtype.
    estimate_gradient_variance : bool
        Static solver configuration. When true, the caller also needs a
        separate parameter gradient for every compiled lane.

    Returns
    -------
    value : jax.Array
        Rank-zero arithmetic mean of the losses in lanes where ``mask`` is
        true, with dtype ``dtype``.
    losses : jax.Array
        Array with shape ``(capacity,)`` and dtype ``dtype``. Entry ``i`` is
        the objective value for compiled lane ``i``. Padded lanes remain in
        this fixed-size array; the caller masks them when forming statistics.
    grads : PyTree[jax.Array]
        Gradient of ``value`` with respect to ``params``. Its container
        structure, leaf shapes, and leaf dtypes exactly match ``params``.
    individual_grads : PyTree[jax.Array] | None
        When ``estimate_gradient_variance`` is true, this has the same tree as
        ``params`` and each leaf has shape
        ``(capacity,) + corresponding_parameter_leaf.shape``. Lane ``i`` is
        associated with loss lane ``i`` and uses the corresponding parameter
        leaf's dtype. Return ``None`` when individual gradients were not
        requested.

    Notes
    -----
    The function must be pure and compatible with JAX JIT and automatic
    differentiation. It must retain fixed-capacity output shapes. Inactive
    lanes may appear in ``losses`` and ``individual_grads``, but they must not
    contribute to ``value`` or ``grads``.
    """
    # STUDENT TODO (Problem 1 - SGD): evaluate the scalar objective across the
    # prepared samples, differentiate it, and form the active minibatch
    # estimate. The result must support array or PyTree parameters, JIT, and
    # both float32/float64.
    def lane_loss(parameters, lane):
        sample = jax.tree.map(lambda values: values[lane], samples)
        return jnp.asarray(loss_fn(parameters, sample, args), dtype=dtype)

    lanes = jnp.arange(mask.shape[0])
    if estimate_gradient_variance:
        losses, individual_grads = jax.vmap(
            jax.value_and_grad(lane_loss), in_axes=(None, 0)
        )(params, lanes)
        return (
            masked_mean(losses, mask), losses,
            masked_mean(individual_grads, mask), individual_grads,
        )

    # Differentiate the mean directly when per-sample gradients are unnecessary.
    def mean_loss(parameters):
        losses = jax.vmap(lane_loss, in_axes=(None, 0))(parameters, lanes)
        return masked_mean(losses, mask), losses

    (value, losses), grads = jax.value_and_grad(mean_loss, has_aux=True)(params)
    return value, losses, grads, None


def make_solver(
    objective,
    sampler: Callable,
    *,
    steps: int,
    optimizer: Optimizer | None = None,
    gradient_fn: Callable | None = None,
    update_transform: Callable | None = None,
    learning_rate=1e-2,
    batch_size=32,
    max_batch_size: int | None = None,
    batch_buckets: Sequence[int] | None = None,
    projection: Callable | None = None,
    project_initial: bool = True,
    summary_fn: Callable | None = None,
    summary_init: Any = (),
    estimate_gradient_variance: bool = False,
    record_params: bool = False,
    checkpoint: bool = False,
    jit: bool = True,
    backend: str | None = None,
) -> Solver:
    """Construct an optimizer for E[objective(params, sampler(key, args), args)].

    Parameters
    ----------
    objective
        A scalar callable (params, sample, args), or a Reduce objective tree.
        May be None when gradient_fn supplies the vector field. Without a
        scalar objective, loss diagnostics are NaN and loss_available is false.
    sampler
        A pure function (key, args) returning ONE fixed-shape sample PyTree.
        All samples are generated on-device; no dataset or Python loader exists.
    steps
        Static nonnegative scan length; static lengths support reverse-mode AD.
    optimizer
        Object with init/update methods; defaults to SGD(). update receives
        value_fn for the same frozen minibatch and returns a direction to
        subtract. Custom and momentum directions need not be descent directions.
    gradient_fn
        Optional (params, sample, args, before_context) -> gradient PyTree,
        averaged directly over the batch instead of differentiating objective.
        It must match parameter structure, shapes, and dtypes. The context
        contains previous statistics and the old optimizer state. If objective
        is also supplied, it provides scalar diagnostics and value_fn; the
        caller is responsible for consistency when a rule uses its Hessian.
        Rules requiring value_fn fail explicitly when objective is absent.
    update_transform
        Optional (candidate_params, UpdateContext) -> (params, auxiliary_data),
        applied after the deterministic optimizer update and before projection.
        The context provides before/after schedule contexts, gradients, update
        direction, learning rate, and an independent key. The auxiliary PyTree
        is recorded in history.update_data; its structure/shapes must be static.
    learning_rate, batch_size
        Scalars or pure ScheduleContext -> scalar callables. Batch sizes are
        ceiled and clipped to [1, max_batch_size]. Invalid nonfinite requests
        use capacity (NaN/+inf) or one (-inf), and remain visible in history.
        Learning rates must be finite and nonnegative; they are not clipped.
    max_batch_size
        Required for an adaptive batch schedule unless batch_buckets supplies
        the capacity. With no buckets, every step computes this many lanes.
    batch_buckets
        Strictly increasing static capacities. lax.switch selects the smallest
        capacity >= the effective requested count. Within it, only exactly that
        count contributes to the objective. Outer vmap can evaluate all branches.
    projection
        Pure (candidate_params, context) -> projected_params. Built-in
        projections also accept an omitted context when called independently.
        Applied initially by default and after every update. Momentum is retained.
    summary_fn, summary_init
        Optional (samples, mask, args) -> data PyTree; reduce the leading batch
        axis using mask. summary_init is a matching initial PyTree or args ->
        PyTree callable. Shape/structure/dtype must stay fixed across capacities.
    estimate_gradient_variance
        Materialize per-sample gradients for their covariance trace. Costs
        O(capacity * parameter_count) memory. Otherwise differentiate the scalar
        batch mean, and grad_variance_available is false. A supplied vector field
        is always vmapped directly, irrespective of this variance option.
    record_params, checkpoint
        Save post-update parameter history (off by default); recompute scan-body
        residuals during reverse AD (off by default), trading time for memory.
    jit, backend
        Compile by default. None uses JAX's selected device; 'cpu' or 'gpu'
        places inputs on its first device for direct calls and fails if
        unavailable. An enclosing jit controls the complete graph's placement;
        place that outer computation's inputs on the desired device too.

    Notes
    -----
    Parameter leaves must share a real float32/float64 dtype. Enable JAX x64
    before creating float64 arrays. Pass differentiable numeric quantities via
    args; configuration and function topology remain static. Integer batch-size
    choices and hard active-set changes do not have smooth derivatives. Keys
    must use Threefry (the standard JAX default); experimental RBG keys are
    rejected because their vmap behavior breaks independent keyed sampling.
    """
    if not isinstance(steps, Integral) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a nonnegative static integer")
    if not callable(sampler):
        raise TypeError("sampler must be callable")
    if gradient_fn is not None and not callable(gradient_fn):
        raise TypeError("gradient_fn must be callable")
    if objective is None and gradient_fn is None:
        raise ValueError("objective may be None only when gradient_fn is supplied")
    if update_transform is not None and not callable(update_transform):
        raise TypeError("update_transform must be callable")
    if not callable(batch_size) and isinstance(batch_size, Real):
        if not math.isfinite(batch_size) or batch_size < 1:
            raise ValueError("constant batch_size must be finite and at least one")
    if isinstance(learning_rate, Real):
        if not math.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError("constant learning_rate must be finite and nonnegative")

    if batch_buckets is not None:
        capacities = tuple(batch_buckets)
        if not capacities or any(not isinstance(x, Integral) or x < 1 for x in capacities):
            raise ValueError("batch_buckets must be a nonempty sequence of positive integers")
        if any(a >= b for a, b in zip(capacities, capacities[1:])):
            raise ValueError("batch_buckets must be strictly increasing")
        if max_batch_size is None:
            max_batch_size = capacities[-1]
        elif max_batch_size != capacities[-1]:
            raise ValueError("max_batch_size must equal the largest batch bucket")
    else:
        capacities = ()
    if max_batch_size is None:
        if isinstance(batch_size, Integral):
            max_batch_size = int(batch_size)
        else:
            raise ValueError("an adaptive or traced batch_size needs max_batch_size or batch_buckets")
    if not isinstance(max_batch_size, Integral) or not 1 <= max_batch_size < 2**31:
        raise ValueError("max_batch_size must be a positive static int32-sized integer")
    if not capacities:
        capacities = (int(max_batch_size),)

    loss_fn = None if objective is None else compile_objective(objective)
    rule = SGD() if optimizer is None else optimizer
    lr_schedule = _as_schedule(learning_rate)
    batch_schedule = _as_schedule(batch_size)
    project = projection if projection is not None else lambda p, context: p

    def initialize(params, key, args=None):
        _validate_key(key)
        params, dtype = _validate_params(params)
        zero = jnp.zeros((), dtype)
        count = jnp.zeros((), jnp.int32)
        data = summary_init(args) if callable(summary_init) else summary_init
        data = _summary_arrays(data, dtype)
        initial_loss = zero if loss_fn is not None else jnp.full((), jnp.nan, dtype)
        stats = BatchStats(initial_loss, initial_loss, zero, zero,
                           jnp.asarray(estimate_gradient_variance), count, count,
                           data, jnp.asarray(loss_fn is not None))
        opt_state = rule.init(params)
        if project_initial:
            context = ScheduleContext(count, params, opt_state, stats, args)
            params = _validate_like(params, project(params, context), "projection")
            opt_state = rule.init(params)
        return SolverState(params, opt_state, key, count, stats)

    def make_branch(capacity):
        def branch(operand):
            state, args, count, requested, batch_key, next_key = operand
            params = state.params
            dtype = jax.tree.leaves(params)[0].dtype
            indices = jnp.arange(capacity, dtype=jnp.uint32)
            mask = indices < count.astype(jnp.uint32)
            # Pad by repeating an ACTIVE random draw before evaluating any
            # sampler/objective. A final where alone cannot repair NaN gradients
            # from invalid, unused samples. fold_in also makes all active random
            # draws invariant to capacity under the validated Threefry keys.
            safe_indices = jnp.where(mask, indices, jnp.zeros_like(indices))
            keys = jax.vmap(lambda i: jax.random.fold_in(batch_key, i))(safe_indices)
            samples = jax.vmap(sampler, in_axes=(0, None))(keys, args)

            def one_loss(p, sample, a):
                return loss_fn(p, sample, a).astype(dtype)

            def losses_for(p):
                if loss_fn is None:
                    return jnp.full((capacity,), jnp.nan, dtype)
                return jax.vmap(one_loss, in_axes=(None, 0, None), axis_size=capacity)(p, samples, args)

            def value_fn(p):
                if loss_fn is None:
                    raise ValueError(
                        "This optimizer requested a scalar value_fn, but no objective "
                        "was supplied. Provide a consistent scalar objective for "
                        "Hessian-based rules, or use a vector-field-compatible rule."
                    )
                return masked_mean(losses_for(p), mask)

            if gradient_fn is not None:
                def one_gradient(p, sample, a):
                    before = ScheduleContext(state.step, p, state.optimizer_state, state.stats, a)
                    return _validate_like(p, gradient_fn(p, sample, a, before), "gradient_fn")

                individual_grads = jax.vmap(
                    one_gradient, in_axes=(None, 0, None), axis_size=capacity
                )(params, samples, args)
                grads = masked_mean(individual_grads, mask)
                losses = losses_for(params)
                value = masked_mean(losses, mask)
            else:
                value, losses, grads, individual_grads = _evaluate_objective_minibatch(
                    loss_fn,
                    params,
                    samples,
                    args,
                    mask,
                    dtype,
                    estimate_gradient_variance,
                )

            if estimate_gradient_variance:
                def centered_ss(g, mean):
                    weights = mask.reshape((capacity,) + (1,) * mean.ndim)
                    return jnp.sum(jnp.where(weights, (g - mean) ** 2, jnp.zeros_like(g)))

                squares = jax.tree.leaves(jax.tree.map(centered_ss, individual_grads, grads))
                grad_variance = sum(squares[1:], squares[0]) / jnp.maximum(count - 1, 1).astype(dtype)
            else:
                grad_variance = jnp.zeros((), dtype)

            loss_variance = jnp.sum(jnp.where(mask, (losses - value) ** 2, 0))
            loss_variance = loss_variance / jnp.maximum(count - 1, 1).astype(dtype)
            data = state.stats.data if summary_fn is None else summary_fn(samples, mask, args)
            data = _summary_arrays(data, dtype)
            stats = BatchStats(value, loss_variance, tree_norm(grads), grad_variance,
                               jnp.asarray(estimate_gradient_variance), count,
                               jnp.asarray(capacity, jnp.int32), data,
                               jnp.asarray(loss_fn is not None))
            context = ScheduleContext(state.step, params, state.optimizer_state, stats, args)
            direction, opt_state = rule.update(
                params, grads, state.optimizer_state, value_fn=value_fn, context=context
            )
            direction = _validate_like(params, direction, "optimizer direction")
            context = context._replace(optimizer_state=opt_state)
            rate = _scalar(lr_schedule(context), dtype, "learning-rate schedule")
            candidate = jax.tree.map(lambda p, d: p - rate * d, params, direction)
            update_data = ()
            if update_transform is not None:
                before = ScheduleContext(state.step, params, state.optimizer_state, state.stats, args)
                # Sample indices are strictly below 2**31. This disjoint tag
                # preserves the base sampler and resume streams exactly.
                transform_key = jax.random.fold_in(batch_key, jnp.uint32(0x80000000))
                update_context = UpdateContext(before, context, grads, direction, rate, transform_key)
                candidate, update_data = update_transform(candidate, update_context)
                candidate = _validate_like(params, candidate, "update_transform")
            new_params = _validate_like(params, project(candidate, context), "projection")
            new_state = SolverState(new_params, opt_state, next_key, state.step + 1, stats)
            info = StepInfo(stats, rate, tree_norm(tree_sub(new_params, params)), requested,
                            new_params if record_params else (), update_data)
            return new_state, info

        return branch

    branches = tuple(make_branch(int(capacity)) for capacity in capacities)

    def take_step(state, args=None):
        _validate_key(state.key)
        dtype = jax.tree.leaves(state.params)[0].dtype
        context = ScheduleContext(state.step, state.params, state.optimizer_state, state.stats, args)
        requested = jnp.ceil(_scalar(batch_schedule(context), dtype, "batch-size schedule"))
        finite = jnp.nan_to_num(requested, nan=float(max_batch_size),
                               posinf=float(max_batch_size), neginf=1.0)
        count = jnp.clip(finite, 1, max_batch_size).astype(jnp.int32)
        next_key, batch_key = jax.random.split(state.key)
        operand = (state, args, count, requested, batch_key, next_key)
        if len(branches) == 1:
            return branches[0](operand)
        index = jnp.sum(count > jnp.asarray(capacities[:-1], dtype=jnp.int32), dtype=jnp.int32)
        return jax.lax.switch(index, branches, operand)

    step_for_scan = jax.checkpoint(take_step) if checkpoint else take_step

    def resume(state, args=None):
        final, history = jax.lax.scan(lambda carry, _: step_for_scan(carry, args),
                                      state, xs=None, length=int(steps))
        return Result(final, history)

    def run(params, key, args=None):
        return resume(initialize(params, key, args), args)

    # Explicit placement also works with jit=False and is composable with AD.
    # Avoid deprecated jit(backend=...) while leaving the default unconstrained.
    device = jax.devices(backend)[0] if backend is not None else None

    def prepare(fn):
        compiled = jax.jit(fn) if jit else fn
        if device is None:
            return compiled

        @wraps(fn)
        def placed(*positional, **keywords):
            positional, keywords = jax.device_put((positional, keywords), device)
            with jax.default_device(device):
                return compiled(*positional, **keywords)

        return placed

    return Solver(prepare(initialize), prepare(take_step), prepare(run), prepare(resume))

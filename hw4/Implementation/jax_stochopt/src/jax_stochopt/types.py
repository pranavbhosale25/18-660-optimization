"""All dynamic solver state is an ordinary JAX-compatible NamedTuple PyTree."""
from __future__ import annotations

from typing import Any, NamedTuple

import jax


# JAX accepts nested tuples, lists, dictionaries, named tuples, and registered
# custom containers whose leaves are arrays or other JAX-compatible values.
# Python's type system cannot express that recursive family precisely, so the
# alias gives signatures a meaningful name while docstrings specify each
# function's leaf-level requirements.
PyTree = Any


class BatchStats(NamedTuple):
    """Statistics measured BEFORE the parameter update on one fresh minibatch.

    All built-in statistic, count, and availability fields listed below are
    rank-zero JAX arrays. ``loss``, ``loss_variance``, ``grad_norm``, and
    ``grad_variance`` use the parameter floating-point dtype. ``batch_size``
    and ``capacity`` are integer arrays; the availability fields are Boolean
    arrays. ``data`` is the one exception and may contain nonscalar leaves.

    Attributes
    ----------
    loss
        Mean scalar objective over the active samples at the pre-update
        parameters. It is NaN when no scalar objective was supplied.
    loss_variance
        Unbiased sample variance of the individual active-sample losses when
        at least two samples are active. It is zero for a one-sample minibatch
        and NaN when no scalar objective exists.
    grad_norm
        Global Euclidean norm of the active-minibatch mean gradient across all
        parameter leaves.
    grad_variance
        Trace of the unbiased individual-gradient sample covariance when at
        least two samples are active and the statistic was requested. It is
        zero when unavailable or when the minibatch has one sample.
    grad_variance_available
        Whether the individual-gradient covariance statistic was requested and
        is available for this minibatch.
    batch_size
        Number of active sample lanes used in the estimates.
    capacity
        Number of statically compiled lanes. This may exceed ``batch_size``;
        the extra lanes are padding and are not included in the estimates.
    data
        Fixed-structure PyTree returned by the optional batch-summary callback.
    loss_available
        Whether a scalar objective exists. When false, loss fields are NaN.
    """

    loss: jax.Array
    loss_variance: jax.Array
    grad_norm: jax.Array
    grad_variance: jax.Array
    grad_variance_available: jax.Array
    batch_size: jax.Array
    capacity: jax.Array
    data: PyTree
    loss_available: jax.Array | bool = True


class ScheduleContext(NamedTuple):
    """Read-only inputs to schedules, update rules, and projections.

    The meaning of ``optimizer_state`` and ``stats`` depends on the documented
    call site. Batch-size schedules, supplied gradient fields, and before-step
    temperature policies see the previous statistics and old optimizer state.
    Optimizer updates see current minibatch statistics and old optimizer state.
    Learning-rate schedules and post-update projections see current minibatch
    statistics and the newly updated optimizer state.

    Attributes
    ----------
    step
        Rank-zero integer JAX array containing the zero-based absolute step
        that is about to execute. Resumed runs continue this counter.
    params
        Current pre-update parameter PyTree. It does not become the candidate
        or post-update parameters as the context moves through a step.
    optimizer_state
        Optimizer-state PyTree at the stage described above.
    stats
        ``BatchStats`` from either the previous or current minibatch, as
        described above. At initialization numerical statistics are zero,
        except unavailable losses are NaN, and ``batch_size`` is zero.
    args
        User-supplied run-time problem data, forwarded as a fixed-structure
        PyTree or ``None``.
    """

    step: jax.Array
    params: PyTree
    optimizer_state: PyTree
    stats: BatchStats
    args: PyTree | None


class UpdateContext(NamedTuple):
    """Inputs to a transformation after the optimizer update, before projection.

    A transform receives the deterministic candidate as a separate positional
    argument and returns ``(transformed_candidate, auxiliary_data)``. The
    transformed candidate must be parameter-like, and the auxiliary data must
    retain a fixed PyTree structure across steps.

    Attributes
    ----------
    before
        ``ScheduleContext`` containing previous statistics and the old
        optimizer state.
    after
        ``ScheduleContext`` containing current minibatch statistics and the
        newly updated optimizer state. Both contexts still contain the same
        pre-update parameters.
    grads
        Current active-minibatch mean gradient, as a parameter-like PyTree.
    direction
        Parameter-like direction returned by the optimizer.
    learning_rate
        Rank-zero floating JAX array selected for this update, using the
        parameter dtype.
    key
        Threefry JAX PRNG key reserved for the transform. Typed keys are
        recommended; compatible legacy Threefry keys are also accepted. This
        key is independent of those used to generate the objective minibatch.
    """

    before: ScheduleContext
    after: ScheduleContext
    grads: PyTree
    direction: PyTree
    learning_rate: jax.Array
    key: jax.Array


class SolverState(NamedTuple):
    """Complete state required to execute or resume the next solver step.

    ``params`` is the current parameter PyTree; ``optimizer_state`` is the
    state carried by the selected optimizer; ``key`` seeds the next step;
    ``step`` is its absolute zero-based index; and ``stats`` describes the
    most recently evaluated minibatch. Before the first step, ``stats`` holds
    the initialized placeholder values documented by ``BatchStats``.
    """

    params: PyTree
    optimizer_state: PyTree
    key: jax.Array
    step: jax.Array
    stats: BatchStats


class StepInfo(NamedTuple):
    """Data recorded for one completed solver step.

    ``stats`` describes the pre-update parameters and fresh minibatch.
    ``learning_rate`` is a rank-zero floating array. ``update_norm`` is the
    norm of the actual parameter change after transforms and projection.
    ``requested_batch_size`` is the rank-zero, ceiled request before capacity
    clipping. ``params`` contains post-update parameters only when
    ``record_params=True``; otherwise it is ``()``. ``update_data`` is the
    fixed-structure auxiliary record returned by an update transform, or
    ``()`` when no transform is configured.
    """

    stats: BatchStats
    learning_rate: jax.Array
    update_norm: jax.Array
    requested_batch_size: jax.Array
    params: PyTree
    update_data: PyTree = ()


class Result(NamedTuple):
    """Final solver state together with step-stacked execution history.

    ``state`` can be passed to ``Solver.resume``. ``history`` has the same
    fields as ``StepInfo``; every recorded array or PyTree leaf has a leading
    axis whose length is the configured number of executed steps. The
    ``params`` property is a convenience alias for ``state.params``.
    """

    state: SolverState
    history: StepInfo

    @property
    def params(self):
        return self.state.params


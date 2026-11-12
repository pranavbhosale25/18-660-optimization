"""Optional schedules; any pure JAX function of ScheduleContext also works."""
from __future__ import annotations

from collections.abc import Callable
from numbers import Integral, Real

import jax.numpy as jnp

from .types import ScheduleContext

Schedule = Callable[[ScheduleContext], object]


def constant(value) -> Schedule:
    return lambda context: value


def polynomial_decay(initial, *, decay_steps: int, power=1.0, final=0.0) -> Schedule:
    """Decay from initial at step 0 to final at/after decay_steps."""
    if not isinstance(decay_steps, Integral) or decay_steps <= 0:
        raise ValueError("decay_steps must be a positive static integer")

    def schedule(context):
        fraction = jnp.minimum(context.step.astype(context.stats.loss.dtype) / decay_steps, 1.0)
        return final + (initial - final) * (1.0 - fraction) ** power

    return schedule


def cosine_decay(initial, *, decay_steps: int, final=0.0) -> Schedule:
    if not isinstance(decay_steps, Integral) or decay_steps <= 0:
        raise ValueError("decay_steps must be a positive static integer")

    def schedule(context):
        fraction = jnp.minimum(context.step.astype(context.stats.loss.dtype) / decay_steps, 1.0)
        return final + (initial - final) * (1 + jnp.cos(jnp.pi * fraction)) / 2

    return schedule


def geometric_batch_size(initial: int, *, every: int, growth=2.0) -> Schedule:
    """Grow the requested batch size; the solver clips to its static capacity."""
    if not isinstance(initial, Integral) or initial < 1:
        raise ValueError("initial must be a positive static integer")
    if not isinstance(every, Integral) or every < 1:
        raise ValueError("every must be a positive static integer")
    if isinstance(growth, Real) and growth <= 0:
        raise ValueError("growth must be positive")
    return lambda context: initial * jnp.asarray(growth, context.stats.loss.dtype) ** (context.step // every)


def variance_batch_size(*, initial: int, relative_error=0.25, floor=1e-12) -> Schedule:
    """Heuristic n ~ trace(Cov[g_i]) / (relative_error^2 * ||mean(g_i)||^2).

    Requires make_solver(..., estimate_gradient_variance=True). This consumes
    previous-batch statistics, avoiding sample-dependent stopping within the
    current batch. At initialization or after a singleton batch it uses initial.
    This noisy plug-in rule is a heuristic, not a confidence guarantee. The
    returned discrete decision has no useful ordinary derivative.
    """
    if not isinstance(initial, Integral) or initial < 2:
        raise ValueError("initial must be at least 2 to estimate gradient variance")
    if isinstance(relative_error, Real) and relative_error <= 0:
        raise ValueError("relative_error must be positive")
    if isinstance(floor, Real) and floor <= 0:
        raise ValueError("floor must be positive")

    def schedule(context):
        stats = context.stats
        denominator = jnp.maximum(relative_error**2 * stats.grad_norm**2, floor)
        estimate = stats.grad_variance / denominator
        usable = stats.grad_variance_available & (stats.batch_size >= 2)
        return jnp.where(usable, jnp.maximum(estimate, 2.0), initial)

    return schedule


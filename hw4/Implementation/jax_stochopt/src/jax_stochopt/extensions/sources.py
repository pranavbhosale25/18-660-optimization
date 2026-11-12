"""Explicit, differentiable gradient sources for energy and score objectives.

Energy leaves return a scalar and have signature ``(params, sample, args)``.
Score leaves have signature ``(params, sample, args, temperature)`` and return
the same PyTree, array shapes, and float dtype as ``params``.

These adapters use the base solver's generic ``gradient_fn`` hook.  They do
not modify optimizer rules, invent scalar losses, or install custom AD rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable
import math

import jax
import jax.numpy as jnp

from ..objectives import Reduce, compile_objective
from ..types import PyTree, ScheduleContext


def _validate_temperature_spec(temperature, *, positive: bool) -> None:
    """Reject invalid constants while allowing traced run-time parameters."""
    if callable(temperature):
        return
    # Validate ordinary Python numbers without first rounding to JAX's
    # configured default precision. Numeric JAX/NumPy scalars are handled
    # below; a factory may legitimately be constructed inside jax.grad.
    if isinstance(temperature, Real) and not isinstance(temperature, bool):
        concrete = float(temperature)
        if not math.isfinite(concrete) or (concrete <= 0 if positive else concrete < 0):
            bound = "positive" if positive else "nonnegative"
            raise ValueError(f"temperature must be finite and {bound}")
        return
    array = jnp.asarray(temperature)
    if array.shape != ():
        raise ValueError("temperature must be a real scalar or a scalar schedule")
    if not (jnp.issubdtype(array.dtype, jnp.floating) or jnp.issubdtype(array.dtype, jnp.integer)):
        raise TypeError("temperature must be real and numeric")
    if not isinstance(array, jax.core.Tracer):
        concrete = float(array)
        if not math.isfinite(concrete) or (concrete <= 0 if positive else concrete < 0):
            bound = "positive" if positive else "nonnegative"
            raise ValueError(f"temperature must be finite and {bound}")


def _temperature_value(temperature, context, dtype, *, positive: bool):
    """Evaluate a policy without silently clipping invalid dynamic values."""
    value = temperature(context) if callable(temperature) else temperature
    value = jnp.asarray(value)
    if value.shape != ():
        raise ValueError(f"temperature must return a scalar, got shape {value.shape}")
    if not (jnp.issubdtype(value.dtype, jnp.floating) or jnp.issubdtype(value.dtype, jnp.integer)):
        raise TypeError("temperature must return a real numeric scalar")
    value = value.astype(dtype)
    valid = jnp.isfinite(value) & (value > 0 if positive else value >= 0)
    return value, valid


def _parameter_dtype(params):
    leaves = jax.tree.leaves(params)
    if not leaves:
        raise ValueError("params must have at least one array leaf")
    return jnp.asarray(leaves[0]).dtype


def _check_score_like(params, score):
    score = jax.tree.map(jnp.asarray, score)
    if jax.tree.structure(params) != jax.tree.structure(score):
        raise ValueError("score must preserve the parameter PyTree structure")
    for param, value in zip(jax.tree.leaves(params), jax.tree.leaves(score)):
        param = jnp.asarray(param)
        if value.shape != param.shape or value.dtype != param.dtype:
            raise ValueError("score must preserve every parameter leaf's shape and dtype")
    return score


def compile_score(tree: Callable | Reduce) -> Callable:
    """Compile a callable/``Reduce`` score tree with a PyTree-valued root.

    Leaves receive ``(params, sample, args, temperature)``. Each reducer is
    called with one tuple of child outputs, exactly as in an objective tree.
    Intermediate results may have any structure understood by their parent;
    only the root must match ``params``. For a PyTree sum, for example, use
    ``lambda values: jax.tree.map(lambda *xs: sum(xs), *values)``.

    Tree composition happens within each stochastic sample, before the solver
    averages sample scores. It does not turn arbitrary vector fields into
    gradients of a normalized probability density.
    """
    def build(node, path):
        if isinstance(node, Reduce):
            label = node.name or path
            children = tuple(
                build(child, f"{label}.children[{index}]")
                for index, child in enumerate(node.children)
            )
            reducer = node.reducer

            def composed(params, sample, args, temperature):
                return reducer(tuple(
                    child(params, sample, args, temperature) for child in children
                ))

            return composed
        if not callable(node):
            raise TypeError(f"Score-tree node {path!r} must be callable or Reduce")
        return node

    evaluate = build(tree, "root")

    def score(params, sample, args, temperature):
        return _check_score_like(params, evaluate(params, sample, args, temperature))

    return score


@dataclass(frozen=True)
class EnergyGradient:
    """Use an energy/tree as an explicit, ordinary-AD gradient source.

    Call signature is the base hook's ``(params, sample, args, context)``.
    ``make_energy_langevin`` normally uses the base scalar-objective path
    directly, avoiding unnecessary per-sample gradient materialization.
    """

    energy: Any
    _energy_fn: Callable = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_energy_fn", compile_objective(self.energy))

    def __call__(self, params, sample, args, context):
        del context
        return jax.grad(self._energy_fn)(params, sample, args)


@dataclass(frozen=True)
class ScoreGradient:
    """Adapt a temperature-dependent score to the explicit-gradient API.

    ``temperature`` is positive, either a scalar or a pure function of the
    BEFORE-step ``ScheduleContext``. The same policy/context is used by the
    Langevin transform for its noise. Scalar score temperature zero is
    undefined; use the energy path for the zero-temperature limit.

    Invalid constants raise at construction. A dynamic result with the wrong
    shape or a non-real dtype raises when evaluated. A real numeric scalar
    outside the permitted domain produces NaN gradient leaves rather than
    being clipped or silently replaced. All continuous score, temperature,
    parameter, and argument derivatives are retained.
    """

    score: Any
    temperature: Any = 1.0
    _score_fn: Callable = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        _validate_temperature_spec(self.temperature, positive=True)
        object.__setattr__(self, "_score_fn", compile_score(self.score))

    def __call__(
        self,
        params: PyTree,
        sample: PyTree,
        args: PyTree | None,
        context: ScheduleContext,
    ) -> PyTree:
        """Produce one parameter-shaped gradient contribution from a score.

        The generic solver calls this method for one sampler realization and
        vectorizes it over the compiled minibatch. This method therefore
        handles one sample; the solver averages active lanes afterward.

        Parameters
        ----------
        params : PyTree[jax.Array]
            Current parameters before this optimization step. The tree is
            nonempty, all leaves use one float32 or float64 dtype, and scalar
            or multidimensional leaves are allowed.
        sample : PyTree
            One unbatched realization returned by the configured sampler. It
            may be an array, a nested PyTree, or ``()`` when the default empty
            sampler is used. Its structure, leaf shapes, and leaf dtypes remain
            fixed, but its dtype need not match the parameter dtype.
        args : PyTree | None
            User-supplied run-time problem data. Numeric leaves may be traced
            and differentiated; they are not host-side configuration values.
        context : ScheduleContext
            State before the current optimization step. ``context.step`` is
            the zero-based step; ``context.params`` is the same parameter
            tree; ``context.optimizer_state`` is the old optimizer state;
            ``context.stats`` describes the preceding completed minibatch (or
            initialized statistics at step zero); and ``context.args`` is the
            same run-time problem data supplied as ``args``.

            If ``self.temperature`` is a callable, it receives this context
            and returns one real scalar. The evaluated scalar is represented
            in the parameter dtype and is also passed to ``self.score``.

        Returns
        -------
        PyTree[jax.Array]
            Per-sample gradient-like value for the generic solver. Its
            structure, leaf shapes, and leaf dtypes exactly match ``params``.
            The surrounding solver validates it, averages it over active
            samples, and passes that average to the selected optimizer. The
            optimizer later produces the distinct optimizer direction.

        Notes
        -----
        The configured score receives
        ``(params, sample, args, evaluated_temperature)`` and returns a
        parameter-like PyTree. This adapter does not average a minibatch,
        apply a learning rate, update parameters, add random noise, or apply
        constraints.

        Constant temperatures are validated at construction. A dynamic result
        must be one real numeric scalar; a wrong shape raises ``ValueError``
        and a non-real dtype raises ``TypeError``. If a scalar numeric result
        is nonfinite or not strictly positive, every returned floating leaf
        must contain NaNs while preserving the parameter tree's structure,
        shapes, and dtypes. The function must remain pure and compatible with
        JAX JIT, vectorization, and automatic differentiation.
        """
        # STUDENT TODO (Problem 2 - Langevin): convert the supplied score field
        # into the gradient-like value expected by the generic solver.
        # Preserve the parameter PyTree's structure, shapes, and dtypes,
        # including invalid-temperature behavior. See the assignment PDF for
        # the score convention.
        temperature, valid = _temperature_value(
            self.temperature, context, _parameter_dtype(params), positive=True
        )
        score = self._score_fn(params, sample, args, temperature)
        return jax.tree.map(
            lambda value: jnp.where(valid, -temperature * value, jnp.full_like(value, jnp.nan)),
            score,
        )


__all__ = ["EnergyGradient", "ScoreGradient", "compile_score"]

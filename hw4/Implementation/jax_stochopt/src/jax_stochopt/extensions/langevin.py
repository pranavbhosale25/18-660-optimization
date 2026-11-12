"""Energy/score Langevin extensions composed above the generic optimizer.

Temperature policies read BEFORE-step state. Learning-rate policies retain
the base solver's current-data/new-momentum timing. Langevin randomness uses a
stream disjoint from the objective's sample stream and is inserted after the
optimizer update and before projection.

At finite step size these are unadjusted Langevin discretizations, not exact
Gibbs samplers. Stochastic-gradient noise adds further approximation. Adam,
momentum, Muon, Newton or arbitrary preconditioners combined with this
isotropic noise are optimization heuristics; their invariant distributions
are not automatically the intended Gibbs distribution. Hard projection,
state-dependent policies, and arbitrary supplied score fields also require
their own sampling analysis. No acceptance correction is implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .._tree import safe_sqrt, tree_norm
from ..core import make_solver
from ..types import PyTree, UpdateContext
from .sources import (
    ScoreGradient,
    _parameter_dtype,
    _temperature_value,
    _validate_temperature_spec,
)


class LangevinInfo(NamedTuple):
    """Per-step diagnostics stored in ``result.history.update_data``.

    ``temperature``, ``noise_scale``, and ``noise_norm`` are rank-zero arrays
    in the parameter dtype. ``noise_scale`` describes the perturbation before
    projection. ``noise_norm`` is the global Euclidean norm of the actual
    sampled perturbation across every parameter leaf, not the norm of the
    projected or net update. ``valid_temperature`` is a rank-zero Boolean
    recording whether the scheduled temperature satisfied its configured
    domain.
    """

    temperature: jax.Array
    noise_scale: jax.Array
    noise_norm: jax.Array
    valid_temperature: jax.Array


@dataclass(frozen=True)
class LangevinNoise:
    """Generic Langevin proposal transform.

    Call as ``transform(candidate, UpdateContext) -> (candidate, diagnostics)``
    or pass the object to ``make_solver(..., update_transform=...)``. It never
    changes optimizer state and supports independent noise for all PyTree
    leaves in their existing float32/float64 dtype.

    ``temperature`` may depend on iteration, previous batch statistics, and
    old optimizer state via ``context.before``. Zero is valid by default and
    gives exactly the original deterministic candidate, including under JAX
    differentiation.

    Constants outside the domain raise. A dynamic result with the wrong shape
    or a non-real dtype raises when evaluated. A real numeric scalar outside
    the permitted domain gives NaN perturbations plus
    ``valid_temperature=False``. Rates follow the base solver's contract:
    finite, nonnegative, and not silently clipped.
    """

    temperature: Any = 1.0
    require_positive_temperature: bool = False

    def __post_init__(self):
        if not isinstance(self.require_positive_temperature, bool):
            raise TypeError("require_positive_temperature must be a bool")
        _validate_temperature_spec(
            self.temperature, positive=self.require_positive_temperature
        )

    def __call__(
        self,
        candidate: PyTree,
        context: UpdateContext,
    ) -> tuple[PyTree, LangevinInfo]:
        """Apply the stochastic proposal transform for one solver step.

        This method runs after the optimizer has returned a direction and the
        solver has applied the learning rate, but before projection. It does
        not update optimizer state or advance the solver's base key stream.

        Parameters
        ----------
        candidate : PyTree[jax.Array]
            Deterministic parameter candidate for this step. It has the same
            structure, leaf shapes, and float32/float64 dtypes as the current
            parameters. It has not yet been projected and need not already
            satisfy the problem constraints.
        context : UpdateContext
            Read-only information for this proposal:

            ``context.before``
                A ``ScheduleContext`` containing statistics from the previous
                minibatch and the old optimizer state.
            ``context.after``
                A ``ScheduleContext`` containing statistics from the current
                minibatch and the newly updated optimizer state. Its
                ``params`` field is still the parameter tree at the beginning
                of the step, not ``candidate``.
            ``context.grads``
                Current active-minibatch gradient estimate, as a
                parameter-like PyTree.
            ``context.direction``
                Parameter-like direction returned by the optimizer.
            ``context.learning_rate``
                Rank-zero learning rate for this step, in the parameter dtype.
            ``context.key``
                JAX PRNG key reserved for this transform and independent of
                the keys used to generate the objective minibatch.

            A callable temperature receives ``context.before``. It therefore
            observes previous statistics and the old optimizer state.

        Returns
        -------
        updated_candidate : PyTree[jax.Array]
            Stochastic candidate before projection. Its tree structure, leaf
            shapes, and leaf dtypes exactly match ``candidate``. Perturbations
            for distinct parameter entries are independent.
        info : LangevinInfo
            Diagnostics for the unprojected proposal. ``temperature`` is the
            evaluated temperature; ``noise_scale`` is the scalar standard
            deviation used for parameter entries; ``noise_norm`` is the
            global norm of the sampled perturbation; and
            ``valid_temperature`` is a Boolean validity flag.

        Notes
        -----
        When zero temperature is permitted and selected,
        ``updated_candidate`` must equal ``candidate`` exactly. Constant
        temperatures are validated at construction. A dynamic result must be
        one real numeric scalar; a wrong shape raises ``ValueError`` and a
        non-real dtype raises ``TypeError``. A scalar numeric result must be
        finite and nonnegative, or strictly positive when
        ``require_positive_temperature`` is true. Values outside that domain
        are not clipped: the returned parameter leaves, ``noise_scale``, and
        ``noise_norm`` expose NaNs, while ``valid_temperature`` is false.

        Diagnostics describe the perturbation before projection. A later
        projection can change the final displacement, so ``noise_norm`` need
        not equal the solver's recorded ``update_norm``. This method must be
        pure and compatible with JAX JIT, vectorization, and automatic
        differentiation.
        """
        # STUDENT TODO (Problem 2 - Langevin): apply one noise update to the
        # deterministic candidate and return its LangevinInfo diagnostics.
        # Support arbitrary parameter PyTrees and independent randomness for
        # every leaf. See the assignment PDF for the required distribution and
        # scaling.
        dtype = _parameter_dtype(candidate)
        temperature, valid = _temperature_value(
            self.temperature, context.before, dtype,
            positive=self.require_positive_temperature,
        )
        scale = jnp.where(
            valid, safe_sqrt(2 * context.learning_rate * temperature),
            jnp.asarray(jnp.nan, dtype=dtype),
        )
        leaves, structure = jax.tree.flatten(candidate)
        keys = jax.random.split(context.key, len(leaves))
        noise = jax.tree.unflatten(structure, [
            scale * jax.random.normal(key, leaf.shape, dtype=leaf.dtype)
            for leaf, key in zip(leaves, keys)
        ])
        updated = jax.tree.map(jnp.add, candidate, noise)
        return updated, LangevinInfo(temperature, scale, tree_norm(noise), valid)


def _empty_sampler(key, args):
    del key, args
    return ()


def _factory_options(solver_options):
    options = dict(solver_options)
    for hook in ("gradient_fn", "update_transform"):
        if options.pop(hook, None) is not None:
            raise ValueError(
                f"Langevin factories reserve {hook!r}; use make_solver with "
                "EnergyGradient/ScoreGradient and LangevinNoise directly. "
                "Use chain_update_transforms to compose multiple transforms."
            )
    options.setdefault("batch_size", 1)
    return options


def make_energy_langevin(energy, sampler=None, *, temperature=1.0, **solver_options):
    """Compose energy-based SGLD above any unchanged base optimizer rule.

    ``energy`` is a scalar callable or Reduce tree. The default sampler
    returns ``()`` and default batch size is one; provide a sampler for a
    stochastic objective. Remaining options are forwarded to ``make_solver``
    (including ``steps``, schedules, optimizer, projections, precision/device
    placement, buckets, continuation, and checkpointing).

    With plain SGD this gives the assignment's energy-based Langevin method.
    Other rules remain available as noisy optimization heuristics, with no
    claim to a calibrated invariant distribution. Temperature zero recovers
    the base optimizer with identical objective-sampling keys.

    The two generic hooks are reserved by this convenience factory. For a
    different source or additional transforms, build the same composition
    explicitly with ``make_solver`` and ``chain_update_transforms``.
    """
    options = _factory_options(solver_options)
    return make_solver(
        energy,
        _empty_sampler if sampler is None else sampler,
        update_transform=LangevinNoise(temperature),
        **options,
    )


def make_score_langevin(
    score, sampler=None, *, temperature=1.0, energy=None, **solver_options
):
    """Compose score-based SGLD without requiring an available scalar energy.

    A score uses four arguments ``(params, sample, args, temperature)`` and
    returns the parameter PyTree. The default sampler returns ``()`` and the
    default batch size is one.

    When ``energy`` is omitted, scalar loss statistics are NaN with
    ``loss_available=False``; no scalar surrogate is invented. A consistent
    energy may be supplied for real loss diagnostics and Hessian-based rules.
    The user is responsible for consistency when an optimizer uses both the
    scalar energy and score. A score alone does not satisfy the scalar-objective
    contract required by NewtonCG.

    Temperatures must be strictly positive. All base options and the sampling
    limitations in the module documentation apply.
    """
    options = _factory_options(solver_options)
    return make_solver(
        energy,
        _empty_sampler if sampler is None else sampler,
        gradient_fn=ScoreGradient(score, temperature),
        update_transform=LangevinNoise(temperature, require_positive_temperature=True),
        **options,
    )


__all__ = [
    "LangevinInfo", "LangevinNoise", "make_energy_langevin", "make_score_langevin"
]

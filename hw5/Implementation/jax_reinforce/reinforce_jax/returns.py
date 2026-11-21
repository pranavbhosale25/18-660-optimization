"""Differentiable return construction; NO stop_gradient inside these functions.

A true terminal state has zero tail, regardless of the bootstrap supplied.
A truncation/data boundary uses its actual final observation for bootstrapping.
Drop mode discards an entry only if it lacks its requested reward horizon AND
has not reached a true terminal state. Masks keep shapes static under JIT.
"""
from dataclasses import dataclass
from typing import Literal
import jax
import jax.numpy as jnp
from .types import Array, Trajectory, ReturnTargets, Estimates


@dataclass(frozen=True)
class ReturnConfig:
    gamma: float = 0.99
    horizon: int | None = None
    missing: Literal["bootstrap", "drop"] = "bootstrap"

    def __post_init__(self):
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if self.horizon is not None and (not isinstance(self.horizon, int) or self.horizon < 1):
            raise ValueError("horizon must be None or a positive integer")
        if self.missing not in ("bootstrap", "drop"):
            raise ValueError("missing must be 'bootstrap' or 'drop'")


def n_step_returns(batch: Trajectory, next_values: Array,
                   config: ReturnConfig = ReturnConfig()) -> ReturnTargets:
    """Construct a discounted target from every recorded starting transition.

    Parameters
    ----------
    batch:
        Time-major trajectory data. ``batch.rewards`` is a floating-point array
        with shape ``[T, B]``, where ``T`` is the stored time-axis length and
        ``B`` is the number of independent environment lanes. ``terminated``,
        ``truncated``, and ``valid`` are Boolean arrays with the same leading
        shape. Every other trajectory field also has leading ``[T, B]`` axes.
        ``valid=False`` marks padding or unavailable data, not a transition with
        a zero reward.
    next_values:
        Floating-point array with shape ``[T, B]`` and normally the same dtype
        as ``batch.rewards``. Entry ``next_values[t, b]`` is the supplied
        bootstrap value at the *actual post-step observation*
        ``batch.next_observations[t, b]``. It is not a value at an automatically
        reset observation. A Q-based bootstrap has already been converted to an
        on-policy state value before this function is called.
    config:
        Immutable Python configuration. ``gamma`` is the discount in ``[0, 1]``.
        ``horizon=n`` requests at most ``n >= 1`` recorded rewards per starting
        entry; ``horizon=None`` uses all rewards available before the recorded
        boundary. ``missing`` is either ``"bootstrap"`` or ``"drop"`` and
        controls the returned training mask as described below. When this
        function is jitted, capture ``config`` in the jitted callable or mark it
        static; it is not an array-valued dynamic argument.

    Returns
    -------
    ReturnTargets
        A tuple of three arrays, each with shape ``[T, B]``:

        ``returns``
            Floating-point discounted targets. For a valid start ``[t, b]``,
            let ``k`` be the number of consecutive rewards actually included.
            The numeric target contains those ``k`` rewards with successive
            powers of ``gamma``, followed by ``gamma**k`` times the tail value.
            The tail is zero after a true termination and is otherwise
            ``next_values`` from the last included transition. Numeric targets
            are still constructed for incomplete entries whose returned mask is
            false. An invalid starting entry has numeric return zero.
        ``mask``
            Boolean array selecting targets that participate in actor and critic
            losses. With ``missing="bootstrap"``, every valid starting entry is
            selected. With ``missing="drop"``, a valid start is selected only if
            it reaches a true termination or collects the complete finite
            horizon. Consequently, ``horizon=None`` with drop mode selects only
            starts that reach a recorded true termination. Invalid starts are
            always false.
        ``steps``
            Integer array giving ``k``, the number of recorded rewards included
            in each numeric target. Count rewards even when ``gamma == 0`` and
            even when drop mode makes the corresponding mask false. Invalid
            starts have zero steps.

    Raises
    ------
    ValueError
        If the stored time or batch axis is empty, or if ``next_values`` does
        not have exactly the same shape as ``batch.rewards``. ``ReturnConfig``
        validates its own fields when it is constructed. Callers are responsible
        for supplying trajectory fields with consistent leading axes and dtypes.

    Notes
    -----
    Beginning at each valid ``[t, b]``, include its current reward and continue
    only within lane ``b``. Stop after the requested finite horizon, a true
    termination, a truncation, padding/invalid data, or the stored-axis end.
    Never continue through a reset into another episode.

    A true termination uses a zero tail. Any other boundary uses the bootstrap
    value at the last included transition. If ``terminated`` and ``truncated``
    are both true, termination takes precedence. Reaching a finite requested
    horizon exactly on a truncated transition counts as a complete horizon and
    still uses that transition's bootstrap value.

    The implementation must preserve the fixed ``[T, B]`` output shapes under
    ``jax.jit``. It must remain differentiable with respect to rewards and
    ``next_values``; do not apply ``stop_gradient`` or convert those arrays to
    NumPy/Python values. Boundary flags, masks, step counts, and configuration
    fields are not differentiable inputs.
    """
    T, B = batch.rewards.shape
    if T < 1 or B < 1:
        raise ValueError("Trajectory time and batch axes must be nonempty")
    if next_values.shape != batch.rewards.shape:
        raise ValueError("next_values must have shape [time, batch]")
    # STUDENT TODO: construct all return values, masks, and step counts.
    gamma = jnp.asarray(config.gamma, dtype=batch.rewards.dtype)
    zeros = jnp.zeros_like(batch.rewards[0])
    counts = jnp.zeros((B,), dtype=jnp.int32)
    false = jnp.zeros((B,), dtype=bool)

    def extend(future, record):
        future_return, future_steps, future_terminal, future_valid = future
        reward, value, terminal, truncated, valid = record
        continues = valid & ~terminal & ~truncated & future_valid
        tail = jnp.where(terminal, 0., value)
        tail = jnp.where(continues, future_return, tail)
        returns = jnp.where(valid, reward + gamma * tail, 0.)
        steps = jnp.where(valid, 1 + jnp.where(continues, future_steps, 0), 0)
        reached_terminal = valid & (terminal | (continues & future_terminal))
        result = (returns, steps, reached_terminal, valid)
        return result, result

    records = (batch.rewards, next_values, batch.terminated,
               batch.truncated, batch.valid)
    if config.horizon is None:
        # One backward pass computes every full available episode segment.
        _, (returns, steps, terminal, _) = jax.lax.scan(
            extend, (zeros, counts, false, false), records, reverse=True)
    else:
        empty = (jnp.zeros_like(batch.rewards),
                 jnp.zeros((T, B), dtype=jnp.int32),
                 jnp.zeros((T, B), dtype=bool),
                 jnp.zeros((T, B), dtype=bool))

        def add_reward(_, previous):
            # Shift within lanes; extend stops at resets, padding, and terminals.
            future = tuple(jnp.concatenate((a[1:], jnp.zeros_like(a[:1])))
                           for a in previous)
            return extend(future, records)[0]

        returns, steps, terminal, _ = jax.lax.fori_loop(
            0, min(config.horizon, T), add_reward, empty)

    complete = terminal
    if config.horizon is not None:
        complete = complete | (steps >= config.horizon)
    mask = batch.valid & (complete if config.missing == "drop" else True)
    return ReturnTargets(returns, mask, steps)


@dataclass(frozen=True)
class NStepEstimator:
    config: ReturnConfig = ReturnConfig()
    use_baseline: bool = True

    def __call__(self, batch: Trajectory, state_values: Array, next_values: Array) -> Estimates:
        result = n_step_returns(batch, next_values, self.config)
        advantage = result.returns - state_values if self.use_baseline else result.returns
        return Estimates(result.returns, advantage, result.mask)


@dataclass(frozen=True)
class GAEEstimator:
    """GAE/TD(lambda). Trace stops at reset/data boundaries, but truncations bootstrap."""
    gamma: float = 0.99
    lam: float = 0.95
    missing: Literal["bootstrap", "drop"] = "bootstrap"

    def __post_init__(self):
        ReturnConfig(self.gamma, None, self.missing)
        if not 0 <= self.lam <= 1:
            raise ValueError("lam must be in [0, 1]")

    def __call__(self, batch: Trajectory, state_values: Array, next_values: Array) -> Estimates:
        delta = batch.rewards + self.gamma * jnp.where(batch.terminated, 0., next_values) - state_values
        false = jnp.zeros_like(batch.valid[0])

        def step(carry, x):
            next_adv, next_valid, next_complete = carry
            td, terminal, truncated, valid = x
            continues = valid & ~terminal & ~truncated & next_valid
            adv = jnp.where(valid, td + self.gamma * self.lam * jnp.where(continues, next_adv, 0.), 0.)
            complete = valid & (terminal | (continues & next_complete))
            return (adv, valid, complete), (adv, complete)

        _, (adv, complete) = jax.lax.scan(
            step, (jnp.zeros_like(delta[0]), false, false),
            (delta, batch.terminated, batch.truncated, batch.valid), reverse=True)
        mask = batch.valid & (complete if self.missing == "drop" else True)
        return Estimates(adv + state_values, adv, mask)

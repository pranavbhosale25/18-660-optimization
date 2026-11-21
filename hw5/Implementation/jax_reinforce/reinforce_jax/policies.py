"""Categorical and (optionally affine-tanh transformed) diagonal Gaussian policies.

Raw/pre-tanh samples are stored to avoid unstable atanh reconstruction near
bounds. Environment actions are NOT clipped after sampling. For bounded
Gaussians, log_prob includes the transform Jacobian.
"""
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
import jax
import jax.numpy as jnp
from .types import ActionSample, Array, ModelApply, PyTree


@dataclass(frozen=True)
class CategoricalPolicy:
    apply: ModelApply
    n_actions: int
    is_discrete: ClassVar[bool] = True
    reparameterizable: ClassVar[bool] = False

    def __post_init__(self):
        if self.n_actions < 1:
            raise ValueError("n_actions must be positive")

    def log_probs(self, params: PyTree, obs: Array) -> Array:
        return jax.nn.log_softmax(self.apply(params, obs))

    def probs(self, params: PyTree, obs: Array) -> Array:
        return jax.nn.softmax(self.apply(params, obs))

    def sample(self, params: PyTree, obs: Array, key: Array) -> ActionSample:
        lp = self.log_probs(params, obs)
        a = jax.random.categorical(key, lp).astype(jnp.int32)
        return ActionSample(a, a, lp[a])

    def log_prob(self, params: PyTree, obs: Array, raw_action: Array) -> Array:
        return self.log_probs(params, obs)[raw_action.astype(jnp.int32)]

    def entropy(self, params: PyTree, obs: Array, key: Array) -> Array:
        del key
        lp = self.log_probs(params, obs)
        return -jnp.sum(jnp.exp(lp) * lp)

    def mode(self, params: PyTree, obs: Array) -> Array:
        return jnp.argmax(self.apply(params, obs)).astype(jnp.int32)


@dataclass(frozen=True)
class GaussianPolicy:
    """apply(params, obs) returns concatenated [mean, log_std].

    low/high=None: unbounded Gaussian. Otherwise every bound must be finite.
    Partially bounded spaces need a custom distribution/transform adapter.
    """
    apply: ModelApply
    action_dim: int
    low: tuple[float, ...] | None = None
    high: tuple[float, ...] | None = None
    min_log_std: float = -5.0
    max_log_std: float = 2.0
    is_discrete: ClassVar[bool] = False
    reparameterizable: ClassVar[bool] = True

    def __post_init__(self):
        if self.action_dim < 1:
            raise ValueError("action_dim must be positive")
        if (self.low is None) != (self.high is None):
            raise ValueError("Supply both low and high, or neither")
        if self.min_log_std >= self.max_log_std:
            raise ValueError("min_log_std must be less than max_log_std")
        if self.low is not None:
            lo, hi = np.asarray(self.low), np.asarray(self.high)
            if lo.shape != (self.action_dim,) or hi.shape != lo.shape:
                raise ValueError("Bounds must have action_dim entries")
            if not (np.isfinite(lo).all() and np.isfinite(hi).all() and (hi > lo).all()):
                raise ValueError("Tanh Gaussian requires finite, strictly ordered bounds")

    def statistics(self, params: PyTree, obs: Array) -> tuple[Array, Array]:
        out = self.apply(params, obs)
        if out.shape != (2 * self.action_dim,):
            raise ValueError("Gaussian model must output 2 * action_dim scalars")
        mean, log_std = jnp.split(out, 2)
        return mean, jnp.clip(log_std, self.min_log_std, self.max_log_std)

    def transform(self, raw: Array) -> Array:
        if self.low is None:
            return raw
        lo, hi = jnp.asarray(self.low, raw.dtype), jnp.asarray(self.high, raw.dtype)
        return (lo + hi) / 2 + (hi - lo) / 2 * jnp.tanh(raw)

    def sample(self, params: PyTree, obs: Array, key: Array) -> ActionSample:
        mean, log_std = self.statistics(params, obs)
        raw = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape, mean.dtype)
        return ActionSample(self.transform(raw), raw, self.log_prob(params, obs, raw))

    def log_prob(self, params: PyTree, obs: Array, raw_action: Array) -> Array:
        mean, log_std = self.statistics(params, obs)
        z = (raw_action - mean) * jnp.exp(-log_std)
        log_base = -0.5 * z**2 - log_std - 0.5 * jnp.log(jnp.asarray(2 * np.pi, mean.dtype))
        if self.low is None:
            return jnp.sum(log_base)
        scale = (jnp.asarray(self.high, mean.dtype) - jnp.asarray(self.low, mean.dtype)) / 2
        # log(1 - tanh(x)^2), stable even for saturated samples.
        log_jac = jnp.log(scale) + 2 * (jnp.log(jnp.asarray(2., mean.dtype))
                                     - raw_action - jax.nn.softplus(-2 * raw_action))
        return jnp.sum(log_base - log_jac)

    def entropy(self, params: PyTree, obs: Array, key: Array) -> Array:
        _, log_std = self.statistics(params, obs)
        if self.low is None:
            c = jnp.asarray(0.5 * np.log(2 * np.pi * np.e), log_std.dtype)
            return jnp.sum(log_std + c)
        # Reparameterized one-sample entropy estimate of the TRANSFORMED policy.
        raw = self.sample(params, obs, key).raw_action
        return -self.log_prob(params, obs, raw)

    def mode(self, params: PyTree, obs: Array) -> Array:
        # Transform of the base mean: deterministic evaluation action. This is
        # not generally the density mode/mean of the transformed distribution.
        mean, _ = self.statistics(params, obs)
        return self.transform(mean)

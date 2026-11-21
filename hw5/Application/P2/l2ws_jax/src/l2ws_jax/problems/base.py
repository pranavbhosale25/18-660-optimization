"""Pure single-instance functions with vmap batching and immutable data."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np

from ..numerics import euclidean_norm
from ..runtime import Runtime


class Problem(ABC):
    name: str
    context_dim: int
    decision_dim: int
    convex = False
    constraint = "unconstrained"
    strong_convexity = 0.0
    sampled_optimum_value = None

    def __init__(self, *, dtype="float64", device="auto"):
        self._frozen = False
        self.runtime = Runtime.create(dtype, device)
        self.dtype = self.runtime.dtype
        self.device = self.runtime.device
        self._data = {}
        self._config = {}

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError(
                "Problem data are immutable; use .to(...) or make a new problem"
            )
        object.__setattr__(self, name, value)

    @property
    def config(self):
        return copy.deepcopy(self._config)

    @config.setter
    def config(self, value):
        self._config = copy.deepcopy(value)

    @property
    def data(self):
        return MappingProxyType(self._data)

    def register(self, name, value, *, dtype=None):
        if self._frozen:
            raise AttributeError(
                "Problem data are immutable; use .to(...) or make a new problem"
            )
        array = self.runtime.array(value, dtype=dtype)
        setattr(self, name, array)
        self._data[name] = array
        return array

    def finish(self):
        """Call at the end of a subclass constructor, before compilation."""
        self._frozen = True
        return self

    def array(self, value):
        return self.runtime.array(value)

    def key(self, seed):
        return self.runtime.key(seed)

    def state_dict(self):
        return {
            name: np.array(jax.device_get(value), copy=True)
            for name, value in self._data.items()
        }

    def with_state(self, state, *, dtype=None, device=None):
        from . import make_problem

        result = make_problem(
            self.name,
            dtype=self.dtype if dtype is None else dtype,
            device=self.device if device is None else device,
            **self.config,
        )
        if set(state) != set(result._data):
            raise ValueError(
                "Problem-state fields do not match the constructor"
            )
        arrays = {}
        for name, value in state.items():
            expected = result._data[name]
            host = np.asarray(jax.device_get(value))
            if host.shape != expected.shape or not np.isfinite(host).all():
                raise ValueError(f"Invalid problem-state array {name}")
            array = result.runtime.array(host, dtype=expected.dtype)
            arrays[name] = array
            object.__setattr__(result, name, array)
        object.__setattr__(result, "_data", arrays)
        return result

    def to(self, *, dtype=None, device=None):
        """Return a new problem, preserving fixed data before moving it."""
        return self.with_state(self._data, dtype=dtype, device=device)

    @abstractmethod
    def sample(self, n, key, *, shift=False):
        """Return contexts [n, context_dim] using an explicit JAX key."""

    @abstractmethod
    def objective_one(self, y, context):
        """Pure scalar objective for one decision and one context vector."""

    @abstractmethod
    def baseline_one(self, context):
        """Meaningful feasible classical initialization for one context."""

    @abstractmethod
    def step_size_one(self, context):
        """A positive scalar fixed step, depending only on the context."""

    def features_one(self, context):
        return context

    def project_one(self, y, context):
        return y

    def decode_one(self, raw, context):
        return self.project_one(self.baseline_one(context) + raw, context)

    def certificate_one(self, y, context):
        return jnp.asarray(jnp.nan, dtype=self.dtype)

    def solution_distance_one(self, y, reference):
        return euclidean_norm(y - reference)

    def extra_metrics_one(self, y, context):
        return {}

    def _batch(self, function, *arguments):
        # Constant outputs (uniform starts, fixed steps, absent certificates)
        # otherwise use the process default device outside whole-function JIT.
        with jax.default_device(self.device):
            # Commit constant outputs too, so a later reshape/index outside the
            # default_device scope does not move an uncommitted array back.
            return jax.device_put(jax.vmap(function)(*arguments), self.device)

    def objective(self, y, context):
        return self._batch(self.objective_one, y, context)

    def gradient(self, y, context):
        return self._batch(jax.grad(self.objective_one), y, context)

    def baseline(self, context):
        return self._batch(self.baseline_one, context)

    def step_size(self, context):
        return self._batch(self.step_size_one, context)[:, None]

    def features(self, context):
        return self._batch(self.features_one, context)

    def project(self, y, context):
        return self._batch(self.project_one, y, context)

    def decode(self, raw, context):
        return self._batch(self.decode_one, raw, context)

    def certificate(self, y, context):
        return self._batch(self.certificate_one, y, context)

    def feasibility(self, y, context):
        return euclidean_norm(y - self.project(y, context))

    def solution_distance(self, y, reference):
        return self._batch(self.solution_distance_one, y, reference)

    def extra_metrics(self, y, context):
        return self._batch(self.extra_metrics_one, y, context)

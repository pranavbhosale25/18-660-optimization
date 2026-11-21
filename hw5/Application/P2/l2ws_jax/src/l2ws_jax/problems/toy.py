"""Completed diagnostic: a strongly convex diagonal quadratic."""

import jax
import jax.numpy as jnp
import numpy as np

from .base import Problem

_NOMINAL_CONTEXT_SCALE = 0.75
_SHIFTED_CONTEXT_SCALE = 1.50


class QuadraticDemo(Problem):
    name = "toy"
    convex = True
    strong_convexity = 1.0

    def __init__(self, dimension=3, *, dtype="float64", device="auto"):
        super().__init__(dtype=dtype, device=device)
        if not isinstance(dimension, int) or dimension < 1:
            raise ValueError("dimension must be a positive integer")
        self.context_dim = self.decision_dim = dimension
        self.config = {"dimension": dimension}
        self.register("curvature", np.logspace(0.0, 1.0, dimension))
        self.finish()

    def sample(self, n, key, *, shift=False):
        if n < 0:
            raise ValueError("n must be nonnegative")
        scale = _SHIFTED_CONTEXT_SCALE if shift else _NOMINAL_CONTEXT_SCALE
        return (
            jax.random.normal(key, (n, self.context_dim), dtype=self.dtype)
            * scale
        )

    def objective_one(self, y, context):
        return 0.5 * jnp.sum(self.curvature * jnp.square(y - context))

    def baseline_one(self, context):
        return jnp.zeros(self.decision_dim, dtype=self.dtype)

    def step_size_one(self, context):
        return jnp.reciprocal(jnp.max(self.curvature))

    def certificate_one(self, y, context):
        gradient = self.curvature * (y - context)
        return jnp.sum(jnp.square(gradient)) / 2.0

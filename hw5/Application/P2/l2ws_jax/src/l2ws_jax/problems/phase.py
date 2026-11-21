"""Real phase retrieval with spectral initialization and JAX refinement."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from ..numerics import euclidean_norm
from ..projections import box_projection
from .base import Problem

_NOMINAL_COEFFICIENT_FRACTION = 0.20
_SHIFTED_COEFFICIENT_FRACTION = 0.40
_NOMINAL_NOISE_FRACTION = 0.04
_SHIFTED_NOISE_FRACTION = 0.08
_NOMINAL_OFFSET_FRACTION = 0.35
_SHIFTED_OFFSET_FRACTION = 0.20
_SIGNAL_INTERIOR_FRACTION = 0.95
_MINIMUM_SIGNAL_DIMENSION = 3


class PhaseRetrieval(Problem):
    name = "phase"
    convex = False
    constraint = "box"
    sampled_optimum_value = 0.0
    objective_lower_bound = 0.0

    def __init__(
        self,
        dimension: int = 6,
        measurements: int = 48,
        radius: float = 1.5,
        seed: int = 23,
        *,
        dtype="float64",
        device="auto",
    ):
        super().__init__(dtype=dtype, device=device)
        if dimension < _MINIMUM_SIGNAL_DIMENSION or int(dimension) != dimension:
            raise ValueError("dimension must be an integer at least three")
        if (
            int(measurements) != measurements
            or measurements < 2 * dimension - 1
        ):
            raise ValueError(
                "measurements must be an integer at least 2*dimension-1"
            )
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and strictly positive")
        self.decision_dim = int(dimension)
        self.context_dim = int(measurements)
        self.radius = float(radius)
        self.seed = int(seed)
        self.config = {
            "dimension": self.decision_dim,
            "measurements": self.context_dim,
            "radius": self.radius,
            "seed": self.seed,
        }
        fixed_rng = np.random.default_rng(self.seed)
        matrix = fixed_rng.standard_normal(
            (self.context_dim, self.decision_dim)
        )
        matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        stored_matrix = matrix.astype(self.dtype)
        self.register("A", stored_matrix)
        self.register("lower", np.full(self.decision_dim, -self.radius))
        self.register("upper", np.full(self.decision_dim, self.radius))
        grid = np.arange(self.decision_dim, dtype=np.float64) * (
            2.0 * math.pi / self.decision_dim
        )
        self.register(
            "signal_modes",
            np.stack((np.cos(grid), np.sin(grid), np.cos(2.0 * grid))),
        )
        self.finish()

    @property
    def measurement_matrix(self):
        return self.A

    def intensities(self, context):
        return context

    def predict_intensities(self, y):
        return jnp.square(y @ self.A.T)

    def sample(self, n, key, *, shift=False):
        """Return intensities; n and shift are static under JIT."""
        if n < 0:
            raise ValueError("n must be nonnegative")
        coefficient_key, noise_key = jax.random.split(key)
        coefficient_fraction = (
            _SHIFTED_COEFFICIENT_FRACTION
            if shift
            else _NOMINAL_COEFFICIENT_FRACTION
        )
        noise_fraction = (
            _SHIFTED_NOISE_FRACTION if shift else _NOMINAL_NOISE_FRACTION
        )
        offset_fraction = (
            _SHIFTED_OFFSET_FRACTION if shift else _NOMINAL_OFFSET_FRACTION
        )
        coefficient_scale = coefficient_fraction * self.radius
        noise_scale = noise_fraction * self.radius
        offset = offset_fraction * self.radius
        mode_count = self.signal_modes.shape[0]
        coefficients = coefficient_scale * (
            2.0
            * jax.random.uniform(
                coefficient_key, (n, mode_count), dtype=self.dtype
            )
            - 1.0
        )
        noise = noise_scale * (
            2.0
            * jax.random.uniform(
                noise_key, (n, self.decision_dim), dtype=self.dtype
            )
            - 1.0
        )
        hidden_signal = offset + coefficients @ self.signal_modes + noise
        interior_radius = _SIGNAL_INTERIOR_FRACTION * self.radius
        hidden_signal = jnp.clip(
            hidden_signal, -interior_radius, interior_radius
        )
        return self.predict_intensities(hidden_signal)

    def objective_one(self, y, context):
        """STUDENT TODO: implement phase.objective_one.

        Inputs:
            y: signal with shape [decision_dim].
            context: observed intensities with shape [context_dim].
        Returns:
            A rank-0 JAX array on self.device, with self.dtype.
        Requirements:
            Use JAX operations and preserve differentiation through y.

        See the separate homework PDF for the mathematical definition.
        """
        residual = jnp.square(self.A @ y) - self.intensities(context)
        return 0.25 * jnp.mean(jnp.square(residual))

    def project_one(self, y, context):
        return box_projection(y, self.lower, self.upper)

    def spectral_initialization_one(self, context):
        """One spectral start; its eigensolve belongs in inference timing."""
        spectral_matrix = (
            (self.A.T * context[None, :]) @ self.A / self.context_dim
        )
        _, eigenvectors = jnp.linalg.eigh(spectral_matrix)
        direction = eigenvectors[:, -1]
        largest_coordinate = jnp.argmax(jnp.abs(direction))
        one = jnp.asarray(1.0, dtype=self.dtype)
        direction = direction * jnp.where(
            direction[largest_coordinate] < 0, -one, one
        )
        amplitude_squared = self.decision_dim * jnp.mean(context)
        amplitude = jnp.sqrt(
            jnp.maximum(amplitude_squared, jnp.asarray(0.0, dtype=self.dtype))
        )
        return self.project_one(amplitude * direction, context)

    def spectral_initialization(self, context):
        """Accept either one intensity vector or a batch."""
        if context.ndim == 1:
            return self.spectral_initialization_one(context)
        return jax.vmap(self.spectral_initialization_one)(context)

    def baseline_one(self, context):
        return self.spectral_initialization_one(context)

    def step_size_one(self, context):
        """STUDENT TODO: implement phase.step_size_one.

        Input:
            context: observed intensities with shape [context_dim].
        Returns:
            A finite, strictly positive rank-0 JAX array on self.device,
            with self.dtype.
        Requirements:
            Derive a finite upper bound on the Hessian spectral norm that
            is valid at every point in the complete signal box for this
            context, and return its reciprocal. Use JAX operations and
            clamp the denominator away from zero in self.dtype.

        See the separate homework PDF for the mathematical definition.
        """
        # |a_i^T y| <= r ||a_i||_1 throughout the signal box.
        max_squared_measurement = jnp.square(
            self.radius * jnp.sum(jnp.abs(self.A), axis=1)
        )
        intensities = self.intensities(context)
        diagonal_bound = jnp.maximum(
            jnp.abs(intensities),
            jnp.abs(3.0 * max_squared_measurement - intensities),
        )
        # A^T diag(diagonal_bound) A dominates the Hessian in absolute value.
        bound_matrix = (self.A.T * diagonal_bound) @ self.A / self.context_dim
        curvature_bound = jnp.linalg.eigvalsh(bound_matrix)[-1]
        return jnp.reciprocal(
            jnp.maximum(curvature_bound, jnp.finfo(self.dtype).eps)
        )

    def solution_distance_one(self, y, reference):
        return jnp.minimum(
            euclidean_norm(y - reference), euclidean_norm(y + reference)
        )

    def extra_metrics_one(self, y, context):
        residual = jnp.square(self.A @ y) - context
        return {
            "intensity_rmse": euclidean_norm(residual)
            / jnp.sqrt(jnp.asarray(self.context_dim, dtype=self.dtype))
        }

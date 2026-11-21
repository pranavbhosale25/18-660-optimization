"""Three-link planar inverse kinematics with pure JAX kernels."""

from __future__ import annotations

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..numerics import euclidean_norm
from ..projections import box_projection
from .base import Problem

_CARTESIAN_DIMENSION = 2
_JOINT_COUNT = 3
_SAMPLE_RANDOM_STREAMS = 3
_SHIFTED_POSE_MINIMUM_FRACTION = 0.60
_SHIFTED_POSE_FRACTION_RANGE = 0.40
_NOMINAL_POSE_FRACTION = 0.85
_SHIFTED_PREFERRED_POSE_FRACTION = 0.95
_NOMINAL_PREFERRED_POSE_FRACTION = 0.65
_UNIFORM_SIGN_SPLIT = 0.5


class InverseKinematics(Problem):
    """Three-link planar arm whose relative joint angles are in radians."""

    name = "kinematics"
    context_dim = _CARTESIAN_DIMENSION + _JOINT_COUNT
    decision_dim = _JOINT_COUNT
    convex = False
    constraint = "box"

    def __init__(
        self,
        regularization: float = 0.03,
        lengths: Sequence[float] = (1.0, 0.8, 0.6),
        joint_limits: Sequence[float] = (1.4, 1.6, 1.5),
        *,
        dtype="float64",
        device="auto",
    ):
        super().__init__(dtype=dtype, device=device)
        lengths = [float(value) for value in lengths]
        joint_limits = [float(value) for value in joint_limits]
        if len(lengths) != self.decision_dim or any(
            not math.isfinite(x) or x <= 0 for x in lengths
        ):
            raise ValueError(
                "lengths must contain three finite positive numbers"
            )
        if len(joint_limits) != self.decision_dim or any(
            not math.isfinite(x) or x <= 0 for x in joint_limits
        ):
            raise ValueError(
                "joint_limits must contain three finite positive numbers"
            )
        if not math.isfinite(regularization) or regularization < 0:
            raise ValueError("regularization must be finite and nonnegative")
        self.regularization = float(regularization)
        self.config = {
            "regularization": self.regularization,
            "lengths": lengths,
            "joint_limits": joint_limits,
        }
        self.register("lengths", np.asarray(lengths, dtype=np.float64))
        limits = np.asarray(joint_limits, dtype=np.float64)
        self.register("lower", -limits)
        self.register("upper", limits)
        self.finish()

    def target(self, context):
        """Cartesian target, supporting a single context or a context batch."""
        return context[..., :_CARTESIAN_DIMENSION]

    def preferred_pose(self, context):
        """Preferred joint angles in radians for one context or a batch."""
        return context[..., _CARTESIAN_DIMENSION:]

    def forward_kinematics(self, q):
        """Map relative angles in radians, [..., 3], to positions [..., 2]."""
        angles = jnp.cumsum(q, axis=-1)
        directions = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)
        return jnp.sum(directions * self.lengths[:, None], axis=-2)

    def joint_positions(self, q):
        """Joint positions for relative angles in radians; shape [..., 4, 2]."""
        angles = jnp.cumsum(q, axis=-1)
        directions = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)
        displacements = directions * self.lengths[:, None]
        origin = jnp.zeros(q.shape[:-1] + (1, 2), dtype=q.dtype)
        return jnp.concatenate(
            (origin, jnp.cumsum(displacements, axis=-2)), axis=-2
        )

    def sample(self, n, key, *, shift=False):
        """Sample observations; n and shift are static under JIT."""
        if n < 0:
            raise ValueError("n must be nonnegative")
        pose_key, magnitude_key, preference_key = jax.random.split(
            key, _SAMPLE_RANDOM_STREAMS
        )
        shape = (n, self.decision_dim)
        draws = jax.random.uniform(pose_key, shape, dtype=self.dtype)
        if shift:
            signs = jnp.where(
                draws < _UNIFORM_SIGN_SPLIT,
                -jnp.ones_like(draws),
                jnp.ones_like(draws),
            )
            magnitudes = (
                _SHIFTED_POSE_MINIMUM_FRACTION
                + _SHIFTED_POSE_FRACTION_RANGE
                * jax.random.uniform(magnitude_key, shape, dtype=self.dtype)
            )
            hidden_pose = signs * magnitudes * self.upper
            preference_fraction = _SHIFTED_PREFERRED_POSE_FRACTION
        else:
            hidden_pose = (2.0 * draws - 1.0) * (
                _NOMINAL_POSE_FRACTION * self.upper
            )
            preference_fraction = _NOMINAL_PREFERRED_POSE_FRACTION
        preference_draws = jax.random.uniform(
            preference_key, shape, dtype=self.dtype
        )
        preferred_pose = (2.0 * preference_draws - 1.0) * (
            preference_fraction * self.upper
        )
        targets = self.forward_kinematics(hidden_pose)
        return jnp.concatenate((targets, preferred_pose), axis=-1)

    def features_one(self, context):
        return jnp.concatenate(
            (
                self.target(context) / jnp.sum(self.lengths),
                self.preferred_pose(context) / self.upper,
            )
        )

    def objective_one(self, y, context):
        """STUDENT TODO: implement kinematics.objective_one.

        Inputs:
            y: relative joint angles in radians with shape [3].
            context: shape [5], containing a 2-D target followed by three
                preferred joint angles in radians.
        Returns:
            A rank-0 JAX array on self.device, with self.dtype.
        Requirements:
            Use JAX operations and preserve differentiation through y.

        See the separate homework PDF for the mathematical definition.
        """
        endpoint_error = self.forward_kinematics(y) - self.target(context)
        posture_error = y - self.preferred_pose(context)
        return 0.5 * (
            jnp.sum(jnp.square(endpoint_error))
            + self.regularization * jnp.sum(jnp.square(posture_error))
        )

    def project_one(self, y, context):
        return box_projection(y, self.lower, self.upper)

    def baseline_one(self, context):
        return self.project_one(self.preferred_pose(context), context)

    def step_size_one(self, context):
        """STUDENT TODO: implement kinematics.step_size_one.

        Input:
            context: one kinematics context with shape [5].
        Returns:
            A finite, strictly positive rank-0 JAX array on self.device,
            with self.dtype.
        Requirements:
            Derive a finite upper bound on the Hessian spectral norm that
            is valid at every point in the complete joint-limit box for
            this context, and return its reciprocal. Use JAX operations
            and clamp the denominator away from zero in self.dtype.

        See the separate homework PDF for the mathematical definition.
        """
        # Column j of the endpoint Jacobian has norm at most suffix_lengths[j].
        suffix_lengths = jnp.cumsum(self.lengths[::-1])[::-1]
        jacobian_bound = jnp.sum(jnp.square(suffix_lengths))
        residual_bound = jnp.sum(self.lengths) + jnp.linalg.norm(
            self.target(context)
        )
        angle_counts = jnp.arange(1, self.decision_dim + 1, dtype=self.dtype)
        curvature_bound = (
            jacobian_bound
            + residual_bound * jnp.sum(angle_counts * self.lengths)
            + self.regularization
        )
        return jnp.reciprocal(
            jnp.maximum(curvature_bound, jnp.finfo(self.dtype).eps)
        )

    def extra_metrics_one(self, y, context):
        return {
            "endpoint_error": euclidean_norm(
                self.forward_kinematics(y) - self.target(context)
            ),
            "posture_deviation": euclidean_norm(
                y - self.preferred_pose(context)
            ),
        }

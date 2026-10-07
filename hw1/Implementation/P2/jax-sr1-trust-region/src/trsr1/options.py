"""Static solver options and validation."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class SR1Options:
    """Configuration for the full-memory SR1 trust-region method.

    The defaults follow Nocedal--Wright Algorithm 6.2: accept a trial step when
    ``rho > eta``; halve the radius when ``rho < 0.1``; preserve it for ratios
    in ``[0.1, 0.75]``; and double it when ``rho > 0.75`` and the trial step is
    close to the trust-region boundary.
    """

    max_iterations: int = 250
    gradient_tolerance: float = 1.0e-6

    initial_radius: float = 1.0
    maximum_radius: float = 1.0e6
    minimum_radius: float = 1.0e-12

    acceptance_threshold: float = 1.0e-4
    poor_ratio_threshold: float = 0.1
    good_ratio_threshold: float = 0.75
    boundary_fraction: float = 0.8
    shrink_factor: float = 0.5
    expansion_factor: float = 2.0

    sr1_skip_tolerance: float = 1.0e-8
    initial_matrix_scale: float = 1.0
    symmetrize_matrices: bool = True

    store_history: bool = True

    def __post_init__(self) -> None:
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be nonnegative")
        if not _positive_finite(self.gradient_tolerance):
            raise ValueError("gradient_tolerance must be finite and positive")
        if not _positive_finite(self.initial_radius):
            raise ValueError("initial_radius must be finite and positive")
        if not _positive_finite(self.maximum_radius):
            raise ValueError("maximum_radius must be finite and positive")
        if self.initial_radius > self.maximum_radius:
            raise ValueError("initial_radius cannot exceed maximum_radius")
        if not _positive_finite(self.minimum_radius):
            raise ValueError("minimum_radius must be finite and positive")
        if self.minimum_radius >= self.initial_radius:
            raise ValueError("minimum_radius must be smaller than initial_radius")

        if not 0.0 <= self.acceptance_threshold < self.poor_ratio_threshold:
            raise ValueError(
                "acceptance_threshold must lie in [0, poor_ratio_threshold)"
            )
        if not (
            0.0
            < self.poor_ratio_threshold
            < self.good_ratio_threshold
            < 1.0
        ):
            raise ValueError(
                "ratio thresholds must satisfy 0 < poor < good < 1"
            )
        if not 0.0 < self.boundary_fraction <= 1.0:
            raise ValueError("boundary_fraction must lie in (0, 1]")
        if not 0.0 < self.shrink_factor < 1.0:
            raise ValueError("shrink_factor must lie in (0, 1)")
        if self.expansion_factor <= 1.0 or not math.isfinite(
            self.expansion_factor
        ):
            raise ValueError("expansion_factor must be finite and greater than 1")

        if not 0.0 < self.sr1_skip_tolerance < 1.0:
            raise ValueError("sr1_skip_tolerance must lie in (0, 1)")
        if not math.isfinite(self.initial_matrix_scale):
            raise ValueError("initial_matrix_scale must be finite")


def _positive_finite(value: float) -> bool:
    return value > 0.0 and math.isfinite(value)

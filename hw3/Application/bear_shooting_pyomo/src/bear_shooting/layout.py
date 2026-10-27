"""Public array layouts. Controls and observations always use NumPy C order."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import numpy as np


def vector(value, size: int, name: str, *, finite: bool = True) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != size:
        raise ValueError(f"{name} has {result.size} entries; expected {size}.")
    invalid = not np.all(np.isfinite(result)) if finite else np.any(np.isnan(result))
    if invalid:
        raise ValueError(f"{name} contains invalid numerical values.")
    return result.copy()


def broadcast(value, shape: tuple[int, ...], name: str, *, finite: bool = True) -> np.ndarray:
    try:
        result = np.broadcast_to(np.asarray(value, dtype=np.float64), shape).copy()
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to {shape}.") from exc
    invalid = not np.all(np.isfinite(result)) if finite else np.any(np.isnan(result))
    if invalid:
        raise ValueError(f"{name} contains invalid numerical values.")
    return result


@dataclass(frozen=True)
class Layout:
    horizon: int
    actuator_names: tuple[str, ...]
    part_names: tuple[str, ...]
    control_dt: float

    def __post_init__(self):
        if isinstance(self.horizon, bool) or int(self.horizon) != self.horizon or self.horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        if not np.isfinite(self.control_dt) or self.control_dt <= 0:
            raise ValueError("control_dt must be positive and finite.")
        object.__setattr__(self, "horizon", int(self.horizon))
        object.__setattr__(self, "actuator_names", tuple(self.actuator_names))
        object.__setattr__(self, "part_names", tuple(self.part_names))
        for names, label in [(self.actuator_names, "actuators"), (self.part_names, "parts")]:
            if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
                raise ValueError(f"{label} must have nonempty, unique names.")

    @property
    def control_shape(self):
        return self.horizon, len(self.actuator_names)

    @property
    def n_controls(self):
        return self.horizon * len(self.actuator_names)

    @property
    def observation_shape(self):
        # [knot, part, (px,py,pz,vx,vy,vz)]
        return self.horizon + 1, len(self.part_names), 6

    @property
    def n_observations(self):
        return int(np.prod(self.observation_shape))

    @property
    def times(self):
        return np.arange(self.horizon + 1, dtype=float) * self.control_dt

    def knot(self, knot: int) -> int:
        if isinstance(knot, bool) or not isinstance(knot, (int, np.integer)):
            raise TypeError("A knot index must be an integer; use -1 for the terminal knot.")
        knot = int(knot)
        if knot < 0:
            knot += self.horizon + 1
        if not 0 <= knot <= self.horizon:
            raise IndexError(f"Knot must be in [0, {self.horizon}], or a valid negative index.")
        return knot

    def observation_slice(self, part: str, knot: int, *, velocity: bool = False) -> slice:
        try:
            index = self.part_names.index(part)
        except ValueError as exc:
            raise KeyError(f"Unknown part {part!r}; choose from {self.part_names}.") from exc
        start = (self.knot(knot) * len(self.part_names) + index) * 6 + (3 if velocity else 0)
        return slice(start, start + 3)

    def trajectory(self, observations) -> "Trajectory":
        return Trajectory(self, vector(observations, self.n_observations, "observations"))


@dataclass(frozen=True)
class Trajectory:
    layout: Layout
    values: np.ndarray

    @property
    def array(self):
        return self.values.reshape(self.layout.observation_shape)

    @property
    def positions(self):
        return self.array[:, :, :3]

    @property
    def velocities(self):
        return self.array[:, :, 3:]

    def position(self, part: str, knot: int = -1):
        return self.values[self.layout.observation_slice(part, knot)].copy()

    def velocity(self, part: str, knot: int = -1):
        return self.values[self.layout.observation_slice(part, knot, velocity=True)].copy()


@runtime_checkable
class RolloutBackend(Protocol):
    """Each instance owns one mutable rollout tape; do not share across threads.

    rollout(U) resets the initial condition and returns observations in C order.
    vjp(seed) returns d(seed dot observations)/dU at the LAST rollout, in C order.
    vjp must be repeatable and must clear old adjoints without changing primals.
    """
    layout: Layout

    def rollout(self, controls: np.ndarray) -> np.ndarray: ...
    def vjp(self, observation_seed: np.ndarray) -> np.ndarray: ...

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


Objective = Callable[..., jax.Array]


@dataclass(frozen=True)
class TestProblem:
    """Metadata for a standard unconstrained optimization test problem."""

    name: str
    objective: Objective
    start: tuple[float, ...]
    minimizer: tuple[float, ...]
    minimum: float = 0.0
    gtol32: float = 5.0e-4
    gtol64: float = 1.0e-5

    def x0(self, dtype: Any | None = None) -> jax.Array:
        return jnp.asarray(self.start, dtype=dtype)

    def x_star(self, dtype: Any | None = None) -> jax.Array:
        return jnp.asarray(self.minimizer, dtype=dtype)


def rosenbrock(x: jax.Array) -> jax.Array:
    """The n-dimensional Rosenbrock chain function."""

    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def rosenbrock_hessian_diagonal(x: jax.Array) -> jax.Array:
    """Diagonal of the Hessian of :func:`rosenbrock`."""

    n = x.shape[0]
    diagonal = jnp.zeros_like(x)
    diagonal = diagonal.at[:-1].add(1200.0 * x[:-1] ** 2 - 400.0 * x[1:] + 2.0)
    diagonal = diagonal.at[1:].add(200.0)
    return diagonal


def extended_rosenbrock_start(n: int) -> tuple[float, ...]:
    if n < 2:
        raise ValueError("Rosenbrock dimension must be at least two")
    values = np.ones(n, dtype=float)
    values[::2] = -1.2
    return tuple(float(v) for v in values)


def powell_singular(x: jax.Array) -> jax.Array:
    """Extended Powell singular function; dimension must be a multiple of four."""

    blocks = x.reshape((-1, 4))
    x1, x2, x3, x4 = blocks.T
    return jnp.sum(
        (x1 + 10.0 * x2) ** 2
        + 5.0 * (x3 - x4) ** 2
        + (x2 - 2.0 * x3) ** 4
        + 10.0 * (x1 - x4) ** 4
    )


def beale(x: jax.Array) -> jax.Array:
    x1, x2 = x
    return (
        (1.5 - x1 + x1 * x2) ** 2
        + (2.25 - x1 + x1 * x2**2) ** 2
        + (2.625 - x1 + x1 * x2**3) ** 2
    )


def wood(x: jax.Array) -> jax.Array:
    x1, x2, x3, x4 = x
    return (
        100.0 * (x2 - x1**2) ** 2
        + (1.0 - x1) ** 2
        + 90.0 * (x4 - x3**2) ** 2
        + (1.0 - x3) ** 2
        + 10.1 * ((x2 - 1.0) ** 2 + (x4 - 1.0) ** 2)
        + 19.8 * (x2 - 1.0) * (x4 - 1.0)
    )


def brown_badly_scaled(x: jax.Array) -> jax.Array:
    x1, x2 = x
    return (x1 - 1.0e6) ** 2 + (x2 - 2.0e-6) ** 2 + (x1 * x2 - 2.0) ** 2


def freudenstein_roth(x: jax.Array) -> jax.Array:
    x1, x2 = x
    r1 = -13.0 + x1 + ((5.0 - x2) * x2 - 2.0) * x2
    r2 = -29.0 + x1 + ((x2 + 1.0) * x2 - 14.0) * x2
    return r1**2 + r2**2


def diagonal_quadratic(x: jax.Array, diagonal: jax.Array) -> jax.Array:
    """``0.5 * x.T @ diag(diagonal) @ x``."""

    return 0.5 * jnp.vdot(x, diagonal * x).real


def ill_conditioned_quadratic_diagonal(
    n: int, condition_number: float = 1.0e6
) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    if condition_number < 1.0:
        raise ValueError("condition_number must be at least one")
    return np.geomspace(1.0, condition_number, n, dtype=float)


ROSENBROCK_2 = TestProblem(
    name="Rosenbrock (2D)",
    objective=rosenbrock,
    start=(-1.2, 1.0),
    minimizer=(1.0, 1.0),
)

ROSENBROCK_10 = TestProblem(
    name="Extended Rosenbrock (10D)",
    objective=rosenbrock,
    start=extended_rosenbrock_start(10),
    minimizer=(1.0,) * 10,
)

POWELL_SINGULAR_4 = TestProblem(
    name="Powell singular (4D)",
    objective=powell_singular,
    start=(3.0, -1.0, 0.0, 1.0),
    minimizer=(0.0, 0.0, 0.0, 0.0),
)

BEALE_2 = TestProblem(
    name="Beale (2D)",
    objective=beale,
    start=(1.0, 1.0),
    minimizer=(3.0, 0.5),
)

WOOD_4 = TestProblem(
    name="Wood (4D)",
    objective=wood,
    start=(-3.0, -1.0, -3.0, -1.0),
    minimizer=(1.0, 1.0, 1.0, 1.0),
)

BROWN_BADLY_SCALED_2 = TestProblem(
    name="Brown badly scaled (2D)",
    objective=brown_badly_scaled,
    start=(1.0, 1.0),
    minimizer=(1.0e6, 2.0e-6),
    gtol32=2.0e-2,
)

FREUDENSTEIN_ROTH_2 = TestProblem(
    name="Freudenstein--Roth (2D)",
    objective=freudenstein_roth,
    start=(0.5, -2.0),
    # The standard start converges to this well-known local minimizer.
    minimizer=(11.41277475, -0.89680551),
    minimum=48.984253679240,
    gtol32=5.0e-2,
)

STANDARD_PROBLEMS: tuple[TestProblem, ...] = (
    ROSENBROCK_2,
    ROSENBROCK_10,
    POWELL_SINGULAR_4,
    BEALE_2,
    WOOD_4,
    FREUDENSTEIN_ROTH_2,
    BROWN_BADLY_SCALED_2,
)

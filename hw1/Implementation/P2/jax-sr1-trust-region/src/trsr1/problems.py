"""Standard unconstrained optimization test problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

Array = jax.Array
Objective = Callable[[Array], Array]


@dataclass(frozen=True, slots=True)
class TestProblem:
    name: str
    fun: Objective
    start: tuple[float, ...]
    known_minimum: float
    gradient_tolerance: float = 1.0e-7
    max_iterations: int = 500

    def initial_point(self, dtype: jnp.dtype | None = None) -> Array:
        return jnp.asarray(self.start, dtype=dtype)


def rosenbrock(x: Array) -> Array:
    return jnp.sum(
        100.0 * (x[1:] - x[:-1] ** 2) ** 2
        + (1.0 - x[:-1]) ** 2
    )


def beale(x: Array) -> Array:
    x1, x2 = x
    return (
        (1.5 - x1 + x1 * x2) ** 2
        + (2.25 - x1 + x1 * x2**2) ** 2
        + (2.625 - x1 + x1 * x2**3) ** 2
    )


def himmelblau(x: Array) -> Array:
    x1, x2 = x
    return (x1**2 + x2 - 11.0) ** 2 + (x1 + x2**2 - 7.0) ** 2


def powell_singular(x: Array) -> Array:
    x1, x2, x3, x4 = x
    return (
        (x1 + 10.0 * x2) ** 2
        + 5.0 * (x3 - x4) ** 2
        + (x2 - 2.0 * x3) ** 4
        + 10.0 * (x1 - x4) ** 4
    )


def wood(x: Array) -> Array:
    x1, x2, x3, x4 = x
    return (
        100.0 * (x1**2 - x2) ** 2
        + (x1 - 1.0) ** 2
        + (x3 - 1.0) ** 2
        + 90.0 * (x3**2 - x4) ** 2
        + 10.1 * ((x2 - 1.0) ** 2 + (x4 - 1.0) ** 2)
        + 19.8 * (x2 - 1.0) * (x4 - 1.0)
    )


def brown_badly_scaled(x: Array) -> Array:
    x1, x2 = x
    return (x1 - 1.0e6) ** 2 + (x2 - 2.0e-6) ** 2 + (x1 * x2 - 2.0) ** 2


def freudenstein_roth(x: Array) -> Array:
    x1, x2 = x
    first = -13.0 + x1 + ((5.0 - x2) * x2 - 2.0) * x2
    second = -29.0 + x1 + ((x2 + 1.0) * x2 - 14.0) * x2
    return first**2 + second**2


def extended_rosenbrock(x: Array) -> Array:
    """Even-dimensional extended Rosenbrock problem."""

    odd = x[0::2]
    even = x[1::2]
    return jnp.sum(100.0 * (even - odd**2) ** 2 + (1.0 - odd) ** 2)


def standard_problems() -> tuple[TestProblem, ...]:
    """Return a deterministic benchmark suite with conventional starts."""

    return (
        TestProblem(
            "rosenbrock",
            rosenbrock,
            (-1.2, 1.0),
            0.0,
            gradient_tolerance=1.0e-7,
            max_iterations=300,
        ),
        TestProblem(
            "beale",
            beale,
            (1.0, 1.0),
            0.0,
            gradient_tolerance=1.0e-7,
            max_iterations=200,
        ),
        TestProblem(
            "himmelblau",
            himmelblau,
            (-3.0, -3.0),
            0.0,
            gradient_tolerance=1.0e-7,
            max_iterations=200,
        ),
        TestProblem(
            "powell-singular",
            powell_singular,
            (3.0, -1.0, 0.0, 1.0),
            0.0,
            gradient_tolerance=1.0e-7,
            max_iterations=400,
        ),
        TestProblem(
            "wood",
            wood,
            (-3.0, -1.0, -3.0, -1.0),
            0.0,
            gradient_tolerance=1.0e-7,
            max_iterations=7000,
        ),
        TestProblem(
            "brown-badly-scaled",
            brown_badly_scaled,
            (1.0, 1.0),
            0.0,
            gradient_tolerance=1.0e-6,
            max_iterations=500,
        ),
        TestProblem(
            "freudenstein-roth",
            freudenstein_roth,
            (0.5, -2.0),
            48.9842536792400,
            gradient_tolerance=1.0e-7,
            max_iterations=200,
        ),
    )


def problem_by_name(name: str) -> TestProblem:
    normalized = name.strip().lower()
    for problem in standard_problems():
        if problem.name == normalized:
            return problem
    available = ", ".join(problem.name for problem in standard_problems())
    raise KeyError(f"unknown problem {name!r}; available problems: {available}")

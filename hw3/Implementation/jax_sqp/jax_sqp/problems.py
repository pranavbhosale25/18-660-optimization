"""Standard HS/CUTEst problems and explicitly labelled custom scaling examples.

Sources: ralna/SIF HS6.SIF, HS7.SIF, HS26.SIF, HS71.SIF, HS100.SIF,
BROYDN3D.SIF; Nocedal & Wright, Numerical Optimization (2nd ed.), Ex.18.3.
The control, conflicting-linearization, and constrained-Rosenbrock examples
are custom problems, not claims of additional canonical CUTEst benchmarks.
"""

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .solver import SQP, Options


@dataclass(frozen=True)
class Problem:
    name: str
    fun: Callable
    x0: jax.Array
    eq: Callable | None = None
    ineq: Callable | None = None
    lower: Any = None
    upper: Any = None
    params: Any = None
    reference_objective: float | None = None
    source: str = ""
    notes: str = ""

    def solver(self, options: Options | None = None) -> SQP:
        return SQP(
            self.fun,
            self.eq,
            self.ineq,
            lower=self.lower,
            upper=self.upper,
            options=options,
        )

    @property
    def dimensions(self):
        n = self.x0.size
        ne = 0 if self.eq is None else self.eq(self.x0, self.params).size
        ni = 0 if self.ineq is None else self.ineq(self.x0, self.params).size
        for b in (self.lower, self.upper):
            if b is not None:
                ni += int(np.sum(np.isfinite(np.broadcast_to(b, (n,)))))
        return n, ne, ni


SIF = "https://raw.githubusercontent.com/ralna/SIF/master/"


def hs06(dtype=jnp.float64):
    return Problem(
        "HS06",
        lambda x, _: (1 - x[0]) ** 2,
        jnp.array([-1.2, 1.0], dtype),
        eq=lambda x, _: jnp.array([10 * (x[1] - x[0] ** 2)]),
        reference_objective=0.0,
        source=SIF + "HS6.SIF",
    )


def hs07(dtype=jnp.float64):
    return Problem(
        "HS07",
        lambda x, _: jnp.log1p(x[0] ** 2) - x[1],
        jnp.array([2.0, 2.0], dtype),
        eq=lambda x, _: jnp.array([(1 + x[0] ** 2) ** 2 + x[1] ** 2 - 4]),
        reference_objective=-np.sqrt(3.0),
        source=SIF + "HS7.SIF",
    )


def hs26(dtype=jnp.float64):
    return Problem(
        "HS26",
        lambda x, _: (x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 4,
        jnp.array([-2.6, 2.0, 2.0], dtype),
        eq=lambda x, _: jnp.array([(1 + x[1] ** 2) * x[0] + x[2] ** 4 - 3]),
        reference_objective=0.0,
        source=SIF + "HS26.SIF",
        notes="Uses the official SIF start (-2.6,2,2). The minimum is very flat.",
    )


def hs71(dtype=jnp.float64):
    return Problem(
        "HS71",
        lambda x, _: x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2],
        jnp.array([1.0, 5.0, 5.0, 1.0], dtype),
        eq=lambda x, _: jnp.array([jnp.sum(x * x) - 40]),
        ineq=lambda x, _: jnp.array([25 - jnp.prod(x)]),
        lower=1.0,
        upper=5.0,
        reference_objective=17.014017289156,
        source=SIF + "HS71.SIF",
    )


def hs100(dtype=jnp.float64):
    def f(x, _):
        a, b, c, d, e, f, g = x
        return (
            (a - 10) ** 2
            + 5 * (b - 12) ** 2
            + c**4
            + 3 * (d - 11) ** 2
            + 10 * e**6
            + 7 * f * f
            + g**4
            - 4 * f * g
            - 10 * f
            - 8 * g
        )

    def ci(x, _):
        a, b, c, d, e, f, g = x
        return jnp.array(
            [
                2 * a * a + 3 * b**4 + c + 4 * d * d + 5 * e - 127,
                7 * a + 3 * b + 10 * c * c + d - e - 282,
                23 * a + b * b + 6 * f * f - 8 * g - 196,
                4 * a * a + b * b - 3 * a * b + 2 * c * c + 5 * f - 11 * g,
            ]
        )

    return Problem(
        "HS100",
        f,
        jnp.array([1.0, 2.0, 0.0, 4.0, 0.0, 1.0, 1.0], dtype),
        ineq=ci,
        reference_objective=680.6300573,
        source=SIF + "HS100.SIF",
    )


def nw_ex18_3(dtype=jnp.float64):
    def f(x, _):
        return jnp.exp(jnp.prod(x)) - 0.5 * (x[0] ** 3 + x[1] ** 3 + 1) ** 2

    def c(x, _):
        return jnp.array(
            [
                jnp.sum(x * x) - 10,
                x[1] * x[2] - 5 * x[3] * x[4],
                x[0] ** 3 + x[1] ** 3 + 1,
            ]
        )

    return Problem(
        "NW-Ex18.3",
        f,
        jnp.array([-1.71, 1.59, 1.82, -0.763, -0.763], dtype),
        eq=c,
        source="Nocedal & Wright, Numerical Optimization, 2nd ed., Exercise 18.3, pp.562.",
        notes="The textbook solution is rounded; do not treat it as an exact feasible reference.",
    )


def conflicting_linearizations(dtype=jnp.float64):
    # At x=0: c1+J1 p=-1, c2+J2 p=-1+p. No hard QP can satisfy both.
    return Problem(
        "Conflicting-linearizations",
        lambda x, _: (x[0] - 2) ** 2,
        jnp.array([0.0], dtype),
        eq=lambda x, _: jnp.array([x[0] ** 2 - 1, x[0] - 1]),
        reference_objective=1.0,
        source="Custom regression problem",
        notes="Initially inconsistent linearizations.",
    )


def circle(dtype=jnp.float64):
    return Problem(
        "Circle",
        lambda x, _: x[0],
        jnp.array([0.5, 0.8], dtype),
        eq=lambda x, _: jnp.array([jnp.dot(x, x) - 1]),
        reference_objective=-1.0,
        source="Custom equality-constrained example",
    )


def broyden_tridiagonal(n=1000, dtype=jnp.float64):
    if n < 2:
        raise ValueError("n must be at least 2.")

    def c(x, _):
        padded = jnp.pad(x, (1, 1))
        return (3 - 2 * x) * x - padded[:-2] - 2 * padded[2:] + 1

    return Problem(
        f"BROYDN3D-{n}",
        lambda x, _: jnp.array(0.0, x.dtype),
        -jnp.ones(n, dtype),
        eq=c,
        reference_objective=0.0,
        source=SIF + "BROYDN3D.SIF",
        notes="Canonical nonlinear-equation feasibility problem: n equalities, zero objective.",
    )


def optimal_control(horizon=1000, dtype=jnp.float64):
    """Custom stable nonlinear direct transcription; 2T variables, 3T constraints.

    y[t+1] = .8*y[t] + u[t] - .05*y[t]**3, y[0]=0,
    |u[t]| <= .18. Cost = .5 sum((y-ref)**2 + .1*u**2)
    + .025 sum(diff(u)**2). Coupled, nonlinear equalities and active bounds.
    """
    if horizon < 2:
        raise ValueError("horizon must be at least 2.")
    ref = jnp.asarray(0.8 + 0.25 * np.sin(np.linspace(0, 6 * np.pi, horizon)), dtype)

    def f(x, reference):
        y, u = x[:horizon], x[horizon:]
        return 0.5 * jnp.sum((y - reference) ** 2 + 0.1 * u * u) + 0.025 * jnp.sum(
            jnp.diff(u) ** 2
        )

    def ce(x, _):
        y, u = x[:horizon], x[horizon:]
        yp = jnp.concatenate((jnp.zeros(1, x.dtype), y[:-1]))
        return y - 0.8 * yp - u + 0.05 * yp**3

    lo = np.concatenate((np.full(horizon, -np.inf), np.full(horizon, -0.18)))
    up = np.concatenate((np.full(horizon, np.inf), np.full(horizon, 0.18)))
    return Problem(
        f"Nonlinear-control-{2*horizon}",
        f,
        jnp.zeros(2 * horizon, dtype),
        eq=ce,
        lower=lo,
        upper=up,
        params=ref,
        source="Custom nonlinear optimal-control transcription",
        notes="A real optimization objective; no claim of a known global optimum.",
    )


def constrained_rosenbrock(n=1000, dtype=jnp.float64):
    """Return a custom constrained chain Rosenbrock problem."""
    if n < 2:
        raise ValueError("n must be at least 2.")

    def f(x, _):
        return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    def ce(x, _):
        return jnp.array([jnp.sum(x - 1) / jnp.sqrt(n)])

    def ci(x, _):
        return jnp.array([jnp.sum((x - 1) ** 2) / n - 0.25])

    return Problem(
        f"Constrained-Rosenbrock-{n}",
        f,
        jnp.full(n, 0.8, dtype),
        eq=ce,
        ineq=ci,
        reference_objective=0.0,
        source="Custom constrained variant of the chain Rosenbrock objective",
    )


def small_suite(dtype=jnp.float64):
    return [
        hs06(dtype),
        hs07(dtype),
        hs26(dtype),
        hs71(dtype),
        hs100(dtype),
        nw_ex18_3(dtype),
        circle(dtype),
        conflicting_linearizations(dtype),
    ]

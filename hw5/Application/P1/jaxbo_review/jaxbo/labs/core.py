"""Public laboratory contracts and an accounting boundary around a simulator.

Numerical simulator functions are pure JAX. ExperimentSession is deliberately
host-side: real experiments, budget enforcement, and provenance are effects.
This boundary prevents accidental oracle use, NOT hostile Python introspection.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Protocol
import math
import jax
import jax.numpy as jnp
from ..spaces import Box, MixedSpace


@dataclass(frozen=True)
class TaskSpec:
    """Public, read-only description of one application problem.

    Attributes:
        name: Stable command-line problem identifier.
        title: Human-readable problem name.
        space: Physical decision domain. A ``Box`` contains continuous bounded
            coordinates; a ``MixedSpace`` additionally describes integer,
            categorical, and conditionally active coordinates.
        budget: Total resource available to one independent episode.
        budget_unit: Human-readable unit used by ``budget`` and observation
            costs.
        n_initial: Requested number of supplied initial-design evaluations. The
            policy caps this at the number affordable under the episode budget.
        target: Public quality threshold used by post-run reporting. Qualities
            are maximized.
        feature_names: Physical-coordinate names in the same order as vectors
            accepted by ``space``.
        constraint_name: Name of the measured scalar constraint, or ``None``
            when the task is unconstrained. A reported value at or below zero
            is feasible.
        reports_variance: Whether observations include a calibrated variance
            for the average of their independent replicates.
        minimum_cost: Smallest charge for one physical run. The experiment loop
            stops when less than this amount remains.
        safe_start: Optional known-feasible physical point placed first in the
            ordinary initial design, or ``None`` when no such point is supplied.

    ``dimension`` is the physical vector length. ``capacity`` is the largest
    number of minimum-cost runs that could fit in the budget; it is a storage
    bound, not a promise that every task will complete that many observations.
    """

    name: str
    title: str
    space: Box | MixedSpace
    budget: float
    budget_unit: str
    n_initial: int
    target: float
    feature_names: tuple[str, ...]
    constraint_name: str | None = None
    reports_variance: bool = False
    minimum_cost: float = 1.0
    safe_start: tuple[float, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.space,(Box,MixedSpace)):
            raise TypeError("Task space must be a Box or MixedSpace")
        if (not math.isfinite(self.budget) or not math.isfinite(self.minimum_cost)
                or self.minimum_cost<=0 or self.budget<self.minimum_cost):
            raise ValueError("Invalid budget or minimum cost")
        if len(self.feature_names)!=self.space.dimension:
            raise ValueError("One feature name is required per physical dimension")
        if not math.isfinite(self.target) or self.n_initial<1:
            raise ValueError("Target must be finite and initial-design size positive")

    @property
    def dimension(self) -> int:
        return self.space.dimension

    @property
    def capacity(self) -> int:
        return int(math.floor(self.budget / self.minimum_cost + 1e-9))


class Truth(NamedTuple):
    """Internal simulator outputs. All are scalar arrays for one input."""
    mean: jax.Array
    variance: jax.Array
    constraint: jax.Array  # g <= 0; constant -1 for unconstrained tasks
    cost: jax.Array


@dataclass(frozen=True)
class Problem:
    spec: TaskSpec
    _truth: Callable[[jax.Array], Truth] = field(repr=False)
    _initial: Callable[[jax.Array, int], jax.Array] = field(repr=False)
    metadata: dict = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class Action:
    """One requested physical point and its positive replicate count."""

    x: jax.Array
    replicates: int = 1


@dataclass(frozen=True)
class Observation:
    """The charged feedback returned for one completed action.

    Attributes:
        x: One physical decision vector in the task's declared coordinate order.
        y: Mean scalar quality across the action's independent replicates.
        noise_variance: Calibrated variance of that mean, not of one physical
            run, or ``None`` when the task does not report a variance.
        constraint: Measured scalar constraint, with values at or below zero
            feasible, or ``None`` for an unconstrained task.
        cost: Total resource charged for all replicates in this action.
        replicates: Number of independent physical runs averaged into ``y``.
        cumulative_cost: Total charged resource through this observation.
        cumulative_runs: Total physical runs through this observation.
    """

    x: jax.Array
    y: float                         # average of the independent replicates
    noise_variance: float | None      # variance OF THE AVERAGE, not of one run
    constraint: float | None
    cost: float                      # total cost of these replicates
    replicates: int
    cumulative_cost: float
    cumulative_runs: int


class BudgetExhausted(RuntimeError):
    """Terminal; no outcome is returned for an unaffordable request."""


class Policy(Protocol):
    def ask(self, remaining: float) -> Action: ...
    def tell(self, observation: Observation) -> None: ...
    def recommend(self) -> jax.Array: ...


def validate_point(spec: TaskSpec, x) -> jax.Array:
    x = jnp.asarray(x, dtype=float)
    if x.shape != (spec.dimension,) or not bool(jnp.all(jnp.isfinite(x))):
        raise ValueError(f"x must be a finite vector of shape ({spec.dimension},)")
    lo, hi = jnp.asarray(spec.space.lower), jnp.asarray(spec.space.upper)
    if not bool(jnp.all((x >= lo) & (x <= hi))):
        raise ValueError("Point outside the declared domain")
    if isinstance(spec.space, MixedSpace):
        for i, kind in enumerate(spec.space.kinds):
            if kind != "real" and float(x[i]) != round(float(x[i])):
                raise ValueError("Integer/categorical variables must be integral")
        x = spec.space.canonicalize(x[None, :])[0]
    return x


class ExperimentSession:
    """The student-facing experiment interface.

    Every accepted observation is charged, including initialization, repeats and
    infeasible recipes. An unaffordable request terminates the session and burns
    the remaining reservation; it reveals no output and cannot be retried as a
    free cost oracle. Completed-run count excludes a canceled reservation.
    Independent validation is available only after finalize(), which irrevocably
    closes the session. Only the first recommendation can be finalized.
    """
    def __init__(self, problem: Problem, *, key: jax.Array):
        self.spec = problem.spec
        self.__truth = jax.jit(problem._truth)
        self.__key = key
        self.__history: list[Observation] = []
        self.__spent = 0.0
        self.__runs = 0
        self.__closed = False
        self.__finalized = False
        self.__canceled = 0.0

    @property
    def history(self) -> tuple[Observation, ...]:
        return tuple(self.__history)

    @property
    def spent(self) -> float:
        return self.__spent

    @property
    def remaining(self) -> float:
        return max(0.0, self.spec.budget - self.__spent)

    @property
    def completed_runs(self) -> int:
        return self.__runs

    @property
    def canceled_reservation(self) -> float:
        return self.__canceled

    def evaluate(self, action: Action | jax.Array, *, replicates: int = 1) -> Observation:
        if self.__closed:
            raise BudgetExhausted("Session is closed")
        if isinstance(action, Action):
            x, r = action.x, action.replicates
        else:
            x, r = action, replicates
        if isinstance(r, bool) or not isinstance(r, int) or r < 1:
            raise ValueError("replicates must be a positive integer")
        x = validate_point(self.spec, x)
        t = self.__truth(x)
        if not bool(jnp.all(jnp.isfinite(jnp.asarray(t)))) or float(t.variance) < 0:
            raise FloatingPointError("Invalid simulator output")
        if float(t.cost) < self.spec.minimum_cost - 1e-8:
            raise ValueError("Simulator violated the public minimum-cost bound")
        price = r * float(t.cost)
        if price > self.remaining + 1e-8:
            self.__canceled = self.remaining
            self.__spent = self.spec.budget
            self.__closed = True
            raise BudgetExhausted("Request exceeds remaining budget; reservation canceled and session closed")
        # One folded key per physical run: grouping repeats does not change their
        # random draws. Simulator seeds and algorithm keys are separate streams.
        keys = jax.vmap(lambda i: jax.random.fold_in(self.__key, i))(
            jnp.arange(self.__runs, self.__runs + r, dtype=jnp.uint32))
        eps = jax.vmap(lambda k: jax.random.normal(k, (), dtype=x.dtype))(keys)
        y = float(t.mean + jnp.sqrt(t.variance) * eps.mean())
        self.__spent = min(self.spec.budget, self.__spent + price)
        self.__runs += r
        obs = Observation(x, y, float(t.variance) / r if self.spec.reports_variance else None,
                          float(t.constraint) if self.spec.constraint_name else None,
                          price, r, self.__spent, self.__runs)
        self.__history.append(obs)
        if self.remaining < self.spec.minimum_cost - 1e-8:
            self.__closed = True
        return obs

    def finalize(self, x, *, validation_key: jax.Array, validation_runs: int = 256) -> dict:
        if self.__finalized:
            raise RuntimeError("The recommendation was already finalized")
        if isinstance(validation_runs,bool) or not isinstance(validation_runs,int) or validation_runs < 2:
            raise ValueError("validation_runs must be an integer of at least two")
        x = validate_point(self.spec, x)
        self.__finalized = True
        self.__closed = True
        t = self.__truth(x)
        feasible = bool(t.constraint <= 0)
        samples = t.mean + jnp.sqrt(t.variance) * jax.random.normal(
            validation_key, (validation_runs,), dtype=x.dtype)
        # Oracle score is deliberately post-run only and never sent to a policy.
        return dict(recommendation=x.tolist(), true_mean=float(t.mean), feasible=feasible,
                    passed=feasible and float(t.mean) >= self.spec.target,
                    target=self.spec.target, constraint=float(t.constraint),
                    validation_mean=float(samples.mean()),
                    validation_standard_error=float(samples.std(ddof=1) / jnp.sqrt(validation_runs)),
                    validation_runs=validation_runs, spent=self.spent,
                    completed_runs=self.completed_runs, observations=len(self.__history),
                    canceled_reservation=self.canceled_reservation)

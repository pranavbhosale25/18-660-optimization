"""Veinott boundary supporting cuts, with LP epigraph objective cuts.

The chance constraint is separated at the intersection of a ray from a FIXED
strictly feasible anchor to an infeasible LP candidate and the chance boundary.
This is not linearization of the probability at arbitrary infeasible points.
A convex quadratic objective is handled using its epigraph and globally valid
objective tangents. SciPy/HiGHS solves LPs; all nonlinear derivatives are JAX AD.
"""

from dataclasses import dataclass, field
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import linprog, minimize


@dataclass
class SolverOptions:
    max_iterations: int = 800
    absolute_gap: float = 2e-6
    relative_gap: float = 2e-5
    interior_margin: float = 1e-6
    bisection_steps: int = 42
    probability_buffer: float = 0.0
    verbose: bool = False


@dataclass
class VeinottResult:
    x: np.ndarray
    objective: float
    probability: float
    working_target: float
    lower_bound: float
    absolute_gap: float
    iterations: int
    status: str
    exact_oracle: bool
    anchor: np.ndarray
    history: list[dict] = field(default_factory=list)
    chance_cuts: list[dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def certificate_label(self):
        return (
            "numerical LP gap for exact-probability convex model"
            if self.exact_oracle
            else "diagnostic LP gap only: approximate probability cuts are not certified"
        )


def _interior_point(problem, oracle, log_target, options, supplied=None):
    x = problem.center_guess() if supplied is None else np.asarray(supplied, float)
    if x.shape != (problem.n_decisions,):
        raise ValueError("Incorrect interior point dimension")
    if (
        problem.linear_feasible(x)
        and oracle.log_probability(x) > log_target + options.interior_margin
    ):
        return x

    # Phase I ONLY. SLSQP is not the compensation optimizer.
    def fun(x):
        v, g = oracle.logp_and_grad(x)
        return -v, -g

    cons = []
    if len(problem.A_ub):
        cons = [
            {
                "type": "ineq",
                "fun": lambda x: problem.b_ub - problem.A_ub @ x,
                "jac": lambda x: -problem.A_ub,
            }
        ]
    attempt = minimize(
        fun,
        np.clip(x, *problem.bounds.T),
        jac=True,
        method="SLSQP",
        bounds=problem.bounds,
        constraints=cons,
        options={"maxiter": 300, "ftol": 1e-12},
    )
    x = attempt.x
    if (
        not problem.linear_feasible(x)
        or oracle.log_probability(x) <= log_target + options.interior_margin
    ):
        raise ValueError(
            "No strictly chance-feasible anchor was found. This is not an "
            "infeasibility certificate. Check tolerances, variance, bounds, "
            "or supply an interior point. Phase I: " + attempt.message
        )
    return x


def solve_veinott(problem, oracle, options=None, interior_point=None):
    """Solve the fixed-covariance convex model, returning a feasible incumbent.

    'converged' means the LP-gap stopping rule was met. For a Genz/QMC oracle,
    this is NOT a rigorous global-optimality or chance-feasibility certificate.
    A probability buffer helps numerically but is not a proved integration bound.
    """
    start = time.perf_counter()
    options = options or SolverOptions()
    target = problem.alpha + options.probability_buffer
    if not 0 < target < 1:
        raise ValueError("alpha + probability_buffer must be in (0,1)")
    log_target = np.log(target)
    anchor = _interior_point(problem, oracle, log_target, options, interior_point)
    d = problem.n_decisions
    objective_vg = jax.jit(jax.value_and_grad(problem.objective))

    def objective(x):
        v, g = objective_vg(jnp.asarray(x))
        return float(v), np.asarray(g)

    def chance(x):
        return log_target - oracle.log_probability_jax(x)

    # Compile the ENTIRE segment bisection, not one Python call per bisection step.
    anchor_j = jnp.asarray(anchor)

    @jax.jit
    def boundary(outside):
        delta = outside - anchor_j

        def body(_, interval):
            lo, hi = interval
            mid = 0.5 * (lo + hi)
            inside = chance(anchor_j + mid * delta) <= 0.0
            return jnp.where(inside, mid, lo), jnp.where(inside, hi, mid)

        lo, hi = jax.lax.fori_loop(0, options.bisection_steps, body, (0.0, 1.0))
        return anchor_j + lo * delta

    cut_A, cut_b = [], []
    if len(problem.A_ub):
        cut_A.extend(np.column_stack([problem.A_ub, np.zeros(len(problem.A_ub))]))
        cut_b.extend(problem.b_ub)

    def add_objective_tangent(x):
        f, g = objective(x)
        cut_A.append(np.r_[g, -1.0])
        cut_b.append(float(g @ x - f))
        return f

    best_x = anchor.copy()
    upper = add_objective_tangent(anchor)
    add_objective_tangent(np.clip(np.zeros(d), *problem.bounds.T))
    # Since Q >= 0, min(c'x) over the box is a valid finite epigraph lower bound.
    theta_lower = float(
        np.sum(
            np.minimum(
                problem.c * problem.bounds[:, 0], problem.c * problem.bounds[:, 1]
            )
        )
    )
    lp_bounds = [tuple(v) for v in problem.bounds] + [(theta_lower, None)]
    lp_cost = np.r_[np.zeros(d), 1.0]
    history, chance_cuts = [], []
    lower = theta_lower
    status = "iteration_limit"
    for k in range(options.max_iterations):
        lp = linprog(
            lp_cost,
            A_ub=np.asarray(cut_A),
            b_ub=np.asarray(cut_b),
            bounds=lp_bounds,
            method="highs",
            options={
                "dual_feasibility_tolerance": 1e-9,
                "primal_feasibility_tolerance": 1e-9,
            },
        )
        if not lp.success:
            raise RuntimeError(f"LP iteration {k} failed: {lp.message}")
        candidate = lp.x[:d]
        lower = max(lower, float(lp.fun))
        logp = oracle.log_probability(candidate)
        f_candidate = add_objective_tangent(candidate)
        boundary_point = None
        if logp >= log_target - 1e-12:
            if f_candidate < upper:
                best_x, upper = candidate.copy(), f_candidate
        else:
            y = np.asarray(boundary(jnp.asarray(candidate)))
            logpy, gradlogp = oracle.logp_and_grad(y)
            grad = -gradlogp
            norm = np.linalg.norm(grad)
            if not np.isfinite(norm) or norm < 1e-12:
                raise RuntimeError(
                    "Degenerate chance-boundary gradient; cannot construct a cut"
                )
            a = grad / norm
            # A tiny outward tolerance absorbs floating-point boundary error.
            rhs = float(a @ y + 2e-11)
            cut_A.append(np.r_[a, 0.0])
            cut_b.append(rhs)
            chance_cuts.append(
                {
                    "iteration": k,
                    "normal": a.tolist(),
                    "rhs": rhs,
                    "boundary": y.tolist(),
                    "logp": logpy,
                }
            )
            fy, _ = objective(y)
            if logpy >= log_target - 1e-10 and fy < upper:
                best_x, upper = y.copy(), fy
            boundary_point = y.tolist()
        raw_gap = upper - lower
        if raw_gap < -1e-6:
            raise RuntimeError(
                "Outer bound exceeds feasible cost. Probability/cut numerical "
                "error may invalidate the relaxation; increase QMC accuracy."
            )
        gap = max(0.0, raw_gap)
        history.append(
            {
                "iteration": k,
                "lower_bound": lower,
                "upper_bound": upper,
                "absolute_gap": gap,
                "candidate_probability": float(np.exp(logp)),
                "candidate_objective": f_candidate,
                "candidate": candidate.tolist(),
                "boundary": boundary_point,
                "n_cuts": len(cut_A),
            }
        )
        if options.verbose and (k % 25 == 0 or gap < options.absolute_gap):
            print(
                f"  {k:4d}  lower={lower:.7f} upper={upper:.7f} gap={gap:.3g}",
                flush=True,
            )
        if gap <= options.absolute_gap + options.relative_gap * max(1.0, abs(upper)):
            status = "converged"
            break
    p_final = oracle.probability(best_x)
    if not problem.linear_feasible(best_x) or p_final < target - 1e-8:
        raise RuntimeError("Returned incumbent failed the numerical feasibility check")
    return VeinottResult(
        best_x,
        upper,
        p_final,
        target,
        lower,
        max(0.0, upper - lower),
        len(history),
        status,
        oracle.exact,
        anchor,
        history,
        chance_cuts,
        time.perf_counter() - start,
    )

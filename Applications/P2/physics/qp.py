"""Shared QP boundary: min .5*x.T Q x + c.T x, lower <= A x <= upper.

JAXopt BoxOSQP is the production implementation. SciPy (HiGHS for LPs,
SLSQP for QPs) is an independent, host-only validation backend. Both return
signed row multipliers y in Qx+c+A.T@y=0. There is no second optimization
for multiplier recovery: each backend supplies its own multipliers.
"""
from dataclasses import dataclass
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np


class QP(NamedTuple):
    Q: jax.Array
    c: jax.Array
    A: jax.Array
    lower: jax.Array
    upper: jax.Array


class QPResult(NamedTuple):
    x: jax.Array
    status: jax.Array             # 1=success; anything else is not success
    iterations: jax.Array
    error: jax.Array              # backend metric, NOT comparable across backends
    feasibility: jax.Array
    dual: jax.Array               # signed y: negative at lower, positive at upper
    stationarity: jax.Array
    complementarity: jax.Array


def _max(a):
    return jnp.max(a, initial=0.)


def violation(problem: QP, x):
    ax = problem.A @ x
    return jnp.maximum(0., jnp.maximum(_max(problem.lower-ax),
                                     _max(ax-problem.upper)))


def kkt_residuals(p: QP, x, y):
    """Original-unit feasibility, stationarity, and dual/complementarity errors.

    Infinite sides contribute neither products nor admissible multipliers.
    Equality multipliers are free; the two-sided split handles them as well.
    """
    ax = p.A @ x
    finite_lo, finite_hi = jnp.isfinite(p.lower), jnp.isfinite(p.upper)
    alpha, eta = jnp.maximum(-y, 0.), jnp.maximum(y, 0.)
    lower_slack = ax - jnp.where(finite_lo, p.lower, ax)
    upper_slack = jnp.where(finite_hi, p.upper, ax) - ax
    comp = jnp.maximum(_max(jnp.abs(alpha * lower_slack)),
                       _max(jnp.abs(eta * upper_slack)))
    comp = jnp.maximum(comp, _max(jnp.where(finite_lo, 0., alpha)))
    comp = jnp.maximum(comp, _max(jnp.where(finite_hi, 0., eta)))
    stationarity = _max(jnp.abs(p.Q @ x + p.c + p.A.T @ y))
    return violation(p, x), stationarity, comp


@dataclass(frozen=True)
class SolverConfig:
    backend: str = "jaxopt"
    tol: float = 1e-8
    maxiter: int = 20000
    check_infeasibility: bool = True


class Solver:
    """Construct outside jit and pass this object to the supplied drivers."""
    def __init__(self, config: SolverConfig = SolverConfig()):
        if config.backend not in ("jaxopt", "scipy"):
            raise ValueError("backend must be 'jaxopt' or 'scipy'")
        if config.tol <= 0 or config.maxiter < 1:
            raise ValueError("tol and maxiter must be positive")
        self.config = config
        self.compilable = config.backend == "jaxopt"
        if self.compilable:
            try:
                from jaxopt import BoxOSQP
            except ImportError as exc:
                raise ImportError("Install jaxopt==0.8.5; backend='scipy' is "
                                  "available for independent CPU validation.") from exc
            self._solver = BoxOSQP(
                tol=config.tol, maxiter=config.maxiter, eq_qp_solve="lu",
                jit=True, implicit_diff=True,
                check_primal_dual_infeasability=config.check_infeasibility,
                primal_infeasible_tol=min(1e-7, config.tol),
                dual_infeasible_tol=min(1e-7, config.tol),
                termination_check_frequency=5)

    def solve(self, p: QP) -> QPResult:
        if self.compilable:
            out = self._solver.run(params_obj=(p.Q, p.c), params_eq=p.A,
                                   params_ineq=(p.lower, p.upper))
            x, y = out.params.primal[0], out.params.dual_eq
            primal, stationarity, comp = kkt_residuals(p, x, y)
            return QPResult(x, out.state.status, out.state.iter_num,
                            out.state.error, primal, y, stationarity, comp)
        return self._scipy(p)

    def _scipy(self, p: QP) -> QPResult:
        """One CPU LP/QP solve, with explicit conversion of signed multipliers.

        Redundant equalities are removed by QR, not by another optimizer.
        SciPy >=1.16 is needed for SLSQP's returned KKT multipliers.
        """
        from scipy.linalg import qr
        from scipy.optimize import linprog, minimize
        Q, c, A, lo, hi = (np.asarray(v, dtype=float) for v in p)
        n, r = len(c), len(A)
        dtype = p.c.dtype

        def finish(x, y, success, nit=0):
            xx, yy = jnp.asarray(x, dtype=dtype), jnp.asarray(y, dtype=dtype)
            primal, stationarity, comp = kkt_residuals(p, xx, yy)
            return QPResult(xx, jnp.asarray(1 if success else -1),
                            jnp.asarray(nit), jnp.maximum(primal, stationarity),
                            primal, yy, stationarity, comp)

        def failure():
            return finish(np.full(n, np.nan), np.zeros(r), False)

        if not (np.isfinite(Q).all() and np.isfinite(c).all() and np.isfinite(A).all()):
            raise ValueError("Q, c, A must be finite")
        if np.isnan(lo).any() or np.isnan(hi).any() or (lo > hi).any():
            return failure()
        zero = np.all(A == 0, axis=1)
        if ((lo[zero] > 0) | (hi[zero] < 0)).any():
            return failure()
        eq = np.isfinite(lo) & np.isfinite(hi) & (lo == hi) & ~zero
        eqidx = np.flatnonzero(eq)
        # Select an independent equality basis, and test consistency first.
        if len(eqidx):
            Efull, bfull = A[eqidx], lo[eqidx]
            feasible_x = np.linalg.lstsq(Efull, bfull, rcond=None)[0]
            if np.max(np.abs(Efull @ feasible_x-bfull)) > 1e-9:
                return failure()
            _, rr, piv = qr(Efull.T, mode="economic", pivoting=True)
            diag = np.abs(np.diag(rr))
            rank = int(np.sum(diag > 1e-11 * max(1., diag.max(initial=0.))))
            eqidx = eqidx[piv[:rank]]
        E, beq = A[eqidx], lo[eqidx]
        upperidx = np.flatnonzero(np.isfinite(hi) & ~eq & ~zero)
        loweridx = np.flatnonzero(np.isfinite(lo) & ~eq & ~zero)
        C = np.concatenate((A[upperidx], -A[loweridx]), axis=0)
        b = np.concatenate((hi[upperidx], -lo[loweridx]))
        y = np.zeros(r)

        if np.all(Q == 0):
            out = linprog(c, A_ub=C if len(C) else None,
                          b_ub=b if len(C) else None,
                          A_eq=E if len(E) else None,
                          b_eq=beq if len(E) else None,
                          bounds=[(None, None)]*n, method="highs",
                          options={"maxiter": self.config.maxiter})
            if out.x is None:
                return failure()
            if out.success:
                y[eqidx] = -out.eqlin.marginals
                nu = -out.ineqlin.marginals
                np.add.at(y, upperidx, nu[:len(upperidx)])
                np.add.at(y, loweridx, -nu[len(upperidx):])
        else:
            x0 = np.linalg.lstsq(E, beq, rcond=None)[0] if len(E) else np.zeros(n)
            if len(E) == n:
                # Equalities uniquely determine x; no remaining optimization.
                y[eqidx] = np.linalg.solve(E.T, -(Q @ x0+c))
                feasible = np.max(C @ x0-b, initial=0.) <= 1e-9
                return finish(x0, y, feasible)
            constraints = []
            if len(E):
                constraints.append({"type": "eq", "fun": lambda x: E @ x-beq,
                                    "jac": lambda x: E})
            if len(C):
                constraints.append({"type": "ineq", "fun": lambda x: b-C @ x,
                                    "jac": lambda x: -C})
            out = minimize(lambda x: .5*x @ Q @ x+c @ x, x0,
                           jac=lambda x: Q @ x+c, method="SLSQP",
                           constraints=constraints,
                           options={"ftol": min(self.config.tol, 1e-12),
                                    "maxiter": self.config.maxiter})
            if not hasattr(out, "multipliers"):
                raise RuntimeError("The SciPy oracle requires scipy>=1.16 "
                                   "for SLSQP multipliers. Upgrade SciPy.")
            nu = np.asarray(out.multipliers)
            if len(nu) == len(E)+len(C):
                y[eqidx] = -nu[:len(E)]
                dual_ineq = nu[len(E):]
                np.add.at(y, upperidx, dual_ineq[:len(upperidx)])
                np.add.at(y, loweridx, -dual_ineq[len(upperidx):])
        return finish(out.x, y, out.success, getattr(out, "nit", 0))

    def while_loop(self, cond, body, initial):
        if self.compilable:
            return jax.lax.while_loop(cond, body, initial)
        state = initial
        while bool(cond(state)):
            state = body(state)
        return state

    def cond(self, pred, yes, no, operand):
        if self.compilable:
            return jax.lax.cond(pred, yes, no, operand)
        return yes(operand) if bool(pred) else no(operand)

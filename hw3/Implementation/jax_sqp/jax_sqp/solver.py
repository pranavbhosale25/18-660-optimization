"""Nocedal--Wright style line-search SQP with elastic QPs and damped L-BFGS.

Constraint convention: equalities == 0, inequalities <= 0.
Lagrangian: L(x,lambda) = f(x) + lambda.T c(x).
All iteration state is a pytree of fixed-size arrays. No host callbacks or
SciPy calls occur inside a solve. ``run`` supports jit and vmap.
"""

import math
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from . import lbfgs
from .linalg import norm_inf
from .merit import violation
from .qp import QPOptions, solve_qp


class Status(IntEnum):
    CONVERGED = 0
    MAX_ITER = 1
    LINE_SEARCH_FAILED = 2
    QP_FAILED = 3
    NONFINITE = 4
    STALLED_INFEASIBLE = 5


STATUS_MESSAGES = {
    0: "First-order KKT tolerances satisfied.",
    1: "Maximum SQP iterations reached.",
    2: "No acceptable l1-merit line-search step.",
    3: "QP solver did not reach its requested accuracy.",
    4: "Nonfinite objective, derivatives, constraints, or step.",
    5: "Stalled at an infeasible point; this is NOT a global infeasibility certificate.",
}
HISTORY_COLUMNS = (
    "objective",
    "primal_inf",
    "stationarity_inf",
    "complementarity_inf",
    "kkt_inf",
    "alpha",
    "penalty",
    "linear_violation_l1",
    "qp_residual",
    "elastic",
    "qp_iterations",
    "constraint_violation_l1",
    "merit",
)


@dataclass(frozen=True)
class Options:
    max_iter: int = 250
    history_size: int = 10
    tolerance: float = 1e-6
    initial_hessian_scale: float = 1.0
    initial_penalty: float = 10.0
    max_penalty: float = 1e8
    penalty_growth: float = 5.0
    penalty_margin: float = 1.2
    elastic_retries: int = 2
    elastic_target: float = 0.1
    qp_mode: str = "auto"  # hard first, elastic on failure; or 'elastic'
    hard_qp_max_iter: int = 80
    qp: QPOptions = field(default_factory=QPOptions)
    armijo: float = 1e-4
    backtrack: float = 0.5
    max_linesearch: int = 24
    linesearch_batch: int = 4
    second_order_correction: bool = True
    max_soc: int = 3
    # The model slope is conservative; "exact" uses the one-sided derivative.
    merit_slope: str = "model"
    step_tolerance: float = 1e-12

    def __post_init__(self):
        if self.hard_qp_max_iter < 1 or self.max_soc < 1 or self.elastic_retries < 0:
            raise ValueError("Invalid QP/SOC/elastic iteration limits.")
        if self.initial_hessian_scale <= 0 or self.step_tolerance <= 0:
            raise ValueError(
                "Initial Hessian scale and step tolerance must be positive."
            )
        if not 0 < self.elastic_target < 1 or self.penalty_margin <= 1:
            raise ValueError("Require 0 < elastic_target < 1 and penalty_margin > 1.")
        if self.max_iter < 1 or self.history_size < 1:
            raise ValueError("max_iter and history_size must be positive.")
        if not (0 < self.tolerance and 0 < self.qp.tolerance):
            raise ValueError("Tolerances must be positive.")
        if not 0 < self.armijo < 0.5 or not 0 < self.backtrack < 1:
            raise ValueError("Require 0 < armijo < .5 and 0 < backtrack < 1.")
        if self.merit_slope not in ("model", "exact"):
            raise ValueError("merit_slope must be 'model' or 'exact'.")
        if self.qp_mode not in ("auto", "elastic"):
            raise ValueError("qp_mode must be 'auto' or 'elastic'.")
        if self.linesearch_batch < 1 or self.max_linesearch < 1:
            raise ValueError("Line-search limits must be positive.")
        if not 0 < self.initial_penalty <= self.max_penalty or self.penalty_growth <= 1:
            raise ValueError("Invalid penalty parameters.")

    @classmethod
    def for_dtype(cls, precision: int, **overrides):
        if precision == 64:
            opts = cls()
        elif precision == 32:
            # The QP tolerance is tighter than the outer KKT tolerance while
            # remaining above the residual floor observed across CPU backends.
            opts = cls(
                tolerance=3e-4,
                step_tolerance=1e-7,
                max_penalty=1e5,
                qp=QPOptions(
                    tolerance=5e-5, cg_tolerance=2e-6, regularization=1e-7, max_iter=300
                ),
            )
        else:
            raise ValueError("precision must be 32 or 64.")
        return replace(opts, **overrides)


class KKT(NamedTuple):
    primal: jax.Array
    stationarity: jax.Array
    dual: jax.Array
    complementarity: jax.Array
    error: jax.Array


class Result(NamedTuple):
    x: jax.Array
    objective: jax.Array
    multipliers: jax.Array
    iterations: jax.Array
    status: jax.Array
    success: jax.Array
    kkt: KKT
    penalty: jax.Array
    elastic_steps: jax.Array
    qp_iterations: jax.Array
    cg_iterations: jax.Array
    line_search_evaluations: jax.Array
    history: jax.Array


class _State(NamedTuple):
    x: jax.Array
    multipliers: jax.Array
    memory: lbfgs.Memory
    iteration: jax.Array
    status: jax.Array
    penalty: jax.Array
    elastic_steps: jax.Array
    qp_iterations: jax.Array
    cg_iterations: jax.Array
    evaluations: jax.Array
    history: jax.Array


def kkt_residual(g, c, at, multipliers, neq):
    """Measure first-order KKT residuals for the current primal-dual point.

    ``g`` is the objective gradient. ``c`` contains equality values first,
    followed by inequalities in ``<= 0`` form. ``at(v)`` evaluates the
    transpose-Jacobian product for the complete constraint vector, and
    ``multipliers`` follows the same ordering as ``c``. Equality multipliers
    are unrestricted, while inequality multipliers must be nonnegative.
    ``neq`` is the number of leading equality constraints.

    Return scalar JAX infinity-norm measures for primal feasibility,
    stationarity, inequality-dual feasibility, complementarity, and their
    maximum in a :class:`KKT` tuple.
    """
    # STUDENT TODO: Compute and return all five KKT residual measures.
    # norm_inf handles empty groups, so unconstrained and equality-only
    # problems use exactly the same expressions as mixed problems.
    inequalities = c[neq:]
    inequality_multipliers = multipliers[neq:]
    primal = jnp.maximum(
        norm_inf(c[:neq]), norm_inf(jnp.maximum(inequalities, 0.0))
    )
    # L = f + lambda.T c: equality multipliers are unrestricted and
    # multipliers for <= 0 inequalities must be nonnegative.
    stationarity = norm_inf(g + at(multipliers))
    dual = norm_inf(jnp.minimum(inequality_multipliers, 0.0))
    complementarity = norm_inf(inequality_multipliers * inequalities)
    error = jnp.max(jnp.stack((primal, stationarity, dual, complementarity)))
    return KKT(primal, stationarity, dual, complementarity, error)


def merit_eval(objective, constraints, neq, penalty):
    """Evaluate the scalar l1 exact-penalty merit function.

    ``objective`` is the scalar objective value. ``constraints`` contains
    equalities first and ``<= 0`` inequalities second, ``neq`` separates the
    two groups, and ``penalty`` is the current nonnegative penalty weight.
    Return a scalar JAX value.
    """
    # STUDENT TODO: Evaluate the merit function at the supplied values.
    return objective + penalty * violation(constraints, neq)


def merit_directional_derivative(g, p, c, ap, neq, penalty):
    """Evaluate the exact one-sided merit derivative along ``p``.

    ``g`` is the objective gradient, ``p`` is the proposed search direction,
    ``c`` is the ordered constraint vector, and ``ap`` is its Jacobian-vector
    product along ``p``. ``neq`` is the number of leading equalities and
    ``penalty`` is the current nonnegative penalty weight. Return the scalar
    JAX right-sided derivative, including the equality and inequality penalty
    terms at nonsmooth zero constraint values.
    """
    # STUDENT TODO: Compute the exact one-sided directional derivative.
    equalities, equality_slopes = c[:neq], ap[:neq]
    inequalities, inequality_slopes = c[neq:], ap[neq:]

    # At an equality kink, |0 + t*a| has right derivative |a|.
    # Away from zero, the usual sign(c)*a rule applies.
    equality_terms = jnp.where(
        equalities == 0.0,
        jnp.abs(equality_slopes),
        jnp.sign(equalities) * equality_slopes,
    )
    # max(c, 0) has slope a when c>0, zero when c<0, and
    # max(a, 0) at c=0. Array selection keeps this jit/vmap compatible.
    inequality_terms = jnp.where(
        inequalities > 0.0,
        inequality_slopes,
        jnp.where(inequalities == 0.0, jnp.maximum(inequality_slopes, 0.0), 0.0),
    )
    return jnp.dot(g, p) + penalty * (
        jnp.sum(equality_terms) + jnp.sum(inequality_terms)
    )


class SQP:
    """Construct reusable single-problem and batched compiled solvers.

    ``fun(x, params) -> scalar`` and ``eq/ineq(x, params) -> 1-D arrays`` must
    be pure JAX functions with static output shapes. ``params`` can be any
    JAX pytree (including None). Bounds are static and may contain +/-inf.
    The result's multipliers are ordered: eq, ineq, finite lower bounds,
    finite upper bounds. Bounds use the same merit/elastic treatment as all
    other constraints, so intermediate iterates need not satisfy them.
    """

    def __init__(
        self,
        fun: Callable,
        eq: Callable | None = None,
        ineq: Callable | None = None,
        *,
        lower=None,
        upper=None,
        options: Options | None = None,
    ):
        self.fun, self.eq, self.ineq = fun, eq, ineq
        self.lower = None if lower is None else np.asarray(lower)
        self.upper = None if upper is None else np.asarray(upper)
        self.options = options
        for b in (self.lower, self.upper):
            if b is not None and np.any(np.isnan(b)):
                raise ValueError("Bounds must not contain NaN.")
        self.solve = jax.jit(self.run)
        self.solve_batch = jax.jit(jax.vmap(self.run, in_axes=(0, None)))

    def run(self, x0, params: Any = None) -> Result:
        x0 = jnp.asarray(x0)
        o = self.options or Options.for_dtype(64 if x0.dtype == jnp.float64 else 32)
        if x0.ndim != 1 or x0.size == 0:
            raise ValueError("x0 must be a nonempty 1-D vector.")
        if x0.dtype not in (jnp.dtype("float32"), jnp.dtype("float64")):
            raise TypeError("x0 must have float32 or float64 dtype.")
        n = x0.size

        # Bounds become ordinary inequality rows, so all constraint handling
        # below can use a single equality-first, inequality-second convention.
        low = np.broadcast_to(-np.inf if self.lower is None else self.lower, (n,))
        upp = np.broadcast_to(np.inf if self.upper is None else self.upper, (n,))
        if np.any(np.isposinf(low)) or np.any(np.isneginf(upp)):
            raise ValueError(
                "Lower bounds cannot be +inf; upper bounds cannot be -inf."
            )
        if np.any(low > upp):
            raise ValueError("A lower bound exceeds an upper bound.")
        il = np.flatnonzero(np.isfinite(low))
        iu = np.flatnonzero(np.isfinite(upp))
        lo = jnp.asarray(low[il], dtype=x0.dtype)
        up = jnp.asarray(upp[iu], dtype=x0.dtype)

        def empty(x):
            return jnp.zeros(0, x.dtype)

        if self.eq is None:
            ef = empty
        else:

            def ef(x):
                return self.eq(x, params)

        if self.ineq is None:
            gf = empty
        else:

            def gf(x):
                return self.ineq(x, params)

        e0, g0 = ef(x0), gf(x0)
        if e0.ndim != 1 or g0.ndim != 1:
            raise ValueError(
                "Constraint functions must return 1-D arrays (not scalars)."
            )
        neq = e0.shape[0]

        def cons(x):
            return jnp.concatenate((ef(x), gf(x), lo - x[il], x[iu] - up))

        def fun(x):
            return self.fun(x, params)

        valgrad = jax.value_and_grad(fun)

        def evaluate(x):
            f, g = valgrad(x)
            # Keep the constraint Jacobian matrix-free. ``av`` computes A @ v,
            # while its linear transpose computes A.T @ w.
            c, av = jax.linearize(cons, x)
            transpose = jax.linear_transpose(av, jnp.zeros_like(x))

            def at(w):
                return transpose(w)[0]

            return f, g, c, av, at

        f, g, c, av, at = evaluate(x0)
        lam = jnp.zeros_like(c)
        kkt = kkt_residual(g, c, at, lam, neq)

        # Establish the initial termination state before entering JAX's fixed-
        # shape loop, and reserve one history row per possible SQP iteration.
        finite = (
            jnp.isfinite(f)
            & jnp.all(jnp.isfinite(g))
            & jnp.all(jnp.isfinite(c))
            & jnp.isfinite(kkt.error)
        )
        status = jnp.where(
            ~finite, 4, jnp.where(kkt.error <= o.tolerance, 0, -1)
        ).astype(jnp.int32)
        history = jnp.full((o.max_iter + 1, len(HISTORY_COLUMNS)), jnp.nan, x0.dtype)
        row = jnp.array(
            [
                f,
                kkt.primal,
                kkt.stationarity,
                kkt.complementarity,
                kkt.error,
                0.0,
                o.initial_penalty,
                0.0,
                0.0,
                0.0,
                0.0,
                violation(c, neq),
                merit_eval(f, c, neq, o.initial_penalty),
            ],
            dtype=x0.dtype,
        )
        history = history.at[0].set(row)
        zeroi = jnp.array(0, jnp.int32)
        initial = _State(
            x0,
            lam,
            lbfgs.initialize(n, o.history_size, x0.dtype, o.initial_hessian_scale),
            zeroi,
            status,
            jnp.array(o.initial_penalty, x0.dtype),
            zeroi,
            zeroi,
            zeroi,
            zeroi,
            history,
        )
        hard_options = replace(o.qp, max_iter=o.hard_qp_max_iter)

        def body(st):
            x = st.x
            f, g, c, av, at = evaluate(x)

            # L-BFGS supplies both B and inverse-Hessian products without ever
            # materializing a dense n-by-n Hessian.
            b = lbfgs.hessian_operator(st.memory)

            def h(v):
                return lbfgs.inverse_product(st.memory, v)

            v0 = violation(c, neq)

            def qp(penalty, multipliers, opts=o.qp, cc=c):
                return solve_qp(g, cc, av, at, h, neq, penalty, multipliers, opts)

            def elastic(_):
                # If an elastic step does not reduce the linearized violation
                # sufficiently, increase the shared penalty and solve again.
                r = qp(st.penalty, st.multipliers)

                def cond(t):
                    j, mu, r, qpi, cgi = t
                    return (
                        (j < o.elastic_retries)
                        & r.success
                        & (
                            r.linear_violation
                            > jnp.maximum(o.qp.tolerance * 10, o.elastic_target * v0)
                        )
                        & (mu < o.max_penalty)
                    )

                def increase(t):
                    j, mu, r, qpi, cgi = t
                    mu = jnp.minimum(mu * o.penalty_growth, o.max_penalty)
                    rr = qp(mu, r.multipliers)
                    return j + 1, mu, rr, qpi + rr.iterations, cgi + rr.cg_iterations

                _, mu, r, qpi, cgi = lax.while_loop(
                    cond,
                    increase,
                    (zeroi, st.penalty, r, r.iterations, r.cg_iterations),
                )
                return r, mu, jnp.array(True), qpi, cgi

            if o.qp_mode == "auto":
                # Try the ordinary hard QP first. A failed or insufficiently
                # feasible hard solve falls back to the elastic formulation.
                hard = qp(jnp.array(jnp.inf, x.dtype), st.multipliers, hard_options)
                hard_ok = hard.success & (
                    hard.linear_violation_inf
                    <= jnp.maximum(10 * o.qp.tolerance, o.tolerance * 0.05)
                )

                def regular(_):
                    # Eq. (18.36), sigma=1, generalized to inequalities.
                    bp = lbfgs.product(b, hard.step)
                    numerator = jnp.dot(g, hard.step) + 0.5 * jnp.dot(hard.step, bp)
                    required = jnp.where(
                        v0 > o.qp.tolerance,
                        numerator / jnp.maximum(0.9 * v0, jnp.finfo(x.dtype).tiny),
                        0.0,
                    )
                    mu = jnp.minimum(
                        o.max_penalty,
                        jnp.maximum(
                            st.penalty,
                            jnp.maximum(
                                o.penalty_margin * norm_inf(hard.multipliers) + 1.0,
                                required,
                            ),
                        ),
                    )
                    return (
                        hard,
                        mu,
                        jnp.array(False),
                        hard.iterations,
                        hard.cg_iterations,
                    )

                r, mu, is_elastic, qpi, cgi = lax.cond(
                    hard_ok, regular, elastic, operand=None
                )
                qpi = qpi + jnp.where(hard_ok, 0, hard.iterations)
                cgi = cgi + jnp.where(hard_ok, 0, hard.cg_iterations)
            else:
                r, mu, is_elastic, qpi, cgi = elastic(None)
            p, lamhat = r.step, r.multipliers
            ap = av(p)
            # Convexity gives D phi(x;p) <= g'p + mu*(v(c+A p)-v(c)).
            # The model bound is conservative and remains valid for elastic QPs.
            slope = (
                jnp.dot(g, p) + mu * (violation(c + ap, neq) - v0)
                if o.merit_slope == "model"
                else merit_directional_derivative(g, p, c, ap, neq, mu)
            )
            if o.merit_slope == "model":
                # An inexact QP can give a nonnegative conservative bound near
                # feasibility, especially in float32. Check the actual slope
                # before declaring that no descent direction is available.
                slope = jnp.where(
                    slope >= 0.0,
                    merit_directional_derivative(g, p, c, ap, neq, mu),
                    slope,
                )
            current_kkt = kkt_residual(g, c, at, lamhat, neq)
            already = current_kkt.error <= o.tolerance
            phi0 = merit_eval(f, c, neq, mu)
            roundoff = 5 * jnp.finfo(x.dtype).eps * (1 + jnp.abs(f) + mu * (1 + v0))

            def values(alpha):
                xx = x + alpha * p
                ff = fun(xx)
                return merit_eval(ff, cons(xx), neq, mu)

            # vmap four candidate alphas at a time; short circuit between blocks.
            batches = math.ceil(o.max_linesearch / o.linesearch_batch)

            def ls_body(ls):
                k, alpha, accepted, neval = ls
                ids = k * o.linesearch_batch + jnp.arange(o.linesearch_batch)
                aa = jnp.asarray(o.backtrack, x.dtype) ** ids
                vv = jax.vmap(values)(aa)
                good = (
                    (ids < o.max_linesearch)
                    & jnp.isfinite(vv)
                    & (vv <= phi0 + o.armijo * aa * slope + roundoff)
                )
                idx = jnp.argmax(good)
                return (
                    k + 1,
                    jnp.where(jnp.any(good), aa[idx], 0.0),
                    jnp.any(good),
                    neval + o.linesearch_batch,
                )

            allow = r.success & (slope < 0) & (~already) & jnp.all(jnp.isfinite(p))
            ls0 = (zeroi, jnp.array(0.0, x.dtype), jnp.array(False), zeroi)
            _, alpha, accepted, neval = lax.while_loop(
                lambda s: (s[0] < batches) & (~s[2]) & allow, ls_body, ls0
            )
            xn = jnp.where(accepted, x + alpha * p, x)
            lamn = jnp.where(
                accepted,
                st.multipliers + alpha * (lamhat - st.multipliers),
                st.multipliers,
            )
            # A multiplier-only correction permits termination at a primal optimum
            # even when the preceding damped dual step lagged behind.
            lamn = jnp.where(already, lamhat, lamn)
            alpha = jnp.where(already, 0.0, alpha)
            if o.second_order_correction:
                # Re-solve the same quadratic objective with shifted constraints:
                # c(x+p) + A(q-p) = 0 on equalities/active inequalities.
                # q is the corrected TOTAL step, not merely the correction.
                try_soc = allow & (alpha < 1) & (v0 < 0.01) & (norm_inf(p) <= 1.0)

                def soc(_):
                    def correction(t):
                        j, q, qlam, ok, qi, ci = t
                        shifted = cons(x + q) - av(q)
                        rs = solve_qp(
                            g,
                            shifted,
                            av,
                            at,
                            h,
                            neq,
                            jnp.where(is_elastic, mu, jnp.inf),
                            qlam,
                            o.qp,
                        )
                        qq = rs.step
                        xx = x + qq
                        phi = merit_eval(fun(xx), cons(xx), neq, mu)
                        good = (
                            rs.success
                            & jnp.isfinite(phi)
                            & (
                                norm_inf(qq - p)
                                <= 0.5 * jnp.maximum(norm_inf(p), o.step_tolerance)
                            )
                            & (phi <= phi0 + o.armijo * slope + roundoff)
                        )
                        return (
                            j + 1,
                            qq,
                            rs.multipliers,
                            good,
                            qi + rs.iterations,
                            ci + rs.cg_iterations,
                        )

                    j, q, qlam, ok, qi, ci = lax.while_loop(
                        lambda t: (t[0] < o.max_soc) & (~t[3]),
                        correction,
                        (zeroi, p, lamhat, jnp.array(False), zeroi, zeroi),
                    )
                    return (
                        jnp.where(ok, x + q, xn),
                        jnp.where(ok, qlam, lamn),
                        jnp.where(ok, 1.0, alpha),
                        accepted | ok,
                        qpi + qi,
                        cgi + ci,
                        neval + j,
                    )

                xn, lamn, alpha, accepted, qpi, cgi, neval = lax.cond(
                    try_soc,
                    soc,
                    lambda _: (xn, lamn, alpha, accepted, qpi, cgi, neval),
                    operand=None,
                )
            fn, gn, cn, avn, atn = evaluate(xn)
            kn = kkt_residual(gn, cn, atn, lamn, neq)

            # Form the Lagrangian-gradient secant difference with the new
            # multiplier at both endpoints, as required by the SQP update.
            y = gn + atn(lamn) - (g + at(lamn))
            sn = xn - x
            newmem, updated = lbfgs.update(st.memory, b, sn, y)
            newmem = lax.cond(
                accepted, lambda _: newmem, lambda _: st.memory, operand=None
            )
            finite = (
                jnp.isfinite(fn)
                & jnp.all(jnp.isfinite(gn))
                & jnp.all(jnp.isfinite(cn))
                & jnp.isfinite(kn.error)
                & jnp.all(jnp.isfinite(p))
                & jnp.all(jnp.isfinite(lamhat))
            )
            stalled = (norm_inf(sn) <= o.step_tolerance * (1 + norm_inf(x))) & (
                ~already
            )

            # Status precedence matters: convergence wins, followed by
            # nonfinite data, QP failure, infeasible stagnation, and line-search
            # failure. A negative status means that the outer loop continues.
            status = jnp.where(
                kn.error <= o.tolerance,
                0,
                jnp.where(
                    ~finite,
                    4,
                    jnp.where(
                        ~r.success,
                        3,
                        jnp.where(
                            stalled
                            & (kn.primal > o.tolerance)
                            & (norm_inf(p) <= o.step_tolerance * (1 + norm_inf(x))),
                            5,
                            jnp.where(~accepted, 2, -1),
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            row = jnp.array(
                [
                    fn,
                    kn.primal,
                    kn.stationarity,
                    kn.complementarity,
                    kn.error,
                    alpha,
                    mu,
                    r.linear_violation,
                    r.residual,
                    is_elastic,
                    qpi,
                    # Record actual nonlinear violation at the accepted point;
                    # the QP's linear_violation_l1 is a different quantity.
                    violation(cn, neq),
                    merit_eval(fn, cn, neq, mu),
                ],
                dtype=x.dtype,
            )
            return _State(
                xn,
                lamn,
                newmem,
                st.iteration + 1,
                status,
                mu,
                st.elastic_steps + is_elastic.astype(jnp.int32),
                st.qp_iterations + qpi,
                st.cg_iterations + cgi,
                st.evaluations + neval,
                st.history.at[st.iteration + 1].set(row),
            )

        st = lax.while_loop(
            lambda st: (st.iteration < o.max_iter) & (st.status < 0), body, initial
        )
        status = jnp.where(st.status < 0, jnp.array(1, jnp.int32), st.status)
        f, g, c, av, at = evaluate(st.x)
        kkt = kkt_residual(g, c, at, st.multipliers, neq)
        return Result(
            st.x,
            f,
            st.multipliers,
            st.iteration,
            status,
            status == 0,
            kkt,
            st.penalty,
            st.elastic_steps,
            st.qp_iterations,
            st.cg_iterations,
            st.evaluations,
            st.history,
        )

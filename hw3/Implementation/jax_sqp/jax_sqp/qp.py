"""Matrix-free elastic QP using a box-constrained dual Newton-CG solve.

No slack-variable matrices or dense primal Hessians are formed.
"""

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .linalg import norm_inf, pcg
from .merit import violation


@dataclass(frozen=True)
class QPOptions:
    max_iter: int = 200
    tolerance: float = 1e-8
    cg_max_iter: int = 200
    cg_tolerance: float = 1e-10
    dense_threshold: int = 64
    regularization: float = 1e-10
    max_linesearch: int = 30
    multiplier_limit: float = 1e9
    stall_iterations: int = 30
    jacobi_probes: int = 0

    def __post_init__(self):
        if (
            min(
                self.max_iter,
                self.cg_max_iter,
                self.max_linesearch,
                self.stall_iterations,
            )
            < 1
        ):
            raise ValueError("QP iteration limits must be positive.")
        if self.tolerance <= 0 or self.cg_tolerance <= 0 or self.regularization < 0:
            raise ValueError(
                "QP tolerances must be positive and regularization nonnegative."
            )
        if (
            self.dense_threshold < 0
            or self.jacobi_probes < 0
            or self.multiplier_limit <= 0
        ):
            raise ValueError("Invalid QP storage/preconditioner settings.")


class QPResult(NamedTuple):
    step: jax.Array
    multipliers: jax.Array
    residual: jax.Array
    iterations: jax.Array
    linear_violation: jax.Array
    linear_violation_inf: jax.Array
    success: jax.Array
    cg_iterations: jax.Array


def solve_qp(
    g,
    c,
    av: Callable,
    at: Callable,
    hinv: Callable,
    neq: int,
    penalty,
    multipliers,
    options: QPOptions,
) -> QPResult:
    """Solve one SQP quadratic subproblem through its box-constrained dual.

    ``av`` and ``at`` apply the constraint Jacobian and its transpose, while
    ``hinv`` applies the inverse L-BFGS Hessian model. Constraint values in
    ``c`` are ordered as ``neq`` equalities followed by ``<= 0`` inequalities.
    """
    m = c.shape[0]
    if m == 0:
        # With no constraints, the inverse Hessian directly gives the step.
        p = -hinv(g)
        zero = jnp.array(0.0, g.dtype)
        return QPResult(
            p,
            c,
            zero,
            jnp.array(0, jnp.int32),
            zero,
            zero,
            jnp.array(True),
            jnp.array(0, jnp.int32),
        )
    # The l1 penalty becomes box bounds on dual multipliers: equalities are
    # signed, while inequality multipliers remain nonnegative.
    lower = jnp.where(jnp.arange(m) < neq, -penalty, 0.0)
    upper = jnp.full_like(c, penalty)

    def op(z):
        # Reduced dual Hessian product A H A.T z after eliminating the primal.
        return av(hinv(at(z)))

    # Small constraint spaces use an explicit reduced matrix; larger spaces
    # retain only operator products and solve their Newton systems with PCG.
    dense = m <= options.dense_threshold
    if dense:
        mat = jax.vmap(op)(jnp.eye(m, dtype=g.dtype)).T
        mat = (mat + mat.T) / 2

        def mv(z):
            return mat @ z

        lipschitz = jnp.maximum(jnp.max(jnp.sum(jnp.abs(mat), axis=1)), 1e-12)
        diagonal = jnp.diag(mat)
    else:
        mv = op
        # Deterministic power iteration; majorization backtracking safeguards this estimate.
        v = jnp.sin(jnp.arange(m, dtype=g.dtype) + 1)
        v = v / jnp.linalg.norm(v)

        def power(_, v):
            w = mv(v)
            return w / jnp.maximum(jnp.linalg.norm(w), 1e-30)

        v = lax.fori_loop(0, 12, power, v)
        lipschitz = jnp.maximum(1.1 * jnp.dot(v, mv(v)), 1e-8)
        if options.jacobi_probes > 0:
            probes = jax.random.rademacher(
                jax.random.key(1729), (options.jacobi_probes, m), dtype=g.dtype
            )
            diagonal = jnp.mean(probes * jax.vmap(mv)(probes), axis=0)
            # The stochastic estimate may be negative: clipping makes M SPD.
            diagonal = jnp.clip(diagonal, 1e-3 * lipschitz, lipschitz)
        else:
            diagonal = jnp.full_like(c, lipschitz)
    # Linear term of the reduced dual objective.
    d = av(hinv(g)) - c

    def obj(z):
        return 0.5 * jnp.dot(z, mv(z)) + jnp.dot(d, z)

    def residual(z, h):
        # Natural (unscaled) projected KKT residual of the dual box QP.
        return norm_inf(z - jnp.clip(z - h, lower, upper))

    z = jnp.clip(multipliers, lower, upper)
    h = mv(z) + d
    eps = jnp.finfo(g.dtype).eps
    reg = jnp.maximum(options.regularization * lipschitz, 5 * eps * lipschitz)
    # State fields: iteration, multipliers, gradient, residual, cumulative CG
    # work, Lipschitz estimate, best residual, and consecutive stalled steps.
    state = (
        jnp.array(0, jnp.int32),
        z,
        h,
        residual(z, h),
        jnp.array(0, jnp.int32),
        lipschitz,
        residual(z, h),
        jnp.array(0, jnp.int32),
    )

    def cond(s):
        return (
            (s[0] < options.max_iter)
            & ((s[0] == 0) | (s[3] > options.tolerance))
            & jnp.isfinite(s[3])
            & (norm_inf(s[1]) < options.multiplier_limit)
            & (s[7] < options.stall_iterations)
        )

    def body(s):
        k, z, h, error, cg_total, L, best, stalled = s
        # Generalized active face of the projection. Binding bounds stay fixed.
        active_eps = jnp.maximum(10 * eps, 0.1 * options.tolerance)
        free = ((z > lower + active_eps) | (h < 0)) & (
            (z < upper - active_eps) | (h > 0)
        )
        ff = free.astype(g.dtype)

        def restricted(v):
            return ff * mv(ff * v) + (1 - ff) * v + reg * ff * v

        rhs = -ff * h
        if dense:
            # Fixed multipliers receive identity rows; free multipliers use the
            # regularized reduced Newton system.
            kk = mat * ff[:, None] * ff[None, :] + jnp.diag(1 - ff + reg * ff)
            direction = jnp.linalg.solve(kk, rhs)
            n_cg = jnp.array(0, jnp.int32)
        else:
            sol = pcg(
                restricted,
                rhs,
                lambda v: v / (1 - ff + ff * (diagonal + reg)),
                options.cg_max_iter,
                options.cg_tolerance,
                jnp.minimum(
                    options.tolerance * 0.01,
                    options.cg_tolerance * jnp.linalg.norm(rhs),
                ),
            )
            direction, n_cg = sol.x, sol.iterations
        # A projected Newton direction can cease to be descending after clipping.
        trial = jnp.clip(z + direction, lower, upper)
        descent = jnp.dot(h, trial - z) < -1e-12 * jnp.dot(trial - z, trial - z)
        direction = jnp.where(
            descent & jnp.all(jnp.isfinite(direction)), direction, -h / L
        )
        f0 = obj(z)

        def ls_cond(ls):
            return (ls[0] < options.max_linesearch) & (~ls[3])

        def ls_body(ls):
            j, a, zz, ok = ls
            zz = jnp.clip(z + a * direction, lower, upper)
            delta = zz - z
            f = obj(zz)
            ok = (
                f
                <= f0
                + 1e-4 * jnp.dot(h, delta)
                + 10 * eps * jnp.maximum(1.0, jnp.abs(f0))
            ) & (norm_inf(delta) > 0)
            return j + 1, jnp.where(ok, a, 0.5 * a), zz, ok

        _, a, zn, ok = lax.while_loop(
            ls_cond,
            ls_body,
            (jnp.array(0, jnp.int32), jnp.array(1.0, g.dtype), z, jnp.array(False)),
        )
        # Safeguarded projected-gradient step, including majorization line search.
        # Used also when Newton makes no meaningful progress.
        hn = mv(zn) + d
        need_pg = (~ok) | (residual(zn, hn) > 0.99 * error)

        def pg(_):
            def b(ls):
                j, LL, zz, valid = ls
                zz = jnp.clip(z - h / LL, lower, upper)
                delta = zz - z
                valid = obj(zz) <= f0 + jnp.dot(h, delta) + 0.5 * LL * jnp.dot(
                    delta, delta
                ) + 10 * eps * jnp.maximum(1.0, jnp.abs(f0))
                return j + 1, jnp.where(valid, LL, 2 * LL), zz, valid

            _, LL, zz, valid = lax.while_loop(
                lambda a: (a[0] < 30) & (~a[3]),
                b,
                (jnp.array(0, jnp.int32), L, z, jnp.array(False)),
            )
            # Prefer Newton when its objective decrease is at least as good.
            choose_pg = (obj(zz) < obj(zn)) | (~ok)
            return jnp.where(choose_pg, zz, zn), LL

        zn, L = lax.cond(need_pg, pg, lambda _: (zn, L), operand=None)
        hn = mv(zn) + d
        rn = residual(zn, hn)
        improved = rn < 0.99 * best
        return (
            k + 1,
            zn,
            hn,
            rn,
            cg_total + n_cg,
            L,
            jnp.minimum(best, rn),
            jnp.where(improved, 0, stalled + 1),
        )

    k, z, h, error, cg_total, _, _, _ = lax.while_loop(cond, body, state)
    # Recover the primal step from the final dual multipliers, then report the
    # feasibility of the linearized constraints used by the SQP outer loop.
    p = -hinv(g + at(z))
    lin = c + av(p)
    return QPResult(
        p,
        z,
        error,
        k,
        violation(lin, neq),
        jnp.maximum(norm_inf(lin[:neq]), norm_inf(jnp.maximum(lin[neq:], 0.0))),
        (error <= options.tolerance) & jnp.all(jnp.isfinite(p)),
        cg_total,
    )

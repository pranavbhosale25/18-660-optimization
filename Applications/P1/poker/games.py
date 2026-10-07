"""Zero-sum games over products of simplices; pure JAX problem assembly.

A conventional matrix game has one information type per player. A one-stage
Bayesian game has one simplex per private type. K includes the joint chance
probabilities, NOT conditional probabilities normalized separately by row.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from qp import QP, Solver


class GameResult(NamedTuple):
    row: jax.Array       # (row_types, row_actions), conditional action probabilities
    col: jax.Array       # (col_types, col_actions)
    payoff: jax.Array
    lower: jax.Array     # guarantee of row against an exact best response
    upper: jax.Array     # best-response payoff against col
    gap: jax.Array       # Nash gap = upper - lower, in original payoff units
    status: jax.Array   # (2,), both entries must be 1
    lp_residual: jax.Array
    raw_simplex_error: jax.Array
    iterations: jax.Array


def row_lp(K: jax.Array, row_types: int, row_actions: int,
           col_types: int, col_actions: int) -> QP:
    """STUDENT PART: formulate max_p min_q p.T K q as an LP.

    Variables are [p.ravel(), z], with one free z per opponent type.
    A common scale improves constraint magnitudes without changing the game.
    """
    # STUDENT TODO: Implement row_lp.
    # Keep one probability distribution per OWN private type. The auxiliary
    # z[j] is the guaranteed payoff contribution from opponent type j; it
    # must be free (possibly negative), rather than a probability variable.
    # Integer payoff matrices also need floating-point arrays and infinite bounds.
    K = jnp.asarray(K, dtype=jnp.result_type(K, 1.0))
    dtype = K.dtype
    num_probabilities = row_types * row_actions
    num_variables = num_probabilities + col_types

    # K already contains joint deal probabilities. Scale the entire kernel
    # together for conditioning; never normalize individual rows or types.
    scale = jnp.max(jnp.abs(K))
    scale = jnp.where(scale > 0, scale, jnp.asarray(1.0, dtype=dtype))
    scaled_kernel = K / scale

    # Q=0 makes the adapter's quadratic objective an LP. Maximize the average
    # scaled guarantee: multiplying sum(z) by 1/col_types preserves the optimum.
    # Keep these coefficients independent of K's magnitude, so very small
    # payoffs do not make the linear objective numerically negligible.
    Q = jnp.zeros((num_variables, num_variables), dtype=dtype)
    c = jnp.concatenate((jnp.zeros(num_probabilities, dtype=dtype),
                         -jnp.ones(col_types, dtype=dtype) / col_types))

    # sum_a p[i,a] = 1 for each private type i.
    simplex = jnp.kron(jnp.eye(row_types, dtype=dtype),
                       jnp.ones((1, row_actions), dtype=dtype))
    simplex = jnp.concatenate((simplex,
                              jnp.zeros((row_types, col_types), dtype=dtype)), axis=1)

    # p >= 0. Only probability variables get these bounds: z stays free.
    nonnegative = jnp.concatenate((jnp.eye(num_probabilities, dtype=dtype),
                                  jnp.zeros((num_probabilities, col_types), dtype=dtype)), axis=1)

    # For every opponent type j and action b, require
    # z[j] <= sum_{i,a} scaled_K[(i,a),(j,b)] * p[i,a].
    # Repeating identity rows matches the type-then-action flattening order.
    opponent_type = jnp.repeat(jnp.eye(col_types, dtype=dtype), col_actions, axis=0)
    payoff_bounds = jnp.concatenate((scaled_kernel.T, -opponent_type), axis=1)

    A = jnp.concatenate((simplex, nonnegative, payoff_bounds), axis=0)
    num_inequalities = num_probabilities + col_types * col_actions
    lower = jnp.concatenate((jnp.ones(row_types, dtype=dtype),
                             jnp.zeros(num_inequalities, dtype=dtype)))
    upper = jnp.concatenate((jnp.ones(row_types, dtype=dtype),
                             jnp.full((num_inequalities,), jnp.inf, dtype=dtype)))
    return QP(Q, c, A, lower, upper)


def simplex_error(x):
    return jnp.maximum(jnp.max(jnp.maximum(-x, 0.0)),
                       jnp.max(jnp.abs(x.sum(axis=-1) - 1.0)))


def normalize_roundoff(x):
    """Convert tiny solver violations to a strategy; inspect raw error as well."""
    positive = jnp.maximum(x, 0.0)
    total = positive.sum(axis=-1, keepdims=True)
    return jnp.where(total > 0, positive / jnp.maximum(total, 1e-30),
                     jnp.ones_like(positive) / positive.shape[-1])


def evaluate(K, row, col):
    """Exact expectation and exact unilateral best-response bounds."""
    r, c = row.ravel(), col.ravel()
    lower = jnp.min((K.T @ r).reshape(col.shape), axis=1).sum()
    upper = jnp.max((K @ c).reshape(row.shape), axis=1).sum()
    return r @ K @ c, lower, upper


def solve_game(K, row_types, row_actions, col_types, col_actions, solver: Solver):
    pr = row_lp(K, row_types, row_actions, col_types, col_actions)
    pc = row_lp(-K.T, col_types, col_actions, row_types, row_actions)
    r, c = solver.solve(pr), solver.solve(pc)
    rr = r.x[:row_types * row_actions].reshape(row_types, row_actions)
    cc = c.x[:col_types * col_actions].reshape(col_types, col_actions)
    raw_error = jnp.maximum(simplex_error(rr), simplex_error(cc))
    row, col = normalize_roundoff(rr), normalize_roundoff(cc)
    payoff, lower, upper = evaluate(K, row, col)
    return GameResult(row, col, payoff, lower, upper, upper-lower,
                      jnp.stack((r.status, c.status)),
                      jnp.maximum(r.feasibility, c.feasibility), raw_error,
                      jnp.stack((r.iterations, c.iterations)))


def solve_matrix_game(A, solver):
    return solve_game(A, 1, A.shape[0], 1, A.shape[1], solver)


def make_batched_game_solver(solver, row_types, row_actions, col_types, col_actions):
    if not solver.compilable:
        raise ValueError("SciPy is a host oracle; batching requires JAXopt")
    return jax.jit(jax.vmap(lambda K: solve_game(
        K, row_types, row_actions, col_types, col_actions, solver)))

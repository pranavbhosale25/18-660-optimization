"""Constraint-violation utilities used by the SQP and QP solvers."""

import jax.numpy as jnp


def violation(constraints, neq):
    """Return the scalar total violation of an ordered constraint vector.

    ``constraints`` contains the ``neq`` equality values first, followed by
    inequalities in ``<= 0`` form. Return a nonnegative scalar JAX value with
    the same floating dtype as ``constraints``. The function must also handle
    either constraint group being empty, including a completely empty vector.
    """
    # STUDENT TODO: Compute the total constraint violation.
    # Equalities contribute their distance from zero. Only positive values
    # violate the <= 0 inequalities. Empty slices sum to a typed JAX zero.
    equality_violation = jnp.sum(jnp.abs(constraints[:neq]))
    inequality_violation = jnp.sum(jnp.maximum(constraints[neq:], 0.0))
    return equality_violation + inequality_violation

"""JIT/vmap-safe preconditioned conjugate gradients with explicit diagnostics."""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


class CGResult(NamedTuple):
    x: jax.Array
    iterations: jax.Array
    residual: jax.Array
    success: jax.Array


def norm_inf(x):
    return jnp.max(jnp.abs(x), initial=0.0)


def pcg(
    matvec: Callable,
    rhs: jax.Array,
    precondition: Callable,
    max_iter: int,
    rtol=1e-8,
    atol=1e-12,
    x0=None,
) -> CGResult:
    """Solve a positive-definite linear system with left-preconditioned CG."""
    x = jnp.zeros_like(rhs) if x0 is None else x0
    r = rhs - matvec(x)
    z = precondition(r)
    rz = jnp.dot(r, z)
    threshold = jnp.maximum(atol, rtol * jnp.linalg.norm(rhs))
    # Fixed-shape loop state keeps the routine compatible with jit and vmap.
    initial = (jnp.array(0, jnp.int32), x, r, z, rz, jnp.array(True))

    def cond(s):
        k, x, r, p, rz, ok = s
        return (k < max_iter) & (jnp.linalg.norm(r) > threshold) & ok

    def body(s):
        k, x, r, p, rz, ok = s
        ap = matvec(p)
        pap = jnp.dot(p, ap)
        # Nonpositive curvature or nonfinite arithmetic invalidates the PCG
        # assumptions and terminates the loop with success=False.
        good = (pap > 0) & jnp.isfinite(pap) & jnp.isfinite(rz)
        alpha = jnp.where(good, rz / jnp.where(good, pap, 1.0), 0.0)
        xn = x + alpha * p
        rn = r - alpha * ap
        zn = precondition(rn)
        rzn = jnp.dot(rn, zn)
        beta = rzn / jnp.where(rz > 0, rz, 1.0)
        return k + 1, xn, rn, zn + beta * p, rzn, good

    k, x, r, _, _, ok = lax.while_loop(cond, body, initial)
    residual = jnp.linalg.norm(rhs - matvec(x))
    return CGResult(x, k, residual, ok & (residual <= 5 * threshold))

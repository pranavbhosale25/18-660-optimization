"""Fixed-storage damped L-BFGS operators; never construct an n-by-n matrix."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


class Memory(NamedTuple):
    s: jax.Array
    y: jax.Array
    count: jax.Array
    scale: jax.Array  # B_0 = scale * I, H_0 = I / scale


class Hessian(NamedTuple):
    scale: jax.Array
    u: jax.Array
    v: jax.Array


def initialize(n: int, history: int, dtype, scale: float = 1.0) -> Memory:
    """Create empty fixed-shape memory suitable for JIT-compiled loops."""
    z = jnp.zeros((history, n), dtype)
    return Memory(z, z, jnp.array(0, jnp.int32), jnp.array(scale, dtype))


def inverse_product(mem: Memory, vector: jax.Array) -> jax.Array:
    """Standard L-BFGS two-loop recursion, ordered oldest to newest."""
    h = mem.s.shape[0]
    sy = jnp.sum(mem.s * mem.y, axis=1)
    valid = jnp.arange(h) < mem.count
    rho = jnp.where(valid, 1.0 / jnp.where(valid, sy, 1.0), 0.0)

    # First pass: newest pair to oldest pair.
    def backward(k, carry):
        q, alpha = carry
        i = h - 1 - k
        a = rho[i] * jnp.dot(mem.s[i], q)
        return q - a * mem.y[i], alpha.at[i].set(a)

    q, alpha = lax.fori_loop(0, h, backward, (vector, jnp.zeros(h, vector.dtype)))

    # Second pass: oldest pair to newest pair.
    def forward(i, r):
        beta = rho[i] * jnp.dot(mem.y[i], r)
        return r + mem.s[i] * (alpha[i] - beta)

    return lax.fori_loop(0, h, forward, q / mem.scale)


def hessian_operator(mem: Memory) -> Hessian:
    """Rebuild B = theta I - U.T U + V.T V in O(history**2 * n)."""

    # Rebuild compact factors one stored curvature pair at a time.
    def body(i, uv):
        u, v = uv
        s, y = mem.s[i], mem.y[i]
        bs = mem.scale * s - u.T @ (u @ s) + v.T @ (v @ s)
        valid = i < mem.count
        sbs = jnp.where(valid, jnp.dot(s, bs), 1.0)
        sy = jnp.where(valid, jnp.dot(s, y), 1.0)
        ui = jnp.where(
            valid, bs / jnp.sqrt(jnp.maximum(sbs, jnp.finfo(s.dtype).tiny)), 0.0
        )
        vi = jnp.where(
            valid, y / jnp.sqrt(jnp.maximum(sy, jnp.finfo(s.dtype).tiny)), 0.0
        )
        return u.at[i].set(ui), v.at[i].set(vi)

    u, v = lax.fori_loop(
        0, mem.s.shape[0], body, (jnp.zeros_like(mem.s), jnp.zeros_like(mem.y))
    )
    return Hessian(mem.scale, u, v)


def product(b: Hessian, vector: jax.Array) -> jax.Array:
    return b.scale * vector - b.u.T @ (b.u @ vector) + b.v.T @ (b.v @ vector)


def update(
    mem: Memory, b: Hessian, s: jax.Array, y: jax.Array, damping: float = 0.2
) -> tuple[Memory, jax.Array]:
    """Powell-damp positive-curvature pairs; skip zero/negative-curvature pairs."""
    bs = product(b, s)
    sy, sbs = jnp.dot(s, y), jnp.dot(s, bs)
    tiny = jnp.finfo(s.dtype).tiny
    # Powell damping replaces a weak-curvature y with a convex combination of
    # y and B s so that accepted updates preserve positive definiteness.
    theta = jnp.where(
        sy >= damping * sbs, 1.0, (1.0 - damping) * sbs / jnp.maximum(sbs - sy, tiny)
    )
    yd = theta * y + (1.0 - theta) * bs
    syd = jnp.dot(s, yd)
    ok = (
        (sbs > tiny)
        & (syd > tiny)
        & jnp.all(jnp.isfinite(yd))
        & (sy > 1e-8 * jnp.linalg.norm(s) * jnp.linalg.norm(y))
    )

    def insert(_):
        # Once full, discard the oldest pair before appending the new one.
        full = mem.count == mem.s.shape[0]
        ss = jnp.where(full, jnp.roll(mem.s, -1, axis=0), mem.s)
        yy = jnp.where(full, jnp.roll(mem.y, -1, axis=0), mem.y)
        idx = jnp.minimum(mem.count, mem.s.shape[0] - 1)
        scale = jnp.clip(jnp.dot(yd, yd) / syd, 1e-4, 1e4)
        return Memory(
            ss.at[idx].set(s),
            yy.at[idx].set(yd),
            jnp.minimum(mem.count + 1, mem.s.shape[0]),
            scale,
        )

    return lax.cond(ok, insert, lambda _: mem, operand=None), ok

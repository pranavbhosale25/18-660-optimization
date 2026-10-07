"""Provided random worlds and validation; no optimizer dependency."""
import numpy as np
import jax
import jax.numpy as jnp
from physics import Ball, World


def make_world(normals, offsets, friction=.3, restitution=.5):
    """Host-side constructor normalizes planes (INCLUDING their offsets)."""
    a = np.asarray(normals, dtype=float)
    b = np.asarray(offsets, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or b.shape != (len(a),) or len(a) < 1:
        raise ValueError("Expected normals (n,3) and offsets (n,), n>=1")
    norm = np.linalg.norm(a, axis=1)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or (norm <= 0).any():
        raise ValueError("Plane data must be finite, with nonzero normals")
    mu = np.broadcast_to(np.asarray(friction, float), (len(a),)).copy()
    e = np.broadcast_to(np.asarray(restitution, float), (len(a),)).copy()
    if not np.isfinite(mu).all() or not np.isfinite(e).all() or (mu < 0).any() or ((e<0)|(e>1)).any():
        raise ValueError("Need finite mu>=0 and 0<=e<=1")
    return World(*(jnp.asarray(x) for x in (a/norm[:,None], b/norm, mu, e)))


def make_ball(position, velocity, radius=0., mass=1.):
    """Point mass; radius is optional geometric clearance, not rotational inertia."""
    if not np.isfinite(radius) or not np.isfinite(mass) or radius < 0 or mass <= 0:
        raise ValueError("Need finite radius >= 0 and mass > 0")
    arrays = [np.asarray(x, dtype=float) for x in (position, velocity)]
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in arrays):
        raise ValueError("Position and velocity must be finite 3-vectors")
    return Ball(*(jnp.asarray(x) for x in (*arrays, radius, mass)))


def random_worlds(key, batch=16, planes=10, extent=2., radius=0.,
                  friction=.25, restitution=.6):
    """Bounded worlds containing a common interior ball; no rejection loop.

    Six box walls guarantee boundedness. Extra planes clip corners. Starting
    centers are close to the origin, guaranteed feasible for extent>2*radius.
    All worlds share shapes, making vmap straightforward.
    """
    if batch < 1 or planes < 6 or extent <= 2*radius or radius < 0:
        raise ValueError("Need batch>=1, planes>=6, extent>2*radius>=0")
    if friction < 0 or not 0 <= restitution <= 1:
        raise ValueError("Invalid friction or restitution")
    ka, kb, kx, kv, ks = jax.random.split(key, 5)
    base = jnp.concatenate((jnp.eye(3), -jnp.eye(3)), axis=0)
    base = jnp.broadcast_to(base, (batch, 6, 3))
    extra = jax.random.normal(ka, (batch, planes-6, 3))
    extra = extra / jnp.maximum(jnp.linalg.norm(extra, axis=-1, keepdims=True), 1e-20)
    normals = jnp.concatenate((base, extra), axis=1)
    offsets = jnp.concatenate((jnp.full((batch, 6), -extent),
                               -extent*jax.random.uniform(kb, (batch, planes-6),
                                                         minval=.8, maxval=1.2)), axis=1)
    worlds = World(normals, offsets, jnp.full((batch, planes), friction),
                   jnp.full((batch, planes), restitution))
    x = jax.random.uniform(kx, (batch, 3), minval=-.08*extent, maxval=.08*extent)
    direction = jax.random.normal(kv, (batch, 3))
    direction /= jnp.maximum(jnp.linalg.norm(direction, axis=1, keepdims=True), 1e-20)
    speed = jax.random.uniform(ks, (batch, 1), minval=1., maxval=5.)
    balls = Ball(x, direction*speed, jnp.full(batch, radius), jnp.ones(batch))
    return worlds, balls

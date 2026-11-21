import numpy as np
import jax.numpy as jnp
from reinforce_jax.types import Trajectory


def batch(rewards, *, terminated=None, truncated=None, valid=None, dtype=jnp.float64,
          observations=None, next_observations=None, actions=None, raw_actions=None):
    r = jnp.asarray(rewards, dtype)
    if r.ndim == 1:
        r = r[:, None]
    T, B = r.shape
    shape = (T, B)
    def flags(x, default=False):
        return jnp.full(shape, default, bool) if x is None else jnp.asarray(x, bool).reshape(shape)
    obs = jnp.ones((T, B, 1), dtype) if observations is None else jnp.asarray(observations, dtype).reshape(T, B, -1)
    nxt = obs if next_observations is None else jnp.asarray(next_observations, dtype).reshape(T, B, -1)
    a = jnp.zeros(shape, jnp.int32) if actions is None else jnp.asarray(actions).reshape(T, B, -1)
    if actions is not None and np.asarray(actions).ndim <= 2:
        a = jnp.asarray(actions).reshape(shape)
    raw = a if raw_actions is None else jnp.asarray(raw_actions)
    return Trajectory(obs, a, raw, r, nxt, flags(terminated), flags(truncated),
                      flags(valid, True), jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32)[:, None], shape),
                      jnp.zeros(shape, dtype))


def assert_trees_equal(a, b, *, rtol=0, atol=0):
    import jax
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)

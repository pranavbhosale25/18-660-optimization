"""Executed against real Optax when installed; otherwise explicitly skipped."""
import pytest
import jax
import jax.numpy as jnp
optax = pytest.importorskip("optax")
from reinforce_jax import ObjectiveResult, make_direct_learner
from reinforce_jax.optim import adam
from .helpers import assert_trees_equal


def test_optax_optimizer_is_drop_in_and_follows_native_adam():
    def objective(p, context, key):
        return ObjectiveResult(-jnp.sum((p - context)**2), jnp.asarray(1))
    ox = make_direct_learner(objective, optax.adam(0.01))
    native = make_direct_learner(objective, adam(0.01))
    a, b = ox.init(jnp.ones(2)), native.init(jnp.ones(2))
    for _ in range(3):
        a, _ = ox.update(a, jnp.zeros(2), jax.random.key(0))
        b, _ = native.update(b, jnp.zeros(2), jax.random.key(0))
    assert_trees_equal(a.params, b.params, rtol=1e-10, atol=1e-10)

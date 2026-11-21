"""Host collector tests use an explicit Gymnasium-API test double, NOT fake Gymnasium."""
import numpy as np
import jax
import jax.numpy as jnp
from reinforce_jax import CategoricalPolicy
from reinforce_jax.gymnasium_adapter import GymnasiumCollector


class CountingEnv:
    def __init__(self, terminal=False):
        self.terminal = terminal
        self.closed = False
        self.t = 0
    def reset(self, *, seed=None, options=None):
        self.t = 0
        return np.asarray([0.], np.float64), {}
    def step(self, action):
        self.t += 1
        end = self.t == 2
        return np.asarray([float(self.t)]), 1., end and self.terminal, end and not self.terminal, {}
    def close(self):
        self.closed = True


def collector(envs):
    policy = CategoricalPolicy(lambda p, obs: p, 2)
    return GymnasiumCollector(envs, policy, lambda obs: np.asarray(obs), int, dtype=jnp.float64)


def test_collector_keeps_final_observation_before_manual_reset():
    env = CountingEnv()
    with collector([env]) as c:
        data, _, returns = c.collect(jnp.zeros(2), jax.random.key(0), steps=5)
        np.testing.assert_array_equal(data.observations[:, 0, 0], [0., 1., 0., 1., 0.])
        np.testing.assert_array_equal(data.next_observations[:, 0, 0], [1., 2., 1., 2., 1.])
        np.testing.assert_array_equal(data.episode_steps[:, 0], [0, 1, 0, 1, 0])
        np.testing.assert_array_equal(data.truncated[:, 0], [False, True, False, True, False])
        assert not bool(data.terminated.any())
        np.testing.assert_array_equal(returns, [2., 2.])
    assert env.closed


def test_episodic_collection_pads_and_can_resume_chunks_safely():
    with collector([CountingEnv(True), CountingEnv(False)]) as c:
        data, key, returns = c.collect(jnp.zeros(2), jax.random.key(0), steps=5, episodes=True)
        np.testing.assert_array_equal(data.valid[:, 0], [True, True, False, False, False])
        assert int(data.valid.sum()) == 4
        assert data.rewards.dtype == jnp.float64
        np.testing.assert_array_equal(returns, [2., 2.])
        chunks, _, _ = c.collect(jnp.zeros(2), key, steps=1)
        np.testing.assert_array_equal(chunks.observations, [[[0.], [0.]]])


def test_collection_cap_does_not_invent_terminal():
    with collector([CountingEnv()]) as c:
        data, _, _ = c.collect(jnp.zeros(2), jax.random.key(0), steps=1, episodes=True)
        assert bool(data.valid.all())
        assert not bool(data.terminated.any())
        assert not bool(data.truncated.any())

"""Executed when real Gymnasium is installed; otherwise explicitly skipped."""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
gym = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env
from reinforce_jax import init_mlp
from reinforce_jax.gymnasium_adapter import from_gym_spaces, GymnasiumCollector
from examples.gymnasium_train import make_env, make_discrete_context_env


@pytest.mark.parametrize("case", ["dd", "dc", "cd", "cc"])
def test_real_gymnasium_cases(case):
    envs = [make_env(case) for _ in range(2)]
    adapter = from_gym_spaces(envs[0].observation_space, envs[0].action_space)
    key = jax.random.key(0)
    params = init_mlp(key, adapter.input_dim, adapter.actor_output_dim, (8,))
    with GymnasiumCollector(envs, adapter.policy, adapter.encode_observation, adapter.decode_action) as collector:
        data, _, _ = collector.collect(params, key, steps=5)
        assert data.rewards.shape == (5, 2)
        assert bool(data.valid.all())
        assert bool(jnp.isfinite(data.rewards).all())


def test_custom_gymnasium_environment_checker():
    env = make_discrete_context_env()
    check_env(env, skip_render_check=True)
    env.close()


def test_nonzero_discrete_starts():
    adapter = from_gym_spaces(gym.spaces.Discrete(3, start=5), gym.spaces.Discrete(2, start=4))
    np.testing.assert_array_equal(adapter.encode_observation(6), [0., 1., 0.])
    assert adapter.decode_action(1) == 5

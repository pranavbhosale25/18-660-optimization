"""Small pure-JAX environments for tests and fully accelerated examples."""
from dataclasses import dataclass
from typing import NamedTuple
import jax
import jax.numpy as jnp


class LQState(NamedTuple):
    x: jax.Array
    t: jax.Array


@dataclass(frozen=True)
class LinearQuadraticEnv:
    """Finite-horizon control task. Time is observed; its end is a true terminal."""
    episode_length: int = 24
    noise: float = 0.02
    dtype: object = jnp.float32

    def reset(self, key):
        return LQState(jax.random.normal(key, (), self.dtype), jnp.asarray(0, jnp.int32))

    def observe(self, state):
        time_left = 1 - state.t.astype(self.dtype) / self.episode_length
        return jnp.stack((state.x, time_left))

    def step(self, state, action, key):
        a = jnp.ravel(action)[0]
        x = 0.9 * state.x + 0.25 * a + self.noise * jax.random.normal(key, (), self.dtype)
        reward = -(x**2 + 0.02 * a**2)
        next_state = LQState(x, state.t + 1)
        return next_state, reward, next_state.t >= self.episode_length, jnp.asarray(False)


@dataclass(frozen=True)
class ContextBandit:
    """One-step JAX task with any of the four observation/action space combinations."""
    discrete_observations: bool = True
    discrete_actions: bool = True
    dtype: object = jnp.float32

    def reset(self, key):
        if self.discrete_observations:
            return jax.random.randint(key, (), 0, 3, dtype=jnp.int32)
        return jax.random.uniform(key, (), self.dtype, minval=-1., maxval=1.)

    def observe(self, state):
        if self.discrete_observations:
            return jax.nn.one_hot(state, 3, dtype=self.dtype)
        return jnp.reshape(state, (1,))

    def step(self, state, action, key):
        del key
        target = state.astype(self.dtype) - 1 if self.discrete_observations else state
        if self.discrete_actions:
            reward = (action == (target > 0).astype(jnp.int32)).astype(self.dtype)
        else:
            reward = 1 - (jnp.ravel(action)[0] - 0.8 * target)**2
        return state, reward, jnp.asarray(True), jnp.asarray(False)

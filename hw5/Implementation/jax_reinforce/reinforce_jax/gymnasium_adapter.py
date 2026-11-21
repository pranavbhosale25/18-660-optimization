"""Host Gymnasium bridge. No host callbacks masquerading as differentiable physics.

Ordinary env.step executes in Python on the CPU. Policy inference is jitted
and vmapped, and the finished trajectory is sent to the JAX learner. Explicit
reset handling preserves the final observation for truncation bootstrapping.
"""
from dataclasses import dataclass
from collections.abc import Callable, Sequence
import numpy as np
import jax
import jax.numpy as jnp
from .types import Trajectory, Policy, ModelApply
from .models import mlp_apply
from .policies import CategoricalPolicy, GaussianPolicy


@dataclass(frozen=True)
class SpaceAdapters:
    encode_observation: Callable
    decode_action: Callable
    input_dim: int
    actor_output_dim: int
    policy: Policy


def from_gym_spaces(observation_space, action_space, *, apply: ModelApply = mlp_apply,
                    dtype=jnp.float32) -> SpaceAdapters:
    """Support Box and Discrete spaces (including nonzero Discrete.start).

    Supply custom encoding/decoding and a custom policy for structured spaces.
    """
    try:
        from gymnasium import spaces
    except ImportError as exc:
        raise ImportError('Install Gymnasium with: pip install -e ".[gym]"') from exc
    np_dtype = np.dtype(dtype)
    if isinstance(observation_space, spaces.Discrete):
        n, start = int(observation_space.n), int(observation_space.start)
        def encode(obs):
            result = np.zeros(n, np_dtype)
            result[int(obs) - start] = 1
            return result
        input_dim = n
    elif isinstance(observation_space, spaces.Box):
        input_dim = int(np.prod(observation_space.shape))
        def encode(obs):
            return np.asarray(obs, np_dtype).reshape(-1)
    else:
        raise TypeError("Default encoder supports Discrete and Box observations only")

    if isinstance(action_space, spaces.Discrete):
        policy = CategoricalPolicy(apply, int(action_space.n))
        output_dim = int(action_space.n)
        def decode(action):
            return int(action) + int(action_space.start)
    elif isinstance(action_space, spaces.Box):
        dim = int(np.prod(action_space.shape))
        low, high = action_space.low.reshape(-1), action_space.high.reshape(-1)
        if np.isfinite(low).all() and np.isfinite(high).all():
            policy = GaussianPolicy(apply, dim, tuple(map(float, low)), tuple(map(float, high)))
        elif np.isneginf(low).all() and np.isposinf(high).all():
            policy = GaussianPolicy(apply, dim)
        else:
            raise ValueError("Partially bounded Box actions require a custom policy transform")
        output_dim = 2 * dim
        def decode(action):
            return np.asarray(action, action_space.dtype).reshape(action_space.shape)
    else:
        raise TypeError("Default policies support Discrete and Box actions only")
    return SpaceAdapters(encode, decode, input_dim, output_dim, policy)


class GymnasiumCollector:
    """Batch independent Gymnasium Env instances; manually manage every reset.

    collect(..., episodes=True) collects at most one episode per lane, padded
    to `steps`. Otherwise it collects a fixed-length segment with manual resets.
    `steps` hitting its cap is a DATA boundary, not fabricated termination.
    Do not pass an auto-reset wrapper; pass ordinary Env instances instead.
    """
    def __init__(self, envs: Sequence, policy: Policy,
                 encode_observation: Callable, decode_action: Callable, *,
                 seed: int = 0, dtype=jnp.float32):
        if not envs:
            raise ValueError("At least one environment is required")
        self.envs = list(envs)
        self.policy = policy
        self.encode = encode_observation
        self.decode = decode_action
        self.dtype = np.dtype(dtype)
        self.obs = []
        self.episode_steps = np.zeros(len(envs), np.int32)
        self.running_returns = np.zeros(len(envs), np.float64)
        self.needs_reset = np.zeros(len(envs), bool)
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=seed + i)
            self.obs.append(self.encode(obs))
        self.sample_batch = jax.jit(jax.vmap(
            lambda p, o, k: policy.sample(p, o, k), in_axes=(None, 0, 0)))

    def _reset(self, i):
        obs, _ = self.envs[i].reset()
        self.obs[i] = self.encode(obs)
        self.episode_steps[i] = 0
        self.running_returns[i] = 0
        self.needs_reset[i] = False

    def collect(self, params, key, *, steps: int, episodes: bool = False):
        if steps < 1:
            raise ValueError("steps must be positive")
        B = len(self.envs)
        for i in range(B):
            if episodes or self.needs_reset[i]:
                self._reset(i)
        live = np.ones(B, bool)
        records, completed_returns = [], []
        for _ in range(steps):
            obs = np.stack(self.obs).astype(self.dtype)
            episode_steps = self.episode_steps.copy()
            key, action_key = jax.random.split(key)
            sample = jax.device_get(self.sample_batch(
                params, jnp.asarray(obs), jax.random.split(action_key, B)))
            valid = live.copy()
            rewards = np.zeros(B, self.dtype)
            terminated, truncated = np.zeros(B, bool), np.zeros(B, bool)
            next_obs = obs.copy()
            for i, env in enumerate(self.envs):
                if not live[i]:
                    continue
                raw_obs, reward, terminal, truncate, _ = env.step(self.decode(sample.action[i]))
                final_obs = self.encode(raw_obs)
                next_obs[i] = final_obs  # Capture BEFORE any reset below.
                rewards[i], terminated[i], truncated[i] = reward, terminal, truncate
                self.episode_steps[i] += 1
                self.running_returns[i] += float(reward)
                self.obs[i] = final_obs
                if terminal or truncate:
                    completed_returns.append(self.running_returns[i])
                    self.needs_reset[i] = True
                    if episodes:
                        live[i] = False
                    else:
                        self._reset(i)
            records.append(Trajectory(obs, np.asarray(sample.action), np.asarray(sample.raw_action),
                                      rewards, next_obs, terminated, truncated, valid,
                                      episode_steps, np.asarray(sample.log_prob)))
            if episodes and not live.any():
                break
        # Padding contains finite placeholders. valid=False prevents every loss/trace.
        while len(records) < steps:
            records.append(jax.tree.map(np.zeros_like, records[-1]))
        batch = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *records)
        return batch, key, np.asarray(completed_returns, np.float64)

    def close(self):
        for env in self.envs:
            env.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

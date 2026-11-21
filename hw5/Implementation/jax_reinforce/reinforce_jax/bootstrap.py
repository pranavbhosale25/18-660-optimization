"""Value/Q tail adapters. Q means on-policy expectation, never max_a Q."""
from typing import Literal
import jax
import jax.numpy as jnp
from .types import Bootstrap, CriticApply, Policy


def make_bootstrap(policy: Policy, kind: Literal["zero", "v", "q"] = "zero", *,
                   critic_apply: CriticApply | None = None,
                   q_mode: Literal["expectation", "sample"] = "expectation",
                   q_samples: int = 1) -> Bootstrap:
    if kind not in ("zero", "v", "q"):
        raise ValueError("kind must be zero, v, or q")
    if q_mode not in ("expectation", "sample") or q_samples < 1:
        raise ValueError("q_mode must be expectation/sample and q_samples must be positive")
    if kind != "zero" and critic_apply is None:
        raise ValueError("A v/q bootstrap requires critic_apply(params, obs, action)")

    if kind == "zero":
        def zero(actor_params, critic_params, obs, key):
            del actor_params, critic_params, key
            return jnp.zeros((), obs.dtype)
        return zero

    if kind == "v":
        def value(actor_params, critic_params, obs, key):
            del actor_params, key
            return critic_apply(critic_params, obs, jnp.zeros((), obs.dtype))
        return value

    def q_value(actor_params, critic_params, obs, key):
        if policy.is_discrete and q_mode == "expectation":
            actions = jnp.arange(policy.n_actions, dtype=jnp.int32)
            qs = jax.vmap(lambda a: critic_apply(critic_params, obs, a))(actions)
            return jnp.sum(policy.probs(actor_params, obs) * qs)
        keys = jax.random.split(key, q_samples)
        qs = jax.vmap(lambda k: critic_apply(
            critic_params, obs, policy.sample(actor_params, obs, k).action))(keys)
        return jnp.mean(qs)

    return q_value

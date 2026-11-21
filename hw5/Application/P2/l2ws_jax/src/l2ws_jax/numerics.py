"""Numerical primitives with a finite, explicitly chosen derivative at zero."""

import jax
import jax.numpy as jnp


@jax.custom_jvp
def euclidean_norm(vector):
    """Last-axis Euclidean norm with zero subgradient at the origin."""
    return jnp.sqrt(jnp.sum(jnp.square(vector), axis=-1))


@euclidean_norm.defjvp
def _euclidean_norm_jvp(primals, tangents):
    (vector,), (tangent,) = primals, tangents
    value = euclidean_norm(vector)
    denominator = jnp.where(value > 0, value, jnp.ones_like(value))
    derivative = jnp.sum(vector * tangent, axis=-1) / denominator
    return value, jnp.where(value > 0, derivative, jnp.zeros_like(derivative))

"""JAX-native box projection supporting arbitrary leading batch axes."""

import jax.numpy as jnp


def box_projection(value, lower, upper):
    return jnp.clip(value, lower, upper)

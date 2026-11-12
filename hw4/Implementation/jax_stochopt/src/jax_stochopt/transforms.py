"""Generic, composable transformations of optimizer proposals."""
from __future__ import annotations

import jax


def chain_update_transforms(*transforms):
    """Apply proposal transforms in order, with independent keys and tuple data.

    Each transform follows ``(candidate, UpdateContext) -> (candidate, data)``.
    Before/after contexts retain their original meanings throughout the chain;
    only the candidate changes. Each component receives a distinct subkey.
    An empty chain is the identity. A one-component chain preserves its key.
    Projection remains the solver's final operation, outside this chain.
    """
    if any(not callable(transform) for transform in transforms):
        raise TypeError("every update transform must be callable")

    def apply(candidate, context):
        if not transforms:
            return candidate, ()
        keys = (context.key,) if len(transforms) == 1 else jax.random.split(context.key, len(transforms))
        data = []
        for transform, key in zip(transforms, keys):
            candidate, auxiliary = transform(candidate, context._replace(key=key))
            data.append(auxiliary)
        return candidate, tuple(data)

    return apply

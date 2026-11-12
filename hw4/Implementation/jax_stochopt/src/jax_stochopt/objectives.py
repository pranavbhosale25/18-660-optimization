"""Static trees for composing differentiable, per-sample objectives.

A leaf has signature ``leaf(params, sample, args)``. An internal ``Reduce``
receives a tuple containing its children's outputs. Internal values may have
any shape that their parent reducer understands; the root must return one
real floating-point scalar. Compilation builds ordinary Python closures once,
so JAX sees one inlinable numerical function, not a dynamic tree traversal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax.numpy as jnp


Objective = Callable[[Any, Any, Any], Any]
Reducer = Callable[[tuple[Any, ...]], Any]


@dataclass(frozen=True)
class Reduce:
    """An internal objective-tree node.

    ``reducer`` is called as ``reducer(tuple(child_outputs))``. It can, for
    example, sum scalar losses, combine vector residuals, or apply a nonlinear
    risk transform. Each child receives the same parameters, random sample,
    and auxiliary arguments. Put separate random draws in the sampler's
    returned PyTree when children require independent noise.

    Python tree topology and reducer callables are static. Put differentiable
    problem inputs in ``args``; do not mutate the tree after compilation.
    """

    reducer: Reducer
    children: Sequence[Objective | "Reduce"]
    name: str = ""

    def __post_init__(self) -> None:
        if not callable(self.reducer):
            raise TypeError("Reduce.reducer must be callable.")
        if not isinstance(self.children, (tuple, list)):
            raise TypeError("Reduce.children must be a nonempty tuple or list.")
        if not self.children:
            raise ValueError("Reduce.children must not be empty.")
        if not isinstance(self.name, str):
            raise TypeError("Reduce.name must be a string.")
        object.__setattr__(self, "children", tuple(self.children))


def compile_objective(tree: Objective | Reduce) -> Objective:
    """Build a JIT/vmap/autodiff-compatible callable from an objective tree.

    Plain callables are valid one-leaf trees. Root shape and dtype validation
    takes place when the callable is evaluated, including during JAX tracing;
    it does not require transferring numerical results to the host. All
    callables must themselves use JAX-compatible numerical operations.

    Composition is *within each sample*. The solver subsequently averages the
    resulting scalar across samples. Consequently, a nonlinear root reducer
    computes ``E[reducer(children(sample))]``, which is generally different
    from ``reducer(E[children(sample)])``.
    """

    def build(node: Objective | Reduce, path: str) -> Objective:
        if isinstance(node, Reduce):
            label = node.name or path
            children = tuple(
                build(child, f"{label}.children[{i}]")
                for i, child in enumerate(node.children)
            )
            reducer = node.reducer

            def composed(params: Any, sample: Any, args: Any) -> Any:
                return reducer(tuple(child(params, sample, args) for child in children))

            return composed
        if not callable(node):
            raise TypeError(f"Objective-tree node {path!r} must be callable or Reduce.")
        return node

    evaluate = build(tree, "root")

    def objective(params: Any, sample: Any, args: Any = None) -> Any:
        value = jnp.asarray(evaluate(params, sample, args))
        if value.ndim != 0:
            raise ValueError(
                f"Objective root must return a scalar; received shape {value.shape}."
            )
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError(
                "Objective root must return a real floating-point scalar; "
                f"received dtype {value.dtype}."
            )
        return value

    return objective


def sum_reduce(values: tuple[Any, ...]) -> Any:
    """Sum same-shaped child values elementwise, retaining their common shape."""

    return jnp.sum(jnp.stack(values, axis=0), axis=0)


def mean_reduce(values: tuple[Any, ...]) -> Any:
    """Average same-shaped child values elementwise."""

    return jnp.mean(jnp.stack(values, axis=0), axis=0)


def product_reduce(values: tuple[Any, ...]) -> Any:
    """Multiply same-shaped child values elementwise."""

    return jnp.prod(jnp.stack(values, axis=0), axis=0)


def weighted_sum(weights: Any) -> Reducer:
    """Return a reducer with one real scalar weight per child.

    Child values must share a shape. Weights are cast to the stacked values'
    dtype at evaluation time, so switching a floating-point objective between
    float32 and float64 also switches arithmetic in this reducer. Weights can
    themselves be traced differentiable inputs if this factory is constructed
    inside the function being differentiated.
    """

    weights_array = jnp.asarray(weights)
    if weights_array.ndim != 1:
        raise ValueError("weighted_sum weights must be a one-dimensional sequence.")
    if jnp.issubdtype(weights_array.dtype, jnp.complexfloating):
        raise TypeError("weighted_sum weights must be real.")
    count = weights_array.shape[0]

    def reduce(values: tuple[Any, ...]) -> Any:
        if len(values) != count:
            raise ValueError(
                f"weighted_sum received {len(values)} children but {count} weights."
            )
        stacked = jnp.stack(values, axis=0)
        weight_values = jnp.asarray(weights, dtype=stacked.dtype)
        return jnp.tensordot(weight_values, stacked, axes=((0,), (0,)))

    return reduce


__all__ = [
    "Objective",
    "Reduce",
    "Reducer",
    "compile_objective",
    "mean_reduce",
    "product_reduce",
    "sum_reduce",
    "weighted_sum",
]

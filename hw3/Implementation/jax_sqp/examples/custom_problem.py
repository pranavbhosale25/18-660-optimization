"""Run after installing this package: python examples/custom_problem.py."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"  # Remove for auto/GPU; set before JAX imports.
from jax_sqp import SQP, STATUS_MESSAGES, Options, configure

rt = configure(platform="cpu", precision=64)
import jax
import jax.numpy as jnp

# Minimize distance to a target, on an affine hyperplane and inside a ball.
# Constraint convention: equality == 0; inequality <= 0.
solver = SQP(
    fun=lambda x, p: 0.5 * jnp.sum((x - p["target"]) ** 2),
    eq=lambda x, p: jnp.array([jnp.sum(x) - p["total"]]),
    ineq=lambda x, p: jnp.array([jnp.dot(x, x) - p["radius"] ** 2]),
    lower=0.0,
    options=Options.for_dtype(rt.precision),
)
params = {
    "target": jnp.array([2.0, 1.0, 0.0], dtype=rt.dtype),
    "total": jnp.array(1.0, dtype=rt.dtype),
    "radius": jnp.array(0.8, dtype=rt.dtype),
}
r = solver.solve(jnp.ones(3, rt.dtype) / 3, params)
jax.block_until_ready(r)
print(STATUS_MESSAGES[int(r.status)])
print("x =", r.x, "objective =", r.objective, "KKT =", r.kkt)
assert bool(r.success)

# Batch *different parameter pytrees*: leading dimension indexes problems.
parameters = jax.tree.map(lambda a: jnp.stack([a, a]), params)
parameters["target"] = jnp.array([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]], dtype=rt.dtype)
starts = jnp.ones((2, 3), rt.dtype) / 3
batch_solve = jax.jit(jax.vmap(solver.run, in_axes=(0, 0)))
batch = batch_solve(starts, parameters)
jax.block_until_ready(batch)
print("Batched solutions:", batch.x)
assert bool(jnp.all(batch.success))

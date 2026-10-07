import trsr1

trsr1.configure_runtime(platform="cpu", precision=64)

import jax.numpy as jnp


def quadratic(x, matrix, linear):
    return 0.5 * x @ matrix @ x + linear @ x


matrices = jnp.asarray(
    [
        [[2.0, 0.0], [0.0, 5.0]],
        [[4.0, 1.0], [1.0, 3.0]],
    ]
)
linear = jnp.asarray([[1.0, -2.0], [-3.0, 1.0]])
starts = jnp.zeros_like(linear)

solver = trsr1.make_batched_solver(
    quadratic,
    trsr1.SR1Options(
        initial_radius=10.0,
        gradient_tolerance=1.0e-10,
        store_history=False,
    ),
    arg_in_axes=(0, 0),
)
result = solver(starts, matrices, linear)
print(result.x)
print(result.success)

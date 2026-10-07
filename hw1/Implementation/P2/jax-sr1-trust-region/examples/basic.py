import trsr1

trsr1.configure_runtime(platform="cpu", precision=64)

import jax.numpy as jnp


def rosenbrock(x):
    return jnp.sum(
        100.0 * (x[1:] - x[:-1] ** 2) ** 2
        + (1.0 - x[:-1]) ** 2
    )


options = trsr1.SR1Options(
    max_iterations=300,
    gradient_tolerance=1.0e-8,
)
solver = trsr1.make_solver(rosenbrock, options)
result = solver(jnp.asarray([-1.2, 1.0]))

print("x:", result.x)
print("f:", result.fun)
print("gradient norm:", result.grad_norm)
print("status:", trsr1.status_message(result.status))

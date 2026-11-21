"""User-supplied exact derivatives instead of likelihood-ratio or autodiff gradients.

For a ~ Normal(mean, exp(log_std)^2), E[-(a-target)^2] is available in closed form.
python -m examples.analytic_gradient
"""
import jax
import jax.numpy as jnp
from reinforce_jax import configure, ObjectiveResult, make_direct_learner, adam


def expected_return(params, target, key):
    del key
    value = -jnp.sum((params["mean"] - target)**2 + jnp.exp(2 * params["log_std"]))
    return ObjectiveResult(value, jnp.asarray(1, jnp.int32))


def analytic_gradient(params, target, key):
    del key
    return {"mean": -2 * (params["mean"] - target),
            "log_std": -2 * jnp.exp(2 * params["log_std"])}


def run():
    dtype = configure(dtype="float64", platform="cpu")
    params = {"mean": jnp.asarray([0.], dtype), "log_std": jnp.asarray([0.], dtype)}
    target, key = jnp.asarray([0.7], dtype), jax.random.key(0)
    learner = make_direct_learner(expected_return, adam(0.03), analytic_grad=analytic_gradient)
    state = learner.init(params)
    before = float(expected_return(params, target, key).value)
    for _ in range(200):
        state, _ = learner.update(state, target, key)
    after = float(expected_return(state.params, target, key).value)
    print(f"exact expected return: {before:.6f} -> {after:.6f}")
    print("parameters:", state.params)
    return before, after


if __name__ == "__main__":
    run()

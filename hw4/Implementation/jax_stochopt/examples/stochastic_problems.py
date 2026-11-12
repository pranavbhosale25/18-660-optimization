"""Dataset-free stochastic optimization, projections, and hypergradients.

Run after ``pip install -e .``::

    python examples/stochastic_problems.py --dtype float64 --backend cpu
    python examples/stochastic_problems.py --dtype float32 --backend gpu
    python examples/stochastic_problems.py --learning-rate 0.1 --batch-size 32

Every sample is generated from a JAX PRNG key. No dataset is downloaded or
constructed. JSON output includes analytic reference solutions and a shared-key
finite-difference check of differentiation through the optimization trajectory.
The optional learning-rate and batch-size overrides apply only to the two
assigned SGD experiments; the other optimizer examples retain their built-in
settings.
GPU execution requires a suitable JAX GPU installation and an available device.
"""

from __future__ import annotations

import argparse
import json
import math

import jax
import jax.numpy as jnp

from jax_stochopt import (
    Adam,
    Muon,
    NewtonCG,
    Reduce,
    SGD,
    make_solver,
    masked_mean,
    sum_reduce,
    tree_norm,
)
from jax_stochopt.projections import box, simplex


def normal_forcing(key, args):
    """One fixed-shape realization of a random external force."""
    return jax.random.normal(key, args["target"].shape, dtype=args["target"].dtype)


def quadratic_energy(params, sample, args):
    """Elastic energy with random forcing; its expected minimizer is target.

    E[0.5 * (x-target)^T H (x-target) - sigma * sample^T (x-target)]
        = 0.5 * (x-target)^T H (x-target).
    """
    displacement = params - args["target"]
    return (
        0.5 * jnp.sum(args["curvature"] * displacement**2)
        - args["noise_scale"] * jnp.sum(sample * displacement)
    )


def expected_quadratic_gap(params, args):
    return 0.5 * jnp.sum(args["curvature"] * (params - args["target"]) ** 2)


def forcing_summary(samples, mask, args):
    """An example data statistic usable by both kinds of schedule."""
    sample_axes = tuple(range(1, samples.ndim))
    mean_component_energy = jnp.mean(samples**2, axis=sample_axes)
    return {
        "forcing_energy": args["noise_scale"] ** 2
        * masked_mean(mean_component_energy, mask)
    }


def scalar(value):
    return float(jax.device_get(value))


def vector(value):
    return jax.device_get(value).tolist()


def random_forcing_quadratic(
    key,
    dtype,
    backend,
    steps,
    learning_rate_override=None,
    batch_size_override=None,
):
    """Compare rules on a stochastic energy with a known exact minimizer.

    Optional constant overrides affect only the SGD branch.
    """
    args = {
        "target": jnp.asarray([0.8, -0.6, 0.3, 1.1], dtype=dtype),
        "curvature": jnp.asarray([0.7, 1.3, 2.2, 4.0], dtype=dtype),
        "noise_scale": jnp.asarray(0.35, dtype=dtype),
    }
    initial = jnp.zeros_like(args["target"])
    summary_init = {"forcing_energy": args["noise_scale"] ** 2}

    def cooldown(context):
        # Explicitly promote integer counters before continuous arithmetic:
        # int32 / a weak Python scalar can otherwise use float32 under x64.
        fraction = context.step.astype(dtype) / max(steps - 1, 1)
        return 0.04 + 0.96 * (1 - fraction) ** 2

    def iteration_batch(context):
        return 16 + (112 * context.step) // max(steps - 1, 1)

    if learning_rate_override is None:
        sgd_learning_rate = (
            lambda context: jnp.asarray(0.12, dtype=dtype) * cooldown(context)
        )
    else:
        sgd_learning_rate = learning_rate_override
    sgd_batch_size = (
        iteration_batch if batch_size_override is None else batch_size_override
    )

    # Both adaptive rules below use the iteration, sampled data, and momentum.
    # The batch rule sees PREVIOUS statistics/state; the LR rule sees CURRENT
    # statistics and the first moment AFTER incorporating the current gradient.
    def adam_learning_rate(context):
        first_moment_size = tree_norm(context.optimizer_state.m)
        forcing_energy = context.stats.data["forcing_energy"]
        return (
            jnp.asarray(0.055, dtype=dtype)
            * cooldown(context)
            / (1 + 0.08 * first_moment_size + 0.1 * forcing_energy)
        )

    def adam_batch_size(context):
        first_moment_size = tree_norm(context.optimizer_state.m)
        relative_forcing = context.stats.data["forcing_energy"] / (
            context.args["noise_scale"] ** 2 + jnp.asarray(1e-8, dtype=dtype)
        )
        return (
            16
            + (90 * context.step) // max(steps - 1, 1)
            + 8 * relative_forcing
            + 16 / (1 + first_moment_size)
        )

    rules = {
        "sgd": (
            SGD(momentum=0.6),
            sgd_learning_rate,
            sgd_batch_size,
        ),
        "adam": (Adam(), adam_learning_rate, adam_batch_size),
        "newton_cg": (
            NewtonCG(damping=0.05, maxiter=8, tol=1e-6),
            lambda context: jnp.asarray(0.45, dtype=dtype) * cooldown(context),
            iteration_batch,
        ),
    }
    results = {}
    for name, (optimizer, lr, batch_size) in rules.items():
        exact_sgd_batch = name == "sgd" and batch_size_override is not None
        solver = make_solver(
            quadratic_energy,
            normal_forcing,
            optimizer=optimizer,
            learning_rate=lr,
            batch_size=batch_size,
            max_batch_size=None if exact_sgd_batch else 128,
            batch_buckets=None if exact_sgd_batch else (16, 32, 64, 128),
            steps=steps,
            summary_fn=forcing_summary,
            summary_init=summary_init,
            estimate_gradient_variance=True,
            backend=backend,
        )
        # Deliberately share a key across rules: overlapping sample indices use
        # common random numbers, even if their adaptive batch sizes differ.
        result = solver.run(initial, key, args)
        entry = {
            "solution": vector(result.params),
            "solution_l2_error": scalar(tree_norm(result.params - args["target"])),
            "expected_objective_gap": scalar(expected_quadratic_gap(result.params, args)),
            "last_learning_rate": scalar(result.history.learning_rate[-1]),
            "last_batch_size": int(result.history.stats.batch_size[-1]),
            "effective_sample_count": int(jnp.sum(result.history.stats.batch_size)),
            "compiled_lane_count": int(jnp.sum(result.history.stats.capacity)),
            "last_gradient_variance_estimate": scalar(result.state.stats.grad_variance),
        }
        if name == "newton_cg":
            entry["last_linear_solve_residual"] = scalar(
                result.state.optimizer_state.residual_norm
            )
        results[name] = entry
    return {
        "interpretation": "Elastic equilibrium under independent mean-zero random forces",
        "exact_solution": vector(args["target"]),
        "true_individual_gradient_variance_trace": scalar(
            args["noise_scale"] ** 2 * args["target"].size
        ),
        "optimizers": results,
    }


def simplex_resource_allocation(
    key,
    dtype,
    backend,
    steps,
    learning_rate_override=None,
    batch_size_override=None,
):
    """Allocate one unit of a resource under random marginal benefits.

    min_{x >= 0, sum(x) = 1} E[0.5 ||x||^2 - (a + sigma*z)^T x].
    The exact minimizer is the Euclidean projection of a onto the simplex.
    For the a below, the threshold is 0.1 and the last coordinate is inactive.
    """
    args = {
        "target": jnp.asarray([0.70, 0.45, 0.15, -0.30], dtype=dtype),
        "noise_scale": jnp.asarray(0.15, dtype=dtype),
    }
    exact = jnp.asarray([0.60, 0.35, 0.05, 0.0], dtype=dtype)

    def congestion_energy(params, sample, args):
        del sample, args
        return 0.5 * jnp.sum(params**2)

    def random_benefit(params, sample, args):
        return -jnp.sum((args["target"] + args["noise_scale"] * sample) * params)

    # Non-leaf nodes reduce their children's values for each realization;
    # leaves all see the same realization and can extract its relevant parts.
    objective = Reduce(sum_reduce, (congestion_energy, random_benefit))
    if learning_rate_override is None:
        learning_rate = lambda context: 0.01 + 0.29 * (
            1 - context.step.astype(dtype) / max(steps - 1, 1)
        ) ** 2
    else:
        learning_rate = learning_rate_override
    batch_size = (
        (lambda context: 32 + (96 * context.step) // max(steps - 1, 1))
        if batch_size_override is None
        else batch_size_override
    )
    exact_batch = batch_size_override is not None

    solver = make_solver(
        objective,
        normal_forcing,
        optimizer=SGD(),
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_batch_size=None if exact_batch else 128,
        batch_buckets=None if exact_batch else (32, 64, 128),
        steps=steps,
        projection=simplex(total=1.0),
        record_params=True,
        backend=backend,
    )
    result = solver.run(jnp.full((4,), 0.25, dtype=dtype), key, args)
    actual_energy = 0.5 * jnp.sum((result.params - args["target"]) ** 2)
    optimal_energy = 0.5 * jnp.sum((exact - args["target"]) ** 2)
    return {
        "interpretation": "Resource allocation with quadratic congestion and random benefits",
        "exact_solution": vector(exact),
        "solution": vector(result.params),
        "solution_l2_error": scalar(tree_norm(result.params - exact)),
        "expected_objective_gap": scalar(actual_energy - optimal_energy),
        "sum_of_allocations": scalar(jnp.sum(result.params)),
        "minimum_allocation": scalar(jnp.min(result.params)),
        "maximum_mass_error_across_iterates": scalar(
            jnp.max(jnp.abs(jnp.sum(result.history.params, axis=-1) - 1))
        ),
        "minimum_allocation_across_iterates": scalar(jnp.min(result.history.params)),
    }


def stochastic_matrix_design(key, dtype, backend, steps):
    """Use Muon on a matrix-valued design variable under random loading.

    A quadratic restoring energy is perturbed by random matrix forces. The
    minimizer is inside the box, avoiding an assumption that an arbitrary
    matrix preconditioner plus Euclidean projection has the same constrained
    fixed points as ordinary projected gradient descent.
    """
    args = {
        "target": jnp.asarray(
            [[0.60, -0.40, 0.30, 0.10], [-0.50, 0.20, 0.45, -0.20], [0.20, -0.10, 0.50, -0.55]],
            dtype=dtype,
        ),
        "curvature": jnp.asarray(1.0, dtype=dtype),
        "noise_scale": jnp.asarray(0.08, dtype=dtype),
    }
    solver = make_solver(
        quadratic_energy,
        normal_forcing,
        optimizer=Muon(momentum=0.9, ns_steps=5),
        learning_rate=lambda context: 0.00015
        + 0.045 * (1 - context.step.astype(dtype) / max(steps - 1, 1)) ** 2,
        batch_size=lambda context: 32 + (96 * context.step) // max(steps - 1, 1),
        max_batch_size=128,
        batch_buckets=(32, 64, 128),
        steps=steps,
        projection=box(-1.0, 1.0),
        backend=backend,
    )
    result = solver.run(jnp.zeros_like(args["target"]), key, args)
    return {
        "interpretation": "Matrix design energy with random forcing; no learned model",
        "exact_solution": vector(args["target"]),
        "solution": vector(result.params),
        "solution_frobenius_error": scalar(tree_norm(result.params - args["target"])),
        "expected_objective_gap": scalar(expected_quadratic_gap(result.params, args)),
        "largest_absolute_entry": scalar(jnp.max(jnp.abs(result.params))),
    }


def trajectory_hypergradient_and_vmap(key, dtype, backend):
    """Differentiate the final objective with respect to a scheduled base LR.

    The minibatch size and key are held fixed in this check. Differentiating
    rounded integer batch choices would not account for transitions between
    those discrete choices. The reported derivative is pathwise for this
    realized random trajectory, not an automatically unbiased derivative of
    an expectation over arbitrarily parameter-dependent random distributions.
    """
    args = {
        "target": jnp.asarray([0.8, -0.6, 0.3, 1.1], dtype=dtype),
        "curvature": jnp.asarray([0.7, 1.3, 2.2, 4.0], dtype=dtype),
        "noise_scale": jnp.asarray(0.25, dtype=dtype),
    }
    initial = jnp.zeros_like(args["target"])
    solver = make_solver(
        quadratic_energy,
        normal_forcing,
        optimizer=SGD(momentum=0.5),
        learning_rate=lambda context: context.args["base_lr"]
        / (1 + 0.01 * context.step.astype(dtype)),
        batch_size=32,
        steps=25,
        checkpoint=True,
        backend=backend,
    )

    def final_expected_energy(base_lr):
        result = solver.run(initial, key, dict(args, base_lr=base_lr))
        return expected_quadratic_gap(result.params, args)

    base_lr = jnp.asarray(0.03, dtype=dtype)
    derivative = jax.grad(final_expected_energy)(base_lr)
    difference_step = jnp.asarray(1e-4 if dtype == jnp.float32 else 1e-5, dtype=dtype)
    finite_difference = (
        final_expected_energy(base_lr + difference_step)
        - final_expected_energy(base_lr - difference_step)
    ) / (2 * difference_step)

    independent_keys = jax.random.split(key, 4)
    run_args = dict(args, base_lr=base_lr)
    batched_solutions = jax.vmap(lambda run_key: solver.run(initial, run_key, run_args).params)(
        independent_keys
    )
    return {
        "base_learning_rate": scalar(base_lr),
        "final_expected_energy": scalar(final_expected_energy(base_lr)),
        "autodiff_d_final_energy_d_base_lr": scalar(derivative),
        "shared_key_central_finite_difference": scalar(finite_difference),
        "relative_derivative_discrepancy": scalar(
            jnp.abs(derivative - finite_difference) / jnp.maximum(jnp.abs(derivative), 1e-12)
        ),
        "vmap_independent_solution_shape": list(batched_solutions.shape),
        "vmap_independent_solutions": vector(batched_solutions),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        metavar="RATE",
        help=(
            "constant learning-rate override for the randomly forced quadratic's "
            "SGD branch and simplex resource allocation; omit to use each built-in "
            "schedule"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="SIZE",
        help=(
            "constant batch-size override for the randomly forced quadratic's SGD "
            "branch and simplex resource allocation; omit to use each built-in schedule"
        ),
    )
    options = parser.parse_args()
    if options.steps < 2:
        parser.error("--steps must be at least 2")
    if options.learning_rate is not None and (
        not math.isfinite(options.learning_rate) or options.learning_rate < 0
    ):
        parser.error("--learning-rate must be finite and nonnegative")
    if options.batch_size is not None and not 1 <= options.batch_size < 2**31:
        parser.error("--batch-size must be a positive 32-bit integer")

    # Configure precision before creating arrays or compiling a solver.
    jax.config.update("jax_enable_x64", options.dtype == "float64")
    dtype = jnp.float64 if options.dtype == "float64" else jnp.float32
    try:
        devices = jax.devices(options.backend)
    except RuntimeError as error:
        parser.error(str(error))
    with jax.default_device(devices[0]):
        keys = jax.random.split(jax.random.key(options.seed, impl="threefry2x32"), 4)
        runtime = {
            "jax_version": jax.__version__,
            "dtype": options.dtype,
            "backend": options.backend,
            "devices": [str(device) for device in devices],
            "steps_per_problem": options.steps,
            "seed": options.seed,
        }
        if options.learning_rate is not None:
            runtime["sgd_learning_rate_override"] = options.learning_rate
        if options.batch_size is not None:
            runtime["sgd_batch_size_override"] = options.batch_size
        report = {
            "runtime": runtime,
            "random_forcing_quadratic": random_forcing_quadratic(
                keys[0],
                dtype,
                options.backend,
                options.steps,
                options.learning_rate,
                options.batch_size,
            ),
            "simplex_resource_allocation": simplex_resource_allocation(
                keys[1],
                dtype,
                options.backend,
                options.steps,
                options.learning_rate,
                options.batch_size,
            ),
            "stochastic_matrix_design": stochastic_matrix_design(
                keys[2], dtype, options.backend, options.steps
            ),
            "trajectory_hypergradient_and_vmap": trajectory_hypergradient_and_vmap(
                keys[3], dtype, options.backend
            ),
        }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

"""Examples using the supplied energy- and score-based Langevin factories.

Run after ``python -m pip install -e .``::

    python examples/langevin_dynamics.py --dtype float64 --backend cpu
    python examples/langevin_dynamics.py --dtype float32 --backend gpu

GPU execution requires a compatible JAX installation and an available device.
"""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp

from jax_stochopt import Reduce, sum_reduce
from jax_stochopt.extensions import make_energy_langevin, make_score_langevin
from jax_stochopt.projections import box


def scalar(value):
    return float(jax.device_get(value))


def vector(value):
    return jax.device_get(value).tolist()


def harmonic_energy(params, sample, args):
    """A scalar energy supplied to the energy-based solver."""
    del sample
    return 0.5 * jnp.sum(args["curvature"] * (params - args["mean"]) ** 2)


def harmonic_score(params, sample, args, temperature):
    """A score field supplied to the score-based solver."""
    del sample
    return -args["curvature"] * (params - args["mean"]) / temperature


def force_sampler(key, args):
    """Generate one random force realization."""
    return args["force_scale"] * jax.random.normal(
        key, args["mean"].shape, dtype=args["mean"].dtype
    )


def random_work(params, sample, args):
    return -jnp.sum(sample * (params - args["mean"]))


def forced_harmonic_score(params, sample, args, temperature):
    return (-args["curvature"] * (params - args["mean"]) + sample) / temperature


def energy_and_score_comparison(key, dtype, backend):
    """Run equivalent supplied energy and score descriptions with one key."""
    steps = 64
    args = {
        "mean": jnp.asarray([0.8, -0.6, 0.3], dtype=dtype),
        "curvature": jnp.asarray([0.7, 1.3, 2.2], dtype=dtype),
        "force_scale": jnp.asarray(0.2, dtype=dtype),
        "initial_temperature": jnp.asarray(0.75, dtype=dtype),
    }
    energy = Reduce(sum_reduce, (harmonic_energy, random_work))

    def temperature(context):
        fraction = context.step.astype(dtype) / (steps - 1)
        return context.args["initial_temperature"] * (1 - 0.6 * fraction)

    options = dict(
        steps=steps,
        learning_rate=jnp.asarray(0.04, dtype=dtype),
        temperature=temperature,
        batch_size=lambda context: 4 + context.step // 4,
        batch_buckets=(4, 8, 16, 32),
        record_params=True,
        backend=backend,
    )
    energy_solver = make_energy_langevin(energy, force_sampler, **options)
    score_solver = make_score_langevin(
        forced_harmonic_score, force_sampler, **options
    )
    initial = jnp.zeros_like(args["mean"])
    energy_run = energy_solver.run(initial, key, args)
    score_run = score_solver.run(initial, key, args)
    return {
        "energy_final_params": vector(energy_run.params),
        "score_final_params": vector(score_run.params),
        "maximum_shared_key_path_difference": scalar(
            jnp.max(jnp.abs(energy_run.history.params - score_run.history.params))
        ),
        "initial_temperature": scalar(
            energy_run.history.update_data.temperature[0]
        ),
        "last_temperature": scalar(
            energy_run.history.update_data.temperature[-1]
        ),
        "last_effective_batch_size": int(
            energy_run.history.stats.batch_size[-1]
        ),
    }


def gaussian_ensemble(key, dtype, backend, chains):
    """Run independent score-based chains and report empirical moments."""
    args = {
        "mean": jnp.asarray(0.7, dtype=dtype),
        "curvature": jnp.asarray(1.4, dtype=dtype),
    }
    solver = make_score_langevin(
        harmonic_score,
        steps=18,
        learning_rate=jnp.asarray(0.07, dtype=dtype),
        temperature=jnp.asarray(0.7, dtype=dtype),
        backend=backend,
    )
    keys = jax.random.split(key, chains)
    initial = jnp.asarray(-1.8, dtype=dtype)
    finals = jax.jit(
        jax.vmap(lambda run_key: solver.run(initial, run_key, args).params)
    )(keys)
    return {
        "independent_chains": chains,
        "observed_mean": scalar(jnp.mean(finals)),
        "observed_variance": scalar(jnp.var(finals, ddof=1)),
    }


def annealed_double_well(key, dtype, backend):
    """Run an annealed energy-based example with a box projection."""
    steps = 400

    def energy(params, sample, args):
        del sample, args
        return 0.25 * jnp.sum((params**2 - 1) ** 2)

    def temperature(context):
        fraction = context.step.astype(dtype) / (steps - 1)
        first = jnp.asarray(0.45, dtype=dtype)
        last = jnp.asarray(0.015, dtype=dtype)
        return first * (last / first) ** fraction

    solver = make_energy_langevin(
        energy,
        steps=steps,
        learning_rate=jnp.asarray(0.025, dtype=dtype),
        temperature=temperature,
        projection=box(-2.0, 2.0),
        record_params=True,
        backend=backend,
    )
    initial = jnp.asarray(
        [-1.8, -1.3, -0.6, -0.15, 0.15, 0.6, 1.3, 1.8], dtype=dtype
    )
    result = solver.run(initial, key)
    return {
        "initial_params": vector(initial),
        "final_params": vector(result.params),
        "final_total_energy": scalar(energy(result.params, (), None)),
        "maximum_absolute_coordinate": scalar(jnp.max(jnp.abs(result.params))),
        "initial_temperature": scalar(result.history.update_data.temperature[0]),
        "last_temperature": scalar(result.history.update_data.temperature[-1]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--chains", type=int, default=4096)
    options = parser.parse_args()
    if options.chains < 2:
        parser.error("--chains must be at least 2")

    jax.config.update("jax_enable_x64", options.dtype == "float64")
    dtype = getattr(jnp, options.dtype)
    try:
        device = jax.devices(options.backend)[0]
    except RuntimeError as error:
        parser.error(str(error))

    with jax.default_device(device):
        keys = jax.random.split(
            jax.random.key(options.seed, impl="threefry2x32"), 3
        )
        report = {
            "runtime": {
                "jax_version": jax.__version__,
                "dtype": options.dtype,
                "device": str(device),
                "seed": options.seed,
            },
            "energy_and_score_comparison": energy_and_score_comparison(
                keys[0], dtype, options.backend
            ),
            "gaussian_ensemble": gaussian_ensemble(
                keys[1], dtype, options.backend, options.chains
            ),
            "annealed_double_well": annealed_double_well(
                keys[2], dtype, options.backend
            ),
        }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

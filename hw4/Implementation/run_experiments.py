"""Reproduce the assigned ten-seed sweeps and Langevin figures.

Run from the submission root with the implementation environment:
    .venv-implementation/bin/python Implementation/run_experiments.py
"""
from pathlib import Path
import csv
import json
import sys

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "jax_stochopt" / "examples"))
from stochastic_problems import normal_forcing, quadratic_energy
from langevin_dynamics import (
    harmonic_energy, harmonic_score, random_work, force_sampler, forced_harmonic_score,
)
from jax_stochopt import SGD, Reduce, sum_reduce, make_solver
from jax_stochopt.extensions import make_energy_langevin, make_score_langevin
from jax_stochopt.projections import simplex

OUT = Path(__file__).parent / "results"
SEEDS = tuple(range(10))
RATES = (0.01, 0.1, 0.3, 0.9)
BATCHES = (1, 16, 64)
STEPS = 400


def stable_sd(values):
    """Avoid squaring enormous gaps in the deliberately unstable run."""
    scale = np.maximum(np.max(np.abs(values), axis=0), np.finfo(float).tiny)
    return (values / scale).std(axis=0, ddof=1) * scale


def save_figure(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def resource_energy(params, sample, args):
    return 0.5 * jnp.sum(params**2) - jnp.sum(
        (args["target"] + args["noise_scale"] * sample) * params
    )


def sgd_sweeps():
    """Reuse the supplied problems, including quadratic momentum 0.6."""
    rows, arrays = [], {}
    for problem in ("quadratic", "simplex"):
        if problem == "quadratic":
            args = dict(target=jnp.array([0.8, -0.6, 0.3, 1.1]),
                        curvature=jnp.array([0.7, 1.3, 2.2, 4.0]),
                        noise_scale=jnp.array(0.35))
            initial, exact = jnp.zeros(4), args["target"]
            objective, projection, momentum = quadratic_energy, None, 0.6
        else:
            args = dict(target=jnp.array([0.70, 0.45, 0.15, -0.30]),
                        noise_scale=jnp.array(0.15))
            initial, exact = jnp.full(4, 0.25), jnp.array([0.6, 0.35, 0.05, 0.0])
            objective, projection, momentum = resource_energy, simplex(), 0.0

        fig, axes = plt.subplots(2, len(RATES), figsize=(17, 7), squeeze=False)
        for col, rate in enumerate(RATES):
            for batch in BATCHES:
                solver = make_solver(
                    objective, normal_forcing, steps=STEPS, optimizer=SGD(momentum),
                    learning_rate=rate, batch_size=batch, projection=projection,
                    record_params=True, backend="cpu",
                )
                keys = jnp.stack([jax.random.key(seed) for seed in SEEDS])
                result = jax.jit(jax.vmap(lambda key: solver.run(initial, key, args)))(keys)
                paths = np.asarray(result.history.params)
                if problem == "quadratic":
                    gaps = 0.5 * np.sum(np.asarray(args["curvature"]) * (paths - exact)**2, axis=-1)
                else:
                    target = np.asarray(args["target"])
                    gaps = (0.5 * np.sum((paths - target)**2, axis=-1)
                            - 0.5 * np.sum((np.asarray(exact) - target)**2))
                    assert np.min(paths) >= -1e-12
                    assert np.max(np.abs(paths.sum(axis=-1) - 1)) < 1e-12
                assert np.all(np.isfinite(gaps))
                tag = f"{problem}_lr{rate}_b{batch}"
                arrays[tag] = gaps
                mean, sd = gaps.mean(axis=0), stable_sd(gaps)
                iterations = np.arange(1, STEPS + 1)
                for row, horizontal in enumerate((iterations, iterations * batch)):
                    ax = axes[row, col]
                    line, = ax.plot(horizontal, mean, label=f"batch {batch}")
                    indices = np.unique(np.geomspace(1, STEPS, 14).astype(int)) - 1
                    # Truncate only the displayed lower error bar for the log axis.
                    ax.errorbar(horizontal[indices], mean[indices],
                                yerr=[np.minimum(sd[indices], 0.99 * mean[indices]), sd[indices]],
                                fmt="none", color=line.get_color(), alpha=0.55, capsize=2)
                    ax.set_yscale("log")
                    ax.set_xlabel("Iterations" if row == 0 else "Samples used (batch × iterations)")
                    ax.set_ylabel("Expected objective gap")
                    ax.grid(alpha=0.2)
                common_index = 384 // batch - 1
                rows.append(dict(problem=problem, learning_rate=rate, batch_size=batch,
                                 iterations=STEPS, samples=STEPS * batch,
                                 final_mean=float(mean[-1]), final_sd=float(sd[-1]),
                                 gap_at_384_samples=float(mean[common_index]),
                                 samples_at_comparison=int((common_index + 1) * batch)))
                print(tag, "final mean gap", mean[-1], flush=True)
            axes[0, col].set_title(f"Learning rate {rate}")
            axes[0, col].legend()
        fig.suptitle(f"{problem.capitalize()}: ten independent seeds; error bars = ±1 run SD", fontsize=14)
        fig.tight_layout()
        save_figure(fig, f"{problem}_sweep")
    with (OUT / "sgd_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(OUT / "sgd_gaps.npz", **arrays)
    return rows


def harmonic_comparison():
    steps = 64
    args = dict(mean=jnp.array([0.8, -0.6, 0.3]), curvature=jnp.array([0.7, 1.3, 2.2]),
                force_scale=jnp.array(0.2), initial_temperature=jnp.array(0.75))

    def temperature(context):
        return context.args["initial_temperature"] * (1 - 0.6 * context.step / (steps - 1))

    options = dict(steps=steps, learning_rate=0.04, temperature=temperature,
                   batch_size=lambda context: 4 + context.step // 4,
                   batch_buckets=(4, 8, 16, 32), record_params=True, backend="cpu")
    energy = make_energy_langevin(Reduce(sum_reduce, (harmonic_energy, random_work)),
                                  force_sampler, **options)
    score = make_score_langevin(forced_harmonic_score, force_sampler, **options)
    keys = jnp.stack([jax.random.key(seed) for seed in SEEDS])
    run = lambda solver: np.asarray(jax.jit(jax.vmap(
        lambda key: solver.run(jnp.zeros(3), key, args).history.params))(keys))
    energy_paths, score_paths = run(energy), run(score)
    errors = np.max(np.abs(energy_paths - score_paths), axis=(1, 2))
    assert np.all(errors < 1e-12)
    np.savez_compressed(OUT / "harmonic_paths.npz", energy=energy_paths, score=score_paths,
                        maximum_errors=errors)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for coordinate, ax in enumerate(axes):
        for seed in range(3):
            line, = ax.plot(np.arange(1, steps + 1), energy_paths[seed, :, coordinate],
                            label=f"energy, seed {seed}")
            ax.plot(np.arange(1, steps + 1), score_paths[seed, :, coordinate], "--",
                    color=line.get_color(), label=f"score, seed {seed}")
        ax.axhline(float(args["mean"][coordinate]), color="black", lw=1, alpha=0.5)
        ax.set(xlabel="Iteration", ylabel=f"Coordinate {coordinate + 1}")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("Harmonic energy and score paths overlap for shared random keys")
    fig.tight_layout()
    save_figure(fig, "harmonic_comparison")
    return errors.tolist()


def gaussian_ensembles():
    """Repeat the supplied 18-step, 4096-chain experiment ten times."""
    steps, chains, eta, temperature = 18, 4096, 0.07, 0.7
    args = dict(mean=jnp.array(0.7), curvature=jnp.array(1.4))
    solver = make_score_langevin(harmonic_score, steps=steps, learning_rate=eta,
                                 temperature=temperature, record_params=True, backend="cpu")
    roots = jnp.stack([jax.random.key(seed) for seed in SEEDS])

    def ensemble(key):
        return jax.vmap(lambda chain_key: solver.run(jnp.array(-1.8), chain_key, args).history.params)(
            jax.random.split(key, chains))

    paths = np.asarray(jax.jit(jax.vmap(ensemble))(roots))
    means, variances = paths.mean(axis=1), paths.var(axis=1, ddof=1)
    t = np.arange(1, steps + 1)
    contraction = 1 - eta * 1.4
    analytic_mean = 0.7 + contraction**t * (-1.8 - 0.7)
    analytic_variance = 2 * eta * temperature * (1 - contraction**(2 * t)) / (1 - contraction**2)
    np.savez_compressed(OUT / "gaussian_ensembles.npz", representative_chains=paths[0, :10],
                        means=means, variances=variances,
                        analytic_mean=analytic_mean, analytic_variance=analytic_variance)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(np.arange(steps + 1), np.column_stack((np.full(10, -1.8), paths[0, :10])).T)
    axes[0].set(title="Ten representative chains", xlabel="Iteration", ylabel="State")
    for ax, values, analytic, label in (
        (axes[1], means, analytic_mean, "Ensemble mean"),
        (axes[2], variances, analytic_variance, "Ensemble variance"),
    ):
        interval = 2.262 * values.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
        ax.errorbar(t, values.mean(axis=0), yerr=interval, fmt="o", ms=3,
                    capsize=3, label="10 ensembles, 95% t CI")
        ax.plot(t, analytic, "--", label="Exact discrete-time moment")
        ax.set(title=label, xlabel="Iteration")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "gaussian_ensemble")
    return dict(chains_per_ensemble=chains, independent_ensembles=len(SEEDS),
                final_mean=float(means[:, -1].mean()),
                final_mean_sd=float(means[:, -1].std(ddof=1)),
                final_variance=float(variances[:, -1].mean()),
                analytic_final_mean=float(analytic_mean[-1]),
                analytic_final_variance=float(analytic_variance[-1]),
                stationary_discrete_variance=float(2 * eta * temperature / (1 - contraction**2)),
                gibbs_variance=0.5)


def main():
    jax.config.update("jax_enable_x64", True)
    OUT.mkdir(exist_ok=True)
    report = dict(seeds=SEEDS, steps=STEPS, dtype="float64", jax_version=jax.__version__,
                  sgd=sgd_sweeps(), harmonic_max_errors=harmonic_comparison(),
                  gaussian=gaussian_ensembles())
    (OUT / "experiment_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print("Saved experiments to", OUT, flush=True)


if __name__ == "__main__":
    main()

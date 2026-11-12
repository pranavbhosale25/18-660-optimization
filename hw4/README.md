# Optimization Homework 4 submission

## Deliverables

- [Implementation writeup](Implementation/writeup.md): Exercises 2.1 and 2.2, including ten-seed sweeps, trajectory comparisons, error bars, and experimental conclusions.
- [Application writeup](Application/writeup.md): Exercise 3.1, including the matrix derivations, all 18 optimized decision vectors, validation tables, and exaggerated geometry figures.
- Completed starter packages in `Implementation/jax_stochopt/` and `Application/geocomp_python/`.
- Numerical data and figures in `Implementation/results/`, `Application/results/`, and the three application probability directories.

## Setup

Use **Python 3.12** and CPU execution. Apple Silicon macOS and supported Linux platforms can run the commands below from this directory. The original package READMEs contain additional platform guidance.

```bash
bash setup.sh
```

The two starter packages require different numerical stacks, so setup creates separate environments:

| Environment | JAX/JAXlib | NumPy | SciPy |
| --- | --- | --- | --- |
| `.venv-implementation` | 0.10.0 | 2.3.4 | 1.17.1 |
| `.venv-application` | 0.12.0.1 | 2.5.4 | 1.18.1 |

The starter `pyproject.toml` files pin the remaining numerical dependencies. The experiment requirements pin Matplotlib and pytest. Installation requires package-index access. Existing generated results can be inspected without rerunning the experiments.

## Reproduce everything

```bash
bash reproduce.sh
```

This runs the tests, both supplied implementation example programs, the ten-seed experiments, all application cases at all three probabilities, and application report generation. It overwrites the generated result files in their existing directories. Runtime depends on JAX compilation and the processor; the manufacturing runs take several minutes.

## Individual commands

```bash
# Implementation correctness checks and assigned repeated-seed experiments
.venv-implementation/bin/python -m pytest Implementation/tests -q
MPLCONFIGDIR=/tmp/hw4-matplotlib .venv-implementation/bin/python Implementation/run_experiments.py

# Supplied examples, including their additional diagnostics
.venv-implementation/bin/python Implementation/jax_stochopt/examples/stochastic_problems.py --dtype float64 --backend cpu
.venv-implementation/bin/python Implementation/jax_stochopt/examples/langevin_dynamics.py --dtype float64 --backend cpu

# Application formulas and all physical/abstract cases
.venv-application/bin/python -m pytest Application/tests -q
.venv-application/bin/python -m geocomp --case all --alpha 0.90 --out Application/results_a090 --verbose
.venv-application/bin/python -m geocomp --case all --alpha 0.95 --out Application/results_a095 --verbose
.venv-application/bin/python -m geocomp --case all --alpha 0.99 --out Application/results_a099 --verbose
MPLCONFIGDIR=/tmp/hw4-matplotlib .venv-application/bin/python Application/summarize_results.py
```

## View the manufacturing results

Open these self-contained HTML files in a browser:

- [90% viewer](Application/results_a090/viewer.html)
- [95% viewer](Application/results_a095/viewer.html)
- [99% viewer](Application/results_a099/viewer.html)

Choose a geometry, a mean or paired random realization, and a displacement scale. The writeup embeds static exaggerated views for all six geometries at all three probability levels. Each case folder contains actual-scale STL exports, full precision model arrays, optimized variables, and validation results.


## Result files

- `Implementation/results/sgd_summary.csv`: all 24 hyperparameter configurations, final gap mean/SD, sample cost, and equal-budget comparison.
- `Implementation/results/sgd_gaps.npz`: ten per-seed expected-gap curves for every configuration.
- `Implementation/results/harmonic_paths.npz`: all ten paired energy/score trajectories and maximum discrepancies.
- `Implementation/results/gaussian_ensembles.npz`: ten ensemble mean/variance curves, exact discrete-time moments, and ten representative chains.
- `Implementation/results/experiment_summary.json`: experiment settings and numerical summary.
- `Application/results/all_results.csv`: all 18 manufacturing results.
- `Application/results/optimized_variables.csv`: every dimensionless decision and corresponding millimeter command, at full saved precision.
- `Application/results_a*/<case>/metrics.json`: complete per-case decisions, objective, probabilities, and independent validation.
- `Application/results_a*/<case>/optimization.json`: solver trace and numerical gap diagnostics.
- `Implementation/results/tests.txt` and `Application/results/tests.txt`: verification results.

ChatGPT was used to write this README and to help develop and check the code and solutions.

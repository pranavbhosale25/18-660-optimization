#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/hw4-matplotlib}"
mkdir -p Implementation/results Application/results

.venv-implementation/bin/python -m pytest Implementation/tests -q > Implementation/results/tests.txt
.venv-application/bin/python -m pytest Application/tests -q > Application/results/tests.txt
.venv-implementation/bin/python Implementation/jax_stochopt/examples/stochastic_problems.py --dtype float64 --backend cpu > Implementation/results/stochastic_examples.json
.venv-implementation/bin/python Implementation/jax_stochopt/examples/langevin_dynamics.py --dtype float64 --backend cpu > Implementation/results/langevin_examples.json
.venv-implementation/bin/python Implementation/jax_stochopt/examples/langevin_dynamics.py --dtype float32 --backend cpu > Implementation/results/langevin_examples_float32.json
.venv-implementation/bin/python Implementation/run_experiments.py > Implementation/results/experiments.log
for level in 090 095 099; do
    case "$level" in
        090) alpha=0.90 ;;
        095) alpha=0.95 ;;
        099) alpha=0.99 ;;
    esac
    .venv-application/bin/python -m geocomp --case all --alpha "$alpha" --out "Application/results_a$level" --verbose > "Application/results_a$level.log" 2>&1
done
.venv-application/bin/python Application/summarize_results.py

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# The two starter packages pin different numerical stacks, so isolate them.
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
"$PYTHON_BIN" -m venv .venv-implementation
"$PYTHON_BIN" -m venv .venv-application
.venv-implementation/bin/python -m pip install -e Implementation/jax_stochopt -r Implementation/requirements-experiments.txt
.venv-application/bin/python -m pip install -e Application/geocomp_python -r Application/requirements-experiments.txt

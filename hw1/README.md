# Homework 1

Writeups:

- **Implementation P1 — Nonlinear CG:** [WRITEUP.md](Implementation/P1/jax-pncg/WRITEUP.md)
- **Implementation P2 — SR1 trust region:** [WRITEUP.md](Implementation/P2/jax-sr1-trust-region/WRITEUP.md)
- **Applications — Physical modeling:** [WRITEUP.md](Applications/unified_jax_optim-0.1.3/WRITEUP.md)

# Running the submission

**NOTE**: Each directory contains a directory level WRITEUP.md with info about the requested questions/discussion points from the homework. This README only contains the necessary commands to run P1, P2 and Applications.

Run the following commands from `hw1/` (first run `cd hw1` from the repository root).
The commands below are for macOS or Linux.

## Implementation P1: preconditioned nonlinear CG

```bash
cd Implementation/P1/jax-pncg

python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e ".[dev]"

./.venv/bin/python -m pytest -q

./.venv/bin/python examples/benchmark.py \
  --device cpu --precision float64 --repeat 5

./.venv/bin/python examples/benchmark.py \
  --device cpu --precision float32 --repeat 5

cd ../../..
```

The P1 convergence plots and CSV files are written under
`Implementation/P1/jax-pncg/benchmark-results/<precision>/convergence/`.

## Implementation P2: trust-region SR1

```bash
cd Implementation/P2/jax-sr1-trust-region

python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e ".[test,plot]"

./.venv/bin/python -m pytest -q

./.venv/bin/python -m trsr1 \
  --platform cpu \
  --precision 64 \
  --plot-directory benchmark-results/float64/convergence

cd ../../..
```

The P2 convergence plots and CSV files are written under
`Implementation/P2/jax-sr1-trust-region/benchmark-results/float64/convergence/`.

## Applications: physical modeling and simulation

The Applications package requires 64-bit CPython 3.12 because the supplied
optimizer runtime is a compiled local wheel.

```bash
cd Applications/unified_jax_optim-0.1.3

python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install "jax==0.11.1" "matplotlib>=3.10"
./.venv/bin/python -m pip install \
  --no-index \
  --no-deps \
  --find-links ./wheels \
  "hw1-optimizer-runtime==0.1.3"
./.venv/bin/python -m pip install --no-deps -e .

./.venv/bin/python -m pip check
./.venv/bin/python -m compileall -q src benchmarks

./.venv/bin/python benchmarks/run_principled_physics_benchmark.py \
  --size full \
  --repeats 5 \
  --warmups 1 \
  --backend cpu \
  --dtype float64 \
  --output-dir benchmark-results/full-cpu-float64

cd ../..
```

The Applications command writes five JSON logs, five convergence PNGs, and ten
CSV files under
`Applications/unified_jax_optim-0.1.3/benchmark-results/full-cpu-float64/`.

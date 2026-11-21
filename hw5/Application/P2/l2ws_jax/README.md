# Learning to warm-start: JAX student starter

Complete both required nonconvex applications: inverse kinematics and phase
retrieval. The common neural network, trainer, projected-gradient solver,
samplers, projections, baselines, numerical references, and benchmark are
supplied.

Your code work is exactly four functions:

1. `kinematics.objective_one`
2. `kinematics.step_size_one`
3. `phase.objective_one`
4. `phase.step_size_one`

Search for `STUDENT TODO` to find these four functions. The separate homework
PDF is authoritative for the mathematical definitions, assignment
requirements, and submission instructions. This README documents the code
contract, installation, and commands.

For each step-size TODO, derive a finite curvature bound that is valid at every
point in the complete feasible box for the current context, then return its
reciprocal. A bound checked only at finitely many sampled decisions is not a
whole-box bound. Use JAX operations so the result remains compatible with
`jax.jit` and `jax.vmap`.

## Code contract

Each TODO operates on one problem instance, not a batch. The base class uses
`jax.vmap` to create the batched methods. Each objective receives one decision
and one context and returns a scalar; each step-size method receives one
context and returns a finite, positive scalar. The kinematics decision and
preferred posture are relative joint angles measured in radians.

Use `jax.numpy` for expressions involving decisions or contexts. Keep each
function pure and differentiable: do not mutate state, convert traced values
to Python or NumPy values, use global randomness, or stop gradients through
the objective or solver path. The TODO docstrings give the exact input and
output shapes.

## Install

Use Python 3.12 or newer in an isolated environment.

Linux or macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c constraints-tested.txt -e ".[dev,plots]"
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -c constraints-tested.txt -e ".[dev,plots]"
```

For NVIDIA GPU installation on Linux, install one compatible project extra:

```bash
python -m pip install -c constraints-tested.txt -e ".[dev,plots,cuda13]"
```

Use `cuda12` instead when required by the supported driver/runtime. Consult
the [official JAX installation
guide](https://docs.jax.dev/en/latest/installation.html), then verify the
actual backend with `python -m l2ws_jax devices`.

## Verify and run

The completed toy problem lets you verify the environment before implementing
the homework:

Linux or macOS:

```bash
python -m l2ws_jax list
python scripts/check_problem.py --problem toy --dtype float64 --device cpu
python -m pytest -q
python examples/walkthrough.py \
  --problem toy \
  --dtype float64 \
  --device cpu \
  --steps 100 \
  --output results/toy_walkthrough
```

Windows PowerShell:

```powershell
python -m l2ws_jax list
python scripts/check_problem.py --problem toy --dtype float64 --device cpu
python -m pytest -q
python examples/walkthrough.py `
  --problem toy `
  --dtype float64 `
  --device cpu `
  --steps 100 `
  --output results/toy_walkthrough
```

After implementing all four TODOs:

Linux or macOS:

```bash
python scripts/check_problem.py \
  --problem kinematics --dtype float64 --device cpu
python scripts/check_problem.py --problem phase --dtype float64 --device cpu
python -m l2ws_jax benchmark \
  --problems kinematics phase \
  --dtype float64 \
  --device cpu \
  --steps 300 \
  --train-size 1024 \
  --test-size 64 \
  --seed 7 \
  --evaluation-seed 107 \
  --output results/nonconvex_seed_7
```

Windows PowerShell:

```powershell
python scripts/check_problem.py `
  --problem kinematics --dtype float64 --device cpu
python scripts/check_problem.py --problem phase --dtype float64 --device cpu
python -m l2ws_jax benchmark `
  --problems kinematics phase `
  --dtype float64 `
  --device cpu `
  --steps 300 `
  --train-size 1024 `
  --test-size 64 `
  --seed 7 `
  --evaluation-seed 107 `
  --output results/nonconvex_seed_7
```

The checker validates shapes, differentiation, numerical self-consistency,
execution, and sampled descent. It does not compare your objective with the
assigned mathematical formula, so passing it does not establish semantic
correctness. Its finite-difference calculation is only an independent check
of JAX's automatic derivatives. Students do not implement finite differences,
and neither the optimizer nor the trainer uses them.

This command runs one training trial. For additional trials, change `--seed`
and use a separate output directory. Keep `--evaluation-seed` fixed when you
want each trial evaluated on the same final-test instances.

Use a new or empty output directory for each trial or configuration. The
displayed values are a reproducible starting point; record any settings you
change. Run `python -m l2ws_jax train --help` and
`python -m l2ws_jax benchmark --help` for all command-line controls.

This independent teaching implementation is provided under the MIT license.

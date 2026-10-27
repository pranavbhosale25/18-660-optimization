# HW3 Implementation: JAX SQP

This directory contains the student version of the SQP implementation. The
separate homework handout is the authoritative description of the assignment.
This README explains how to navigate, install, test, and run the code.

## Student edit locations

Search the repository for the exact tag `STUDENT TODO`. The required functions
are located where their responsibilities naturally belong:

- `jax_sqp/merit.py`: `violation`
- `jax_sqp/solver.py`: `kkt_residual`
- `jax_sqp/solver.py`: `merit_eval`
- `jax_sqp/solver.py`: `merit_directional_derivative`

The function docstrings describe the supplied inputs and required return
values. Keep the public function interfaces unchanged so that the driver and
tests can call them. You may organize the bodies of your implementations as
you prefer.

The starter intentionally raises `NotImplementedError` when execution reaches
unfinished student code. These errors are expected until the corresponding
`STUDENT TODO` sections have been completed.

## Set up

Use Python 3.11, 3.12, 3.13, or 3.14. JAX and JAXlib `0.9.0.1` are pinned to
match the course environment. Run all commands below from this directory.

On Linux or Apple-silicon macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

JAXlib `0.9.0.1` does not provide a prebuilt Intel-macOS wheel. Accelerator
support is platform-specific: the documented NVIDIA GPU path is Linux, with
WSL2 support considered experimental, rather than native Windows. The standard
JAX package does not provide a macOS GPU backend. Consult the [official JAX
installation guide](https://docs.jax.dev/en/latest/installation.html) if you
want an accelerator-enabled JAX installation.

For the pinned JAX version on a compatible Linux or WSL2 system using CUDA 13,
install the matching accelerator components without changing JAX versions:

```text
python -m pip install --upgrade "jax[cuda13]==0.9.0.1"
```

## Test the implementation

The public tests always select the CPU backend, even if a GPU-enabled JAX
installation is present. A useful staged workflow is:

1. Confirm that the supplied L-BFGS implementation works before editing:

   ```text
   python -m pytest -q tests/test_lbfgs.py
   ```

2. Implement `violation`, and then check the supplied quadratic-subproblem
   solver:

   ```text
   python -m pytest -q tests/test_qp.py
   ```

3. Implement the three functions in `solver.py`, and then run the complete
   public test suite:

   ```text
   python -m pytest -q
   ```

During development, `python -m pytest -x -q` stops after the first failure and
can make an unfinished function easier to locate. The first complete run may
take about a minute while JAX compiles the larger test cases.

Student implementations must remain compatible with JAX transformations used
by the solver and tests. Use `jax.numpy` operations and return JAX values. Do
not convert traced values to ordinary NumPy arrays or Python scalars, and do
not branch on a JAX array with a Python `if`. Functions must also handle empty
equality or inequality portions of the constraint vector.

## Run the demonstration problems

Run the default small suite on the CPU in float64:

```text
python demo.py --device cpu --precision 64 --suite small --batch --output results/my_cpu64_small
```

Run the precision suite on the CPU in float32:

```text
python demo.py --device cpu --precision 32 --suite precision --output results/my_cpu32_precision
```

The solver uses its model-based merit slope by default. To exercise the exact
directional-derivative implementation explicitly, run:

```text
python demo.py --device cpu --precision 64 --suite small --merit-slope exact --output results/my_cpu64_exact
```

Each output directory contains `report.json`, `summary.csv`, per-problem
iteration histories under `histories/`, and final solution arrays under
`solutions/`. A nonconverged problem makes the command return a nonzero exit
status unless `--allow-failures` is supplied; its results are still written.
Run `python demo.py --help` to see all available controls.

Demonstrations may use a configured accelerator with `--device gpu`. An
explicit GPU request fails clearly when JAX cannot find a GPU rather than
silently falling back to the CPU. After following the official installation
instructions for your accelerator, a short device check is:

```text
python examples/gpu_smoke.py --precision 64
```

## Code map

- `jax_sqp/merit.py`: student implementation of total constraint violation.
- `jax_sqp/solver.py`: SQP driver, KKT residuals, and merit functions.
- `jax_sqp/qp.py`: supplied quadratic-subproblem solver.
- `jax_sqp/lbfgs.py`: supplied limited-memory Hessian operators.
- `jax_sqp/linalg.py`: supplied PCG implementation.
- `jax_sqp/problems.py`: supplied demonstration problems.
- `jax_sqp/runtime.py`: precision and execution-device selection.
- `demo.py`: command-line experiment driver and result writer.
- `tests/`: public checks for the supplied code and completed implementation.

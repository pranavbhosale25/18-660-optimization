# JAX Trust-Region SR1

This README describes how to navigate, install, and run the supplied code. The
assignment handout and Section 6.2 of the textbook are the authoritative
sources for the mathematical method and assignment requirements.

## Student edit location

The only required implementation is `solve_quadratic_subproblem` in
`src/trsr1/subproblem.py`. Search for the exact tag `STUDENT TODO` to find the
incomplete region. The function signature, docstring, supporting code, callers,
and tests are supplied.

This is the intended edit location, but you may reorganize your own code if
you prefer.

## Project layout

- `src/trsr1/subproblem.py`: student implementation and supplied supporting
  code;
- `src/trsr1/solver.py`: supplied outer solver;
- `src/trsr1/updates.py`: supplied matrix-update code;
- `src/trsr1/options.py`: solver options;
- `src/trsr1/runtime.py`: backend and precision configuration;
- `src/trsr1/problems.py`: supplied problems and starting points;
- `tests/`: automated tests;
- `examples/`: small runnable examples.

## Installation

Create and activate a virtual environment from the project root. Python 3.10
or newer is required.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package and its test dependencies in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test,plot]"
```

Activation is optional. You may instead invoke the virtual environment's
Python executable directly: `.\.venv\Scripts\python.exe` on Windows or
`.venv/bin/python` on macOS and Linux.

JAX accelerator installation depends on the local system. A GPU run requires
a compatible accelerator-enabled JAX installation.

## Tests

Run the complete test suite from the project root:

```bash
python -m pytest -q
```

The unfinished scaffold raises `NotImplementedError` when execution reaches
the `STUDENT TODO` region.

## Running the supplied problems

Display all command-line controls:

```bash
python -m trsr1 --help
```

Run the supplied problems on CPU in 64-bit precision:

```bash
python -m trsr1 --platform cpu --precision 64 --no-history
```

Generate an objective-versus-iteration PNG and matching CSV history for every
supplied problem with:

```bash
python -m trsr1 --platform cpu --precision 64 \
  --plot-directory benchmark-results/float64/convergence
```

Select an individual problem by passing `--problem`, for example:

```bash
python -m trsr1 --platform cpu --precision 64 --problem wood --no-history
```

Use `--precision 32` for 32-bit arithmetic. Use `--platform gpu` only when a
compatible GPU-enabled JAX installation is available.

## Examples

```bash
python examples/basic.py
python examples/batched_quadratics.py
```

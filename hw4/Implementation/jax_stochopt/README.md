# JAX Stochastic Optimization

This folder contains the implementation portion of the assignment. The
separate homework PDF is the authoritative description of the required
algorithms, experiments, and submission materials.

## Setup

Python 3.11 through 3.14 is required.

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

On macOS with Apple Silicon, or on Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For CPU execution, install the project after activating the environment:

```bash
python -m pip install -e .
```

This installs the tested numerical stack: JAX/JAXlib 0.10.0, NumPy 2.3.4,
SciPy 1.17.1, ml-dtypes 0.5.4, and opt-einsum 3.4.0. Ordinary wheels are
available for Windows x86-64, Linux x86-64/ARM64, and Apple Silicon macOS.
Intel macOS is not supported by this pinned release's ordinary wheels.

For a compatible NVIDIA GPU on Linux, install the matching JAX build before
installing this project. For CUDA 13:

```bash
python -m pip install "jax[cuda13]==0.10.0"
python -m pip install -e .
```

Use `jax[cuda12]==0.10.0` instead when CUDA 12 is required. Native Windows GPU
execution is not supported; Windows users need WSL2, where GPU support is
experimental. macOS uses CPU execution. Consult the current
[JAX installation instructions](https://docs.jax.dev/en/latest/installation.html)
for driver and platform requirements without changing the pinned JAX version.

## Student work

Search the source tree for `STUDENT TODO`. The four implementation locations
are organized into two assignment problems.

### Problem 1: SGD

- `src/jax_stochopt/core.py`: `_evaluate_objective_minibatch`
- `src/jax_stochopt/optimizers.py`: `SGD.update`

Problem 1 covers preparation of the scalar-objective minibatch data used by
the optimizer and the SGD optimizer-state update.

### Problem 2: Langevin methods

- `src/jax_stochopt/extensions/sources.py`: `ScoreGradient.__call__`
- `src/jax_stochopt/extensions/langevin.py`: `LangevinNoise.__call__`

Problem 2 covers the score-based gradient contribution and the stochastic
proposal transform shared by the supplied Langevin factories. It reuses the
generic minibatch and optimizer machinery completed in Problem 1.

The surrounding code supplies the solver loop, random-number handling,
schedules, projections, result records, and construction of the energy- and
score-based solvers. You may reorganize the code if you prefer, but all four
required behaviors must be implemented. In the unfinished student
distribution, the corresponding paths deliberately raise
`NotImplementedError`.

Each TODO has a complete `Parameters`, `Returns`, and `Notes` contract in its
source docstring. Those contracts define the required structures, shapes,
dtypes, meanings, and solver lifecycle timing without prescribing an
implementation.

## Data conventions

| Term | Representation and meaning |
| --- | --- |
| Parameter PyTree | Any JAX PyTree whose leaves are nonempty JAX arrays. All leaves use the same `float32` or `float64` dtype. |
| Parameter-like PyTree | Exactly the same container structure, leaf shapes, and leaf dtypes as the parameters. Gradients, optimizer directions, momentum, candidates, and updated parameters are parameter-like. |
| One sample | The fixed-structure PyTree returned by `sampler(key, args)`. It has no minibatch-leading axis, and its leaf dtypes are problem-specific. The default deterministic sampler returns `()`. |
| Prepared samples | A compiled sample batch. Every nonempty sample leaf has a leading axis of length `capacity`; remaining axes describe one sample. |
| Active mask | A Boolean JAX array with shape `(capacity,)`. True lanes belong to the requested minibatch; false lanes are fixed-shape padding and do not contribute to aggregate estimates. |
| Scalar | A rank-zero JAX array, normally in the parameter dtype unless documented as integer or Boolean. |
| `args` | User-supplied run-time problem data. It may be `None` or any fixed-structure JAX-compatible PyTree and is forwarded without reinterpretation. |
| `ScheduleContext` | Read-only data associated with one solver step. Its source docstring specifies whether a particular caller sees previous or current minibatch statistics and old or new optimizer state. |
| `UpdateContext` | Read-only data passed after the deterministic optimizer candidate is formed but before projection. |
| PRNG key | A Threefry JAX key; typed keys are recommended and compatible legacy Threefry keys are accepted. Keys reserved for proposal transforms are independent of objective-sampling keys. |

The TODOs receive and return the following data within their respective
problems.

### Problem 1 data

1. `_evaluate_objective_minibatch` receives the complete prepared batch and
   returns fixed-capacity losses plus active-minibatch derivative data.
2. `SGD.update` receives the already-computed active-minibatch gradient and
   returns an optimizer direction and new optimizer state--not new parameters.

### Problem 2 data

1. `ScoreGradient.__call__` receives one unbatched sample and returns one
   parameter-like gradient contribution; the surrounding solver performs
   vectorization and active-lane averaging before the optimizer chooses its
   direction.
2. `LangevinNoise.__call__` receives the deterministic candidate and returns
   a stochastic candidate plus `LangevinInfo`, still before projection.

The records passed between these stages are documented field by field in
[`src/jax_stochopt/types.py`](src/jax_stochopt/types.py). The authoritative
contracts for the TODOs are next to their implementations:

- [`_evaluate_objective_minibatch`](src/jax_stochopt/core.py)
- [`SGD.update`](src/jax_stochopt/optimizers.py)
- [`ScoreGradient.__call__`](src/jax_stochopt/extensions/sources.py)
- [`LangevinNoise.__call__`](src/jax_stochopt/extensions/langevin.py)

## Running the examples

Run these commands from this directory after activating the virtual
environment:

```bash
python examples/stochastic_problems.py --dtype float64 --backend cpu
python examples/langevin_dynamics.py --dtype float64 --backend cpu
```

For the two assigned SGD experiments, constant learning-rate and batch-size
overrides can be supplied from the command line:

```bash
python examples/stochastic_problems.py --dtype float64 --backend cpu \
    --learning-rate 0.1 --batch-size 32 --seed 7
```

`--learning-rate` and `--batch-size` affect only the SGD branch of the randomly
forced quadratic and simplex resource allocation. Omit either option to retain
that setting's built-in schedule. Adam, Newton-CG, Muon, and the trajectory
diagnostic retain their built-in settings. Use different `--seed` values for
independent trials in a sweep.

Both programs support `--dtype float32` and `--dtype float64`. With a
compatible GPU-enabled JAX installation, replace `--backend cpu` with
`--backend gpu`. Use `--help` to see the other controls.

## Code organization

- `src/jax_stochopt/core.py`: solver construction and minibatch execution
- `src/jax_stochopt/optimizers.py`: optimization update rules
- `src/jax_stochopt/extensions/`: Langevin-specific components
- `src/jax_stochopt/objectives.py`: objective-composition utilities
- `src/jax_stochopt/projections.py`: supplied projection operators
- `examples/stochastic_problems.py`: stochastic-optimization examples
- `examples/langevin_dynamics.py`: energy- and score-based Langevin examples

## Copyright

Copyright (c) 2026 Carnegie Mellon University. All rights reserved. See
[`NOTICE`](NOTICE).

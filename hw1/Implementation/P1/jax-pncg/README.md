# `jax-pncg`

A compact JAX implementation of the preconditioned nonlinear conjugate-gradient method using the Polak–Ribière update, with the PR+ safeguard, a strong-Wolfe line search, standard test problems, explicit CPU/GPU selection, and selectable 32- or 64-bit arithmetic.

## Student work areas

This student copy is intentionally incomplete. Search the repository for the
exact tag `STUDENT TODO`. The required implementation and selection regions are
located in:

- `src/jax_pncg/solver.py` for the nonlinear-CG beta update;
- `src/jax_pncg/preconditioners.py` for the identity, fixed-diagonal, and
  position-dependent Jacobi preconditioners;
- `examples/benchmark.py` for selecting and configuring a preconditioner for
  every packaged benchmark problem.

These are the intended edit locations, but you may reorganize your own code if
you prefer.

### Running the benchmark

The test suite checks the optimizer and preconditioner mechanics, but it does
not exercise your per-problem choices in `choose_preconditioner`. After
completing the `STUDENT TODO` regions and installing the package, run the
benchmark from the project root to exercise those choices for every problem in
`STANDARD_PROBLEMS`.

Display every available command-line option with:

```bash
python examples/benchmark.py --help
```

Run both supported precisions on CPU with:

```bash
python examples/benchmark.py --device cpu --precision float64
python examples/benchmark.py --device cpu --precision float32
```

On a machine with a GPU-enabled JAX installation, run the corresponding GPU
experiments with:

```bash
python examples/benchmark.py --device gpu --precision float64
python examples/benchmark.py --device gpu --precision float32
```

An explicit GPU request fails clearly when JAX cannot use a GPU. You may use
`--device auto` to let JAX select an available device. The driver performs an
untimed compilation/warmup run, repeats each timed solve, and prints the chosen
preconditioner, convergence status, iteration and evaluation counts, final
objective and gradient norm, tolerance, and average elapsed time. Options such
as `--repeat`, `--maxiter`, and `--gtol` let you control the experiment; see
`--help` for the complete list.

Scalar iteration history is recorded by default. Each `PNCGResult.history`
contains the objective value, gradient norm, step size, beta value, line-search
evaluation count, and restart indicator for every valid iteration. The
benchmark saves an objective-versus-iteration plot and the underlying objective,
objective-gap, and gradient-norm data for each standard problem under
`benchmark-results/<precision>/convergence/`.

## What is implemented

For an unconstrained problem

\[
\min_x f(x), \qquad g_k = \nabla f(x_k),
\]

the preconditioner supplies the action

\[
z_k = M_k^{-1} g_k.
\]

The initial direction is \(p_0=-z_0\). A step length is selected with the strong-Wolfe conditions,

\[
f(x_k+\alpha_kp_k) \leq f(x_k)+c_1\alpha_k g_k^Tp_k,
\]

\[
\left|\nabla f(x_k+\alpha_kp_k)^Tp_k\right|
\leq c_2\left|g_k^Tp_k\right|.
\]

The preconditioned Polak–Ribière coefficient is implemented as

\[
\beta_{k+1}^{PR}
=
\frac{z_{k+1}^T(g_{k+1}-g_k)}{g_k^Tz_k}.
\]

For fixed symmetric \(M\), this is equivalent to

\[
\frac{g_{k+1}^T(z_{k+1}-z_k)}{g_k^Tz_k}.
\]

The direction update is

\[
p_{k+1}=-z_{k+1}+\beta_{k+1}p_k.
\]

The default is PR+,

\[
\beta_{k+1}^{PR+}=\max(\beta_{k+1}^{PR},0),
\]

as discussed by Nocedal and Wright. Set `beta_method="pr"` for the untruncated Polak–Ribière formula.

The line search follows Algorithms 3.5 and 3.6 in *Numerical Optimization*, using expansion followed by a bisection-based `zoom`. The default `c2=0.1` is appropriate for nonlinear conjugate-gradient methods.

## JAX execution model

The entire nonlinear-CG iteration and both phases of the line search use `jax.lax.while_loop`, so Python loops are not unrolled into the compiled program. Objective and gradient are evaluated together with `jax.value_and_grad`. `make_solver` returns a reusable jitted callable, while `make_batched_solver` applies `jax.vmap` to solve independent problems or multistart instances in parallel.

`PNCGResult` and its history are composed only of JAX arrays, so results pass cleanly through `jit` and `vmap`. The dynamic `while_loop` solver is not intended to be reverse-mode differentiated through; automatic differentiation is used to construct the objective gradient inside the solver.

## Installation

Create and activate a virtual environment from the project root.

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

Then install the package and its development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Activation is optional. You may instead invoke the virtual environment's
Python executable directly: `.\.venv\Scripts\python.exe` on Windows or
`.venv/bin/python` on macOS and Linux.

For an NVIDIA GPU, install the JAX build appropriate for the machine before installing this package. For example, on a currently supported Linux/CUDA setup:

```bash
python -m pip install --upgrade "jax[cuda13]"
python -m pip install -e ".[dev]"
```

## CPU/GPU and 32/64-bit selection

Configure the runtime before creating arrays or compiling a solver:

```python
from jax_pncg import configure_runtime

runtime = configure_runtime(
    platform="cpu",      # "cpu", "gpu", or "auto"
    precision="float64", # "float32" or "float64"
)
```

`platform="cpu"` explicitly places arrays and compiled calls on CPU, even on a GPU machine. `platform="gpu"` fails with a clear error when no GPU backend is available. `platform="auto"` prefers a GPU and falls back to CPU. The returned `Runtime.asarray` creates an array with the selected dtype on the selected device.

JAX precision is process-global. Configure it once, before array creation and compilation. For strict process-wide CPU-only initialization, the environment variable `JAX_PLATFORMS=cpu` can also be set before starting Python.

## Basic solve

```python
from jax_pncg import (
    PNCGOptions,
    configure_runtime,
    make_solver,
    status_message,
)
from jax_pncg.test_problems import ROSENBROCK_2

runtime = configure_runtime(platform="cpu", precision="float64")

with runtime.default_device():
    x0 = runtime.asarray(ROSENBROCK_2.start)
    solver = make_solver(
        ROSENBROCK_2.objective,
        options=PNCGOptions(
            maxiter=1000,
            gtol=1e-8,
            beta_method="pr+",
            record_history=True,
        ),
    )
    result = solver(x0)

result.x.block_until_ready()
print(status_message(result.status))
print(result.x, result.fun, result.grad_norm)
```

For repeated solves with the same objective and options, build the solver once and reuse it. JAX then reuses the compiled executable for matching shapes and dtypes.

## Fixed diagonal preconditioning

```python
import numpy as np

from jax_pncg import PNCGOptions, diagonal_preconditioner, make_solver
from jax_pncg.test_problems import diagonal_quadratic

n = 128
diagonal_np = np.geomspace(1.0, 1.0e8, n)
diagonal = runtime.asarray(diagonal_np)
x0 = runtime.asarray(np.ones(n))

solver = make_solver(
    diagonal_quadratic,
    preconditioner=diagonal_preconditioner(diagonal_np),
    options=PNCGOptions(maxiter=100, gtol=1e-8),
)
result = solver(x0, diagonal)
```

A preconditioner has signature

```python
def preconditioner(x, gradient, *objective_args):
    return z  # action of M^{-1} on gradient
```

The classical interpretation assumes a fixed SPD preconditioner. An `x`-dependent callable is accepted and can be useful in practice, but is a flexible/preconditioned PR heuristic rather than the fixed-metric theory. Invalid or non-descent preconditioner outputs are safeguarded by restarting with the ordinary gradient.

## Batched multistart with `vmap`

```python
import numpy as np

from jax_pncg import PNCGOptions, make_batched_solver
from jax_pncg.test_problems import rosenbrock

starts = runtime.asarray(
    np.array([[-1.2, 1.0], [-1.0, 2.0], [2.0, 2.0], [0.0, 0.0]])
)

batched_solver = make_batched_solver(
    rosenbrock,
    options=PNCGOptions(maxiter=1000, gtol=1e-8),
)
result = batched_solver(starts)
```

Additional objective arguments can be independently batched through `arg_in_axes`. For example, `arg_in_axes=(0, None)` batches the first extra argument and shares the second.

## Options and safeguards

`PNCGOptions` includes:

- `beta_method="pr"` or `"pr+"`;
- absolute and relative gradient tolerances;
- infinity- or Euclidean-norm stopping tests;
- strong-Wolfe constants and line-search limits;
- periodic restart and the Nocedal–Wright gradient-orthogonality restart test;
- automatic restart when a candidate direction loses descent;
- one retry with the preconditioned steepest-descent direction after a failed conjugate-direction line search;
- scalar history recording, enabled by default; set `record_history=False` to
  disable it.

Float32 arithmetic generally needs a looser stopping tolerance. Values from `gtol=1e-4` to `5e-4` are practical for multistart or badly scaled examples; float64 is recommended when tighter deterministic tolerances are required.

## Included test problems

`jax_pncg.test_problems` contains:

- two- and ten-dimensional Rosenbrock problems;
- Powell singular;
- Beale;
- Wood;
- Brown badly scaled;
- Freudenstein–Roth;
- parameterized ill-conditioned diagonal quadratics.

The package tests exercise PR and PR+, strong-Wolfe progress, preconditioning, jitted composition, vectorized multistart solves, batched objective arguments, CPU placement, GPU selection behavior, and both float32 and float64 modes.

Run them with:

```bash
python -m pytest -q
```

Run the non-benchmark examples with:

```bash
python examples/quickstart.py --device cpu --precision float64
python examples/batched_multistart.py --device auto --precision float32
```

See [Running the benchmark](#running-the-benchmark) for the benchmark driver,
device and precision combinations, and its command-line controls.

## Status codes

- `0`: converged;
- `1`: maximum iterations reached;
- `2`: strong-Wolfe line search failed;
- `3`: a non-finite objective, gradient, or state was encountered.

Use `status_message(result.status)` to obtain a host-side description.

## Reference

Jorge Nocedal and Stephen J. Wright, *Numerical Optimization*, second edition, Springer, 2006. In particular, the implementation follows the strong-Wolfe line search in Algorithms 3.5–3.6 and the nonlinear conjugate-gradient discussion in Section 5.2, including the Polak–Ribière and PR+ formulas.

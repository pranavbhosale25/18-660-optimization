# HW3 Application: constrained bear locomotion

The separate homework PDF is the authoritative assignment description. This
README explains where to work and how to install, run, and inspect the supplied
code. Run all commands from the project root.

## Student work

Search for `STUDENT TODO` in `examples/bear_run.py`. There are exactly three
functions to complete:

- `register_objectives`
- `register_constraints`
- `solve_with_ipopt`

The first two construct the Pyomo formulation for one continuation stage. The
third invokes the supplied IPOPT interface and returns its result. You do not
need to modify the Newton simulator, Warp differentiation code, or native IPOPT
callback layer. The marked functions are the recommended edit locations, but
you may organize your work differently.

Useful supplied interfaces are:

- [`Variables.u` and `Variables.position`](src/bear_shooting/pyomo_task.py)
  for decision and observation variables;
- [`ShootingProblem.add_objective`, `add_constraint`, and
  `solve`](src/bear_shooting/problem.py) for registering and solving the
  problem.

## Installation

Python 3.13 is the tested version. The recommended setup installs Python and
the native IPOPT runtime from conda-forge, then installs this project from
PyPI-hosted dependencies:

```bash
conda create --name bear-shooting --channel conda-forge python=3.13 pip ipopt
conda activate bear-shooting
python -m pip install --upgrade pip
python -m pip install -e ".[bear,test]"
python scripts/check_environment.py --test-ipopt
```

The environment check must succeed before optimization. The optional
`cyipopt` package is not required. For virtual-environment instructions,
IPOPT discovery, and the Windows, Linux, and Apple Silicon support table, see
[`docs/INSTALL.md`](docs/INSTALL.md).

Install the interactive viewer only if you want it:

```bash
python -m pip install -e ".[viewer]"
```

## Run the supplied checks

```bash
python -m pytest -q
```

These public tests check the supplied scaffolding and can pass before the three
assignment TODOs are complete.

## Run the locomotion problem

Before implementing the formulation, you can test scene loading and the
deterministic initial trajectory without calling IPOPT:

```bash
python examples/bear_run.py --size quick --device cpu --evaluate-initial
```

After completing all three TODO functions, start with the quick CPU problem:

```bash
python examples/bear_run.py --size quick --device cpu
```

For the longer visible locomotion run, use an NVIDIA GPU:

```bash
python examples/bear_run.py --size small --device cuda:0
```

The supplied profiles are:

| Profile | Simulation | Target | Intended use |
|---|---|---|---|
| `quick` | 1 second, 15 intervals, 4,800 substeps | 0.075 body lengths | CPU setup and formulation check; motion may look mostly like rocking |
| `small` | 5 seconds, 75 intervals, 24,000 substeps | 0.55 body lengths | Visible forward locomotion on CUDA |
| `full` | 10 seconds, 150 intervals, 48,000 substeps | 1 body length | Complete CUDA experiment |

On course hardware, a complete quick CPU or small CUDA run is expected to take
roughly ten minutes, although runtime varies substantially by machine. The
`small` and `full` profiles require several GiB of free GPU memory and may be
prohibitively slow on CPU. Run the full problem with:

```bash
python examples/bear_run.py --size full --device cuda:0
```

Use `python examples/bear_run.py --help` for all device, size, continuation,
resource-limit, and output controls.

## Inspect a result

Without `--output`, the three profiles write:

- `results/bear_run_quick.npz`
- `results/bear_run_small.npz`
- `results/bear_run.npz`

A successful run also writes a JSON solver report, particle replay, and task
metadata beside the NPZ file. The report distinguishes final-stage solver time
(`elapsed_seconds`) from total process time (`end_to_end_wall_seconds`).

Independently replay the default quick result on CPU:

```bash
python scripts/verify_bear_run.py
```

For a small CUDA result, run:

```bash
python scripts/verify_bear_run.py results/bear_run_small.npz --device cuda:0
```

After installing the viewer extra, open a saved trajectory with:

```bash
python examples/view_bear_result.py results/bear_run_quick.npz
```

Replace the path with `results/bear_run_small.npz` or
`results/bear_run.npz` for the longer profiles. Playback uses the saved
simulation timing unless `--frames-per-knot` is supplied.

## Execution path

`examples/bear_run.py` parses the command line, creates the bear scene and
actuator map, builds the Newton/Warp rollout, and generates an initial control
trajectory. For each continuation stage, `build_problem` creates a
`ShootingProblem`, applies control bounds, calls the two student registration
functions, and passes the result to the student solver function. Newton performs
the simulation, Warp supplies derivatives, and the existing package code
connects those values to IPOPT.

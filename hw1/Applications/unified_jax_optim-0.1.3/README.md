# Physics benchmark code map

The separate assignment PDF is the authoritative description of the homework.
This README only covers installation and explains how the benchmark code is
arranged.

## Supported installations

Use 64-bit CPython 3.12. The release `wheels` directory must contain a compiled
optimizer runtime for each distributed platform:

- Ubuntu Linux on x86-64 or ARM64;
- Windows on x86-64; and
- macOS on Apple silicon.

The Linux x86-64 wheel is used for both native Ubuntu and x86-64 WSL. Intel
macOS is not included. The compiled package hides the readable Python
implementation, but -- as with any native program -- it should not be treated
as impossible to reverse engineer.

### CPU installation

From this directory, create a fresh environment and install the project. On
Windows PowerShell, first confirm that `python --version` reports Python 3.12,
then run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "jax==0.11.1"
.\.venv\Scripts\python.exe -m pip install "matplotlib>=3.10"
.\.venv\Scripts\python.exe -m pip install --no-index --no-deps --find-links .\wheels "hw1-optimizer-runtime==0.1.3"
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

If `python` is not the Python 3.12 installation but the Windows Python Launcher
is available, use `py -3.12 -m venv .venv` for the first command instead.

On Ubuntu, WSL, or macOS:

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install "jax==0.11.1"
./.venv/bin/python -m pip install "matplotlib>=3.10"
./.venv/bin/python -m pip install --no-index --no-deps --find-links ./wheels "hw1-optimizer-runtime==0.1.3"
./.venv/bin/python -m pip install --no-deps -e .
```

These commands obtain JAX from the public package index, but force the private
optimizer runtime to come from the local `wheels` directory. `pip` chooses the
compatible local runtime wheel automatically. It will report that no matching
distribution exists if the Python version, operating system, or processor
architecture is not one of the combinations above.

### NVIDIA GPU installation on Linux or WSL

The runtime wheel itself is the same for CPU and GPU. On a supported NVIDIA
Linux or WSL setup, request JAX's CUDA 13 packages while installing:

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install "jax[cuda13]==0.11.1"
./.venv/bin/python -m pip install "matplotlib>=3.10"
./.venv/bin/python -m pip install --no-index --no-deps --find-links ./wheels "hw1-optimizer-runtime==0.1.3"
./.venv/bin/python -m pip install --no-deps -e .
```

Native Windows uses the CPU backend. The ordinary JAX package used here does
not provide a macOS GPU backend. See the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for
the current driver and CUDA requirements.

Verify the installation before editing the scaffold. On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import hw1_optimizer_runtime as runtime; print(runtime.__version__)"
.\.venv\Scripts\python.exe benchmarks\run_principled_physics_benchmark.py --help
```

On Ubuntu, WSL, or macOS:

```bash
./.venv/bin/python -m pip check
./.venv/bin/python -c "import hw1_optimizer_runtime as runtime; print(runtime.__version__)"
./.venv/bin/python benchmarks/run_principled_physics_benchmark.py --help
```

## Running the benchmarks

Run the commands below from the project directory after completing the marked
objective and optimizer-selection sections. Omitting `--case` runs all five
physics cases sequentially. On Windows PowerShell, run:

```powershell
.\.venv\Scripts\python.exe benchmarks\run_principled_physics_benchmark.py
```

On Ubuntu, WSL, or macOS, run:

```bash
./.venv/bin/python benchmarks/run_principled_physics_benchmark.py
```

With no command-line options, the driver uses full-size cases, prepared mode,
one warmup and five measured runs per starting condition, the CPU backend,
`float64`, and the `benchmark-results` output directory.

For a quicker run of all five cases while working on the code, use:

```powershell
.\.venv\Scripts\python.exe benchmarks\run_principled_physics_benchmark.py --size small --repeats 1 --warmups 1
```

or, on Ubuntu, WSL, or macOS:

```bash
./.venv/bin/python benchmarks/run_principled_physics_benchmark.py --size small --repeats 1 --warmups 1
```

Use `--case` to run only selected cases. Repeat the option to select more than
one case. For example:

```bash
./.venv/bin/python benchmarks/run_principled_physics_benchmark.py --case batched_relaxation --case implicit_dynamics
```

CPU and GPU experiments are separate invocations. On NVIDIA Linux or WSL, for
example:

```bash
./.venv/bin/python benchmarks/run_principled_physics_benchmark.py --backend cpu --dtype float64 --output-dir benchmark-results/cpu-float64
./.venv/bin/python benchmarks/run_principled_physics_benchmark.py --backend gpu --dtype float64 --output-dir benchmark-results/gpu-float64
```

Each completed case writes a JSON record, iteration and wall-clock CSV files,
and a two-panel convergence PNG to the selected output directory.

Use `--dtype float32` for a 32-bit experiment. Native Windows and macOS use the
CPU backend. The driver creates the requested output directory automatically;
the scaffold writes results there only if the benchmark-recording code is
implemented to do so.

## Execution path

`benchmarks/run_principled_physics_benchmark.py` starts the driver without
importing JAX. The driver launches a fresh process for each selected case and
sets that process's backend and floating-point dtype. Inside the child process,
the benchmark scaffold constructs the case, runs the optimizer through the
compiled runtime, and passes each completed run to the recording code.

The main path through the source tree is:

```text
benchmarks/run_principled_physics_benchmark.py
  -> src/unified_jax_optim/benchmarks/principled_benchmark_runner.py
  -> src/unified_jax_optim/benchmarks/principled_benchmark.py
  -> src/unified_jax_optim/benchmarks/principled_physics.py
  -> installed hw1_optimizer_runtime wheel
```

## Suggested edit locations

Search the source tree for the exact tag `STUDENT TODO`. Hooks that are not
needed for every recording design are marked `STUDENT TODO (optional)`.

- `src/unified_jax_optim/benchmarks/principled_physics.py` contains the case
  construction and marked inline objective sections.
- `src/unified_jax_optim/benchmarks/principled_benchmark_runner.py` contains the
  marked optimizer and option-selection section.
- `src/unified_jax_optim/benchmarks/principled_benchmark.py` contains the
  supplied execution loop and marked result-recording sections.

These locations are suggestions. The surrounding code may be reorganized if a
different structure is clearer.

## Command-line controls

Use the environment's Python executable to see the current options and
defaults. On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe benchmarks\run_principled_physics_benchmark.py --help
```

On Ubuntu, WSL, or macOS:

```bash
./.venv/bin/python benchmarks/run_principled_physics_benchmark.py --help
```

Assignment developed by the instructors.  Code written and iterated on by the instructors and with ChatGPT 5.6-Sol.

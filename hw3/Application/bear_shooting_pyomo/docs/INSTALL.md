# Installation

Python 3.13 is the tested version. The project dependencies are downloaded
from the Python Package Index; the native IPOPT library comes from
conda-forge.

## Supported platforms

| Platform | Simulation and optimization | Interactive GL viewer |
|---|---|---|
| Windows x86-64 | CPU and NVIDIA CUDA | Supported |
| Linux x86-64 with glibc 2.28 or newer | CPU and NVIDIA CUDA | Supported |
| Apple Silicon macOS 12 or newer | CPU only | Requires macOS 14 or newer |

Intel (`x86_64`) macOS is not supported by this release because Warp 1.16.0
does not publish an Intel Mac wheel. The project supports Python 3.10 through
3.13; Python 3.13 is the release-tested version.

## Recommended setup: Conda environment plus pip packages

IPOPT is a native library rather than a Python-only dependency. Conda provides
a conventional way to install it on Windows, Linux, and macOS while pip
installs this project and its Python dependencies:

```bash
conda create --name bear-shooting --channel conda-forge python=3.13 pip ipopt
conda activate bear-shooting
python -m pip install --upgrade pip
python -m pip install -e ".[bear,test]"
python scripts/check_environment.py --test-ipopt
```

Run these commands from the project root. The `bear` extra installs Newton
1.5.2 and a compatible USD Core release. The ordinary project dependencies
install the tested Warp and Pyomo versions. The `test` extra installs pytest.

If IPOPT is installed elsewhere, a normal virtual environment also works:

```powershell
# Windows PowerShell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[bear,test]"
```

```bash
# Linux or macOS
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[bear,test]"
```

Using the regular pip build process is intentional: it installs the required
build tools in an isolated environment. Do not add `--no-build-isolation` to a
fresh environment.

## IPOPT discovery

The default solver bridge expects a double-precision, 32-bit-index IPOPT
build. It first checks `BEAR_IPOPT_LIBRARY`, then the system library search
path and common locations in the active environment. If automatic discovery
does not find a Conda or system installation, set the variable to the full
shared-library path before running Python.

Linux:

```bash
export BEAR_IPOPT_LIBRARY="$CONDA_PREFIX/lib/libipopt.so"
```

macOS:

```bash
export BEAR_IPOPT_LIBRARY="$CONDA_PREFIX/lib/libipopt.dylib"
```

Windows PowerShell:

```powershell
$ipoptDll = Get-ChildItem "$env:CONDA_PREFIX\Library\bin" -File |
    Where-Object { $_.Name -match '^(lib)?ipopt(?:-\d+)?\.dll$' } |
    Select-Object -First 1
if ($null -eq $ipoptDll) { throw "No IPOPT DLL found in the active Conda environment." }
$env:BEAR_IPOPT_LIBRARY = $ipoptDll.FullName
```

An explicit invalid path is reported as an error. Verify the Python packages,
Warp device discovery, and a small native IPOPT solve with:

```bash
python scripts/check_environment.py --test-ipopt
python -m pytest -q
```

The environment checker prints a JSON report and exits with a nonzero status
if a required dependency, Warp initialization, or the requested IPOPT probe
fails. The optional `cyipopt` package may be reported as missing without
causing failure.

## Interactive viewer

The optimizer does not require the windowing packages. Install them only if
you want to open the interactive result viewer:

```bash
python -m pip install -e ".[viewer]"
python examples/view_bear_result.py results/bear_run_quick.npz
```

Pass `results/bear_run_small.npz` instead to view the longer CUDA run.

The `viewer` extra installs Newton, USD Core, Pyglet, ImGui Bundle, Pillow, and
PyOpenGL. On a headless machine, the same command can write an animated USD
file without opening a window:

```bash
python examples/view_bear_result.py results/bear_run_quick.npz --viewer usd --output-path results/bear_run_quick.usda
```

## CPU and GPU selection

Use `--device cpu` for CPU execution or `--device cuda:0` for the first NVIDIA
GPU. The runner defaults to the 1-second `quick` problem on CPU. The 5-second
`small` profile uses 75 control intervals and 24,000 physics substeps and is
the intermediate locomotion run, while `full` is the complete problem. The
small run needs several GiB of free GPU memory; its stored state arrays alone
have an approximate 3.2 GiB lower bound. Select both flags explicitly for
larger or GPU runs. CUDA requires a compatible NVIDIA driver on Windows or
Linux/WSL. A local CUDA Toolkit is not required by Warp. Apple Silicon macOS
supports CPU execution only; use the quick profile there. Small and full CPU
optimizations may be prohibitively slow.

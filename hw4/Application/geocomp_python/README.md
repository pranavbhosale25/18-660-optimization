# HW4 Application: Geometry Compensation

This directory contains the student scaffold for the geometry-compensation
application. The separate homework PDF is the authoritative assignment
description.

## Setup

Python 3.12 through 3.14 is required. The pinned environment supports Windows
x86-64, glibc-based Linux x86-64 or ARM64, and Apple Silicon macOS 11 or newer.
It does not provide pinned JAXLIB wheels for Intel macOS, Windows ARM, or
Alpine/musl Linux. Run these commands from this directory.

Windows x86-64 PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Apple Silicon macOS or glibc-based Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Student edit locations

Search the project for `STUDENT TODO`. There are three required functions:

- `formulate_dlp_problem` in `geocomp/processes.py`
- `formulate_cnc_problem` in `geocomp/processes.py`
- `formulate_embedded_problem` in `geocomp/embedded.py`

Each formulation is shared by two cases: DLP handles the ellipsoid and sleeve,
CNC handles the cylinder and pocket plate, and embedded extrusion handles the
helix and serpentine. Implementations must therefore use the supplied data
rather than assumptions about only one shape.

The supplied function docstrings describe the meaning, shape, and units of the
inputs. Each function must return an `AffineGaussianBox` whose `name` and
`alpha` fields preserve the supplied `name` and `alpha` arguments. These
locations are recommendations: you may add private helpers or reorganize your
implementation if you preserve the package interface.

## Data conventions

The numerical array arguments are ordinary NumPy floating-point arrays. The
formulation functions do not receive meshes as flattened arrays, and they do not
return an optimized decision vector. Each function returns one
`AffineGaussianBox` describing the problem; the supplied runner passes that
object to the optimizer afterward.

For DLP and CNC, `m` denotes the number of modes and `v` the number of mesh
vertices. The coefficient vectors have shape `(m,)`, response and covariance
matrices have shape `(m, m)`, interpolation weights have shape `(v, m)`, and
vertex-area weights have shape `(v,)`. All mode-indexed inputs use the same
ordering.

For embedded extrusion, `k` denotes the number of controls, `s` the number of
path sections, and `d = 4k` the number of coefficients and decisions. The
channel order is all `dx` controls, then all `dy`, all `dz`, and all radius
controls. The path weights have shape `(s, k)`, channel-indexed vectors have
shape `(d,)`, and the bath bounds have shape `(2, 3)`, with lower XYZ coordinates
in row zero and upper XYZ coordinates in row one.

The complete returned-object schema—including `bias`, `response`, `covariance`,
feature limits, objective data, decision bounds, and optional linear-constraint
arrays—is documented on `AffineGaussianBox` in `geocomp/problem.py`. The three
TODO function docstrings give the units and meaning of every supplied argument.

The three completed abstract cases provide small examples of how the common
problem class is instantiated. They can be run before the student functions are
complete:

```bash
python -m geocomp --case abstract --out abstract_results
```

## Running the application

After implementing the three functions, run an individual physical case with,
for example:

```bash
python -m geocomp --case dlp_ellipsoid --out dlp_results --verbose
python -m geocomp --case cnc_cylinder --out cnc_results --verbose
python -m geocomp --case embedded_helix --out embedded_results --verbose
```

Run every supplied case with:

```bash
python -m geocomp --case all --out all_results --verbose
```

Use `--alpha` to set the required joint success probability for the physical
manufacturing cases. The abstract examples retain their built-in probability
level. Use a separate output directory for each run so that results are not
overwritten, for example:

```bash
python -m geocomp --case all --alpha 0.90 --out results_a090 --verbose
python -m geocomp --case all --alpha 0.95 --out results_a095 --verbose
python -m geocomp --case all --alpha 0.99 --out results_a099 --verbose
```

The full run took approximately seven minutes on the tested Windows CPU. Actual
runtime depends on the processor and local JAX compilation costs.

The available cases and command-line controls are listed by:

```bash
python -m geocomp --help
```

Each completed physical run writes a self-contained `viewer.html` into its
output directory. Open the combined viewer after the `all` run with one of:

```powershell
# Windows PowerShell
Start-Process .\all_results\viewer.html
```

```bash
# macOS
open all_results/viewer.html

# Linux desktop
xdg-open all_results/viewer.html
```

## Code path

`geocomp/demo.py` selects a case and invokes its model constructor. The
constructors in `geocomp/processes.py` and `geocomp/embedded.py` create the
supplied geometry and process data, then call the student formulation function.
The returned problem is passed to the supplied probability oracle and solver.
The validation and visualization modules process the result afterward.

Copyright (c) 2026 Carnegie Mellon University.


# HW2 Applications P2: Physics

This folder contains the contact-physics starter code. The separate homework PDF
is the authoritative assignment description.

## Student edit locations

Implement `contact_qp` and `time_to_collision` in `physics.py`. Search for the
exact tag `STUDENT TODO` to find both incomplete functions. The solver adapter,
event loop, random-world generator, and viewer are supplied.

## Set up

Use Python 3.11 or newer. Run all commands from this folder.

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project and public-test dependencies:

```bash
python -m pip install -e ".[test]"
```

## Run the code

Run the public checks:

```bash
python -m pytest -q
```

Run the JAXopt simulation:

```bash
python -m examples.physics_demo --backend jaxopt --batch 8 --steps 180
```

The demo writes JSON and NPZ data plus a self-contained HTML viewer under
`results/` by default. Run it with `--help` for all command-line controls.

Use a separate virtual environment for P1 and P2.

## Code structure

- `physics.py`: the two student implementations and supplied event loop;
- `worlds.py`: supplied world and point-mass constructors;
- `qp.py`: the supplied JAXopt/SciPy solver adapter;
- `viewer.py`: the supplied offline HTML viewer;
- `examples/physics_demo.py`: the experiment driver;
- `tests/`: limited public checks.

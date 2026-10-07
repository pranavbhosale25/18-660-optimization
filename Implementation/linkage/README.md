# Four-bar linkage optimization

This folder contains the code for the four-bar linkage implementation problem.
The separate homework PDF is the authoritative assignment description.

## Set up

Use Python 3.11 or newer. Run all commands below from this folder.

On macOS or Linux, create and activate a virtual environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, create and activate it with:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Search the code for `STUDENT TODO` to find the four functions to complete:

- `closure_energy` in `linkage_core.py`;
- `batched_state_sensitivities` in `optimize.py`;
- `design_objective_and_gradient` in `optimize.py`;
- `gradient_descent_update` in `optimize.py`.

Implement `closure_energy` first. It is required by the forward
simulation, which in turn underlies the sensitivity, gradient, and optimization
code.

## Run the code

After implementing `closure_energy`, run the linkage simulation:

```bash
python simulate.py
```

Add `--show` to open the plots and animation in desktop windows:

```bash
python simulate.py --show
```

Run the linkage optimization:

```bash
python optimize.py
```

For a numerical and plot-only run without generating animations:

```bash
python optimize.py --no-animation
```

By default, optimization results are written under `outputs/optimization/`.
Use `--output-dir` to select another location. Run any script with `--help`
to see its available command-line options.

## Code structure

- `optimize.py` contains three student-completed optimization components, the
  supplied outer optimization loop, and the command-line driver.
- `linkage_core.py` contains the linkage model, simulation routines, objective
  components, validation helpers, and the closure-energy student exercise.
- `linkage_viz.py` contains plotting and animation utilities.
- `simulate.py` runs the supplied forward-simulation examples.
- `requirements.txt` lists the Python dependencies.

The implementation and command-line driver intentionally live together in
`optimize.py`.

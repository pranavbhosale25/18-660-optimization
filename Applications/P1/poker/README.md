# HW2 Applications P1: Poker

This folder contains the Poker starter code. The separate homework PDF is the
authoritative assignment description.

## Student edit location

Implement `row_lp` in `games.py`. Search for the exact tag `STUDENT TODO` to
find the incomplete function. The game environment, solver adapter, evaluator,
and drivers are supplied.

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

Run the six-card experiments:

```bash
python -m examples.poker_demo --backend jaxopt --bids 0 1
python -m examples.poker_demo --backend jaxopt --bids 0 1 2
```

Run the 52-card experiments:

```bash
python -m examples.poker_demo --backend jaxopt --ranks 13 --suits 4 --bids 0 1
python -m examples.poker_demo --backend jaxopt --ranks 13 --suits 4 --bids 0 1 2
```

The demo writes a JSON summary and an NPZ file under `results/` by default. The
saved data include strategies and the lower bound, upper bound, and gap for
every public board. Run the demo with `--help` for all command-line controls.

Use a separate virtual environment for P1 and P2.

## Code structure

- `games.py`: the student LP formulation and supplied game-solving helpers;
- `poker.py`: the supplied Poker rules, payoff, and deal construction;
- `qp.py`: the supplied JAXopt/SciPy solver adapter;
- `examples/poker_demo.py`: the experiment driver;
- `tests/`: limited public checks.

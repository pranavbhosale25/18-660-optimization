# JAXBO application laboratories

This distribution contains the `jaxbo` package and five application examples.
Student work is confined to pure configuration-building functions in
`examples/configurations.py`; the package supplies the Bayesian-optimization
algorithms, experiment accounting, data collection, and result serialization.
Search for `STUDENT TODO` to find every intended edit location. The separate
assignment PDF is the authoritative assignment description and submission guide.

## Supported installations

Use 64-bit CPython 3.12. The `wheels` directory contains a compiled simulator
runtime for the supported release platforms:

- Ubuntu Linux on x86-64 or ARM64;
- Windows on x86-64; and
- macOS on Apple silicon.

The Linux x86-64 wheel also supports x86-64 WSL. Intel macOS is not included.
The native runtime prevents ordinary source inspection of the simulator, but it
is source concealment rather than encryption and can still be reverse engineered.

Run the commands for your platform from this directory.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "jax==0.10.0" "pytest>=7,<9" "numpy>=1.26,<3" "scipy>=1.11,<2"
.\.venv\Scripts\python.exe -m pip install --no-index --no-deps --find-links .\wheels "hw5-bo-simulator-runtime==0.2.0"
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\Activate.ps1
```

### macOS or Linux

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install "jax==0.10.0" "pytest>=7,<9" "numpy>=1.26,<3" "scipy>=1.11,<2"
./.venv/bin/python -m pip install --no-index --no-deps --find-links ./wheels "hw5-bo-simulator-runtime==0.2.0"
./.venv/bin/python -m pip install --no-deps -e .
./.venv/bin/python -m pip check
source .venv/bin/activate
```

The plotting dependency is optional:

```bash
python -m pip install "matplotlib>=3.7,<4"
```

Run the tests with:

```bash
python -m pytest -q
```

Verify that the compiled runtime and command-line driver are visible:

```bash
python -c "import hw5_bo_simulator_runtime as runtime; print(runtime.__version__)"
python -m examples.run --help
```

The unfinished distribution raises `NotImplementedError` when a test or example
reaches a `STUDENT TODO`. Consequently, exactly five parameterized plan checks
fail in the untouched starter. The remaining tests should pass. Once all five
experiment-plan functions are complete, the entire test suite should pass.

## Code structure

- `jaxbo/` contains the supplied optimization package.
- `jaxbo/labs/` contains the supplied application API, compiled-simulator
  bridge, experiment accounting, configurable policy, and result writer.
- `examples/configurations.py` contains the five functions students complete.
- `examples/run.py` is the supplied command-line driver.
- `tests/` contains package tests and student-interface checks.

Each function in `examples/configurations.py` returns an `ExperimentPlan` with:

- one primary `PolicyConfig`; and
- exactly two descriptively named ablation configurations.

The primary and both ablations must be distinct `PolicyConfig` values by
serialized value. An ablation should also change an active, intentional part of
the method; the structural checks cannot determine whether a changed field has
any behavioral effect. Every variant runs in a separate episode with a complete
budget, so ablations do not reuse observations from the primary run.

The functions are pure: they receive no observations, run no experiments, write
no files, and use no random state. The supplied driver performs those operations.
Students may add private helpers or reorganize their example code, but the five
public plan functions and their signatures must remain available to the driver
and tests.

`PolicyConfig()` prints all available fields and their defaults:

```bash
python -c "from jaxbo import PolicyConfig; print(PolicyConfig())"
```

To inspect declared field names, types, and defaults, run:

```bash
python -c "from dataclasses import fields; from jaxbo import PolicyConfig; print(*[(f.name, f.type, f.default) for f in fields(PolicyConfig)], sep='\n')"
```

For detailed field meanings, allowed string choices, units, and compatibility
rules, run:

```bash
python -c "from jaxbo import PolicyConfig; help(PolicyConfig)"
```

Students can inspect a problem's public input contract without running an
experiment. Replace `coating` with another problem identifier as needed:

```bash
python -c "from jaxbo import make_problem; print(make_problem('coating', instance=0).spec)"
```

The returned `TaskSpec` describes the domain, feature names, budget, public
feedback, and other execution metadata. Its `spec` is public; underscore-prefixed
simulator fields such as `_truth` and `_initial` are private implementation
details and must not be inspected or called.

Invalid or incompatible component combinations raise an explanatory error when
the configuration or policy is constructed.

## Running an experiment

After implementing the selected problem's plan, run its primary configuration:

```bash
python -m examples.run --problem coating --instance 0 --seed 0 --variant primary --output-dir results
```

Run the primary configuration and both controlled ablations on the same
development instance and algorithm seed:

```bash
python -m examples.run --problem coating --instance 0 --seed 0 --variant all --output-dir results
```

To run one named ablation, replace `all` with the name returned by that task's
plan function. Use the driver's help for all controls:

```bash
python -m examples.run --help
```

The five problem identifiers are `coating`, `pipeline`, `production`,
`materials`, and `controller`. The student driver exposes development instances
only. Separate runs use separate complete budgets; observations are never shared
between the primary configuration and an ablation.

For each executed variant, the driver writes `result.json`,
`observations.json`, `observations.csv`, and `recommendations.json` beneath:

```text
<output-dir>/<problem>/instance-<instance>/seed-<seed>/<variant>/
```

It also exports the current primary configuration to:

```text
<output-dir>/<problem>/submission.json
```

Running the same problem, instance, seed, and variant again uses the same output
directory and replaces the files from the earlier run.

The observation files contain the charged experimental data. Post-run scores
and recommendation diagnostics are generated only after the episode and are not
provided to the policy as optimization feedback.

## Information boundary

The exercises treat the application problems as black boxes: student decisions
should use the public task specification and charged observations. The private
response formulas are supplied in a platform-specific native wheel rather than
as readable Python source. This prevents ordinary source inspection, but it is
not encryption or a hostile security sandbox. Do not inspect, trace, or call
underscore-prefixed response functions or other private runtime fields.

# HW5 REINFORCE implementation

This package supplies the rollout, policy, model, optimizer, and learner
infrastructure for the HW5 implementation problem. Search the source tree for
`STUDENT TODO` to find the four functions you will complete. The separate
homework PDF is the authoritative assignment description; this README only
documents setup, code organization, and the data passed between components.

## Setup

Python 3.10 or newer is required. Run all commands from this directory.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[visualize,test]"
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[visualize,test]"
```

Run the tests with:

```bash
python -m pytest -q
```

The unfinished scaffold raises `NotImplementedError` when execution reaches a
student section. Once the four sections are implemented, the complete test
suite should run normally. The `visualize` extra installed above includes
Gymnasium and its rendering, video, and plotting dependencies. These are used
by the REINFORCE/A2C comparison and saved-policy viewer below. Tests that
require the optional Optax package are reported as skipped rather than failed.

Optax is not required. To make its optimizer selectable as well, install:

```bash
python -m pip install -e ".[optax,visualize,test]"
```

Development validation has covered Python 3.12 and 3.13, JAX/JAXlib 0.9.0.1
and 0.10.0, NumPy 2.3.x, and pytest 8 and 9. The broader minimum versions are
declared in `pyproject.toml`; those minimums do not claim that every intervening
combination was tested.

## Student edit locations

The intended edit locations are:

- `reinforce_jax/returns.py`: `n_step_returns`
- `reinforce_jax/objectives.py`: `weighted_log_probability_gradient`
- `reinforce_jax/objectives.py`: `reinforce_actor_loss`
- `reinforce_jax/objectives.py`: `critic_loss`

Each location is marked `STUDENT TODO`. Its function docstring is the
authoritative code-level contract: it specifies field meanings, shapes, dtypes,
boundary behavior, reductions, differentiation behavior, and the exact output.
You may add private helper functions if useful, but do not change the public
function signatures because the learner and tests call them directly. GAE,
PPO, pathwise gradients, rollout collection, and optimizer updates are supplied;
they are not additional student implementation sections.

## Running the examples

The native examples need only the base installation and JAX. The bandit example
exercises the return, actor-objective, and policy-gradient code for discrete and
continuous observation/action combinations:

```bash
python -m examples.native_bandits --case all --iterations 60 --platform cpu
```

The actor-critic example additionally exercises the critic objective:

```bash
python -m examples.minimal_actor_critic
```

`minimal_actor_critic` performs one compiled update as a smoke test and is
hard-coded to CPU/float64. It is not a convergence benchmark.

### Gymnasium REINFORCE/A2C comparison

The Gymnasium driver supplies one problem for every combination of discrete or
continuous observations and actions:

| Case | Problem | Observation space | Action space |
|---|---|---|---|
| `dd` | deterministic `FrozenLake-v1` | discrete | discrete |
| `dc` | supplied one-step contextual bandit | discrete | continuous `Box` |
| `cd` | `CartPole-v1` | continuous `Box` | discrete |
| `cc` | `Pendulum-v1` | continuous `Box` | continuous `Box` |

Discrete observations are one-hot encoded, `Box` observations are flattened,
discrete actions use a categorical policy, and continuous actions use a
Gaussian policy transformed to the environment bounds. Gymnasium environment
steps execute on the host CPU. JAX performs policy and learner calculations on
the selected platform.

The following paired runs use the same seeds, collection policy, and rollout
cap. REINFORCE uses complete Monte Carlo returns with no critic.

**Windows PowerShell**

```powershell
python -m examples.gymnasium_train `
    --case all `
    --algorithm reinforce `
    --tail zero `
    --horizon mc `
    --collection episodes `
    --steps 500 `
    --iterations 100 `
    --seeds 0 1 2 `
    --output-dir results/gym-comparison `
    --platform cpu
```

**macOS or Linux**

```bash
python -m examples.gymnasium_train \
    --case all \
    --algorithm reinforce \
    --tail zero \
    --horizon mc \
    --collection episodes \
    --steps 500 \
    --iterations 100 \
    --seeds 0 1 2 \
    --output-dir results/gym-comparison \
    --platform cpu
```

A2C uses a learned state-value critic, eight-step bootstrapped targets, and
critic-based advantages.

**Windows PowerShell**

```powershell
python -m examples.gymnasium_train `
    --case all `
    --algorithm a2c `
    --tail v `
    --horizon 8 `
    --collection episodes `
    --steps 500 `
    --iterations 100 `
    --seeds 0 1 2 `
    --output-dir results/gym-comparison `
    --platform cpu
```

**macOS or Linux**

```bash
python -m examples.gymnasium_train \
    --case all \
    --algorithm a2c \
    --tail v \
    --horizon 8 \
    --collection episodes \
    --steps 500 \
    --iterations 100 \
    --seeds 0 1 2 \
    --output-dir results/gym-comparison \
    --platform cpu
```

The value `--steps 500` covers the episode limits of all four supplied cases.
It therefore allows the episode-mode REINFORCE run to collect complete Monte
Carlo episodes rather than discarding an unfinished episode at the rollout cap.
The displayed iteration count is a starting point, not a promise that every
case and seed converges in exactly 100 iterations. The command-line controls
may be adjusted for an experiment; the separate PDF states the assignment
requirements.

In this package, A2C uses the same score-function actor objective as REINFORCE.
The differences are its learned value baseline, finite-horizon bootstrap, and
separately trained critic. Both commands exercise `n_step_returns`,
`reinforce_actor_loss`, and `weighted_log_probability_gradient`; A2C also
exercises `critic_loss`.

Each run creates:

```text
<output-dir>/<case>/<algorithm>/seed-<seed>/history.csv
<output-dir>/<case>/<algorithm>/seed-<seed>/config.json
<output-dir>/<case>/<algorithm>/seed-<seed>/actor.npz
```

The output directory is created when necessary. Running the same case,
algorithm, and seed again under the same output directory replaces that run's
three files. `config.json` records the relevant run configuration.
`actor.npz` safely stores the final actor parameters without Python pickling.
It is intended for evaluation and visualization; it omits the critic and
optimizer state and therefore cannot resume training. `history.csv` contains
one row after every collection-and-update iteration:

- `iteration` is the one-based outer-loop count. One iteration collects a
  batch, prepares it, and attempts one learner update.
- `training_step` is the cumulative number of successful learner updates after
  that iteration. An actor and critic updated together count as one step. An
  entirely masked batch does not advance the counter.
- `rollout_step` is the cumulative number of valid environment transitions
  across all parallel environments. Padding does not count. One rollout step
  means one environment transition, not one vectorized collector call.
- `batch_rollout_steps` is the number of valid transitions in the current
  iteration.
- `completed_episodes` and `cumulative_episodes` count episodes ending in the
  current iteration and since the run began.
- `average_reward` is the arithmetic mean of immediate rewards over valid
  transitions in the current batch.
- `average_return` is the arithmetic mean of the undiscounted total rewards of
  episodes completed in the current iteration. It is `NaN` if no episode ends.
  This is an observed environment statistic, not an n-step training target.
- `actor_loss`, `critic_loss`, and `valid_training_records` are learner
  diagnostics for the update.

The reward and return in a row were generated by the pre-update policy that
collected the batch consumed by that row's numbered training step. They are not
a separate evaluation of the post-update policy. With the configurations above,
`iteration` and `training_step` normally coincide; both are retained so a
skipped update remains visible. `rollout_step` is the useful axis for comparing
how much environment experience the methods consumed.

The CSV also records the effective tail, horizon, collection policy, and
relevant hyperparameters. `--log-every` affects console output only; the CSV
still records every iteration. Combine files using their `case`, `algorithm`,
and `seed` columns, then plot both `average_reward` and `average_return` against
each of `iteration`, `training_step`, and `rollout_step`. The package
intentionally does not choose the aggregation, smoothing, error display, or
plot style for you. Standard CSV readers or plotting tools can read the files.

### Visualizing a saved policy

After a training run, use its run directory to replay the final actor. For
FrozenLake, CartPole, and Pendulum, this opens a live Gymnasium window:

```bash
python -m examples.visualize_policy --run-dir results/gym-comparison/cd/a2c/seed-0 --episodes 3
```

Live playback requires a graphical desktop. On a remote or headless machine,
or to keep the result, supply a video directory to record MP4 files instead:

```bash
python -m examples.visualize_policy --run-dir results/gym-comparison/cd/a2c/seed-0 --episodes 3 --video-dir results/policy-videos/cd-a2c
```

The contextual bandit has no useful animation. The same command writes a CSV
table and PNG plot comparing the policy action with the target action in every
context:

```bash
python -m examples.visualize_policy --run-dir results/gym-comparison/dc/a2c/seed-0
```

By default, visualization uses the policy's deterministic action and runs JAX
in CPU mode, even if training used a GPU. Add `--sample-actions` to sample from
the learned distribution, or `--platform gpu` to use GPU inference. Run
`python -m examples.visualize_policy --help` for the seed, episode-length, and
output controls.

Immediate reward and episodic return need not tell the same story. In
particular, CartPole normally gives the same immediate reward on every valid
step, so its average-reward curve may be nearly flat even while its episode
return changes.

The first execution may pause while JAX compiles the functions. Use `--help` on
an example that accepts command-line options to see its available controls. For
example, after installing a compatible accelerator-enabled JAX build, run:

```bash
python -m examples.native_bandits \
    --case all --iterations 60 --platform gpu --dtype float32
```

Selecting `--platform gpu` does not install the accelerator runtime or drivers,
and GPU support differs by operating system and hardware. Follow the current
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for the machine being used. The supplied `native_pathwise` example exercises
the already-completed pathwise-gradient route, but it still depends on the
assigned `n_step_returns`; it does not exercise the three objective TODOs.
The `analytic_gradient` example bypasses all four student TODOs.

## Execution path

The main data flow is:

1. An example constructs an environment, policy, actor, optional critic, return
   estimator, objective, and optimizers.
2. `make_rollout` in `reinforce_jax/functional.py`, or the Gymnasium collector,
   records a time-major `Trajectory`.
3. On the assigned n-step path, `Learner.prepare` in
   `reinforce_jax/learner.py` evaluates the configured bootstrap and baseline
   functions. `NStepEstimator` calls `n_step_returns` and produces fixed targets,
   advantages, and a validity mask. The supplied GAE path uses its own estimator.
4. On the assigned REINFORCE/A2C path, `Learner.update` evaluates
   `ReinforceObjective`, obtains the weighted log-probability gradient, and
   applies the actor optimizer. If a critic is configured, it also evaluates
   `critic_loss` and applies the critic optimizer. The supplied PPO path uses
   ordinary autodiff of its own objective instead.
5. The update returns a new immutable `TrainState` and a dictionary of scalar
   diagnostics.

`Learner.prepare` and `Learner.update` are JIT-compiled. Student implementations
must therefore use JAX-compatible operations and preserve static array shapes.
To avoid an otherwise unused policy evaluation, the returned `entropy`
diagnostic is the measured entropy only when `entropy_coef` is nonzero; it is
zero when the entropy bonus is disabled.

## Data conventions

The public containers and callable interfaces are defined in
`reinforce_jax/types.py`. A parameter `PyTree` may be a nested combination of
dictionaries, tuples, lists, and other JAX containers whose leaves are arrays.
A gradient PyTree has the same structure and leaf shapes as its parameter
PyTree.

Trajectory arrays are time-major. Their first two axes are `[time, batch]`:

- `observations`: encoded observations, with any feature axes following the
  leading two axes;
- `actions`: actions sent to the environment;
- `raw_actions`: samples in the policy distribution's coordinate system;
- `rewards`: scalar rewards with shape `[time, batch]`;
- `next_observations`: the actual observation returned by each environment
  step, before any reset;
- `terminated`, `truncated`, and `valid`: Boolean arrays with shape
  `[time, batch]`;
- `episode_steps`: zero-based integer within-episode step indices with shape
  `[time, batch]`; the first transition is step 0 and the index resets to 0
  after an episode boundary;
- `behavior_log_probs`: scalar recorded-action log probabilities with shape
  `[time, batch]`.

For categorical policies, an action is an integer category index. For bounded
Gaussian policies, `actions` contains the transformed environment action while
`raw_actions` contains its pre-transform sample. Policy probability evaluation
uses the recorded `raw_actions`.

`n_step_returns` receives a `Trajectory`, a `next_values` array with shape
`[time, batch]`, and a `ReturnConfig`. Each `next_values[t, b]` is the scalar
bootstrap value associated with `next_observations[t, b]`. It returns a
`ReturnTargets` containing numeric `returns`, Boolean `mask`, and integer
`steps`, all with shape `[time, batch]`. A false returned mask does not mean the
numeric return is absent; it means that entry is excluded from training.

`weighted_log_probability_gradient` receives observation and raw-action arrays,
a fixed scalar coefficient for every `[time, batch]` entry, and the actor
parameter PyTree. It returns one aggregate gradient PyTree matching those
parameters, not a separate gradient PyTree per transition. The coefficients
already contain all signs, masks, discounts, advantages, and reduction factors.

`reinforce_actor_loss` receives arrays of log probabilities, advantages, and
entropy values together with a `PreparedBatch` and the objective configuration.
The three arrays and the batch mask all have shape `[time, batch]`. It returns
one scalar JAX array. A `PreparedBatch` contains the trajectory and numeric
`targets`, `advantages`, and Boolean `mask`, each with leading `[time, batch]`
axes. Episode reduction divides by the batch-lane count; transition reduction
divides by the number of true mask entries. See the function docstring for the
complete sign, discount, entropy, and all-masked behavior.

`critic_loss` receives a critic callable, critic parameters, and a
`PreparedBatch`. The critic callable accepts one observation and one action and
returns one scalar prediction. A state-value critic may ignore its action
argument. The function uses recorded environment `actions`, not `raw_actions`,
and returns one half of the masked mean squared prediction error. Prepared data
and targets are fixed; critic differentiation is with respect to critic
parameters only.

## Episode and batch boundaries

The return code must preserve these supplied data conventions:

- A true termination ends the trajectory contribution and has no value beyond
  the terminal transition.
- A truncation ends the recorded episode but retains its actual final
  `next_observation`, which may be used for bootstrapping.
- A return must never continue through a reset into the next episode.
- `valid=False` marks padding or unavailable data. Invalid entries do not
  contribute to actor or critic training.
- With `missing="bootstrap"`, an incomplete recorded future may use the value at
  its last available next observation. With `missing="drop"`, an entry lacking
  the requested future information is excluded through the returned mask.
- A true terminal completes a return even when it occurs before a finite
  horizon, so that entry is retained in drop mode and has a zero tail.
- Reaching the configured horizon makes that horizon available even when the
  last included transition is truncated.
- If `terminated` and `truncated` are both true, termination takes precedence.
- `horizon=None` uses the available episode segment through its recorded
  boundary; it cannot recover rewards that were never collected. In drop mode,
  only starts that reach a recorded true terminal remain active.

Masks retain fixed array shapes under JIT. If a prepared batch contains no
active entries, the supplied learner leaves parameters, optimizer state, and
the update counter unchanged.

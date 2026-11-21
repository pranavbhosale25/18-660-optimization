# Optimization HW5 submission

## Deliverables

- [Complete submission PDF](output/pdf/hw5_complete_submission.pdf): handwritten theory followed by the Q2 and Q3 writeups, with all graphs.
- [Q2 and Q3 writeups PDF](output/pdf/q2_q3_writeups.pdf): implementation and application writeups only.

| Part | Main deliverable | Supporting files |
|---|---|---|
| Theory, Exercise 1.1 | [Handwritten theory PDF](Theory/hw5.pdf) | Submitted handwritten solution |
| Implementation, Exercise 2.1 | [Bullet-point writeup](Implementation/writeup.md) | Completed code, 24 training runs, four six-panel figures, eight MP4 policy videos |
| Application, Exercise 3.1 | [Bullet-point writeup](Application/writeup.md#exercise-31-bayesian-optimization) | Five completed plans, 15 full-budget runs, plots, observations, and configurations |
| Application, Exercise 3.2 | [Bullet-point writeup](Application/writeup.md#exercise-32-learning-to-warm-start) | Four completed functions, step-size derivations, validation tuning, nominal/shifted evaluations, and checkpoints |

- Includes Exercise 3.1, which is required for 18-660 and extra credit for 18-460.
- Does not attempt optional Exercise 2.2 (diffusion reinforcement learning).
- Preserves all 13 assigned TODO locations and their original public signatures.
- The supplied assignment file is named `Optimization_HW5.pdf` but says “Homework #4” inside. This submission uses HW5, following the filename and distribution.

## Install

- Use **64-bit Python 3.12**. The supplied simulator wheels support Apple-silicon macOS, Linux x86-64/ARM64, and Windows x86-64. Intel macOS is not supplied by the course.
- From the extracted submission root, run:

```bash
python3.12 scripts/setup_envs.py
```

- On Windows, replace `python3.12` with `py -3.12` in these commands.
- Setup creates two isolated environments because the supplied packages require different JAX versions:

| Environment | Purpose | JAX | NumPy |
|---|---|---:|---:|
| `.venv` | Bayesian optimization, RL, plotting, video | 0.10.0 | 2.5.5 |
| `.venv-l2ws` | Warm-start training and benchmarks | 0.12.0.1 | 2.5.4 |

- Exact external versions are in [requirements-main.txt](requirements-main.txt) and [requirements-warmstart.txt](requirements-warmstart.txt). Setup also installs the supplied local simulator wheel and editable homework packages.
- Installation downloads public Python packages; the simulator runtime itself comes from the included course wheels.
- The submission excludes installed environments and caches. No environment activation is needed for the commands below.

## Verify

```bash
python3.12 reproduce.py tests
```

- Verified on Apple-silicon macOS, Python 3.12.14, CPU:
  - Bayesian optimization: **112 passed**.
  - REINFORCE/A2C: **90 passed**.
  - Warm-start package: **49 passed**, including four additional mathematical checks.
  - Both supplied nonconvex problem checkers passed float64 gradient, JIT, feasibility, sampled-descent, and short-training checks.
- Total: **251 passing tests**, with no skipped tests in the final run.
- Full output: [final_validation.log](Application/results/logs/final_validation.log).

## Reproduce the full experiment set

- Preserve the submitted evidence by writing a separate artifact tree:

```bash
python3.12 reproduce.py all --output-dir reproduced
```

- This runs the tests, all BO comparisons, all 24 RL training runs, warm-start baseline and tuning benchmarks, eight policy videos, and every custom figure.
- New data and figures go under `reproduced/Application/` and `reproduced/Implementation/`; submitted writeups and original results remain in their original directories.
- Use a **new output directory** for another full rerun. The supplied warm-start benchmark intentionally refuses an already existing problem-output folder.
- The runner prints each exact underlying command. Workload and JAX compilation time depend on the machine; training is CPU-based.
- Small floating-point, timing, or platform-dependent differences are possible. The compiled simulator requires a supported platform.

## Run individual parts

```bash
python3.12 reproduce.py bo
python3.12 reproduce.py rl
python3.12 reproduce.py videos
python3.12 reproduce.py plots
```

- These defaults reproduce BO/RL/video/plot outputs in the submitted locations and replace the corresponding generated files.
- For a separate warm-start rerun, use:

```bash
python3.12 reproduce.py warmstart --output-dir reproduced-warmstart
```

- To rerun any other part under a separate artifact root, use the same `--output-dir` option consistently for its inputs and outputs.
- `plots` expects all BO, RL, and tuned warm-start outputs under the selected root. It regenerates figures and summary CSVs, not the prose writeups.
- The direct baseline warm-start command is also documented in the original `Application/P2/l2ws_jax/README.md`.
- All policy videos use headless Gymnasium rendering with the final seed-0 actor, deterministic actions, and evaluation seeds 1000 and 1001. The video files are playable MP4s; no live window is required.

## Findings and limits

- All five primary BO configurations meet the saved development-instance target; unsuccessful comparisons are included.
- Validation-selected wider warm-start networks improve the tested solution quality; all required refinement horizons and both test distributions are reported.
- The RL experiments learn FrozenLake and useful bandit/CartPole behavior. **Pendulum remains weak within the 100-iteration budget**; its unsuccessful results and final-policy videos are included explicitly.

## AI usage disclosure

**OpenAI Codex (GPT-6)** assisted with mathematical reasoning, implementation, experiments, figures, videos, and writeup editing.

#!/usr/bin/env bash
# Reproduce the two code exercises from the directory containing this script.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_PYTHON="${TASK_PYTHON:-$TASK_ROOT/.venv/bin/python}"
export WARP_CACHE_PATH="$TASK_ROOT/.cache/warp"
export MPLCONFIGDIR="$TASK_ROOT/.cache/matplotlib"
export XDG_CACHE_HOME="$TASK_ROOT/.cache"

cd "$TASK_ROOT/Implementation/jax_sqp"
mkdir -p results
"$TASK_PYTHON" -m pytest -q 2>&1 | tee results/pytest.log
"$TASK_PYTHON" -u demo.py --device cpu --precision 64 --suite all --batch \
  --output results/cpu64_all 2>&1 | tee results/demo_cpu64_all.log
"$TASK_PYTHON" -u demo.py --device cpu --precision 64 --suite small \
  --merit-slope exact --output results/cpu64_exact 2>&1 | tee results/demo_cpu64_exact.log
"$TASK_PYTHON" -u demo.py --device cpu --precision 32 --suite precision \
  --output results/cpu32_precision 2>&1 | tee results/demo_cpu32_precision.log
"$TASK_PYTHON" plot_results.py --input results/cpu64_all 2>&1 | tee results/plot_results.log
"$TASK_PYTHON" plot_results.py --input results/cpu64_exact
"$TASK_PYTHON" plot_results.py --input results/cpu32_precision

cd "$TASK_ROOT/Application/bear_shooting_pyomo"
mkdir -p results
"$TASK_PYTHON" -u scripts/check_environment.py --test-ipopt 2>&1 | tee results/environment.log
"$TASK_PYTHON" -m pytest -q 2>&1 | tee results/pytest.log
"$TASK_PYTHON" -u examples/bear_run.py --size quick --device cpu 2>&1 | tee results/bear_run_quick.log
"$TASK_PYTHON" -u scripts/verify_bear_run.py 2>&1 | tee results/verify_bear_run.log
"$TASK_PYTHON" scripts/render_deliverables.py 2>&1 | tee results/render_deliverables.log
"$TASK_PYTHON" -c 'from bear_shooting.export import replay_to_usd; print(replay_to_usd("results/bear_run_quick_particles.npz", "results/bear_run_quick.usda"))' \
  2>&1 | tee results/export_usd.log

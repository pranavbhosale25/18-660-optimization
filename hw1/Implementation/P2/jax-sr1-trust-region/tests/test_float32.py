from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_float32_solver_smoke_in_fresh_process() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "false"
    code = """
import trsr1
trsr1.configure_runtime('cpu', 32)
import jax.numpy as jnp
problem = trsr1.problem_by_name('rosenbrock')
options = trsr1.SR1Options(
    max_iterations=300,
    gradient_tolerance=5e-4,
    store_history=False,
)
result = trsr1.make_solver(problem.fun, options)(
    problem.initial_point(jnp.float32)
)
assert result.x.dtype == jnp.float32
assert bool(result.success)
assert float(result.grad_norm) <= 5e-4
print('ok')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"

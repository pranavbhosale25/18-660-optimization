"""Run the benchmark driver without importing JAX in the parent process."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runner = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "unified_jax_optim"
        / "benchmarks"
        / "principled_benchmark_runner.py"
    )
    runpy.run_path(str(runner), run_name="__main__")

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(code: str, **environment: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(environment)
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_package_import_is_lazy_with_respect_to_jax() -> None:
    completed = _run(
        "import sys; import trsr1; print('jax' in sys.modules)"
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"


@pytest.mark.parametrize(
    ("precision", "expected_dtype", "expected_x64"),
    [(32, "float32", "False"), (64, "float64", "True")],
)
def test_runtime_precision_and_cpu_selection(
    precision: int, expected_dtype: str, expected_x64: str
) -> None:
    completed = _run(
        f"""
import trsr1
info = trsr1.configure_runtime('cpu', {precision})
import jax.numpy as jnp
print(info.default_backend)
print(info.x64_enabled)
print(jnp.asarray([1.0]).dtype)
""",
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="true" if precision == 64 else "false",
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.strip().splitlines()
    assert lines == ["cpu", expected_x64, expected_dtype]


def test_invalid_runtime_options_fail_early() -> None:
    import trsr1

    with pytest.raises(ValueError):
        trsr1.configure_runtime("quantum", 64)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        trsr1.configure_runtime("cpu", 16)  # type: ignore[arg-type]


def test_explicit_gpu_request_never_silently_falls_back_to_cpu() -> None:
    completed = _run(
        """
import trsr1
try:
    info = trsr1.configure_runtime('gpu', 32)
except RuntimeError:
    print('unavailable')
else:
    print(f'selected:{info.default_backend}')
"""
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() in {"unavailable", "selected:gpu"}

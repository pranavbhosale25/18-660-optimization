from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jax_pncg import PNCGOptions, configure_runtime, make_solver
from jax_pncg.test_problems import ROSENBROCK_2


@pytest.mark.parametrize(
    "precision, dtype, gtol",
    [
        ("float32", jnp.float32, 1.0e-5),
        ("float64", jnp.float64, 1.0e-8),
    ],
)
def test_cpu_and_precision_switches(precision, dtype, gtol: float) -> None:
    runtime = configure_runtime(platform="cpu", precision=precision)
    x0 = runtime.asarray(ROSENBROCK_2.start)
    result = make_solver(
        ROSENBROCK_2.objective,
        options=PNCGOptions(maxiter=1000, gtol=gtol),
    )(x0)

    assert x0.dtype == dtype
    assert result.x.dtype == dtype
    assert runtime.device.platform == "cpu"
    assert all(device.platform == "cpu" for device in result.x.devices())
    assert bool(result.success)


def test_gpu_mode_is_selected_or_fails_clearly() -> None:
    try:
        runtime = configure_runtime(platform="gpu", precision="float32")
    except RuntimeError as error:
        assert "GPU mode was requested" in str(error)
    else:
        assert runtime.platform == "gpu"
        assert runtime.device.platform == "gpu"


def test_auto_mode_returns_a_real_jax_device() -> None:
    runtime = configure_runtime(platform="auto", precision="float32")
    assert runtime.device in jax.devices(runtime.platform)

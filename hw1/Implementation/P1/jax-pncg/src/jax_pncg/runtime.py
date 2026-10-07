from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp


Platform = Literal["auto", "cpu", "gpu"]
Precision = Literal["float32", "float64"]


@dataclass(frozen=True)
class Runtime:
    """Selected JAX precision and execution device."""

    platform: Platform
    precision: Precision
    device: Any

    @property
    def dtype(self) -> Any:
        return jnp.float64 if self.precision == "float64" else jnp.float32

    def asarray(self, value: Any) -> jax.Array:
        """Create a floating-point array directly on this runtime's device."""

        with jax.default_device(self.device):
            array = jnp.asarray(value, dtype=self.dtype)
        return jax.device_put(array, self.device)

    def default_device(self) -> AbstractContextManager[Any]:
        """Context that makes this runtime's device the JAX default."""

        return jax.default_device(self.device)


def _devices_or_empty(platform: str) -> list[Any]:
    try:
        return list(jax.devices(platform))
    except RuntimeError:
        return []


def configure_runtime(
    *,
    platform: Platform = "auto",
    precision: Precision = "float64",
    device_index: int = 0,
) -> Runtime:
    """Configure X64 and select a CPU or GPU device.

    Call this before creating arrays or compiling functions.  ``platform='cpu'``
    keeps all arrays and JIT calls placed through the returned runtime on CPU;
    ``platform='gpu'`` requires an installed GPU-enabled JAX backend; and
    ``platform='auto'`` prefers a GPU when one is available.
    """

    if platform not in ("auto", "cpu", "gpu"):
        raise ValueError("platform must be 'auto', 'cpu', or 'gpu'")
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be 'float32' or 'float64'")
    if device_index < 0:
        raise ValueError("device_index must be nonnegative")

    jax.config.update("jax_enable_x64", precision == "float64")

    if platform == "auto":
        devices = _devices_or_empty("gpu")
        selected_platform: Platform = "gpu" if devices else "cpu"
        if not devices:
            devices = _devices_or_empty("cpu")
    else:
        selected_platform = platform
        devices = _devices_or_empty(platform)

    if not devices:
        if selected_platform == "gpu":
            raise RuntimeError(
                "GPU mode was requested, but JAX did not report a GPU device. "
                "Install a GPU-enabled jaxlib build and verify the driver/runtime."
            )
        raise RuntimeError("JAX did not report any CPU devices")
    if device_index >= len(devices):
        raise IndexError(
            f"device_index={device_index} is out of range for "
            f"{len(devices)} {selected_platform} device(s)"
        )

    return Runtime(
        platform=selected_platform,
        precision=precision,
        device=devices[device_index],
    )

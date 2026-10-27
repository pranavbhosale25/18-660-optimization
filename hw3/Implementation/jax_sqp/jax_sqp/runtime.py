"""Explicit precision and device selection. Configure before creating arrays."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Runtime:
    device: Any
    dtype: Any
    precision: int
    platform: str


def configure(
    *,
    platform: str = "cpu",
    precision: int = 64,
    disable_jit: bool = False,
    device_index: int = 0,
) -> Runtime:
    # Delay JAX import so callers can set JAX_PLATFORMS before backend startup.
    import jax
    import jax.numpy as jnp

    if platform not in ("cpu", "gpu", "auto"):
        raise ValueError("platform must be 'cpu', 'gpu', or 'auto'.")
    if precision not in (32, 64):
        raise ValueError("precision must be 32 or 64.")
    # These options are process-wide and must be set before arrays are created
    # or computations are compiled.
    jax.config.update("jax_enable_x64", precision == 64)
    jax.config.update("jax_disable_jit", disable_jit)
    jax.config.update("jax_default_matmul_precision", "highest")
    chosen = platform
    if chosen == "auto":
        # Auto mode prefers an accelerator but deliberately falls back to CPU.
        try:
            devices = jax.devices("gpu")
            chosen = "gpu"
        except RuntimeError:
            chosen = "cpu"
            devices = jax.devices("cpu")
    else:
        # An explicitly requested backend must exist; never silently substitute.
        try:
            devices = jax.devices(chosen)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Requested {chosen} backend is unavailable. Install a compatible JAX accelerator build or select cpu."
            ) from exc
    if not 0 <= device_index < len(devices):
        raise ValueError(
            f"device_index={device_index} invalid for {len(devices)} {chosen} device(s)."
        )
    device = devices[device_index]
    jax.config.update("jax_default_device", device)
    return Runtime(
        device, jnp.float64 if precision == 64 else jnp.float32, precision, chosen
    )

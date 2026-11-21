"""Explicit precision and device policy with no import-time JAX changes."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

_DEVICE_SPEC_PART_COUNT = 2


def resolve_dtype(dtype="float64") -> np.dtype:
    value = np.dtype(dtype)
    if value not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    # Enable capability once, never disable it when a float32 problem is made.
    # Every floating model/data array still receives its explicit problem dtype.
    # This permits independent CPU float64 reference solves in float32 runs.
    if not jax.config.x64_enabled:
        jax.config.update("jax_enable_x64", True)
    # Request full float32 multiply precision on accelerators rather than
    # silently evaluating nominal float32 objectives with reduced products.
    if jax.config.jax_default_matmul_precision != "highest":
        jax.config.update("jax_default_matmul_precision", "highest")
    return value


def resolve_device(device="auto") -> jax.Device:
    if isinstance(device, jax.Device):
        return device
    if device is None or device == "auto":
        return jax.devices()[0]  # JAX prefers an installed accelerator backend.
    parts = str(device).split(":")
    if len(parts) > _DEVICE_SPEC_PART_COUNT or parts[0] not in ("cpu", "gpu"):
        raise ValueError("device must be auto, cpu, gpu, cpu:N, or gpu:N")
    try:
        index = int(parts[1]) if len(parts) == _DEVICE_SPEC_PART_COUNT else 0
    except ValueError as error:
        raise ValueError(
            "device index must be a nonnegative integer"
        ) from error
    if index < 0:
        raise ValueError("device index must be nonnegative")
    try:
        devices = jax.devices(parts[0])
    except RuntimeError as error:
        raise RuntimeError(
            f"Requested {parts[0]} backend is unavailable. For an NVIDIA GPU, "
            "install the compatible JAX CUDA extra described in README.md; "
            "use device='cpu' for CPU execution."
        ) from error
    if index >= len(devices):
        raise ValueError(
            f"Requested {device}; only {len(devices)} {parts[0]} "
            "device(s) available"
        )
    return devices[index]


@dataclass(frozen=True)
class Runtime:
    dtype: np.dtype
    device: jax.Device

    @classmethod
    def create(cls, dtype="float64", device="auto"):
        return cls(resolve_dtype(dtype), resolve_device(device))

    def array(self, value, dtype=None):
        with jax.default_device(self.device):
            return jax.device_put(
                jnp.asarray(
                    value, dtype=self.dtype if dtype is None else dtype
                ),
                self.device,
            )

    def key(self, seed: int):
        with jax.default_device(self.device):
            return jax.device_put(jax.random.PRNGKey(seed), self.device)

    def info(self):
        return {
            "dtype": self.dtype.name,
            "backend": self.device.platform,
            "device": str(self.device),
            "device_kind": self.device.device_kind,
            "jax": jax.__version__,
            "x64_enabled": bool(jax.config.x64_enabled),
        }


def synchronize(value):
    """Wait for device work on all array leaves; needed for honest timing."""
    return jax.block_until_ready(value)

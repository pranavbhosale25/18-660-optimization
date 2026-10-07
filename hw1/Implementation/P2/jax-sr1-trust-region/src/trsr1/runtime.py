"""Process-wide JAX backend and floating-point configuration.

This module intentionally imports no JAX modules at import time. Call
``configure_runtime`` immediately after importing :mod:`trsr1`, before touching
solver symbols or importing :mod:`jax.numpy`.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Literal

Platform = Literal["auto", "cpu", "gpu", "tpu"]


@dataclass(frozen=True)
class RuntimeInfo:
    """Description of the JAX runtime selected for this process."""

    requested_platform: str
    default_backend: str
    precision_bits: int
    x64_enabled: bool
    devices: tuple[str, ...]
    jax_was_already_imported: bool


def configure_runtime(
    platform: Platform = "auto",
    precision: Literal[32, 64] = 64,
    *,
    validate: bool = True,
) -> RuntimeInfo:
    """Configure JAX platform selection and 32-/64-bit behavior.

    Parameters
    ----------
    platform:
        ``"cpu"`` forces CPU-only initialization, which is the explicit
        GPU-off mode. ``"gpu"`` and ``"tpu"`` require the requested backend.
        ``"auto"`` leaves backend selection to JAX (and preserves an existing
        ``JAX_PLATFORMS`` environment setting).
    precision:
        ``32`` disables JAX x64 mode; ``64`` enables it.
    validate:
        Initialize the backend and verify that an explicitly requested platform
        is actually active. Set this to ``False`` only when deferred backend
        initialization is important.

    Notes
    -----
    JAX configuration is process-wide. Platform configuration should happen
    before a backend is initialized. If JAX was already imported, this function
    still attempts the requested update and raises a clear error if validation
    shows that the request did not take effect.
    """

    normalized_platform = platform.lower()
    if normalized_platform not in {"auto", "cpu", "gpu", "tpu"}:
        raise ValueError("platform must be one of: auto, cpu, gpu, tpu")
    if precision not in {32, 64}:
        raise ValueError("precision must be either 32 or 64")

    jax_was_already_imported = "jax" in sys.modules

    if normalized_platform != "auto":
        os.environ["JAX_PLATFORMS"] = normalized_platform
    os.environ["JAX_ENABLE_X64"] = "true" if precision == 64 else "false"

    import jax  # Imported only after the environment has been configured.

    try:
        if normalized_platform != "auto":
            jax.config.update("jax_platforms", normalized_platform)
        jax.config.update("jax_enable_x64", precision == 64)
    except Exception as exc:  # JAX exception types have changed across releases.
        raise RuntimeError(
            "JAX rejected the requested runtime configuration. Configure "
            "trsr1 before importing JAX or initializing a backend."
        ) from exc

    if validate:
        try:
            devices = tuple(jax.devices())
            default_backend = jax.default_backend()
        except Exception as exc:
            requested = (
                normalized_platform
                if normalized_platform != "auto"
                else "an automatically selected backend"
            )
            raise RuntimeError(
                f"JAX could not initialize {requested}. Ensure the installed "
                "jaxlib build supports the requested hardware."
            ) from exc

        if normalized_platform != "auto":
            platform_devices = tuple(
                device for device in devices if device.platform == normalized_platform
            )
            if default_backend != normalized_platform or not platform_devices:
                raise RuntimeError(
                    f"Requested JAX platform {normalized_platform!r}, but the "
                    f"active default backend is {default_backend!r}. Configure "
                    "the runtime before importing JAX and install a compatible "
                    "accelerator-enabled jaxlib build when needed."
                )
    else:
        devices = ()
        default_backend = normalized_platform

    return RuntimeInfo(
        requested_platform=normalized_platform,
        default_backend=default_backend,
        precision_bits=precision,
        x64_enabled=bool(jax.config.x64_enabled),
        devices=tuple(str(device) for device in devices),
        jax_was_already_imported=jax_was_already_imported,
    )

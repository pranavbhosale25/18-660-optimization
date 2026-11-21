"""Process-level precision and device selection. Call BEFORE creating arrays."""
import jax
import jax.numpy as jnp


def configure(*, dtype: str = "float32", platform: str | None = None):
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")
    if platform not in (None, "cpu", "gpu", "tpu"):
        raise ValueError("platform must be None, 'cpu', 'gpu', or 'tpu'")
    jax.config.update("jax_enable_x64", dtype == "float64")
    if platform is not None:
        try:
            devices = jax.devices(platform)
        except RuntimeError as exc:
            raise RuntimeError(f"Requested {platform!r} backend is unavailable") from exc
        if not devices:
            raise RuntimeError(f"No {platform!r} device is available")
        jax.config.update("jax_default_device", devices[0])
    return jnp.float64 if dtype == "float64" else jnp.float32

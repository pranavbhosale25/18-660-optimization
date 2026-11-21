"""Composable covariance functions: k(parameters, X[n,d], Z[m,d]) -> [n,m].

Lengths are conventional length scales (distance divided by ell). The review's
precision-matrix notation can be matched by taking its diagonal to be ell^-2.
Amplitude is a standard deviation: kernels multiply by exp(2*log_amplitude).
"""
from __future__ import annotations
from typing import Callable
import jax
import jax.numpy as jnp
from .linalg import safe_sqrt

Kernel = Callable[[dict, jax.Array, jax.Array], jax.Array]


def stationary_parameters(d: int, lengthscale=0.2, amplitude=1.0) -> dict:
    return {"log_lengthscale": jnp.log(jnp.broadcast_to(jnp.asarray(lengthscale, dtype=float), (d,))),
            "log_amplitude": jnp.log(jnp.asarray(amplitude, dtype=float))}


def _distance2(p, x, z):
    delta = (x[:, None, :] - z[None, :, :]) / jnp.exp(p["log_lengthscale"])
    return jnp.sum(delta * delta, axis=-1)


def _amplitude(p):
    return jnp.exp(2.0 * p["log_amplitude"])


def rbf(p, x, z):
    return _amplitude(p) * jnp.exp(-0.5 * _distance2(p, x, z))


def matern12(p, x, z):
    return _amplitude(p) * jnp.exp(-safe_sqrt(_distance2(p, x, z)))


def matern32(p, x, z):
    r = safe_sqrt(3.0 * _distance2(p, x, z))
    return _amplitude(p) * (1 + r) * jnp.exp(-r)


def matern52(p, x, z):
    r2 = 5.0 * _distance2(p, x, z)
    r = safe_sqrt(r2)
    return _amplitude(p) * (1 + r + r2 / 3.0) * jnp.exp(-r)


def periodic(p, x, z):
    period = jnp.exp(p["log_period"])
    delta = jnp.sin(jnp.pi * (x[:, None, :] - z[None, :, :]) / period)
    r2 = jnp.sum((delta / jnp.exp(p["log_lengthscale"])) ** 2, axis=-1)
    return _amplitude(p) * jnp.exp(-2.0 * r2)


def linear(p, x, z):
    scale = jnp.exp(p["log_lengthscale"])
    center = p.get("center", jnp.zeros(x.shape[-1], dtype=x.dtype))
    return _amplitude(p) * ((x-center) / scale) @ ((z-center) / scale).T


def constant(p, x, z):
    return jnp.broadcast_to(_amplitude(p), (x.shape[0], z.shape[0]))


def diagonal(kernel: Kernel, p, x):
    # Avoid constructing an n x n Gram matrix to obtain n marginal variances.
    return jax.vmap(lambda t: kernel(p, t[None, :], t[None, :])[0, 0])(x)


def add(left: Kernel, right: Kernel) -> Kernel:
    return lambda p, x, z: left(p["left"], x, z) + right(p["right"], x, z)


def product(left: Kernel, right: Kernel) -> Kernel:
    return lambda p, x, z: left(p["left"], x, z) * right(p["right"], x, z)


def active_dimensions(kernel: Kernel, indices: tuple[int, ...]) -> Kernel:
    idx = jnp.array(indices)
    return lambda p, x, z: kernel(p, x[:, idx], z[:, idx])


def warped(kernel: Kernel, warp: Callable) -> Kernel:
    """Parameters: {'base': kernel params, 'warp': warp params}."""
    return lambda p, x, z: kernel(p["base"], warp(p["warp"], x), warp(p["warp"], z))


def partitioned(kernel: Kernel, region: Callable) -> Kernel:
    """Independent GPs in a USER-SPECIFIED partition; not Bayesian tree learning."""
    return lambda p, x, z: kernel(p, x, z) * (region(x)[:, None] == region(z)[None, :])


def coregionalized(kernel: Kernel, task_column: int = -1) -> Kernel:
    """ICM, Eq. 62. The last column is an integer task ID by default.

    p={'base': ..., 'task_factor': L, 'log_task_diagonal': vector}.
    B = L L^T + diag(exp(log_task_diagonal)); task IDs must be in range.
    """
    def k(p, x, z):
        tx, tz = x[:, task_column].astype(int), z[:, task_column].astype(int)
        xx, zz = jnp.delete(x, task_column, axis=1), jnp.delete(z, task_column, axis=1)
        factor = p["task_factor"]
        b = factor @ factor.T + jnp.diag(jnp.exp(p["log_task_diagonal"]))
        return kernel(p["base"], xx, zz) * b[tx[:, None], tz[None, :]]
    return k


def freeze_thaw(p, x, z):
    """Eq. 63, time-only kernel, t >= 0. x and z have one time column."""
    a, b = jnp.exp(p["log_alpha"]), jnp.exp(p["log_beta"])
    return jnp.exp(a * (jnp.log(b) - jnp.log(x[:, :1] + z[:, 0][None, :] + b)))


KERNELS = {"rbf": rbf, "matern12": matern12, "matern32": matern32,
           "matern52": matern52, "periodic": periodic, "linear": linear}

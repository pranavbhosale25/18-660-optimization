"""Small, differentiable numerical primitives. No NumPy/SciPy host computation."""
from __future__ import annotations
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

Array = jax.Array


def symmetrize(a: Array) -> Array:
    return (a + a.T) * 0.5


@jax.custom_jvp
def safe_sqrt(x: Array) -> Array:
    """sqrt(max(x, 0)), choosing the zero subgradient at zero."""
    return jnp.sqrt(jnp.maximum(x, 0.0))


@safe_sqrt.defjvp
def _safe_sqrt_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    root = safe_sqrt(x)
    denominator = jnp.maximum(root, jnp.finfo(root.dtype).tiny)
    return root, jnp.where(x > 0, 0.5 * dx / denominator, 0.0)


def cholesky(a: Array, jitter: float = 1e-8) -> Array:
    """Cholesky with a dtype-aware, relative diagonal regularizer.

    Jitter is numerical regularization, NOT modeled observation noise. This
    routine does not silently project a genuinely indefinite matrix onto PSD.
    A failed factorization produces nonfinite output (the high-level API checks).
    """
    scale = jnp.maximum(jnp.mean(jnp.abs(jnp.diag(a))), 1.0)
    eps = jnp.maximum(jnp.asarray(jitter, a.dtype), 10 * jnp.finfo(a.dtype).eps)
    return jnp.linalg.cholesky(symmetrize(a) + eps * scale * jnp.eye(a.shape[0], dtype=a.dtype))


def chol_solve(l: Array, b: Array) -> Array:
    return solve_triangular(l.T, solve_triangular(l, b, lower=True), lower=False)


def psd_root(a: Array) -> Array:
    """Root for posterior sampling; clips small negative roundoff eigenvalues.

    This is a sampling helper, not a repair for an invalid prior kernel.
    """
    values, vectors = jnp.linalg.eigh(symmetrize(a))
    return vectors * safe_sqrt(values)[None, :]


def normal_samples(key: Array, mean: Array, covariance: Array, n: int) -> Array:
    z = jax.random.normal(key, (n, mean.shape[0]), dtype=mean.dtype)
    return mean[None, :] + z @ psd_root(covariance).T


def logdet_chol(l: Array) -> Array:
    return 2.0 * jnp.log(jnp.diag(l)).sum()


def gaussian_logpdf(x: Array, mean: Array, variance: Array) -> Array:
    variance = jnp.maximum(variance, jnp.finfo(jnp.result_type(x, mean, variance)).tiny)
    return -0.5 * (jnp.log(2 * jnp.pi * variance) + (x - mean) ** 2 / variance)

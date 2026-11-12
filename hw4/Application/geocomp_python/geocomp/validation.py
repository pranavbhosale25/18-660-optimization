"""Out-of-sample validation kept independent of optimizer quadrature samples."""

import numpy as np
from scipy.stats import beta
import jax
import jax.numpy as jnp


def binomial_summary(successes, n, confidence=0.95):
    """Exact two-sided Clopper-Pearson interval (not a Bayesian interval)."""
    if not 0 <= successes <= n or n < 1:
        raise ValueError("Invalid binomial counts")
    tail = (1.0 - confidence) / 2.0
    lo = 0.0 if successes == 0 else beta.ppf(tail, successes, n - successes + 1)
    hi = 1.0 if successes == n else beta.ppf(1.0 - tail, successes + 1, n - successes)
    return {
        "successes": int(successes),
        "n": int(n),
        "estimate": float(successes / n),
        "confidence": confidence,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
    }


def validate_instance(instance, x, n_samples=100_000, seed=20261, batch_size=2048):
    """Paired baseline/optimized trials, new JAX random keys, exact event checks.

    Full-mesh test uses the maximum norm of corresponding vertex displacement.
    Piecewise-linear interpolation bounds the displacement over every triangle.
    Coefficient-box and mesh events are reported SEPARATELY: the box is sufficient
    but not necessary. Binomial intervals cover MC sampling error only, not model
    misspecification, covariance-estimation error, or CAD tessellation error.
    """
    if n_samples < 1 or batch_size < 1:
        raise ValueError("n_samples and batch_size must be positive")
    p = instance.problem
    n_modes = p.n_features
    L = jnp.linalg.cholesky(jnp.asarray(p.covariance))
    mu0 = p.mean(jnp.zeros(p.n_decisions))
    mu1 = p.mean(jnp.asarray(x))
    lower = jnp.asarray(p.lower)
    upper = jnp.asarray(p.upper)
    tau = instance.command_scale_mm

    @jax.jit
    def batch(key):
        eps = jax.random.normal(key, (batch_size, n_modes), dtype=jnp.float64) @ L.T
        a0 = mu0 + eps
        a1 = mu1 + eps
        box0 = jnp.all((a0 >= lower) & (a0 <= upper), axis=1)
        box1 = jnp.all((a1 >= lower) & (a1 <= upper), axis=1)
        max0 = instance.basis.max_displacement_batch(a0)
        max1 = instance.basis.max_displacement_batch(a1)
        return box0, box1, max0, max1

    counts = {
        "baseline_box": 0,
        "optimized_box": 0,
        "baseline_mesh": 0,
        "optimized_mesh": 0,
    }
    max0s = []
    max1s = []
    key = jax.random.PRNGKey(seed)
    for offset in range(0, n_samples, batch_size):
        key, sub = jax.random.split(key)
        b0, b1, m0, m1 = [np.asarray(v) for v in batch(sub)]
        keep = min(batch_size, n_samples - offset)
        b0, b1, m0, m1 = b0[:keep], b1[:keep], m0[:keep], m1[:keep]
        counts["baseline_box"] += int(b0.sum())
        counts["optimized_box"] += int(b1.sum())
        counts["baseline_mesh"] += int((m0 <= tau).sum())
        counts["optimized_mesh"] += int((m1 <= tau).sum())
        # Logical consistency is checked in every simulated batch.
        if np.any(b0 & (m0 > tau + 1e-12)) or np.any(b1 & (m1 > tau + 1e-12)):
            raise AssertionError("Partition-of-unity whole-mesh bound was violated")
        max0s.extend(m0.tolist())
        max1s.extend(m1.tolist())
    result = {k: binomial_summary(v, n_samples) for k, v in counts.items()}
    result.update(
        {
            "seed": seed,
            "requested_target": p.alpha,
            "tolerance_mm": tau,
            "baseline_max_displacement_quantiles_mm": np.quantile(
                max0s, [0.5, 0.95, 0.99]
            ).tolist(),
            "optimized_max_displacement_quantiles_mm": np.quantile(
                max1s, [0.5, 0.95, 0.99]
            ).tolist(),
            "interval_scope": "MC sampling only; two-sided individual 95% intervals",
        }
    )
    return result


def validate_abstract(problem, x, n_samples=100_000, seed=403):
    key = jax.random.PRNGKey(seed)
    eps = jax.random.normal(key, (n_samples, problem.n_features), dtype=jnp.float64)
    draws = (
        problem.mean(jnp.asarray(x))
        + eps @ jnp.linalg.cholesky(jnp.asarray(problem.covariance)).T
    )
    passed = jnp.all(
        (draws >= jnp.asarray(problem.lower)) & (draws <= jnp.asarray(problem.upper)),
        axis=1,
    )
    return binomial_summary(int(jnp.sum(passed)), n_samples)

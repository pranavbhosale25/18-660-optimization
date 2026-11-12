"""JAX-differentiable Gaussian rectangle probabilities.

Independent coordinates: exact product of univariate interval probabilities.
Correlated coordinates: Genz separation-of-variables transformation, integrated
with fixed scrambled Sobol points. Fixed points give repeatable AD gradients;
independent scrambles and independent binomial simulation check integration error.

The correlated mode is NUMERICAL, not an exact probability/global certificate.
No differentiation through discontinuous Monte Carlo indicators is used.
"""

from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import log_ndtr, ndtri, logsumexp
from scipy.stats import qmc


def log_normal_interval(a, b):
    """Stable log(Phi(b)-Phi(a)); finite ordered limits required.

    Reflect positive intervals into the lower tail to avoid subtracting two
    probabilities that both round to one. expm1 is accurate for narrow intervals.
    Demo bounds are finite; infinities are supported in the independent branch.
    """
    flip = a > 0
    lo = jnp.where(flip, -b, a)
    hi = jnp.where(flip, -a, b)
    lb, la = log_ndtr(hi), log_ndtr(lo)
    delta = jnp.minimum(la - lb, -jnp.finfo(jnp.float64).eps)
    return lb + jnp.log(-jnp.expm1(delta))


@dataclass
class GaussianBoxOracle:
    problem: object
    mode: str = "auto"
    n_points: int = 2048
    seed: int = 17

    def __post_init__(self):
        cov = self.problem.covariance
        diagonal = np.allclose(cov, np.diag(np.diag(cov)), atol=1e-14, rtol=1e-12)
        if self.mode == "auto":
            self.mode = "independent" if diagonal else "genz"
        if self.mode not in ("independent", "genz"):
            raise ValueError("mode must be auto, independent, or genz")
        if self.mode == "independent" and not diagonal:
            raise ValueError(
                "Independent oracle cannot be used with correlated covariance"
            )
        if self.n_points < 2 or self.n_points & (self.n_points - 1):
            raise ValueError("n_points must be a power of two >= 2")
        self.exact = self.mode == "independent"
        sd = np.sqrt(np.diag(cov))
        # Fixed ordering: narrow marginal intervals first; no x-dependent sorting.
        self.order = np.argsort(
            (self.problem.upper - self.problem.lower) / sd, kind="stable"
        )
        self.L = jnp.asarray(np.linalg.cholesky(cov[np.ix_(self.order, self.order)]))
        self.lower = jnp.asarray(self.problem.lower[self.order])
        self.upper = jnp.asarray(self.problem.upper[self.order])
        self.bias = jnp.asarray(self.problem.bias[self.order])
        self.H = jnp.asarray(self.problem.response[self.order])
        self.sd = jnp.asarray(sd[self.order])
        m = self.problem.n_features
        # A full final coordinate is drawn but does not affect the last weight.
        points = qmc.Sobol(d=m, scramble=True, seed=self.seed).random_base2(
            int(np.log2(self.n_points))
        )
        self.points = jnp.asarray(points)
        self.log_probability_jax = jax.jit(self._log_probability)
        self.logp_and_grad_jax = jax.jit(jax.value_and_grad(self._log_probability))

    def _log_probability(self, x):
        mu = self.bias + self.H @ x
        if self.mode == "independent":
            return jnp.sum(
                log_normal_interval(
                    (self.lower - mu) / self.sd, (self.upper - mu) / self.sd
                )
            )
        n, m = self.points.shape
        z0 = jnp.zeros((n, m), dtype=jnp.float64)
        logw0 = jnp.zeros(n, dtype=jnp.float64)

        def step(carry, k):
            z, logw = carry
            conditional_shift = z @ self.L[k]
            a = (self.lower[k] - mu[k] - conditional_shift) / self.L[k, k]
            b = (self.upper[k] - mu[k] - conditional_shift) / self.L[k, k]
            logwidth = log_normal_interval(a, b)
            # Use reflected-tail inversion as well as reflected interval mass.
            # At extreme >8-sigma truncations the inverse is clipped for AD;
            # this is disclosed and tested for moderate, relevant probabilities.
            flip = a > 0
            lo = jnp.where(flip, -b, a)
            hi = jnp.where(flip, -a, b)
            plo = jnp.exp(log_ndtr(lo))
            mass = jnp.exp(log_normal_interval(lo, hi))
            unif = self.points[:, k]
            u = jnp.where(flip, 1.0 - unif, unif)
            quantile = jnp.clip(plo + u * mass, 1e-15, 1.0 - 1e-15)
            zk = ndtri(quantile) * jnp.where(flip, -1.0, 1.0)
            z = z.at[:, k].set(zk)
            return (z, logw + logwidth), None

        (_, logw), _ = lax.scan(step, (z0, logw0), jnp.arange(m))
        return logsumexp(logw) - jnp.log(float(n))

    def log_probability(self, x):
        return float(self.log_probability_jax(jnp.asarray(x)))

    def probability(self, x):
        return float(np.exp(self.log_probability(x)))

    def logp_and_grad(self, x):
        value, grad = self.logp_and_grad_jax(jnp.asarray(x))
        return float(value), np.asarray(grad)

    def independent_scrambles(self, x, n_points=16384, n_replicates=8, seed=500):
        """Numerical integration diagnostic, not a rigorous probability bound."""
        if self.exact:
            p = self.probability(x)
            return {"mean": p, "standard_error": 0.0, "min": p, "max": p, "values": [p]}
        vals = []
        for r in range(n_replicates):
            other = GaussianBoxOracle(self.problem, "genz", n_points, seed + r)
            vals.append(other.probability(x))
        return {
            "mean": float(np.mean(vals)),
            "standard_error": float(np.std(vals, ddof=1) / np.sqrt(len(vals))),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": vals,
        }

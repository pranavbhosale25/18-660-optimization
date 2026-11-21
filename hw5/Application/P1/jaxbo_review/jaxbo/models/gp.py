"""Exact Gaussian regression and inverse-gamma scale mixtures.

Equations 27-35. Masked fixed-capacity data are supported without dynamic shapes.
The latent variance is distinct from observation variance throughout.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple, Callable
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from .. import kernels
from ..linalg import cholesky, chol_solve, symmetrize, logdet_chol
from ..distributions import Gaussian, NormalMarginal, StudentT, StudentMarginal


class GPParams(NamedTuple):
    kernel: dict
    log_noise: jax.Array
    mean: jax.Array


def default_params(d: int, lengthscale=0.2, amplitude=1.0, noise=0.01, mean=0.0):
    return GPParams(kernels.stationary_parameters(d,lengthscale,amplitude),
                    jnp.log(jnp.asarray(noise,dtype=float)),jnp.asarray(mean,dtype=float))


class GPState(NamedTuple):
    params: GPParams
    x: jax.Array
    y: jax.Array
    mask: jax.Array
    noise_variance: jax.Array
    chol: jax.Array
    alpha: jax.Array


@dataclass(frozen=True)
class ExactGP:
    supports_gaussian_conditioning = True
    kernel: Callable = kernels.matern52
    jitter: float = 1e-8
    mean_function: Callable | None = None

    def _mean(self,p,x):
        return jnp.broadcast_to(p.mean,(x.shape[0],)) if self.mean_function is None else self.mean_function(p.mean,x)

    def fit(self,x,y,params=None,*,noise_variance=None,mask=None):
        x,y = jnp.asarray(x,dtype=float),jnp.asarray(y,dtype=float)
        if x.ndim != 2 or y.shape != (x.shape[0],):
            raise ValueError("x must have shape (n,d), y shape (n,).")
        if params is None:
            params = default_params(x.shape[1])
        mask = jnp.ones(y.shape,dtype=bool) if mask is None else jnp.asarray(mask,dtype=bool)
        noise = jnp.broadcast_to(jnp.exp(2*params.log_noise) if noise_variance is None else noise_variance,y.shape)
        # Masked rows must be finite padded values, NOT NaNs.
        m = mask.astype(x.dtype)
        k = self.kernel(params.kernel,x,x)*m[:,None]*m[None,:]
        # Inactive rows form an identity block; they make zero likelihood contribution.
        scale = jnp.maximum(jnp.sum(jnp.diag(k))/jnp.maximum(m.sum(),1),1.0)
        jit = jnp.maximum(jnp.asarray(self.jitter,x.dtype),10*jnp.finfo(x.dtype).eps)*scale
        k = k+jnp.diag(jnp.where(mask,noise+jit,1.0))
        l = jnp.linalg.cholesky(symmetrize(k))
        r = jnp.where(mask,y-self._mean(params,x),0.0)
        alpha = chol_solve(l,r)
        return GPState(params,x,y,mask,noise,l,alpha)

    def log_marginal_likelihood(self,state):
        residual = jnp.where(state.mask,state.y-self._mean(state.params,state.x),0)
        return -0.5*(residual@state.alpha+logdet_chol(state.chol)+state.mask.sum()*jnp.log(2*jnp.pi))

    def _cross(self,state,x):
        cross = self.kernel(state.params.kernel,state.x,x)*state.mask[:,None]
        v = solve_triangular(state.chol,cross,lower=True)
        mean = self._mean(state.params,x)+cross.T@state.alpha
        return mean,v

    def predict(self,state,x,*,observation_noise=False,noise_variance=None):
        mean,v = self._cross(state,x)
        variance = kernels.diagonal(self.kernel,state.params.kernel,x)-jnp.sum(v*v,axis=0)
        if observation_noise:
            variance += jnp.exp(2*state.params.log_noise) if noise_variance is None else noise_variance
        return NormalMarginal(mean,jnp.maximum(variance,0))

    def posterior(self,state,x,*,observation_noise=False,noise_variance=None):
        mean,v = self._cross(state,x)
        covariance = symmetrize(self.kernel(state.params.kernel,x,x)-v.T@v)
        if observation_noise:
            noise = jnp.broadcast_to(jnp.exp(2*state.params.log_noise) if noise_variance is None else noise_variance,(x.shape[0],))
            covariance += jnp.diag(noise)
        return Gaussian(mean,covariance)

    def noise_at(self,state,x):
        return jnp.broadcast_to(jnp.exp(2*state.params.log_noise),(x.shape[0],))


class TPState(NamedTuple):
    gp: GPState
    shape: jax.Array
    rate: jax.Array


@dataclass(frozen=True)
class StudentTProcess:
    """Conjugate global scale mixture, NOT independent Student-t observation noise.

    covariance and observation noise are both multiplied by one inverse-gamma
    scale. Section V-D motivates scale marginalization; this is an explicit choice.
    """
    gp: ExactGP = ExactGP()
    prior_shape: float = 2.0
    prior_rate: float = 1.0

    def fit(self,x,y,params=None,**kwargs):
        state = self.gp.fit(x,y,params,**kwargs)
        r = jnp.where(state.mask,y-self.gp._mean(state.params,x),0)
        return TPState(state,self.prior_shape+state.mask.sum()/2,self.prior_rate+0.5*r@state.alpha)

    def predict(self,state,x,**kwargs):
        normal = self.gp.predict(state.gp,x,**kwargs)
        return StudentMarginal(normal.mean,normal.variance*state.rate/state.shape,2*state.shape)

    def posterior(self,state,x,**kwargs):
        normal = self.gp.posterior(state.gp,x,**kwargs)
        return StudentT(normal.mean,normal.covariance*state.rate/state.shape,2*state.shape)

    def log_marginal_likelihood(self,state):
        from jax.scipy.special import gammaln
        n=state.gp.mask.sum()
        a,b=self.prior_shape,self.prior_rate
        return (gammaln(state.shape)-gammaln(a)+a*jnp.log(b)-state.shape*jnp.log(state.rate)
                -0.5*logdet_chol(state.gp.chol)-n/2*jnp.log(2*jnp.pi))

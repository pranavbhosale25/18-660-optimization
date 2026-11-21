"""Inducing-point regression: DTC, FITC/SPGP and collapsed variational VFE.

O(n m^2 + m^3) fitting; no dense n x n training covariance. Inducing points and
hyperparameters may be optimized externally through log_marginal_likelihood.
VFE returns a variational LOWER BOUND, not an exact marginal likelihood.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple,Callable
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from .. import kernels
from .gp import default_params,GPParams
from ..linalg import cholesky,chol_solve,logdet_chol,symmetrize
from ..distributions import Gaussian,NormalMarginal


class SparseState(NamedTuple):
    params: GPParams
    inducing: jax.Array
    chol_uu: jax.Array
    chol_a: jax.Array
    weights: jax.Array
    log_evidence: jax.Array


@dataclass(frozen=True)
class SparseGP:
    supports_gaussian_conditioning = True
    inducing: object
    kernel: Callable = kernels.matern52
    method: str = "fitc"
    jitter: float = 1e-8

    def fit(self,x,y,params=None,*,noise_variance=None,mask=None,inducing=None):
        if self.method not in ("fitc","dtc","vfe"):
            raise ValueError("SparseGP method must be fitc, dtc, or vfe")
        p=default_params(x.shape[1]) if params is None else params
        z=jnp.asarray(self.inducing if inducing is None else inducing,dtype=x.dtype)
        mask=jnp.ones(y.shape,dtype=bool) if mask is None else mask
        l=cholesky(self.kernel(p.kernel,z,z),self.jitter)
        phi=solve_triangular(l,self.kernel(p.kernel,z,x),lower=True).T*mask[:,None]
        residual=jnp.where(mask,y-p.mean,0)
        residual_diag=jnp.maximum(kernels.diagonal(self.kernel,p.kernel,x)-jnp.sum(phi*phi,axis=1),0)
        noise=jnp.broadcast_to(jnp.exp(2*p.log_noise) if noise_variance is None else noise_variance,y.shape)
        # Sparse models require positive noise variances (the likelihood is nonsingular).
        d=jnp.where(mask,noise+(residual_diag if self.method=="fitc" else 0),1.0)
        a=jnp.eye(z.shape[0],dtype=x.dtype)+(phi.T/d)@phi
        la=cholesky(a,self.jitter)
        rhs=phi.T@(residual/d); w=chol_solve(la,rhs)
        evidence=-.5*(jnp.sum(residual**2/d)-rhs@w+jnp.log(d).sum()+logdet_chol(la)+mask.sum()*jnp.log(2*jnp.pi))
        if self.method=="vfe":
            evidence-=.5*jnp.sum(jnp.where(mask,residual_diag/noise,0))
        return SparseState(p,z,l,la,w,evidence)

    def log_marginal_likelihood(self,state):
        return state.log_evidence

    def _features(self,state,x):
        return solve_triangular(state.chol_uu,self.kernel(state.params.kernel,state.inducing,x),lower=True).T

    def predict(self,state,x,*,observation_noise=False,noise_variance=None):
        phi=self._features(state,x)
        v=solve_triangular(state.chol_a,phi.T,lower=True)
        var=kernels.diagonal(self.kernel,state.params.kernel,x)-jnp.sum(phi*phi,axis=1)+jnp.sum(v*v,axis=0)
        if observation_noise:
            var+=jnp.exp(2*state.params.log_noise) if noise_variance is None else noise_variance
        return NormalMarginal(state.params.mean+phi@state.weights,jnp.maximum(var,0))

    def posterior(self,state,x,*,observation_noise=False,noise_variance=None):
        phi=self._features(state,x); v=solve_triangular(state.chol_a,phi.T,lower=True)
        cov=self.kernel(state.params.kernel,x,x)-phi@phi.T+v.T@v
        if observation_noise:
            cov+=jnp.diag(jnp.broadcast_to(jnp.exp(2*state.params.log_noise) if noise_variance is None else noise_variance,(x.shape[0],)))
        return Gaussian(state.params.mean+phi@state.weights,symmetrize(cov))

    def noise_at(self,state,x):
        return jnp.broadcast_to(jnp.exp(2*state.params.log_noise),(x.shape[0],))

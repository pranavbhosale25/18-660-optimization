"""Conjugate linear models, optional nonlinear features; Eqs. 8-17."""
from __future__ import annotations
from typing import NamedTuple,Callable
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from ..linalg import cholesky,chol_solve,logdet_chol
from ..distributions import Gaussian,StudentT,NormalMarginal,StudentMarginal


def identity_features(x):
    return x


def affine_features(x):
    return jnp.concatenate([jnp.ones((x.shape[0],1),dtype=x.dtype),x],axis=1)


def radial_features(centers,lengthscale=1.0,include_bias=True):
    def f(x):
        squared=jnp.sum(((x[:,None,:]-centers[None,:,:])/lengthscale)**2,axis=-1)
        features=jnp.exp(-.5*squared)
        return jnp.concatenate([jnp.ones((x.shape[0],1),dtype=x.dtype),features],axis=1) if include_bias else features
    return f


def fourier_features(frequencies,include_bias=True):
    def f(x):
        phase=x@frequencies.T
        out=jnp.concatenate([jnp.cos(phase),jnp.sin(phase)],axis=1)/jnp.sqrt(frequencies.shape[0])
        return jnp.concatenate([jnp.ones((x.shape[0],1),dtype=x.dtype),out],axis=1) if include_bias else out
    return f


class LinearState(NamedTuple):
    mean: jax.Array
    covariance: jax.Array  # weight covariance (known noise) or NIG V (unknown noise)
    shape: jax.Array
    rate: jax.Array
    log_evidence: jax.Array


@dataclass(frozen=True)
class BayesianLinear:
    features: Callable = affine_features
    noise_variance: float | None = 0.01
    prior_variance: float = 1.0
    prior_shape: float = 2.0
    prior_rate: float = 1.0
    prior_mean: object = None
    prior_covariance: object = None

    def fit(self,x,y,params=None,*,mask=None,noise_variance=None):
        phi=self.features(x)
        mask=jnp.ones(y.shape,dtype=bool) if mask is None else mask
        phi=phi*mask[:,None]; yy=jnp.where(mask,y,0)
        m=phi.shape[1]
        m0=jnp.zeros(m,dtype=x.dtype) if self.prior_mean is None else jnp.asarray(self.prior_mean,dtype=x.dtype)
        v0=jnp.eye(m,dtype=x.dtype)*self.prior_variance if self.prior_covariance is None else jnp.asarray(self.prior_covariance,dtype=x.dtype)
        l0=cholesky(v0)
        precision0=chol_solve(l0,jnp.eye(m,dtype=x.dtype))
        if self.noise_variance is None:
            if noise_variance is not None:
                raise ValueError("NIG model does not accept separate per-observation variances")
            precision=precision0+phi.T@phi
            l=cholesky(precision)
            v=chol_solve(l,jnp.eye(m,dtype=x.dtype))
            mn=chol_solve(l,precision0@m0+phi.T@yy)
            an=self.prior_shape+mask.sum()/2
            # Numerically stable algebraic equivalent of Eq. 12.
            bn=self.prior_rate+.5*(jnp.sum(jnp.where(mask,(y-phi@mn)**2,0))+(mn-m0)@precision0@(mn-m0))
            from jax.scipy.special import gammaln
            evidence=(gammaln(an)-gammaln(self.prior_shape)+self.prior_shape*jnp.log(self.prior_rate)
                      -an*jnp.log(bn)-.5*(logdet_chol(l0)+logdet_chol(l))-mask.sum()/2*jnp.log(2*jnp.pi))
            return LinearState(mn,v,an,bn,evidence)
        noise=jnp.broadcast_to(self.noise_variance if noise_variance is None else noise_variance,y.shape)
        weights=jnp.where(mask,1/noise,0)
        precision=precision0+(phi.T*weights)@phi
        l=cholesky(precision)
        v=chol_solve(l,jnp.eye(m,dtype=x.dtype))
        mn=chol_solve(l,precision0@m0+phi.T@(weights*yy))
        residual=y-phi@m0
        rhs=phi.T@(weights*residual)
        evidence=-.5*(jnp.sum(weights*residual**2)-rhs@chol_solve(l,rhs)+logdet_chol(l0)+logdet_chol(l)
                        +jnp.sum(jnp.where(mask,jnp.log(noise),0))+mask.sum()*jnp.log(2*jnp.pi))
        return LinearState(mn,v,jnp.asarray(jnp.inf),jnp.asarray(self.noise_variance),evidence)

    def log_marginal_likelihood(self,state):
        return state.log_evidence

    def predict(self,state,x,*,observation_noise=False,noise_variance=None):
        phi=self.features(x); mean=phi@state.mean
        variance=jnp.maximum(jnp.sum((phi@state.covariance)*phi,axis=1),0)
        if self.noise_variance is None:
            return StudentMarginal(mean,(variance+int(observation_noise))*state.rate/state.shape,2*state.shape)
        if observation_noise:
            variance+=self.noise_variance if noise_variance is None else noise_variance
        return NormalMarginal(mean,variance)

    def posterior(self,state,x,*,observation_noise=False,noise_variance=None):
        phi=self.features(x); mean=phi@state.mean
        covariance=phi@state.covariance@phi.T
        if self.noise_variance is None:
            return StudentT(mean,(covariance+int(observation_noise)*jnp.eye(x.shape[0]))*state.rate/state.shape,2*state.shape)
        if observation_noise:
            covariance+=jnp.diag(jnp.broadcast_to(self.noise_variance if noise_variance is None else noise_variance,(x.shape[0],)))
        return Gaussian(mean,covariance)

    def thompson_function(self,state,key):
        if self.noise_variance is None:
            weights=StudentT(state.mean,state.covariance*state.rate/state.shape,2*state.shape).sample(key,1)[0]
        else:
            weights=Gaussian(state.mean,state.covariance).sample(key,1)[0]
        return lambda x:self.features(x)@weights

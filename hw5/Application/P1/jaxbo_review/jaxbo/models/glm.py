"""Logistic/probit models with Laplace or MCMC inference; Section II-C.

The review prints 1/(1+exp(z)) in its logistic paragraph. 'logit' here means the
conventional increasing sigmoid; 'logit_negative' explicitly reproduces the
printed decreasing link. Probit always uses the increasing normal CDF.
"""
from __future__ import annotations
from typing import NamedTuple,Callable
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr,ndtr,ndtri
from jax.scipy.linalg import solve_triangular
from ..inference import laplace,hmc,random_walk_metropolis
from ..linalg import cholesky,chol_solve,normal_samples,safe_sqrt,symmetrize
from ..distributions import Empirical,EmpiricalMarginal,Gaussian
from .linear import affine_features
from .gp import GPParams,default_params
from .. import kernels


def link_value(z,link):
    if link=="logit": return jax.nn.sigmoid(z)
    if link=="logit_negative": return jax.nn.sigmoid(-z)
    if link=="probit": return ndtr(z)
    raise ValueError("link must be logit, logit_negative, or probit")


def bernoulli_log_likelihood(y,z,link):
    if link in ("logit","logit_negative"):
        z=z if link=="logit" else -z
        return y*jax.nn.log_sigmoid(z)+(1-y)*jax.nn.log_sigmoid(-z)
    if link=="probit":
        return y*log_ndtr(z)+(1-y)*log_ndtr(-z)
    raise ValueError("Unknown link")


def _marginal_probabilities(mean,variance,link,n=96):
    # Equal-weight inverse-normal midpoint quadrature, explicitly approximate.
    nodes=ndtri((jnp.arange(n,dtype=mean.dtype)+.5)/n)
    return EmpiricalMarginal(link_value(mean[None,:]+safe_sqrt(variance)[None,:]*nodes[:,None],link))


class GLMState(NamedTuple):
    mean:jax.Array
    covariance:jax.Array
    samples:jax.Array
    acceptance_rate:jax.Array
    converged:jax.Array


@dataclass(frozen=True)
class GeneralizedLinear:
    features:Callable=affine_features
    link:str="logit"
    inference:str="laplace"
    prior_variance:float=4.0
    steps:int=100
    n_samples:int=128
    burnin:int=128
    step_size:float=.05

    @property
    def requires_key(self): return self.inference!="laplace"

    def fit(self,x,y,params=None,*,mask=None,key=None):
        phi=self.features(x); mask=jnp.ones(y.shape,dtype=bool) if mask is None else mask
        initial=jnp.zeros(phi.shape[1],dtype=x.dtype)
        def logp(w):
            return jnp.sum(jnp.where(mask,bernoulli_log_likelihood(y,phi@w,self.link),0))-.5*jnp.sum(w*w)/self.prior_variance
        approximation=laplace(logp,initial,steps=self.steps)
        if self.inference=="laplace":
            return GLMState(approximation.mean,approximation.covariance,jnp.empty((0,phi.shape[1]),dtype=x.dtype),
                            jnp.asarray(jnp.nan),approximation.converged)
        if key is None: raise ValueError("An explicit PRNG key is required for MCMC")
        if self.inference=="hmc":
            chain=hmc(key,logp,approximation.mean,n_samples=self.n_samples,burnin=self.burnin,step_size=self.step_size)
        elif self.inference=="mh":
            chain=random_walk_metropolis(key,logp,approximation.mean,n_samples=self.n_samples,burnin=self.burnin,proposal_scale=self.step_size)
        else: raise ValueError("inference must be laplace, hmc, or mh")
        centered=chain.samples-chain.samples.mean(axis=0)
        return GLMState(chain.samples.mean(axis=0),centered.T@centered/self.n_samples,chain.samples,
                        chain.acceptance_rate,jnp.array(False))

    def predict(self,state,x):
        phi=self.features(x)
        if self.inference!="laplace": return EmpiricalMarginal(link_value(state.samples@phi.T,self.link))
        mean=phi@state.mean; variance=jnp.maximum(jnp.sum((phi@state.covariance)*phi,axis=1),0)
        return _marginal_probabilities(mean,variance,self.link)

    def sample(self,state,x,key,n_samples=1):
        phi=self.features(x)
        if self.inference=="laplace":
            weights=Gaussian(state.mean,state.covariance).sample(key,n_samples)
        else:
            weights=state.samples[jax.random.randint(key,(n_samples,),0,state.samples.shape[0])]
        return link_value(weights@phi.T,self.link)

    def thompson_function(self,state,key):
        if self.inference=="laplace": weights=Gaussian(state.mean,state.covariance).sample(key,1)[0]
        else: weights=state.samples[jax.random.randint(key,(),0,state.samples.shape[0])]
        return lambda x:link_value(self.features(x)@weights,self.link)


class ClassifierState(NamedTuple):
    params:GPParams
    x:jax.Array
    chol_prior:jax.Array
    mode:jax.Array
    covariance:jax.Array
    log_evidence:jax.Array
    converged:jax.Array


@dataclass(frozen=True)
class GPClassifier:
    """Bernoulli GP with a whitened Laplace approximation (for binary feasibility)."""
    kernel:Callable=kernels.matern52
    link:str="probit"
    steps:int=100
    jitter:float=1e-8

    def fit(self,x,y,params=None,*,mask=None):
        p=default_params(x.shape[1]) if params is None else params
        mask=jnp.ones(y.shape,dtype=bool) if mask is None else mask
        l=cholesky(self.kernel(p.kernel,x,x),self.jitter)
        def logp(z):
            ll=jnp.sum(jnp.where(mask,bernoulli_log_likelihood(y,p.mean+l@z,self.link),0))
            return ll-.5*jnp.dot(z,z)-x.shape[0]/2*jnp.log(2*jnp.pi)
        result=laplace(logp,jnp.zeros(y.shape,dtype=x.dtype),steps=self.steps)
        return ClassifierState(p,x,l,result.mean,result.covariance,result.log_normalizer,result.converged)

    def latent_posterior(self,state,x):
        a=solve_triangular(state.chol_prior,self.kernel(state.params.kernel,state.x,x),lower=True).T
        mean=state.params.mean+a@state.mode
        covariance=self.kernel(state.params.kernel,x,x)-a@a.T+a@state.covariance@a.T
        return Gaussian(mean,symmetrize(covariance))

    def predict(self,state,x):
        a=solve_triangular(state.chol_prior,self.kernel(state.params.kernel,state.x,x),lower=True).T
        mean=state.params.mean+a@state.mode
        variance=kernels.diagonal(self.kernel,state.params.kernel,x)-jnp.sum(a*a,axis=1)+jnp.sum((a@state.covariance)*a,axis=1)
        return _marginal_probabilities(mean,jnp.maximum(variance,0),self.link)

    def probability(self,state,x):
        if self.link=="probit":
            a=solve_triangular(state.chol_prior,self.kernel(state.params.kernel,state.x,x),lower=True).T
            mean=state.params.mean+a@state.mode
            v=kernels.diagonal(self.kernel,state.params.kernel,x)-jnp.sum(a*a,axis=1)+jnp.sum((a@state.covariance)*a,axis=1)
            return ndtr(mean/jnp.sqrt(1+jnp.maximum(v,0)))
        return self.predict(state,x).mean

    def sample(self,state,x,key,n_samples=1):
        return link_value(self.latent_posterior(state,x).sample(key,n_samples),self.link)

    def log_marginal_likelihood(self,state): return state.log_evidence

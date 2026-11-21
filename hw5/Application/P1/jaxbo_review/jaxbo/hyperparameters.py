"""ML-II, MAP, and acquisition-level integration over GP hyperparameters.

Priors here are explicitly defined on UNCONSTRAINED coordinates (log scales,
mean, etc.). A prior stated on natural-scale parameters needs its Jacobian when
converted. A Gaussian prior in log coordinates is not a natural-scale MAP prior.
"""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from .optim import minimize_lbfgs,minimize_adam
from .inference import hmc,random_walk_metropolis,tempered_smc,gaussian_product_quadrature,quadrature_posterior
from .acquisitions import expected_improvement,probability_improvement,upper_confidence_bound,posterior_quantile,make_acquisition


class HyperFit(NamedTuple):
    params:object
    state:object
    objective:jax.Array
    converged:jax.Array


class HyperEnsemble(NamedTuple):
    params:object  # batched parameter PyTree
    states:object  # batched posterior PyTree
    weights:jax.Array
    acceptance_rate:jax.Array


def gaussian_log_prior(center,scale):
    def logp(theta):
        return -.5*jnp.sum(((theta-center)/scale)**2)-jnp.sum(jnp.log(jnp.broadcast_to(scale,theta.shape)))-theta.size/2*jnp.log(2*jnp.pi)
    return logp


def fit_hyperparameters(model,x,y,initial,*,method="map",log_prior=None,prior_scale=1.5,
                        steps=80,restarts=1,key=None,lower=None,upper=None,optimizer="lbfgs",
                        noise_variance=None,mask=None):
    if method not in ("ml","map"): raise ValueError("Point method must be ml or map")
    flat,unravel=ravel_pytree(initial)
    # A fixed zero-centered Gaussian in unconstrained coordinates is the
    # implementation default. Supply a scientific prior when appropriate.
    lp=gaussian_log_prior(jnp.zeros_like(flat),prior_scale) if log_prior is None else log_prior
    def objective(theta):
        state=model.fit(x,y,unravel(theta),noise_variance=noise_variance,mask=mask)
        return -(model.log_marginal_likelihood(state)+(lp(theta) if method=="map" else 0))
    # Broad compact bounds stabilize evidence optimization, particularly noise.
    lo=jnp.full_like(flat,-12) if lower is None else lower
    hi=jnp.full_like(flat,12) if upper is None else upper
    minimize={"lbfgs":minimize_lbfgs,"adam":minimize_adam}.get(optimizer)
    if minimize is None: raise ValueError("optimizer must be lbfgs or adam")
    if restarts<1: raise ValueError("restarts must be positive")
    if restarts>1:
        if key is None: raise ValueError("Multistart hyperparameter fitting requires a key")
        initial_points=jnp.clip(flat+.5*jax.random.normal(key,(restarts,flat.size)),lo,hi).at[0].set(flat)
        results=jax.vmap(lambda q:minimize(objective,q,lo,hi,steps=steps))(initial_points)
        idx=jnp.argmin(results.value); result=jax.tree_util.tree_map(lambda a:a[idx],results)
    else: result=minimize(objective,flat,lo,hi,steps=steps)
    params=unravel(result.x)
    state=model.fit(x,y,params,noise_variance=noise_variance,mask=mask)
    return HyperFit(params,state,result.value,result.converged)


def integrate_hyperparameters(key,model,x,y,initial,*,method="hmc",log_prior=None,
                             prior_mean=None,prior_scale=1.5,n_samples=32,burnin=64,
                             step_size=.025,quadrature_order=3,smc_stages=16,
                             noise_variance=None,mask=None):
    flat,unravel=ravel_pytree(initial)
    center=jnp.zeros_like(flat) if prior_mean is None else jnp.asarray(prior_mean)
    scale=jnp.broadcast_to(jnp.asarray(prior_scale,dtype=flat.dtype),flat.shape)
    lp=gaussian_log_prior(center,scale) if log_prior is None else log_prior
    def ll(theta):
        state=model.fit(x,y,unravel(theta),noise_variance=noise_variance,mask=mask)
        return model.log_marginal_likelihood(state)
    logp=lambda theta:ll(theta)+lp(theta)
    if method in ("hmc","mh"):
        if method=="hmc": result=hmc(key,logp,flat,n_samples=n_samples,burnin=burnin,step_size=step_size)
        else: result=random_walk_metropolis(key,logp,flat,n_samples=n_samples,burnin=burnin,proposal_scale=step_size)
        nodes=result.samples; weights=jnp.full(n_samples,1/n_samples); acceptance=result.acceptance_rate
    elif method=="smc":
        if log_prior is not None: raise ValueError("SMC helper's sampler assumes the supplied Gaussian prior parameters")
        k1,k2=jax.random.split(key)
        prior_particles=center+scale*jax.random.normal(k1,(n_samples,flat.size))
        result=tempered_smc(k2,ll,lp,prior_particles,stages=smc_stages,proposal_scale=step_size)
        nodes,weights=result.particles,result.weights; acceptance=result.acceptance_history.mean()
    elif method=="quadrature":
        if log_prior is not None: raise ValueError("Quadrature helper assumes the supplied Gaussian prior parameters")
        nodes,prior_weights=gaussian_product_quadrature(center,scale,quadrature_order)
        weights=quadrature_posterior(ll,nodes,prior_weights); acceptance=jnp.asarray(jnp.nan)
    else: raise ValueError("method must be hmc, mh, smc, or quadrature")
    params=jax.vmap(unravel)(nodes)
    states=jax.vmap(lambda p:model.fit(x,y,p,noise_variance=noise_variance,mask=mask))(params)
    return HyperEnsemble(params,states,weights,acceptance)


def marginalized_acquisition(name,model,ensemble,best,*,maximize=True,xi=0,beta=2.0,quantile=.95,key=None):
    """E_theta[a(x;theta)], NOT a(x; moment-matched posterior). Eqs. 53-54."""
    if name=="ts":
        if key is None: raise ValueError("An explicit key is required")
        k1,k2=jax.random.split(key)
        idx=jax.random.categorical(k1,jnp.log(ensemble.weights))
        state=jax.tree_util.tree_map(lambda a:a[idx],ensemble.states)
        return make_acquisition("ts",model,state,best,key=k2,maximize=maximize)
    def utility(x):
        def one(state):
            prediction=model.predict(state,x)
            if name=="ei": return expected_improvement(prediction,best,maximize=maximize,xi=xi)
            if name=="pi": return probability_improvement(prediction,best,maximize=maximize,xi=xi)
            if name=="ucb": return upper_confidence_bound(prediction,maximize=maximize,beta=beta)
            if name=="quantile": return (1 if maximize else -1)*posterior_quantile(prediction,quantile if maximize else 1-quantile)
            if name=="mean": return (1 if maximize else -1)*prediction.mean
            raise ValueError("Unsupported ensemble acquisition")
        return jnp.tensordot(ensemble.weights,jax.vmap(one)(ensemble.states),axes=1)
    return utility


def ensemble_moments(model,ensemble,x):
    predictions=jax.vmap(lambda state:model.predict(state,x))(ensemble.states)
    means,variances=predictions.mean,predictions.variance
    mean=jnp.tensordot(ensemble.weights,means,axes=1)
    variance=jnp.tensordot(ensemble.weights,variances+means**2,axes=1)-mean**2
    return mean,jnp.maximum(variance,0)

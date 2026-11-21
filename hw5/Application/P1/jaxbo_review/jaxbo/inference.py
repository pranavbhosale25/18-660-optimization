"""Reusable native-JAX inference: Laplace, MH, HMC, SMC and quadrature.

MCMC returns finite-chain approximations, not iid/exact posterior samples.
Diagnostics are deliberately returned rather than hidden.
"""
from __future__ import annotations
from typing import NamedTuple,Callable
import itertools
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from .linalg import cholesky,chol_solve
from .optim import minimize_lbfgs


class MCMCResult(NamedTuple):
    samples:jax.Array
    acceptance_rate:jax.Array
    log_density:jax.Array
    final_position:jax.Array


class LaplaceResult(NamedTuple):
    mean:jax.Array
    covariance:jax.Array
    log_normalizer:jax.Array
    converged:jax.Array


def laplace(log_density,initial,*,steps=100):
    result=minimize_lbfgs(lambda w:-log_density(w),initial,steps=steps)
    precision=jax.hessian(lambda w:-log_density(w))(result.x)
    l=cholesky(precision)
    covariance=chol_solve(l,jnp.eye(initial.size,dtype=initial.dtype))
    logz=log_density(result.x)+initial.size/2*jnp.log(2*jnp.pi)-jnp.log(jnp.diag(l)).sum()
    return LaplaceResult(result.x,covariance,logz,result.converged)


def random_walk_metropolis(key,log_density,initial,*,n_samples=128,burnin=128,thin=1,proposal_scale=0.1):
    if n_samples<1 or thin<1 or burnin<0:
        raise ValueError("Invalid MCMC sample, thinning, or burn-in count")
    def step(carry,_):
        x,lp,key=carry
        key,kp,ku=jax.random.split(key,3)
        proposal=x+proposal_scale*jax.random.normal(kp,x.shape,dtype=x.dtype)
        lp_new=log_density(proposal)
        accept=jnp.isfinite(lp_new)&(jnp.log(jax.random.uniform(ku))<lp_new-lp)
        x=jnp.where(accept,proposal,x); lp=jnp.where(accept,lp_new,lp)
        return (x,lp,key),(x,lp,accept)
    (last,_,_), (samples,lps,accepted)=jax.lax.scan(step,(initial,log_density(initial),key),None,length=burnin+n_samples*thin)
    idx=slice(burnin+thin-1,None,thin)
    return MCMCResult(samples[idx],accepted[burnin:].mean(),lps[idx],last)


def hmc(key,log_density,initial,*,n_samples=128,burnin=128,thin=1,step_size=0.05,leapfrog_steps=10,mass=None):
    """Fixed-step HMC with diagonal mass and Metropolis correction.

    No automatic convergence claim or NUTS adaptation. Tune steps/mass and run
    multiple chains for real inference; inspect acceptance and effective samples.
    """
    if n_samples<1 or thin<1 or burnin<0 or leapfrog_steps<1:
        raise ValueError("Invalid HMC iteration counts")
    mass=jnp.ones_like(initial) if mass is None else jnp.asarray(mass,dtype=initial.dtype)
    value_grad=jax.value_and_grad(log_density)
    def step(carry,_):
        x,lp,key=carry
        key,kp,ku=jax.random.split(key,3)
        p0=jax.random.normal(kp,x.shape,dtype=x.dtype)*jnp.sqrt(mass)
        _,g=value_grad(x); momentum=p0+.5*step_size*g
        def leap(_,qpg):
            q,p,g=qpg
            q=q+step_size*p/mass
            _,g=value_grad(q)
            return q,p+step_size*g,g
        proposal,p,g=jax.lax.fori_loop(0,leapfrog_steps,leap,(x,momentum,g))
        p=p-.5*step_size*g
        lp_new=log_density(proposal)
        log_accept=lp_new-lp+.5*jnp.sum((p0*p0-p*p)/mass)
        accept=jnp.isfinite(log_accept)&(jnp.log(jax.random.uniform(ku))<log_accept)
        x=jnp.where(accept,proposal,x); lp=jnp.where(accept,lp_new,lp)
        return (x,lp,key),(x,lp,accept)
    (last,_,_), (samples,lps,accepted)=jax.lax.scan(step,(initial,log_density(initial),key),None,length=burnin+n_samples*thin)
    idx=slice(burnin+thin-1,None,thin)
    return MCMCResult(samples[idx],accepted[burnin:].mean(),lps[idx],last)


class SMCResult(NamedTuple):
    particles:jax.Array
    weights:jax.Array
    log_evidence:jax.Array
    ess_history:jax.Array
    acceptance_history:jax.Array


def tempered_smc(key,log_likelihood,log_prior,prior_particles,*,stages=16,mcmc_steps=4,proposal_scale=.1,resample_threshold=.5):
    """Tempered SMC from samples of the stated prior, with MH rejuvenation.

    The prior sampler is supplied by the caller; passing arbitrary initial points
    is not valid. A fixed temperature schedule makes the computation JIT-friendly.
    """
    n=prior_particles.shape[0]
    def stage(carry,beta):
        particles,logw,previous_beta,logz,key=carry
        key,kr,km=jax.random.split(key,3)
        ll=jax.vmap(log_likelihood)(particles)
        updated=logw+(beta-previous_beta)*ll
        norm=logsumexp(updated); logw=updated-norm; logz=logz+norm
        w=jnp.exp(logw); ess=1/jnp.sum(w*w)
        positions=(jax.random.uniform(kr)+jnp.arange(n))/n
        indices=jnp.minimum(jnp.searchsorted(jnp.cumsum(w),positions),n-1)
        do_resample=ess<resample_threshold*n
        particles=jnp.where(do_resample,particles[indices],particles)
        logw=jnp.where(do_resample,jnp.full_like(logw,-jnp.log(n)),logw)
        target=lambda q:log_prior(q)+beta*log_likelihood(q)
        keys=jax.random.split(km,n)
        result=jax.vmap(lambda k,q:random_walk_metropolis(k,target,q,n_samples=mcmc_steps,
                            burnin=0,proposal_scale=proposal_scale))(keys,particles)
        return (result.final_position,logw,beta,logz,key),(ess,result.acceptance_rate.mean())
    initial=(prior_particles,jnp.full((n,),-jnp.log(n)),jnp.asarray(0.0),jnp.asarray(0.0),key)
    (particles,logw,_,logz,_),(ess,acceptance)=jax.lax.scan(stage,initial,jnp.linspace(0,1,stages+1)[1:])
    return SMCResult(particles,jnp.exp(logw),logz,ess,acceptance)


def gauss_hermite(n:int):
    """Nodes and weights for an expectation under N(0,1), via Golub-Welsch."""
    if n<1:
        raise ValueError("Quadrature order must be positive")
    off=jnp.sqrt(jnp.arange(1,n,dtype=float)/2)
    jacobi=jnp.diag(off,1)+jnp.diag(off,-1)
    nodes,vectors=jnp.linalg.eigh(jacobi)
    return jnp.sqrt(2.0)*nodes,vectors[0]**2


def gaussian_product_quadrature(mean,scale,order=5,*,max_points=100_000):
    """Tensor Gaussian-prior quadrature. Exponential in dimension, explicitly capped."""
    d=mean.size
    if order**d>max_points:
        raise ValueError("Tensor quadrature is too large; reduce dimension/order or use MC")
    nodes,weights=gauss_hermite(order)
    indices=jnp.asarray(list(itertools.product(range(order),repeat=d)),dtype=int)
    return mean+nodes[indices]*scale,jnp.prod(weights[indices],axis=1)


def quadrature_posterior(log_likelihood,nodes,prior_weights):
    """Normalize likelihood-weighted prior quadrature weights; Eq. 54."""
    logweights=jnp.log(prior_weights)+jax.vmap(log_likelihood)(nodes)
    return jax.nn.softmax(logweights)

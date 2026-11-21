"""Finite-representer information acquisitions (Section IV-C).

Both estimate information about the best member of a FIXED finite set. They are
not the continuous expectation-propagation PES algorithm cited as [69]. The two
estimators below use different MI identities and finite MC can yield negatives.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy,logsumexp
from .distributions import Gaussian
from .linalg import psd_root,cholesky,chol_solve,gaussian_logpdf


def categorical_entropy(labels,n_classes):
    counts=jnp.bincount(labels,length=n_classes)
    p=counts/counts.sum()
    return -jnp.sum(xlogy(p,p))


def _check_gaussian(model,state,representers):
    p=model.posterior(state,representers)
    if not isinstance(p,Gaussian):
        raise TypeError("Information acquisitions currently require a Gaussian joint posterior")
    if not hasattr(model,"noise_at"):
        raise TypeError("Model must expose noise_at(state, X) for observation noise")
    return p


def entropy_search_mc(model,state,representers,key,*,maximize=True,n_functions=128,n_fantasies=16,
                      observation_noise=None):
    """Eq. 47 via explicit Gaussian conditioning and MC winner histograms.

    A common random-number design fixes the function across acquisition searches.
    Histogram argmax is nondifferentiable: use random, DIRECT, or pattern search.
    observation_noise may be a callable X -> [n] for heteroscedastic noise.
    """
    p=_check_gaussian(model,state,representers); r=representers.shape[0]
    k1,k2=jax.random.split(key)
    eps=jax.random.normal(k1,(n_functions,r),dtype=p.mean.dtype)
    fantasies=jax.random.normal(k2,(n_fantasies,),dtype=p.mean.dtype)
    sign=1 if maximize else -1
    original=p.mean+eps@psd_root(p.covariance).T
    base_entropy=categorical_entropy(jnp.argmax(sign*original,axis=1),r)
    def one(x):
        points=jnp.concatenate([representers,x[None,:]],axis=0)
        joint=model.posterior(state,points)
        cross=joint.covariance[:-1,-1]
        noise=model.noise_at(state,x[None,:])[0] if observation_noise is None else observation_noise(x[None,:])[0]
        vy=jnp.maximum(joint.covariance[-1,-1]+noise,1e-14)
        conditioned_cov=p.covariance-jnp.outer(cross,cross)/vy
        root=psd_root(conditioned_cov)
        conditional_means=p.mean[None,:]+fantasies[:,None]*cross[None,:]/jnp.sqrt(vy)
        def conditional_entropy(mean):
            samples=mean+eps@root.T
            return categorical_entropy(jnp.argmax(sign*samples,axis=1),r)
        expected=jax.vmap(conditional_entropy)(conditional_means).mean()
        return base_entropy-expected
    return jax.vmap(one)


def predictive_entropy_search_mc(model,state,representers,key,*,maximize=True,n_functions=128,
                                 observation_noise=None):
    """Finite-support PES identity using conditional Gaussian-mixture entropies.

    Draw F_R, label each draw by its maximizer, and estimate p(y | winner) as the
    mixture of p(y | F_R) belonging to that label. Costs O(S^2) per query and has
    finite-mixture/self-sampling bias. Both entropies use the same finite mixture
    to reduce cancellation error. This is explicitly NOT EP-PES.
    """
    p=_check_gaussian(model,state,representers); r=representers.shape[0]
    k1,k2=jax.random.split(key)
    f=p.sample(k1,n_functions)
    winners=jnp.argmax((1 if maximize else -1)*f,axis=1)
    same=winners[:,None]==winners[None,:]
    counts=same.sum(axis=1)
    l=cholesky(p.covariance)
    epsilon=jax.random.normal(k2,(n_functions,),dtype=p.mean.dtype)
    def one(x):
        joint=model.posterior(state,jnp.concatenate([representers,x[None,:]],axis=0))
        cross=joint.covariance[:-1,-1]
        noise=model.noise_at(state,x[None,:])[0] if observation_noise is None else observation_noise(x[None,:])[0]
        vy=jnp.maximum(joint.covariance[-1,-1]+noise,1e-14)
        coeff=chol_solve(l,cross)
        component_mean=joint.mean[-1]+(f-p.mean)@coeff
        component_variance=jnp.maximum(vy-cross@coeff,1e-12)
        observations=component_mean+jnp.sqrt(component_variance)*epsilon
        logs=gaussian_logpdf(observations[:,None],component_mean[None,:],component_variance)
        conditional_logpdf=logsumexp(jnp.where(same,logs,-jnp.inf),axis=1)-jnp.log(counts)
        unconditional_logpdf=logsumexp(logs,axis=1)-jnp.log(n_functions)
        # Use the SAME finite mixture in both entropies. This is exactly zero
        # when every sampled function has the same winner, unlike mismatched
        # analytic-Gaussian minus finite-mixture entropy estimates.
        return jnp.mean(conditional_logpdf-unconditional_logpdf)
    return jax.vmap(one)


def integrated_conditional_improvement(model,state,reference,weights,best,*,maximize=True):
    """Mean-preserving IECI approximation (Eq. 60).

    The pending observation's value is imputed at its predictive mean, reducing
    reference variances while preserving their means. This is NOT E_y[EI] with
    an unchanged target, which would be unchanged by the tower property.
    The review does not fully specify the conditioning approximation in Eq. 60.
    """
    from .acquisitions import expected_improvement
    from .distributions import NormalMarginal
    prior=_check_gaussian(model,state,reference)
    weights=jnp.asarray(weights,dtype=reference.dtype)
    weights=weights/weights.sum()
    initial=expected_improvement(prior.marginal,best,maximize=maximize)
    def one(x):
        joint=model.posterior(state,jnp.concatenate([reference,x[None,:]],axis=0))
        variance=jnp.maximum(joint.covariance[-1,-1]+model.noise_at(state,x[None,:])[0],1e-14)
        updated=jnp.maximum(jnp.diag(prior.covariance)-joint.covariance[:-1,-1]**2/variance,0)
        remaining=expected_improvement(NormalMarginal(prior.mean,updated),best,maximize=maximize)
        return jnp.dot(weights,initial-remaining)
    return jax.vmap(one)

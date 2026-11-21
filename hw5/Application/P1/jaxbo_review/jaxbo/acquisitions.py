"""Acquisition functions: all return utilities to MAXIMIZE.

The objective direction is controlled independently by maximize. beta in UCB is
an SD multiplier, exactly as in review Eq. 45, not its square.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr,ndtri,betainc,gammaln
from .linalg import safe_sqrt
from .distributions import NormalMarginal,StudentMarginal,EmpiricalMarginal,BetaMarginal


def normal_pdf(x):
    return jnp.exp(-.5*x*x)/jnp.sqrt(2*jnp.pi)


def student_pdf(x,df):
    return jnp.exp(gammaln((df+1)/2)-gammaln(df/2)-.5*jnp.log(df*jnp.pi)-(df+1)/2*jnp.log1p(x*x/df))


def student_cdf(x,df):
    # betainc's derivative at its endpoint is singular; a local Taylor expansion
    # avoids 0*inf at x=0. df is a fixed shape parameter, not differentiated here.
    small=jnp.abs(x)<1e-5
    xx=jnp.where(small,1e-5,x)
    tail=.5*betainc(df/2,.5,df/(df+xx*xx))
    regular=jnp.where(xx>=0,1-tail,tail)
    near=.5+student_pdf(jnp.asarray(0.,dtype=x.dtype),df)*x*(1-(df+1)/(6*df)*x*x)
    return jnp.where(small,near,regular)


def probability_improvement(prediction,best,*,maximize=True,xi=0.0):
    sign=1.0 if maximize else -1.0
    threshold=best+sign*xi
    if isinstance(prediction,EmpiricalMarginal):
        return jnp.mean(sign*(prediction.values-threshold)>0,axis=0)
    if isinstance(prediction,BetaMarginal):
        t=jnp.clip(threshold,0,1)
        cdf=betainc(prediction.alpha,prediction.beta,t)
        return (1-cdf) if maximize else cdf
    delta=sign*(prediction.mean-best)-xi
    scale=safe_sqrt(prediction.scale2 if isinstance(prediction,StudentMarginal) else prediction.variance)
    z=delta/jnp.where(scale>0,scale,1.0)
    p=student_cdf(z,prediction.df) if isinstance(prediction,StudentMarginal) else ndtr(z)
    return jnp.where(scale>0,p,(delta>0).astype(delta.dtype))


def expected_improvement(prediction,best,*,maximize=True,xi=0.0):
    sign=1.0 if maximize else -1.0
    threshold=best+sign*xi
    if isinstance(prediction,EmpiricalMarginal):
        return jnp.maximum(sign*(prediction.values-threshold),0).mean(axis=0)
    if isinstance(prediction,BetaMarginal):
        a,b=prediction.alpha,prediction.beta
        t=jnp.clip(threshold,0,1)
        if maximize:
            result=a/(a+b)*(1-betainc(a+1,b,t))-threshold*(1-betainc(a,b,t))
        else:
            result=threshold*betainc(a,b,t)-a/(a+b)*betainc(a+1,b,t)
        return jnp.maximum(result,0)
    delta=sign*(prediction.mean-best)-xi
    scale=safe_sqrt(prediction.scale2 if isinstance(prediction,StudentMarginal) else prediction.variance)
    z=delta/jnp.where(scale>0,scale,1.0)
    if isinstance(prediction,StudentMarginal):
        # Finite for df>1. User must choose a prior with a finite predictive mean.
        result=delta*student_cdf(z,prediction.df)+scale*(prediction.df+z*z)/(prediction.df-1)*student_pdf(z,prediction.df)
        result=jnp.where(prediction.df>1,result,jnp.inf)
    else:
        result=delta*ndtr(z)+scale*normal_pdf(z)
    # At zero variance, retain a POSITIVE deterministic improvement. Review's
    # zero-variance 'vanishes' statement assumes the incumbent dominates it.
    return jnp.where(scale>0,jnp.maximum(result,0),jnp.maximum(delta,0))


def upper_confidence_bound(prediction,*,maximize=True,beta=2.0):
    sign=1.0 if maximize else -1.0
    return sign*prediction.mean+beta*safe_sqrt(prediction.variance)


def posterior_quantile(prediction,probability=.95):
    """Bayes-UCB quantile. Shape parameters are static model choices."""
    if isinstance(prediction,NormalMarginal):
        return prediction.mean+safe_sqrt(prediction.variance)*ndtri(probability)
    if isinstance(prediction,EmpiricalMarginal):
        return jnp.quantile(prediction.values,probability,axis=0)
    if isinstance(prediction,BetaMarginal):
        lo,hi=jnp.zeros_like(prediction.mean),jnp.ones_like(prediction.mean)
        cdf=lambda t:betainc(prediction.alpha,prediction.beta,t)
    elif isinstance(prediction,StudentMarginal):
        lo,hi=jnp.full_like(prediction.mean,-1e8),jnp.full_like(prediction.mean,1e8)
        cdf=lambda t:student_cdf(t,prediction.df)
    else: raise TypeError("Unsupported marginal distribution")
    def step(_,bounds):
        low,high=bounds; mid=(low+high)/2; below=cdf(mid)<probability
        return jnp.where(below,mid,low),jnp.where(below,high,mid)
    lo,hi=jax.lax.fori_loop(0,80,step,(lo,hi)); value=(lo+hi)/2
    if isinstance(prediction,StudentMarginal):
        return prediction.mean+safe_sqrt(prediction.scale2)*value
    return value


def finite_gp_ucb_multiplier(t,n_arms,delta=.1):
    """Finite-domain Gaussian union-bound schedule; not a universal regret guarantee."""
    return jnp.sqrt(2*jnp.log(n_arms*jnp.pi**2*jnp.maximum(t,1)**2/(6*delta)))


def thompson_on_candidates(model,state,points,key,*,maximize=True):
    """One JOINT draw on a finite set. Never sample independent marginal normals."""
    if hasattr(model,"sample"):
        values=model.sample(state,points,key,1)[0]
    elif hasattr(model,"thompson_function"):
        values=model.thompson_function(state,key)(points)
    else:
        values=model.posterior(state,points).sample(key,1)[0]
    idx=jnp.argmax(values if maximize else -values)
    return points[idx],values


def spectral_thompson(model,state,key,n_features=128):
    """Fixed continuous RFF posterior draw, as motivated by Eq. 46.

    Refits the finite-feature approximation at the existing exact-GP parameters.
    It is an approximate GP posterior function, not an exact continuous GP draw.
    """
    from .models.spectral import SpectralGP
    from .models.gp import ExactGP
    if not isinstance(model,ExactGP):
        raise TypeError("Use finite-candidate Thompson sampling for this model")
    name=getattr(model.kernel,"__name__","")
    if model.mean_function is not None:
        raise ValueError("RFF Thompson currently requires a constant GP mean")
    k1,k2=jax.random.split(key)
    spectral=SpectralGP.create(k1,state.x.shape[1],n_features,kernel=name)
    approximation=spectral.fit(state.x,state.y,state.params,noise_variance=state.noise_variance,mask=state.mask)
    return spectral.thompson_function(approximation,k2)


def make_acquisition(name,model,state,best,*,key=None,maximize=True,xi=0.0,beta=2.0,
                     quantile=.95,n_features=128):
    """Build deterministic a(X[n,d])->[n]. Randomness is fixed at construction."""
    if name=="ei":
        return lambda x:expected_improvement(model.predict(state,x),best,maximize=maximize,xi=xi)
    if name=="pi":
        return lambda x:probability_improvement(model.predict(state,x),best,maximize=maximize,xi=xi)
    if name=="ucb":
        return lambda x:upper_confidence_bound(model.predict(state,x),maximize=maximize,beta=beta)
    if name=="quantile":
        p=quantile if maximize else 1-quantile; sign=1 if maximize else -1
        return lambda x:sign*posterior_quantile(model.predict(state,x),p)
    if name=="mean":
        return lambda x:(1 if maximize else -1)*model.predict(state,x).mean
    if name=="variance":
        return lambda x:model.predict(state,x).variance
    if name=="ts":
        if key is None: raise ValueError("Thompson sampling requires a PRNG key")
        draw=model.thompson_function(state,key) if hasattr(model,"thompson_function") else spectral_thompson(model,state,key,n_features)
        return lambda x:(1 if maximize else -1)*draw(x)
    raise ValueError(f"Unknown acquisition {name!r}; entropy acquisitions have separate builders")


def monte_carlo_qei(model,state,batch,key,best,*,maximize=True,n_samples=256):
    """Additional joint-batch EI, with correlated latent draws."""
    samples=model.posterior(state,batch).sample(key,n_samples)
    improvements=(1 if maximize else -1)*(samples-best)
    return jnp.mean(jnp.maximum(jnp.max(improvements,axis=1),0))

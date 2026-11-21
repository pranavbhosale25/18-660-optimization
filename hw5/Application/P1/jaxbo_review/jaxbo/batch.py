"""Pending-point fantasies, constant liar, kriging believer, and UCB batches."""
from __future__ import annotations
import jax
import jax.numpy as jnp
from .distributions import Gaussian
from .acquisitions import make_acquisition,expected_improvement,probability_improvement,upper_confidence_bound


def _training_noise(model,state,x,y,noise_variance):
    # ExactGP caches individual noise levels; do not silently replace them by
    # its default noise hyperparameter when constructing fantasy models.
    if noise_variance is None and hasattr(state,"noise_variance"):
        noise_variance=state.noise_variance
    if noise_variance is None:
        noise_variance=model.noise_at(state,x)
    return jnp.broadcast_to(noise_variance,y.shape)


def fantasy_states(key,model,state,x,y,pending,*,n_fantasies=16,noise_variance=None,pending_noise=None):
    """Sample pending OBSERVATIONS jointly, then refit each conditional model.

    GP parameters stay fixed across fantasies. Known heteroscedastic training and
    pending variances can be supplied; they are never inferred from latent variance.
    """
    if not getattr(model,"supports_gaussian_conditioning",False):
        raise TypeError("Fantasy refitting supports ExactGP, SparseGP, and SpectralGP; other models need a custom conditioner")
    posterior=model.posterior(state,pending,observation_noise=True,noise_variance=pending_noise)
    if not isinstance(posterior,Gaussian):
        raise TypeError("Fantasy helper currently supports Gaussian models only")
    observations=posterior.sample(key,n_fantasies)
    xx=jnp.concatenate([x,pending],axis=0)
    existing=_training_noise(model,state,x,y,noise_variance)
    new_noise=model.noise_at(state,pending) if pending_noise is None else jnp.broadcast_to(pending_noise,(pending.shape[0],))
    noise=jnp.concatenate([existing,new_noise])
    states=jax.vmap(lambda yy:model.fit(xx,jnp.concatenate([y,yy]),state.params,noise_variance=noise))(observations)
    return states,observations


def fantasy_acquisition(name,key,model,state,x,y,pending,*,maximize=True,n_fantasies=16,xi=0,beta=2.0,
                        noise_variance=None,pending_noise=None,incumbent="observed"):
    states,observations=fantasy_states(key,model,state,x,y,pending,n_fantasies=n_fantasies,
                                     noise_variance=noise_variance,pending_noise=pending_noise)
    allx=jnp.concatenate([x,pending])
    if incumbent not in ("observed","posterior"): raise ValueError("Unknown incumbent policy")
    def utility(points):
        def one(s,yy):
            values=jnp.concatenate([y,yy]) if incumbent=="observed" else model.predict(s,allx).mean
            best=jnp.max(values) if maximize else jnp.min(values)
            prediction=model.predict(s,points)
            if name=="ei": return expected_improvement(prediction,best,maximize=maximize,xi=xi)
            if name=="pi": return probability_improvement(prediction,best,maximize=maximize,xi=xi)
            if name=="ucb": return upper_confidence_bound(prediction,maximize=maximize,beta=beta)
            raise ValueError("Fantasy acquisitions support ei, pi, and ucb")
        return jax.vmap(one)(states,observations).mean(axis=0)
    return utility


def batch_greedy(key,model,state,x,y,lower,upper,optimizer,*,q=4,strategy="kriging_believer",
                 acquisition="ei",maximize=True,liar_value=None,n_fantasies=16,xi=0,beta=2.0,
                 noise_variance=None,pending_noise=None,incumbent="observed"):
    """Sequential batch construction. Returns [q,d]; does not evaluate objective.

    strategy=fantasies integrates all previously selected points jointly against
    the ORIGINAL posterior. This differs from a single sampled hallucination.
    Supports Gaussian regression models with GPParams and noise_at. For sparse
    or spectral models fitted with known training variances, pass those variances
    explicitly; ExactGP recovers them from its state. pending_noise is a scalar
    or [q] variances in proposal order (otherwise use the model noise default).
    """
    if q<1 or strategy not in ("constant_liar","kriging_believer","fantasies"):
        raise ValueError("Invalid batch size or strategy")
    if incumbent not in ("observed","posterior"):
        raise ValueError("incumbent must be observed or posterior")
    if not getattr(model,"supports_gaussian_conditioning",False):
        raise TypeError("Batch conditioning requires a Gaussian GP regression model with GPParams and noise_at")
    original_state=state; original_x=x; original_y=y
    noise=_training_noise(model,state,x,y,noise_variance)
    original_noise=noise
    planned_noise=None if pending_noise is None else jnp.broadcast_to(pending_noise,(q,))
    selected=[]
    for i in range(q):
        key,ka,ko=jax.random.split(key,3)
        if strategy=="fantasies" and selected:
            pending=jnp.stack(selected)
            utility=fantasy_acquisition(acquisition,ka,model,original_state,original_x,original_y,pending,
                      maximize=maximize,n_fantasies=n_fantasies,xi=xi,beta=beta,
                      noise_variance=original_noise,pending_noise=None if planned_noise is None else planned_noise[:i],
                      incumbent=incumbent)
        else:
            target=y if incumbent=="observed" else model.predict(state,x).mean
            best=jnp.max(target) if maximize else jnp.min(target)
            utility=make_acquisition(acquisition,model,state,best,key=ka,maximize=maximize,xi=xi,beta=beta)
        avoid=jnp.concatenate([original_x,jnp.stack(selected)],axis=0) if selected else original_x
        def scalar(point):
            duplicate=jnp.any(jnp.sum(((point-avoid)/(upper-lower))**2,axis=1)<1e-12)
            return jnp.where(duplicate,-jnp.inf,utility(point[None,:])[0])
        proposal=optimizer(ko,scalar,lower,upper).x
        selected.append(proposal)
        if strategy!="fantasies":
            if strategy=="kriging_believer": yy=model.predict(state,proposal[None,:]).mean[0]
            else: yy=jnp.mean(original_y) if liar_value is None else jnp.asarray(liar_value,dtype=y.dtype)
            proposal_noise=model.noise_at(state,proposal[None,:]) if planned_noise is None else planned_noise[i:i+1]
            noise=jnp.concatenate([noise,proposal_noise])
            x=jnp.concatenate([x,proposal[None,:]]); y=jnp.concatenate([y,yy[None]])
            state=model.fit(x,y,state.params,noise_variance=noise)
    return jnp.stack(selected)


def batch_ucb(key,model,state,lower,upper,optimizer,betas,*,maximize=True):
    """Parallel proposals for different exploration multipliers; duplicates possible."""
    keys=jax.random.split(key,len(betas)); candidates=[]
    for k,beta in zip(keys,betas):
        utility=lambda x:upper_confidence_bound(model.predict(state,x),maximize=maximize,beta=beta)
        candidates.append(optimizer(k,lambda x:utility(x[None,:])[0],lower,upper).x)
    return jnp.stack(candidates)

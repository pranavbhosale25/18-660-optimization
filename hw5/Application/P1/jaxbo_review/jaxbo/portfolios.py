"""GP-Hedge and entropy-search portfolio mechanisms, Section IV-D."""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp


class HedgeState(NamedTuple):
    gains:jax.Array


def initialize_hedge(n_strategies):
    return HedgeState(jnp.zeros(n_strategies,dtype=float))


def hedge_choose(key,state,proposals,eta=1.0):
    probabilities=jax.nn.softmax(eta*state.gains)
    index=jax.random.categorical(key,jnp.log(probabilities))
    return proposals[index],index,probabilities


def hedge_update(state,model,updated_posterior,all_proposals,*,maximize=True):
    """Update ALL strategies using new posterior means, not only the chosen arm."""
    rewards=(1 if maximize else -1)*model.predict(updated_posterior,all_proposals).mean
    return HedgeState(state.gains+rewards)


def propose_portfolio(key,acquisitions,optimizer,lower,upper):
    keys=jax.random.split(key,len(acquisitions))
    return jnp.stack([optimizer(k,lambda x:a(x[None,:])[0],lower,upper).x for k,a in zip(keys,acquisitions)])


def entropy_portfolio_choose(proposals,entropy_acquisition):
    """Eq. 48-49: optimize information gain ONLY among base-policy proposals."""
    scores=entropy_acquisition(proposals)
    index=jnp.argmax(scores)
    return proposals[index],index,scores

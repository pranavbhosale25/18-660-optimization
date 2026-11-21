"""Beta-Bernoulli conjugacy and Algorithm 2 (Thompson sampling)."""
from __future__ import annotations
from typing import NamedTuple
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from ..distributions import BetaMarginal


class BetaState(NamedTuple):
    alpha: jax.Array
    beta: jax.Array


def beta_update(state:BetaState,arm,reward):
    return BetaState(state.alpha.at[arm].add(reward),state.beta.at[arm].add(1-reward))


def beta_thompson(key,state:BetaState):
    return jnp.argmax(jax.random.beta(key,state.alpha,state.beta))


@dataclass(frozen=True)
class BetaBernoulli:
    n_arms:int
    alpha:float=1.0
    beta:float=1.0

    def initialize(self):
        if self.n_arms<1 or self.alpha<=0 or self.beta<=0:
            raise ValueError("n_arms and prior parameters must be positive")
        return BetaState(jnp.full(self.n_arms,self.alpha,dtype=float),jnp.full(self.n_arms,self.beta,dtype=float))

    def fit(self,x,y,params=None,*,mask=None):
        # Caller must supply integer arm IDs in [0,n_arms) and Bernoulli 0/1 data.
        arm=x[:,0].astype(int) if x.ndim==2 else x.astype(int)
        weights=jnp.ones_like(y) if mask is None else mask.astype(y.dtype)
        state=self.initialize()
        return BetaState(state.alpha.at[arm].add(weights*y),state.beta.at[arm].add(weights*(1-y)))

    def predict(self,state,x):
        arm=x[:,0].astype(int) if x.ndim==2 else x.astype(int)
        return BetaMarginal(state.alpha[arm],state.beta[arm])

    def thompson_function(self,state,key):
        w=jax.random.beta(key,state.alpha,state.beta)
        return lambda x:w[x[:,0].astype(int)]


def run_beta_bandit(key,probabilities,steps:int,alpha=1.0,beta=1.0):
    """Compiled simulation of Algorithm 2; real experiments should use ask/update."""
    state=BetaBernoulli(probabilities.size,alpha,beta).initialize()
    def step(carry,_):
        state,key=carry
        key,ka,ky=jax.random.split(key,3)
        arm=beta_thompson(ka,state)
        reward=jax.random.bernoulli(ky,probabilities[arm]).astype(float)
        return (beta_update(state,arm,reward),key),(arm,reward)
    (state,key),(arms,rewards)=jax.lax.scan(step,(state,key),None,length=steps)
    return state,arms,rewards

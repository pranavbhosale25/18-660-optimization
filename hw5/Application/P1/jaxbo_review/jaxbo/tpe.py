"""Continuous-box Parzen density-ratio optimizer.

The review only mentions TPE as a conditional-space alternative, without giving
its algorithm. This is an explicitly added independent-coordinate continuous TPE
baseline, NOT the full tree-structured categorical/conditional Hyperopt algorithm.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
import math
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr,ndtri,logsumexp
from .spaces import Box


class TPEState(NamedTuple):
    x:jax.Array
    good:jax.Array


@dataclass(frozen=True)
class ParzenEstimator:
    space:Box
    gamma:float=.2
    n_candidates:int=128
    min_bandwidth:float=.03
    maximize:bool=True

    def fit(self,x,y):
        if not 0<self.gamma<1 or x.shape[0]<2: raise ValueError("TPE requires at least two observations and 0<gamma<1")
        n_good=max(1,min(x.shape[0]-1,int(math.ceil(self.gamma*x.shape[0]))))
        order=jnp.argsort(-y if self.maximize else y)
        good=jnp.zeros(y.shape,dtype=bool).at[order[:n_good]].set(True)
        return TPEState(self.space.normalize(x),good)

    def _bandwidth(self,x,mask):
        count=mask.sum(); mean=(x*mask[:,None]).sum(axis=0)/jnp.maximum(count,1)
        variance=(((x-mean)**2)*mask[:,None]).sum(axis=0)/jnp.maximum(count,1)
        return jnp.maximum(1.06*jnp.sqrt(variance)*jnp.maximum(count,1)**(-.2),self.min_bandwidth)

    def _log_density(self,state,points,mask):
        h=self._bandwidth(state.x,mask)
        z=(points[:,None,:]-state.x[None,:,:])/h
        normalizer=ndtr((1-state.x)/h)-ndtr(-state.x/h)
        logs=-.5*z*z-jnp.log(h)-.5*jnp.log(2*jnp.pi)-jnp.log(normalizer)[None,:,:]
        logs=jnp.where(mask[None,:,None],logs,-jnp.inf)
        # Include one unit-weight uniform-prior component on the unit box.
        logs=jnp.concatenate([logs,jnp.zeros((points.shape[0],1,points.shape[1]))],axis=1)
        return jnp.sum(logsumexp(logs,axis=1)-jnp.log(mask.sum()+1),axis=1)

    def ask(self,key,state):
        n,d=state.x.shape; k1,k2,k3=jax.random.split(key,3)
        h=self._bandwidth(state.x,state.good)
        logits=jnp.concatenate([jnp.where(state.good,0.,-jnp.inf),jnp.zeros(1)])
        component=jax.random.categorical(k1,logits,shape=(self.n_candidates,d))
        centers=state.x[jnp.minimum(component,n-1),jnp.arange(d)[None,:]]
        low=ndtr(-centers/h); high=ndtr((1-centers)/h)
        u=jax.random.uniform(k2,centers.shape)
        probabilities=jnp.clip(low+u*(high-low),1e-12,1-1e-12)
        normal=centers+h*ndtri(probabilities)
        points=jnp.where(component==n,jax.random.uniform(k3,centers.shape),normal)
        score=self._log_density(state,points,state.good)-self._log_density(state,points,~state.good)
        return self.space.denormalize(points[jnp.argmax(score)])

"""Learned nonlinear features with a Bayesian final layer (Section II-B).

This is a small native-JAX MLP feature model, not a replication of DNGO's full
training pipeline or any deep-belief-network training algorithm.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from ..optim import minimize_adam
from .linear import BayesianLinear


def initialize_mlp(key,widths):
    keys=jax.random.split(key,len(widths)-1)
    return tuple((jax.random.normal(k,(a,b))*jnp.sqrt(2/(a+b)),jnp.zeros(b))
                 for k,a,b in zip(keys,widths[:-1],widths[1:]))


def mlp(params,x,*,activate_final=False):
    for i,(weight,bias) in enumerate(params):
        x=x@weight+bias
        if i<len(params)-1 or activate_final: x=jnp.tanh(x)
    return x


class NeuralState(NamedTuple):
    hidden:object
    linear:object
    training_loss:jax.Array


@dataclass(frozen=True)
class NeuralLinear:
    widths:tuple[int,...]=(16,16)
    steps:int=200
    learning_rate:float=.02
    l2:float=1e-4
    noise_variance:float|None=.01
    requires_key:bool=True

    def _model(self,hidden):
        def features(x):
            f=mlp(hidden,x,activate_final=True)
            return jnp.concatenate([jnp.ones((x.shape[0],1),dtype=x.dtype),f],axis=1)
        return BayesianLinear(features=features,noise_variance=self.noise_variance)

    def fit(self,x,y,params=None,*,key=None,mask=None):
        if key is None: raise ValueError("NeuralLinear requires an explicit training key")
        network=initialize_mlp(key,(x.shape[1],)+self.widths+(1,))
        vector,unravel=ravel_pytree(network)
        mask=jnp.ones(y.shape,dtype=bool) if mask is None else mask
        def objective(theta):
            residual=mlp(unravel(theta),x)[:,0]-y
            return jnp.sum(jnp.where(mask,residual**2,0))/jnp.maximum(mask.sum(),1)+self.l2*jnp.mean(theta**2)
        result=minimize_adam(objective,vector,steps=self.steps,learning_rate=self.learning_rate)
        hidden=unravel(result.x)[:-1]
        linear=self._model(hidden).fit(x,y,mask=mask)
        return NeuralState(hidden,linear,result.value)

    def predict(self,state,x,**kwargs): return self._model(state.hidden).predict(state.linear,x,**kwargs)
    def posterior(self,state,x,**kwargs): return self._model(state.hidden).posterior(state.linear,x,**kwargs)
    def thompson_function(self,state,key): return self._model(state.hidden).thompson_function(state.linear,key)

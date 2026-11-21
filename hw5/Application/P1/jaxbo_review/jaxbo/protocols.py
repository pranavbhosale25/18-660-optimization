"""Structural interfaces for custom model implementations.

Prediction distributions must expose .mean and .variance. Analytic acquisitions
support NormalMarginal, StudentMarginal, BetaMarginal and EmpiricalMarginal.
Additional families can be integrated through a custom acquisition builder.
"""
from typing import Protocol,Any
import jax


class Surrogate(Protocol):
    def fit(self,x:jax.Array,y:jax.Array,params:Any=None,**kwargs:Any)->Any: ...
    def predict(self,state:Any,x:jax.Array)->Any: ...


class GaussianSurrogate(Surrogate,Protocol):
    def posterior(self,state:Any,x:jax.Array,**kwargs:Any)->Any: ...
    def noise_at(self,state:Any,x:jax.Array)->jax.Array: ...

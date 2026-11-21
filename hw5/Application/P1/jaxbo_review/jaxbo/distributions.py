"""Predictive distributions and marginal summaries (all states are PyTrees)."""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .linalg import normal_samples


class NormalMarginal(NamedTuple):
    mean: jax.Array
    variance: jax.Array


class StudentMarginal(NamedTuple):
    mean: jax.Array
    scale2: jax.Array
    df: jax.Array

    @property
    def variance(self):
        return jnp.where(self.df>2, self.scale2*self.df/(self.df-2), jnp.inf)


class EmpiricalMarginal(NamedTuple):
    values: jax.Array  # [draw, test point], or [tree, test point]

    @property
    def mean(self):
        return self.values.mean(axis=0)

    @property
    def variance(self):
        return self.values.var(axis=0)


class BetaMarginal(NamedTuple):
    alpha: jax.Array
    beta: jax.Array

    @property
    def mean(self):
        return self.alpha/(self.alpha+self.beta)

    @property
    def variance(self):
        a,b = self.alpha,self.beta
        return a*b/((a+b)**2*(a+b+1))


class Gaussian(NamedTuple):
    mean: jax.Array
    covariance: jax.Array

    def sample(self, key, n: int):
        return normal_samples(key,self.mean,self.covariance,n)

    @property
    def marginal(self):
        return NormalMarginal(self.mean,jnp.maximum(jnp.diag(self.covariance),0))


class StudentT(NamedTuple):
    mean: jax.Array
    scale_matrix: jax.Array
    df: jax.Array

    @property
    def covariance(self):
        return jnp.where(self.df>2,self.scale_matrix*self.df/(self.df-2),jnp.inf)

    def sample(self, key, n: int):
        k1,k2 = jax.random.split(key)
        z = normal_samples(k1,jnp.zeros_like(self.mean),self.scale_matrix,n)
        g = jax.random.gamma(k2,self.df/2,shape=(n,),dtype=self.mean.dtype)/(self.df/2)
        return self.mean[None,:]+z/jnp.sqrt(g[:,None])

    @property
    def marginal(self):
        return StudentMarginal(self.mean,jnp.maximum(jnp.diag(self.scale_matrix),0),self.df)


class Empirical(NamedTuple):
    values: jax.Array

    @property
    def mean(self):
        return self.values.mean(axis=0)

    @property
    def covariance(self):
        centered = self.values-self.mean
        return centered.T@centered/self.values.shape[0]

    def sample(self,key,n:int):
        return self.values[jax.random.randint(key,(n,),0,self.values.shape[0])]

    @property
    def marginal(self):
        return EmpiricalMarginal(self.values)

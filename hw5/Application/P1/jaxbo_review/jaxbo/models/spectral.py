"""Sparse-spectrum GPs and fixed posterior function draws, Eqs. 38-40/46."""
from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .gp import GPParams,default_params
from .linear import BayesianLinear,LinearState
from ..distributions import Gaussian,NormalMarginal


def draw_spectrum(key,d:int,n_features:int,kernel="rbf"):
    """Unscaled spectral frequencies. Matérn uses a multivariate Student-t law."""
    k1,k2=jax.random.split(key)
    normal=jax.random.normal(k1,(n_features,d))
    if kernel=="rbf":
        return normal
    nu={"matern12":.5,"matern32":1.5,"matern52":2.5}.get(kernel)
    if nu is None:
        raise ValueError("Spectrum supported for rbf, matern12, matern32, matern52")
    # ONE chi-square variable per frequency vector, NOT per coordinate.
    chi=jax.random.gamma(k2,nu,shape=(n_features,))/nu
    return normal/jnp.sqrt(chi[:,None])


class SpectralState(NamedTuple):
    params: GPParams
    frequencies: jax.Array
    linear: LinearState


@dataclass(frozen=True)
class SpectralGP:
    supports_gaussian_conditioning = True
    frequencies: object

    @classmethod
    def create(cls,key,d,n_features=128,kernel="rbf"):
        return cls(draw_spectrum(key,d,n_features,kernel))

    def features(self,p,x,frequencies=None):
        w=jnp.asarray(self.frequencies if frequencies is None else frequencies)
        phase=(x/jnp.exp(p.kernel["log_lengthscale"]))@w.T
        return jnp.exp(p.kernel["log_amplitude"])*jnp.concatenate([jnp.cos(phase),jnp.sin(phase)],axis=1)/jnp.sqrt(w.shape[0])

    def fit(self,x,y,params=None,*,noise_variance=None,mask=None,frequencies=None):
        p=default_params(x.shape[1]) if params is None else params
        w=jnp.asarray(self.frequencies if frequencies is None else frequencies)
        model=BayesianLinear(features=lambda xx:self.features(p,xx,w),noise_variance=1.0)
        noise=jnp.exp(2*p.log_noise) if noise_variance is None else noise_variance
        state=model.fit(x,y-p.mean,noise_variance=noise,mask=mask)
        return SpectralState(p,w,state)

    def predict(self,state,x,*,observation_noise=False,noise_variance=None):
        phi=self.features(state.params,x,state.frequencies)
        var=jnp.sum((phi@state.linear.covariance)*phi,axis=1)
        if observation_noise:
            var+=jnp.exp(2*state.params.log_noise) if noise_variance is None else noise_variance
        return NormalMarginal(state.params.mean+phi@state.linear.mean,jnp.maximum(var,0))

    def posterior(self,state,x,*,observation_noise=False,noise_variance=None):
        phi=self.features(state.params,x,state.frequencies)
        cov=phi@state.linear.covariance@phi.T
        if observation_noise:
            cov+=jnp.diag(jnp.broadcast_to(jnp.exp(2*state.params.log_noise) if noise_variance is None else noise_variance,(x.shape[0],)))
        return Gaussian(state.params.mean+phi@state.linear.mean,cov)

    def log_marginal_likelihood(self,state):
        return state.linear.log_evidence

    def thompson_function(self,state,key):
        w=Gaussian(state.linear.mean,state.linear.covariance).sample(key,1)[0]
        return lambda x:state.params.mean+self.features(state.params,x,state.frequencies)@w

    def noise_at(self,state,x):
        return jnp.broadcast_to(jnp.exp(2*state.params.log_noise),(x.shape[0],))

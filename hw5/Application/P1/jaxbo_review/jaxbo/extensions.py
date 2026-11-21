"""Explicit building blocks for Sections V-D and VIII, with scope qualifications."""
from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .models.gp import ExactGP,default_params
from .distributions import Gaussian,NormalMarginal
from .information import entropy_search_mc
from . import kernels


class HeteroscedasticState(NamedTuple):
    mean_state:object
    noise_state:object
    params:object


@dataclass(frozen=True)
class HeteroscedasticGP:
    """Alternating residual/log-noise GP approximation, not joint Bayesian inference.

    Known noise is supported exactly by ExactGP.fit(noise_variance=...). This
    optional learned-noise estimator is an added engineering approximation; it
    does NOT implement the review's CART-based smooth heteroscedastic model.
    """
    gp:ExactGP=ExactGP()
    noise_gp:ExactGP=ExactGP()
    iterations:int=3
    noise_floor:float=1e-5

    def fit(self,x,y,params=None):
        p=default_params(x.shape[1]) if params is None else params
        noise_params=default_params(x.shape[1],lengthscale=.3,amplitude=1,noise=.5,mean=2*p.log_noise)
        variance=jnp.full(y.shape,jnp.exp(2*p.log_noise))
        for _ in range(self.iterations):
            mean_state=self.gp.fit(x,y,p,noise_variance=variance)
            pred=self.gp.predict(mean_state,x)
            target=jnp.log(jnp.maximum((y-pred.mean)**2+pred.variance,self.noise_floor))
            noise_state=self.noise_gp.fit(x,target,noise_params)
            noise_pred=self.noise_gp.predict(noise_state,x)
            variance=jnp.maximum(jnp.exp(noise_pred.mean),self.noise_floor)
        mean_state=self.gp.fit(x,y,p,noise_variance=variance)
        return HeteroscedasticState(mean_state,noise_state,p)

    def noise_at(self,state,x):
        p=self.noise_gp.predict(state.noise_state,x)
        # Plug-in median variance, consistent with fit. Uncertainty in noise GP
        # is NOT integrated into the regression posterior.
        return jnp.maximum(jnp.exp(p.mean),self.noise_floor)

    def predict(self,state,x,*,observation_noise=False,noise_variance=None):
        pred=self.gp.predict(state.mean_state,x)
        noise=self.noise_at(state,x) if noise_variance is None else noise_variance
        return NormalMarginal(pred.mean,pred.variance+(noise if observation_noise else 0))

    def posterior(self,state,x,*,observation_noise=False,noise_variance=None):
        pred=self.gp.posterior(state.mean_state,x)
        noise=self.noise_at(state,x) if noise_variance is None else jnp.broadcast_to(noise_variance,(x.shape[0],))
        return Gaussian(pred.mean,pred.covariance+(jnp.diag(noise) if observation_noise else 0))


def learning_curve_kernel(spatial_kernel=kernels.matern52):
    """Asymptote covariance plus independent within-run decays from Eq. 63.

    Input rows are [configuration..., run_id, time]. Parameters have 'spatial',
    'time', and 'log_curve_amplitude'. This composition is an explicit design
    choice, not the complete freeze-thaw implementation in reference [149].
    """
    def k(p,x,z):
        asymptote=spatial_kernel(p["spatial"],x[:,:-2],z[:,:-2])
        same_run=x[:,-2,None]==z[None,:,-2]
        decay=kernels.freeze_thaw(p["time"],x[:,-1:],z[:,-1:])
        return asymptote+same_run*jnp.exp(2*p["log_curve_amplitude"])*decay
    return k


def choose_training_action(model,state,candidate_next_evaluations,terminal_representers,key,*,costs=None,maximize=False,n_functions=128,n_fantasies=16):
    """Finite-action information policy for new/resumed training runs.

    The caller supplies rows representing legal next measurements and terminal
    reference rows; this function chooses an index. It does not manage processes,
    checkpoints, or freeze/thaw lifecycle state.
    """
    utility=entropy_search_mc(model,state,terminal_representers,key,maximize=maximize,
                             n_functions=n_functions,n_fantasies=n_fantasies)
    score=utility(candidate_next_evaluations)
    if costs is not None: score=score/jnp.asarray(costs)
    index=jnp.argmax(score)
    return index,candidate_next_evaluations[index],score


def run_rembo(objective,embedding,key,*,n_steps=20,n_initial=5,**optimizer_options):
    """Run Algorithm 3 in the caller's chosen latent box; return both coordinate sets."""
    from .api import BayesianOptimizer
    optimizer=BayesianOptimizer(embedding.latent,key=key,**optimizer_options)
    result=optimizer.run(lambda z:objective(embedding.project(z)),n_steps,n_initial=n_initial)
    return result,embedding.project(result.x),embedding.project(result.best_x)

"""Ask/tell orchestration for external experiments and a fully compiled GP loop.

The ask/tell class validates host inputs and grows its data arrays; it is NOT
itself jittable. Numerical subroutines are native JAX. scan_optimize uses fixed
capacity and lax.scan when the objective itself is a pure JAX function.
"""
from __future__ import annotations
from typing import NamedTuple,Callable
import jax
import jax.numpy as jnp
from .spaces import Box,initial_design
from .models.gp import ExactGP,default_params
from .optim import AcquisitionOptimizer,maximize_discrete
from .acquisitions import make_acquisition
from .hyperparameters import fit_hyperparameters,integrate_hyperparameters,marginalized_acquisition,ensemble_moments
from .information import entropy_search_mc,predictive_entropy_search_mc


class BOResult(NamedTuple):
    x:jax.Array
    y:jax.Array
    best_x:jax.Array
    best_y:jax.Array


def _fit(model,x,y,params,key=None,**kwargs):
    if getattr(model,"requires_key",False): return model.fit(x,y,params,key=key,**kwargs)
    return model.fit(x,y,params,**kwargs)


class BayesianOptimizer:
    """Box-domain convenience API. All models/acquisitions remain separately usable.

    maximize=True follows the review. For minimization, use maximize=False.
    PRNG key is mandatory; the class never accesses global random state.
    """
    def __init__(self,space:Box,*,key,model=None,params=None,acquisition="ei",
                 optimizer=None,maximize=True,standardize=True,incumbent="observed",
                 hyper_method="fixed",hyper_options=None,xi=0.0,beta=2.0,
                 n_representers=32,information_samples=128,information_fantasies=16):
        if not isinstance(space,Box): raise TypeError("The convenience optimizer uses Box; mixed spaces use the low-level finite-candidate API")
        if incumbent not in ("observed","posterior"): raise ValueError("incumbent must be observed or posterior")
        self.space=space; self.key=key; self.model=ExactGP() if model is None else model
        self.params=params; self.acquisition=acquisition
        self.optimizer=AcquisitionOptimizer() if optimizer is None else optimizer
        self.maximize=maximize; self.standardize=standardize; self.incumbent=incumbent
        self.hyper_method=hyper_method; self.hyper_options={} if hyper_options is None else dict(hyper_options)
        self.xi=xi; self.beta=beta
        self.n_representers=n_representers; self.information_samples=information_samples; self.information_fantasies=information_fantasies
        self.x=jnp.empty((0,space.dimension),dtype=space.lower.dtype)
        self.y=jnp.empty((0,),dtype=space.lower.dtype)
        self.state=None; self.ensemble=None
        self.offset=jnp.asarray(0.0); self.scale=jnp.asarray(1.0)
        if self.acquisition in ("es","pes") and self.optimizer.method in ("lbfgs","adam"):
            raise ValueError("Histogram/mixture information acquisitions require a derivative-free optimizer")
        from .models.glm import GeneralizedLinear,GPClassifier
        from .models.bandits import BetaBernoulli
        if isinstance(self.model,BetaBernoulli):
            raise TypeError("Beta-Bernoulli arms are discrete; use beta_thompson or finite-candidate utilities")
        if standardize and isinstance(self.model,(GeneralizedLinear,GPClassifier)):
            raise ValueError("Bernoulli responses must not be standardized; set standardize=False")

    def tell(self,x,y):
        xx=jnp.asarray(x,dtype=self.space.lower.dtype)
        yy=jnp.asarray(y,dtype=self.space.lower.dtype)
        xx=xx[None,:] if xx.ndim==1 else xx
        yy=yy[None] if yy.ndim==0 else yy
        if xx.ndim!=2 or xx.shape[1]!=self.space.dimension or yy.shape!=(xx.shape[0],):
            raise ValueError("tell expects x[n,d] and y[n], or a single x[d] and scalar y")
        if not bool(jnp.all(jnp.isfinite(xx))) or not bool(jnp.all(jnp.isfinite(yy))):
            raise ValueError("Only finite observations are accepted; model failures as constraints separately")
        if not bool(jnp.all((xx>=self.space.lower)&(xx<=self.space.upper))):
            raise ValueError("Observed inputs lie outside the search box")
        from .models.glm import GeneralizedLinear,GPClassifier
        if isinstance(self.model,(GeneralizedLinear,GPClassifier)) and not bool(jnp.all((yy==0)|(yy==1))):
            raise ValueError("Bernoulli models require binary observations")
        self.x=jnp.concatenate([self.x,xx]); self.y=jnp.concatenate([self.y,yy])
        self.state=None; self.ensemble=None
        return self

    def fit(self):
        if self.y.size==0: raise ValueError("At least one observation is needed to fit a surrogate")
        self.offset=jnp.mean(self.y) if self.standardize else jnp.asarray(0.,self.y.dtype)
        sd=jnp.std(self.y)
        self.scale=jnp.where(sd>1e-8,sd,1.0) if self.standardize else jnp.asarray(1.,self.y.dtype)
        x=self.space.normalize(self.x); y=(self.y-self.offset)/self.scale
        self.key,kfit,kh=jax.random.split(self.key,3)
        if self.hyper_method=="fixed":
            self.state=_fit(self.model,x,y,self.params,kfit)
        else:
            from .models.sparse import SparseGP
            from .models.spectral import SpectralGP
            from .models.gp import StudentTProcess
            if not isinstance(self.model,(ExactGP,SparseGP,SpectralGP,StudentTProcess)):
                raise ValueError("The hyperparameter helper currently supports GP regression models only")
            p=default_params(self.space.dimension) if self.params is None else self.params
            if self.hyper_method in ("ml","map"):
                result=fit_hyperparameters(self.model,x,y,p,method=self.hyper_method,key=kh,**self.hyper_options)
                self.params=result.params; self.state=result.state
            else:
                self.ensemble=integrate_hyperparameters(kh,self.model,x,y,p,method=self.hyper_method,**self.hyper_options)
                # A representative state is only for inspection; acquisitions use ALL states.
                idx=jnp.argmax(self.ensemble.weights)
                self.state=jax.tree_util.tree_map(lambda a:a[idx],self.ensemble.states)
        leaves=jax.tree_util.tree_leaves(self.state)
        # State diagnostics may legitimately contain NaNs (e.g. no MCMC acceptance
        # for a Laplace fit); check predictions rather than blindly rejecting leaves.
        prediction=self.model.predict(self.state,x)
        if not bool(jnp.all(jnp.isfinite(prediction.mean))) or not bool(jnp.all(jnp.isfinite(prediction.variance))):
            raise FloatingPointError("Surrogate fit produced nonfinite predictions; check noise, priors, conditioning, and x64")
        self._train_x=x; self._train_y=y
        return self.state

    def _best(self):
        if self.incumbent=="observed": values=self._train_y
        elif self.ensemble is None: values=self.model.predict(self.state,self._train_x).mean
        else: values=ensemble_moments(self.model,self.ensemble,self._train_x)[0]
        return jnp.max(values) if self.maximize else jnp.min(values)

    def _utility(self,key):
        if self.ensemble is not None:
            if self.acquisition in ("es","pes"):
                raise ValueError("Compose per-particle information acquisitions explicitly; automatic ensemble ES/PES is not supplied")
            return marginalized_acquisition(self.acquisition,self.model,self.ensemble,self._best(),key=key,
                        maximize=self.maximize,xi=self.xi/self.scale,beta=self.beta)
        if self.acquisition in ("es","pes"):
            kd,ka=jax.random.split(key)
            representers=initial_design(kd,self.n_representers,self.space.dimension,"latin")
            builder=entropy_search_mc if self.acquisition=="es" else predictive_entropy_search_mc
            options={"n_fantasies":self.information_fantasies} if self.acquisition=="es" else {}
            return builder(self.model,self.state,representers,ka,maximize=self.maximize,
                           n_functions=self.information_samples,**options)
        return make_acquisition(self.acquisition,self.model,self.state,self._best(),key=key,
                        maximize=self.maximize,xi=self.xi/self.scale,beta=self.beta)

    def ask(self,*,q=1,batch_strategy="kriging_believer",pending=None,candidates=None):
        """Suggest physical-space points. q=1 returns [d], otherwise [q,d].

        Supply pending explicitly for asynchronous work. This class does not launch
        jobs, keep a worker queue, or silently treat suggestions as observations.
        """
        if q<1: raise ValueError("q must be positive")
        self.key,ka,ko=jax.random.split(self.key,3)
        if self.y.size==0:
            if pending is not None:
                raise ValueError("Add initial observations before using pending-point fantasy conditioning")
            if candidates is None:
                points=self.space.sample(ko,q,"latin")
            else:
                raw=jnp.asarray(candidates,dtype=self.space.lower.dtype)
                if raw.ndim!=2 or raw.shape[1]!=self.space.dimension or raw.shape[0]<q:
                    raise ValueError("candidates must contain at least q points with shape [n,d]")
                if not bool(jnp.all(jnp.isfinite(raw)&(raw>=self.space.lower)&(raw<=self.space.upper))):
                    raise ValueError("Candidate points must be finite and within bounds")
                raw=jnp.unique(raw,axis=0)
                if raw.shape[0]<q: raise ValueError("Not enough distinct candidates for the initial batch")
                indices=jax.random.permutation(ko,raw.shape[0])[:q]
                points=raw[indices]
            return points[0] if q==1 else points
        if self.state is None: self.fit()
        lower=jnp.zeros(self.space.dimension,dtype=self.space.lower.dtype)
        upper=jnp.ones_like(lower)
        if q>1:
            if candidates is not None or pending is not None or self.ensemble is not None:
                raise ValueError("Compose low-level batch utilities for finite candidates, pending jobs, or hyperparameter mixtures")
            from .batch import batch_greedy
            result=batch_greedy(ka,self.model,self.state,self._train_x,self._train_y,lower,upper,self.optimizer,
                               q=q,strategy=batch_strategy,acquisition=self.acquisition,maximize=self.maximize,
                               xi=self.xi/self.scale,beta=self.beta,incumbent=self.incumbent)
            return self.space.denormalize(result)
        if pending is not None:
            if self.ensemble is not None: raise ValueError("Compose fantasy and hyperparameter averaging explicitly")
            from .batch import fantasy_acquisition
            pp=jnp.atleast_2d(jnp.asarray(pending,dtype=lower.dtype))
            if pp.shape[1]!=self.space.dimension or not bool(jnp.all(jnp.isfinite(pp)&(pp>=self.space.lower)&(pp<=self.space.upper))):
                raise ValueError("Pending points must be finite and within bounds")
            pending_normalized=self.space.normalize(pp)
            utility=fantasy_acquisition(self.acquisition,ka,self.model,self.state,self._train_x,self._train_y,
                        pending_normalized,maximize=self.maximize,xi=self.xi/self.scale,beta=self.beta,
                        incumbent=self.incumbent,n_fantasies=self.information_fantasies)
            avoid=jnp.concatenate([self._train_x,pending_normalized])
        else:
            utility=self._utility(ka); avoid=self._train_x
        def scalar(point):
            duplicate=jnp.any(jnp.sum((point-avoid)**2,axis=1)<1e-12)
            return jnp.where(duplicate,-jnp.inf,utility(point[None,:])[0])
        if candidates is None: result=self.optimizer(ko,scalar,lower,upper)
        else:
            raw=jnp.asarray(candidates,dtype=lower.dtype)
            if raw.ndim!=2 or raw.shape[1]!=self.space.dimension or raw.shape[0]==0:
                raise ValueError("candidates must be a nonempty [n,d] array")
            if not bool(jnp.all(jnp.isfinite(raw)&(raw>=self.space.lower)&(raw<=self.space.upper))):
                raise ValueError("Candidate points must be finite and within bounds")
            result=maximize_discrete(scalar,self.space.normalize(raw))
        if not bool(jnp.isfinite(result.value)):
            raise ValueError("No finite nonduplicate acquisition candidate was found")
        return self.space.denormalize(result.x)

    def recommend(self,method="observed"):
        """Return (x,value). Posterior method returns an UNOBSERVED prediction, not a measurement."""
        if self.y.size==0: raise ValueError("No observations to recommend from")
        if method=="observed":
            idx=jnp.argmax(self.y if self.maximize else -self.y)
            return self.x[idx],self.y[idx]
        if method!="posterior": raise ValueError("method must be observed or posterior")
        if self.state is None: self.fit()
        if self.ensemble is None: mean=self.model.predict(self.state,self._train_x).mean
        else: mean=ensemble_moments(self.model,self.ensemble,self._train_x)[0]
        idx=jnp.argmax(mean if self.maximize else -mean)
        return self.x[idx],self.offset+self.scale*mean[idx]

    def run(self,objective,n_steps:int,*,n_initial=5,design="latin"):
        """Host loop for arbitrary expensive objectives. Existing observations are retained."""
        if n_steps<0: raise ValueError("n_steps must be nonnegative")
        if self.y.size==0:
            self.key,k=jax.random.split(self.key)
            initial=self.space.sample(k,n_initial,design)
            for point in initial: self.tell(point,objective(point))
        for _ in range(n_steps):
            point=self.ask(); self.tell(point,objective(point))
        point,value=self.recommend()
        return BOResult(self.x,self.y,point,value)


def scan_optimize(objective,key,lower,upper,initial_x,initial_y,*,n_steps:int,
                  model=ExactGP(),params=None,acquisition="ei",optimizer=AcquisitionOptimizer(n_initial=64,restarts=4,steps=30),
                  maximize=True,xi=0.0,beta=2.0,hyper_method="fixed",hyper_steps=20):
    """Fixed-shape, fully jittable Algorithm 1 for pure-JAX scalar objectives.

    Inputs/outputs are in physical units; GP inputs are normalized to [0,1].
    Responses are NOT standardized. Specify suitable params or transform y.
    The padded solve costs O((n_initial+n_steps)^3) per step, trading work for a
    single compilation. Inactive rows must contain finite values and are masked.
    """
    if initial_x.ndim!=2 or initial_y.shape!=(initial_x.shape[0],) or initial_y.size<1:
        raise ValueError("Provide nonempty initial_x[n,d] and initial_y[n]")
    if n_steps<0 or hyper_method not in ("fixed","ml","map"):
        raise ValueError("Invalid step count or compiled hyperparameter method")
    n,d=initial_x.shape; capacity=n+n_steps
    p=default_params(d) if params is None else params
    xbuf=jnp.zeros((capacity,d),dtype=initial_x.dtype).at[:n].set((initial_x-lower)/(upper-lower))
    ybuf=jnp.zeros(capacity,dtype=initial_y.dtype).at[:n].set(initial_y)
    mask=jnp.arange(capacity)<n
    low=jnp.zeros(d,dtype=initial_x.dtype); high=jnp.ones(d,dtype=initial_x.dtype)
    def step(carry,_):
        xx,yy,mask,count,p,key=carry
        key,ka,ko=jax.random.split(key,3)
        if hyper_method=="fixed": state=model.fit(xx,yy,p,mask=mask)
        else:
            fitted=fit_hyperparameters(model,xx,yy,p,mask=mask,method=hyper_method,steps=hyper_steps)
            state,p=fitted.state,fitted.params
        best=jnp.max(jnp.where(mask,yy,-jnp.inf)) if maximize else jnp.min(jnp.where(mask,yy,jnp.inf))
        a=make_acquisition(acquisition,model,state,best,key=ka,maximize=maximize,xi=xi,beta=beta)
        def scalar(point):
            duplicate=jnp.any(mask&(jnp.sum((xx-point)**2,axis=1)<1e-12))
            return jnp.where(duplicate,-jnp.inf,a(point[None,:])[0])
        proposal=optimizer(ko,scalar,low,high).x
        value=jnp.asarray(objective(lower+proposal*(upper-lower)),dtype=yy.dtype)
        xx=xx.at[count].set(proposal); yy=yy.at[count].set(value); mask=mask.at[count].set(True)
        return (xx,yy,mask,count+1,p,key),(proposal,value)
    (xx,yy,mask,_,_,_),_=jax.lax.scan(step,(xbuf,ybuf,mask,jnp.array(n),p,key),None,length=n_steps)
    physical=lower+xx*(upper-lower); idx=jnp.argmax(yy if maximize else -yy)
    return BOResult(physical,yy,physical[idx],yy[idx])

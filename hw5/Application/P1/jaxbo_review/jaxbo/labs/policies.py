"""Configurable adapters using the existing JAXBO numerical components.

A policy receives ONLY TaskSpec, an optional public initialization design, and
charged Observation objects. It never receives a Problem or an oracle callback.
All GP arrays have fixed capacity to avoid recompiling after every observation.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from contextlib import contextmanager
from typing import Any
import math
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from .. import kernels
from ..warps import kumaraswamy_warp, beta_warp
from ..spaces import MixedSpace, initial_design
from ..models import ExactGP, RandomForest, GPParams, default_params
from ..hyperparameters import fit_hyperparameters, gaussian_log_prior
from ..acquisitions import make_acquisition
from ..constraints import gaussian_feasibility, constrained, cost_sensitive
from ..distributions import NormalMarginal
from ..optim import AcquisitionOptimizer
from .core import Action, Observation, TaskSpec, validate_point


_COMPILED_CACHE = {}


@dataclass(frozen=True)
class PolicyConfig:
    """Configure one experiment-selection policy.

    All five application problems report a scalar quality that the policy seeks
    to maximize. Ordinarily the policy first evaluates the task's supplied
    initial design; a random embedding instead creates its own charged latent
    initial design for each projection. After initialization, ``strategy``
    determines how new experiments are selected. Fields that configure Bayesian
    optimization have no effect on random selection, although every field is
    still validated when this object is constructed.

    Attributes:
        strategy: ``"bo"`` fits a surrogate to completed observations and
            maximizes an acquisition utility. ``"random"`` samples legal
            points after the supplied initial design. Replication still applies
            to random selection; model, acquisition, acquisition-optimizer,
            and hyperparameter fields do not affect its proposals.
        surrogate: Model fitted to observed qualities for Bayesian optimization:
            ``"gp"`` for an exact Gaussian process or ``"forest"`` for the
            package's bootstrap regression forest. A forest requires
            ``warp="none"`` and ``hyper_method="fixed"``; it cannot use the
            gradient-based ``"lbfgs"`` or ``"adam"`` acquisition optimizers.
        kernel: Objective-GP covariance kernel: ``"rbf"``, ``"matern12"``,
            ``"matern32"``, or ``"matern52"``. Coordinates are represented in
            normalized or encoded feature space. This setting does not affect
            a forest or the separate fixed-kernel models for constraints/costs.
        warp: Optional coordinate-wise objective-GP input warp:
            ``"none"``, ``"kumaraswamy"``, or ``"beta"``. Warps are available
            only with the GP surrogate.
        mixed_encoding: Representation for a mixed search space. ``"activity"``
            uses the package's activity-aware conditional encoding; ``"onehot"``
            one-hot encodes categories and scales other coordinates. It matters
            only for mixed-domain tasks.
        acquisition: Utility maximized for a Bayesian-optimization proposal:
            expected improvement (``"ei"``), probability of improvement
            (``"pi"``), upper confidence bound (``"ucb"``), a joint finite-set
            Thompson draw (``"ts"``), posterior mean (``"mean"``), latent
            variance (``"variance"``), or the 0.95 posterior quantile
            (``"quantile"``). Thompson sampling requires finite candidates.
            Feasibility or cost weighting requires ``"ei"`` or ``"pi"``.
        optimizer: Acquisition-maximization method: ``"candidates"``,
            ``"lbfgs"``, ``"adam"``, ``"pattern"``, ``"cma_es"``, ``"direct"``,
            ``"adaptive_grid"``, or ``"soo"``. Every BO step first evaluates
            the finite candidate pool. Other methods may additionally search
            or refine a continuous normalized box. Mixed domains require
            ``"candidates"``. Forests cannot use ``"lbfgs"`` or ``"adam"``.
        hyper_method: Objective-GP hyperparameter treatment: ``"fixed"`` keeps
            current values, ``"map"`` maximizes log marginal likelihood plus a
            fixed Gaussian log-prior on unconstrained coordinates, and ``"ml"``
            maximizes log marginal likelihood without that prior. Forests
            require ``"fixed"``.
        hyper_steps: Positive numerical-iteration allowance for each MAP or ML
            GP hyperparameter fit. It has no effect for fixed hyperparameters.
        refit_every: Positive number of newly recorded observations between
            attempted GP hyperparameter refits. The first surrogate fit also
            fits hyperparameters unless ``hyper_method="fixed"``. Multiple
            embeddings track this schedule independently.
        hyper_optimizer: Numerical optimizer, ``"adam"`` or ``"lbfgs"``, used
            for MAP/ML hyperparameter fitting. It has no effect for fixed
            hyperparameters or a forest.
        lengthscale: Finite positive initial GP kernel length scale in encoded
            feature coordinates. Continuous and latent coordinates are
            normalized. One value initializes all feature dimensions and may
            change during hyperparameter fitting.
        noise: Finite positive initial homoscedastic GP observation-noise
            standard deviation after quality values are standardized. With
            ``known_noise=True``, the GP instead receives each observation's
            reported variance of the replicate average. Forests do not accept
            known-noise fitting or use this GP noise parameter.
        beta: Finite nonnegative multiplier on latent posterior standard
            deviation for ``acquisition="ucb"``. It is dimensionless and has no
            effect on other acquisitions.
        xi: Finite nonnegative improvement margin for ``"ei"`` and ``"pi"`` in
            the task's original quality units. It is converted to standardized
            units internally and has no effect on other acquisitions.
        candidates: Positive size of the finite candidate pool screened at each
            BO step. It is split between global and local points and also
            supplies starting points to relevant acquisition optimizers.
            ``optimizer_restarts`` cannot exceed it.
        local_fraction: Fraction in ``[0, 1)`` of candidates drawn near the
            current posterior-best eligible visited point. The remainder are
            global, so the strict upper bound preserves global candidates.
        local_scale: Finite positive standard deviation of local perturbations,
            measured as a fraction of physical mixed-domain ranges or directly
            in normalized continuous/latent coordinates.
        optimizer_steps: Positive iteration or partition budget for a
            non-candidate acquisition optimizer. It has no effect when
            ``optimizer="candidates"``.
        optimizer_restarts: Positive multistart count for optimizers that use
            restarts, including L-BFGS, Adam, and pattern search. Some other
            methods ignore it. It cannot exceed ``candidates``.
        known_noise: Whether the objective GP uses the calibrated variance
            reported for each averaged observation. Only a task whose public
            specification reports variances may enable it; among the supplied
            tasks, that is ``production``. The value is variance of the average,
            not variance of one physical run. This requires a GP surrogate.
        incumbent: Rule used at visited points to define the incumbent for
            improvement acquisitions. ``"observed"`` uses standardized noisy
            observations; ``"posterior"`` uses latent surrogate means at those
            same points. Feasible visited points are preferred when present.
        recommendation: Rule for selecting the final recommendation from
            already evaluated physical points. ``"observed"`` compares measured
            quality; ``"posterior"`` compares surrogate means. Random search
            falls back to observed quality. Constrained tasks always restrict
            recommendations to observed feasible points when any exist.
        feasibility: Whether EI/PI accounts for predicted probability that a
            measured constraint is at most zero. A separate GP models constraint
            observations. Before a feasible observation exists, selection uses
            feasibility probability alone. Only tasks declaring a measured
            constraint may enable it; among the supplied tasks, that is
            ``materials``.
        cost_aware: Whether EI/PI is weighted using a separate GP fitted to the
            logarithm of observed per-replicate evaluation cost. Only tasks
            declaring variable laboratory-credit costs may enable it; among the
            supplied tasks, that is ``materials``.
        cost_mode: Weighting used when ``cost_aware=True``:
            ``"divide_expected_cost"``, ``"expected_inverse_cost"``, or
            ``"median_cost"``. These differ when predicted log-cost has nonzero
            variance. It has no effect when cost awareness is disabled.
        n_trees: Positive number of bootstrap trees in a forest surrogate. It
            has no effect on a GP surrogate.
        tree_depth: Positive maximum depth of each forest tree. It has no effect
            on a GP surrogate.
        min_leaf: Positive minimum bootstrap-sample count required on both sides
            of a candidate forest split. It has no effect on a GP surrogate.
        embedding_dimension: Random-embedding search dimension. Zero disables
            embedding and searches the physical space. A positive value cannot
            exceed the physical dimension and is allowed only for a continuous
            box task, not a mixed domain.
        embeddings: Positive number of independent random projections and
            surrogate models. Values above one require a positive embedding
            dimension. Models are visited round-robin and see only observations
            from their own projection, but all share one episode budget. Each
            projection also performs its own charged initial design.
        embedding_bound: Finite positive half-width applied to normalized latent
            coordinates before projection. It matters only when embedding is
            enabled.
        replicates: Positive number of independent physical runs averaged into
            each post-initialization observation. Initial-design points always
            use one run. Replicates are charged individually, and a reported
            noise variance is the variance of their average.
        allow_repeats: Whether BO proposals may repeat a physical point already
            evaluated by the same surrogate or embedding. When false, duplicate
            candidates and continuous refinements are rejected. Random search
            does not perform this duplicate check, and distinct embeddings may
            still decode to the same physical point.
    """

    strategy: str = "bo"                 # bo | random
    surrogate: str = "gp"                # gp | forest
    kernel: str = "matern52"
    warp: str = "none"                   # none | kumaraswamy | beta
    mixed_encoding: str = "activity"     # activity | onehot
    acquisition: str = "ei"
    optimizer: str = "candidates"
    hyper_method: str = "map"            # fixed | map | ml
    hyper_steps: int = 45
    refit_every: int = 6
    hyper_optimizer: str = "adam"
    lengthscale: float = .30
    noise: float = .04                   # GP noise SD after response standardization
    beta: float = 2.0                    # UCB latent-standard-deviation multiplier
    xi: float = .01                      # EI/PI margin in original quality units
    candidates: int = 512
    local_fraction: float = .5
    local_scale: float = .12             # local perturbation SD as range fraction
    optimizer_steps: int = 35
    optimizer_restarts: int = 4
    known_noise: bool = False
    incumbent: str = "observed"          # observed | posterior
    recommendation: str = "posterior"    # observed | posterior; at visited points
    feasibility: bool = False
    cost_aware: bool = False
    cost_mode: str = "divide_expected_cost"
    n_trees: int = 32
    tree_depth: int = 5
    min_leaf: int = 2
    embedding_dimension: int = 0
    embeddings: int = 1                  # independent, round-robin; one total budget
    embedding_bound: float = .75
    replicates: int = 1
    allow_repeats: bool = False

    def __post_init__(self):
        for field in ("known_noise", "feasibility", "cost_aware", "allow_repeats"):
            if not isinstance(getattr(self,field),bool):
                raise ValueError(f"{field} must be a boolean")
        choices = {
            "strategy": ("bo", "random"), "surrogate": ("gp", "forest"),
            "kernel": ("rbf", "matern12", "matern32", "matern52"),
            "warp": ("none", "kumaraswamy", "beta"),
            "mixed_encoding": ("activity", "onehot"),
            "acquisition": ("ei", "pi", "ucb", "ts", "mean", "variance", "quantile"),
            "optimizer": ("candidates", "lbfgs", "adam", "pattern", "cma_es", "direct", "adaptive_grid", "soo"),
            "hyper_method": ("fixed", "map", "ml"), "hyper_optimizer": ("adam", "lbfgs"),
            "incumbent": ("observed", "posterior"), "recommendation": ("observed", "posterior"),
            "cost_mode": ("divide_expected_cost", "expected_inverse_cost", "median_cost"),
        }
        for field, values in choices.items():
            if getattr(self, field) not in values:
                raise ValueError(f"{field} must be one of {values}")
        for field in ("hyper_steps", "refit_every", "candidates", "optimizer_steps", "optimizer_restarts",
                      "n_trees", "tree_depth", "min_leaf", "embeddings", "replicates"):
            v = getattr(self, field)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{field} must be a positive integer")
        if isinstance(self.embedding_dimension, bool) or not isinstance(self.embedding_dimension, int) or self.embedding_dimension < 0:
            raise ValueError("embedding_dimension must be a nonnegative integer")
        for field in ("lengthscale", "noise", "embedding_bound", "local_scale"):
            if not math.isfinite(getattr(self, field)) or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if not 0 <= self.local_fraction < 1:
            raise ValueError("local_fraction must be in [0,1)")
        if not math.isfinite(self.beta) or self.beta < 0 or not math.isfinite(self.xi) or self.xi < 0:
            raise ValueError("beta and xi must be finite and nonnegative")
        if self.optimizer_restarts > self.candidates:
            raise ValueError("optimizer_restarts exceeds candidate count")
        if self.surrogate == "forest" and (self.warp != "none" or self.hyper_method != "fixed"):
            raise ValueError("Forest requires warp='none' and hyper_method='fixed'")
        if self.surrogate == "forest" and self.known_noise:
            raise ValueError("Known observation variances require a GP surrogate")
        if self.surrogate == "forest" and self.optimizer in ("lbfgs", "adam"):
            raise ValueError("Forest acquisitions require a derivative-free optimizer")
        if self.acquisition == "ts" and self.optimizer != "candidates":
                raise ValueError("Thompson sampling is a correlated finite-candidate draw; use optimizer='candidates'")
        if (self.feasibility or self.cost_aware) and self.acquisition not in ("ei", "pi"):
            raise ValueError("Use a nonnegative improvement utility (ei/pi) for these weighting adapters")
        if not self.embedding_dimension and self.embeddings != 1:
            raise ValueError("Multiple embeddings require a positive embedding_dimension")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "PolicyConfig":
        # Dataclass construction rejects misspelled/unrecognized options.
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


def _standardize(y, mask):
    n = jnp.maximum(mask.sum(), 1)
    offset = jnp.sum(jnp.where(mask, y, 0)) / n
    variance = jnp.sum(jnp.where(mask, (y-offset)**2, 0)) / n
    scale = jnp.where(variance > 1e-10, jnp.sqrt(variance), 1.)
    return jnp.where(mask, (y-offset)/scale, 0.), offset, scale


def _onehot(space, raw):
    # Legal canonical values and one-hot categories; no bogus ordinal category
    # distance, even in the representation ablation.
    raw = space.canonicalize(raw)
    out = []
    for i, kind in enumerate(space.kinds):
        if kind == "categorical":
            out.append(jax.nn.one_hot(raw[:, i].astype(int), space.categories[i]))
        else:
            out.append(((raw[:, i]-space.lower[i])/(space.upper[i]-space.lower[i]))[:, None])
    return jnp.concatenate(out, axis=1)


class ConfiguredPolicy:
    """Public ask/tell policy with interchangeable components and fixed buffers.

    Call ask(), perform the charged experiment, then tell(). Initial designs and
    all random embeddings consume the SAME experiment budget. For embeddings,
    each GP sees only observations actually made through that embedding; ambient
    initial observations are NOT falsely assigned a pseudoinverse coordinate.
    """
    def __init__(self, spec: TaskSpec, config: PolicyConfig, *, key, initial_points=None):
        self.spec, self.config = spec, config
        self.key, kd, ke = jax.random.split(key, 3)
        mixed = isinstance(spec.space, MixedSpace)
        if mixed and config.optimizer != "candidates":
            raise ValueError("Mixed domains use optimizer='candidates' to preserve legality")
        if mixed and config.embedding_dimension:
            raise ValueError("Random embeddings require a continuous Box")
        if config.embedding_dimension > spec.dimension:
            raise ValueError("Embedding dimension exceeds the physical dimension")
        if config.known_noise and not spec.reports_variance:
            raise ValueError("This task does not report instrument noise variances")
        if config.feasibility and not spec.constraint_name:
            raise ValueError("This task has no measured constraint")
        if config.cost_aware and spec.budget_unit != "laboratory_credits":
            raise ValueError("This task has no variable evaluation costs")
        self.mixed = mixed
        self.search_dim = config.embedding_dimension or spec.dimension
        self.groups = config.embeddings
        self.matrices = (jax.random.normal(ke,(self.groups,spec.dimension,self.search_dim))
                         if config.embedding_dimension else None)
        n0 = min(spec.n_initial, spec.capacity)
        if config.embedding_dimension:
            self.initial = [initial_design(jax.random.fold_in(kd,g),n0,self.search_dim,"latin")
                            for g in range(self.groups)]
        else:
            initial_points = spec.space.sample(kd,n0) if initial_points is None else initial_points[:n0]
            if initial_points.shape != (n0,spec.dimension):
                raise ValueError("Public initial design has incorrect shape")
            if spec.safe_start is not None:
                initial_points = initial_points.at[0].set(jnp.asarray(spec.safe_start))
            self.initial = [self._to_search(initial_points)]
        self.n_initial = n0
        self.feature_dim = self._features(self.initial[0][:1]).shape[1]
        cap = spec.capacity
        self.buffers = [dict(z=jnp.zeros((cap,self.search_dim)), raw=jnp.zeros((cap,spec.dimension)),
                             x=jnp.zeros((cap,self.feature_dim)), y=jnp.zeros(cap),
                             variance=jnp.ones(cap), constraint=-jnp.ones(cap),
                             log_cost=jnp.zeros(cap), mask=jnp.zeros(cap,dtype=bool))
                        for _ in range(self.groups)]
        self.counts = [0] * self.groups
        self.fits = [None] * self.groups
        self.last_hyper = [-10**6] * self.groups
        self.observations: list[Observation] = []
        self._pending = None
        self._next_group = 0
        self._records: list[tuple[int,int]] = []
        base = getattr(kernels, config.kernel)
        params = default_params(self.feature_dim, lengthscale=config.lengthscale, noise=config.noise)
        if config.warp != "none":
            warp = kumaraswamy_warp if config.warp == "kumaraswamy" else beta_warp
            base = kernels.warped(base, warp)
            params = params._replace(kernel={"base": params.kernel, "warp": {
                "log_a": jnp.zeros(self.feature_dim), "log_b": jnp.zeros(self.feature_dim)}})
        self.model = (ExactGP(kernel=base) if config.surrogate == "gp" else
                      RandomForest(n_trees=config.n_trees, max_depth=config.tree_depth,
                                   min_leaf=config.min_leaf))
        self.params = [params] * self.groups
        self.aux_model = ExactGP(kernel=kernels.matern52)
        self._build_compiled_functions(params)

    def _to_search(self, raw):
        if self.mixed:
            return self.spec.space.canonicalize(raw)
        return self.spec.space.normalize(raw)

    def _features(self, z):
        if not self.mixed:
            return z
        if self.config.mixed_encoding == "activity":
            return self.spec.space.encode(self.spec.space.canonicalize(z))
        return _onehot(self.spec.space, z)

    def _decode(self, z, group):
        if self.config.embedding_dimension:
            latent = (2*z-1)*self.config.embedding_bound
            projected = jnp.clip(latent @ self.matrices[group].T, -1, 1)
            return self.spec.space.denormalize((projected+1)/2)
        if self.mixed:
            return self.spec.space.canonicalize(jnp.atleast_2d(z)).reshape(z.shape)
        return self.spec.space.denormalize(z)

    def _build_compiled_functions(self, initial_params):
        cache_key = (self.config, self.feature_dim, self.search_dim, bool(jax.config.jax_enable_x64))
        if cache_key in _COMPILED_CACHE:
            for name, fun in _COMPILED_CACHE[cache_key].items():
                setattr(self, name, fun)
            return
        model, aux, cfg = self.model, self.aux_model, self.config
        search_dimension = self.search_dim
        flat, unravel = ravel_pytree(initial_params)
        # Fixed prior center across BO iterations, not the latest estimate.
        prior_scale = jnp.full_like(flat, 1.0)
        prior = gaussian_log_prior(flat, prior_scale)
        low_tree = jax.tree_util.tree_map(lambda x: jnp.full_like(x,-5.), initial_params)
        high_tree = jax.tree_util.tree_map(lambda x: jnp.full_like(x,3.), initial_params)
        # More conservative warp bounds (shapes .2..5).
        if cfg.warp != "none":
            low_tree.kernel["warp"]["log_a"] = jnp.full(self.feature_dim,-1.6)
            low_tree.kernel["warp"]["log_b"] = jnp.full(self.feature_dim,-1.6)
            high_tree.kernel["warp"]["log_a"] = jnp.full(self.feature_dim,1.6)
            high_tree.kernel["warp"]["log_b"] = jnp.full(self.feature_dim,1.6)
        low, _ = ravel_pytree(low_tree)
        high, _ = ravel_pytree(high_tree)

        def fit(x,y,variance,mask,p,key,refit):
            ys, offset, scale = _standardize(y,mask)
            noise = variance/scale**2 if cfg.known_noise else None
            if cfg.surrogate == "forest":
                state = model.fit(x,ys,key=key,mask=mask)
                return state,p,offset,scale
            if cfg.hyper_method != "fixed":
                def update(p):
                    result = fit_hyperparameters(model,x,ys,p,method=cfg.hyper_method,
                        steps=cfg.hyper_steps,key=key,log_prior=prior,lower=low,upper=high,
                        optimizer=cfg.hyper_optimizer,noise_variance=noise,mask=mask)
                    return result.params
                p = jax.lax.cond(refit,update,lambda p:p,p)
            state = model.fit(x,ys,p,noise_variance=noise,mask=mask)
            return state,p,offset,scale
        self._fit = jax.jit(fit)
        self._predict = jax.jit(model.predict)
        # Constraint and log-cost models have their own response units.
        aux_params = default_params(self.feature_dim,lengthscale=.40,noise=.002)
        def fit_aux(x,y,mask):
            ys,offset,scale = _standardize(y,mask)
            state = aux.fit(x,ys,aux_params,mask=mask)
            return state,offset,scale
        self._fit_aux = jax.jit(fit_aux)

        def values(state,z,best,scale,key,gs,go,gscale,cs,co,cscale,has_feasible):
            if cfg.acquisition == "ts":
                # One joint draw; never independent pointwise predictive samples.
                value = model.posterior(state,z).sample(key,1)[0]
                return value
            base = make_acquisition(cfg.acquisition,model,state,best,xi=cfg.xi/scale,beta=cfg.beta)
            if cfg.feasibility:
                def pof(x):
                    pred = aux.predict(gs,x)
                    return gaussian_feasibility(pred.mean,pred.variance,upper=-go/gscale)
                # has_feasible is a traced scalar; avoid a Python truth-value test.
                old = base
                base = lambda x: jnp.where(has_feasible,old(x)*pof(x),pof(x))
            if cfg.cost_aware:
                def log_cost(x):
                    pred = aux.predict(cs,x)
                    return NormalMarginal(co+cscale*pred.mean,cscale**2*pred.variance)
                base = cost_sensitive(base,log_cost,mode=cfg.cost_mode)
            return base(z)
        self._values = jax.jit(values)
        if cfg.optimizer != "candidates":
            optimizer = AcquisitionOptimizer(cfg.optimizer,n_initial=cfg.candidates,
                                            restarts=cfg.optimizer_restarts,steps=cfg.optimizer_steps)
            def maximize(state,best,scale,key,gs,go,gscale,cs,co,cscale,has_feasible):
                return optimizer(key,lambda z:values(state,z[None,:],best,scale,key,gs,go,gscale,
                                                     cs,co,cscale,has_feasible)[0],
                                 jnp.zeros(search_dimension),jnp.ones(search_dimension)).x
            self._maximize = jax.jit(maximize)
        names = ("_fit", "_predict", "_fit_aux", "_values")
        if cfg.optimizer != "candidates":
            names += ("_maximize",)
        # Bounded cache retains no observations or simulator state. Function
        # closures use only static model configuration and fixed prior constants.
        if len(_COMPILED_CACHE) >= 32:
            _COMPILED_CACHE.pop(next(iter(_COMPILED_CACHE)))
        _COMPILED_CACHE[cache_key] = {name:getattr(self,name) for name in names}

    def _ensure_fit(self, group):
        if self.fits[group] is not None:
            return self.fits[group]
        if not self.counts[group]:
            raise ValueError("Cannot fit an empty group")
        b = self.buffers[group]
        refit = self.counts[group]-self.last_hyper[group] >= self.config.refit_every
        self.key,k = jax.random.split(self.key)
        state,p,offset,scale = self._fit(b["x"],b["y"],b["variance"],b["mask"],
                                        self.params[group],k,jnp.asarray(refit))
        self.params[group] = p
        if refit:
            self.last_hyper[group] = self.counts[group]
        preds = self._predict(state,b["x"])
        if not bool(jnp.all(jnp.isfinite(preds.mean)&jnp.isfinite(preds.variance))):
            raise FloatingPointError("Nonfinite surrogate prediction; inspect noise, precision, and priors")
        feasible = b["mask"] & (b["constraint"] <= 0)
        utility = preds.mean if self.config.incumbent == "posterior" else (b["y"]-offset)/scale
        best = jnp.max(jnp.where(feasible,utility,-jnp.inf))
        has_feasible = jnp.any(feasible)
        best = jnp.where(has_feasible,best,jnp.max(jnp.where(b["mask"],utility,-jnp.inf)))
        # Unused auxiliary state reuses an objective GP state when possible.
        # It is not inspected by statically disabled branches.
        dummy = None
        gs,go,gscale = (self._fit_aux(b["x"],b["constraint"],b["mask"])
                       if self.config.feasibility else (dummy,jnp.asarray(0.),jnp.asarray(1.)))
        cs,co,cscale = (self._fit_aux(b["x"],b["log_cost"],b["mask"])
                       if self.config.cost_aware else (dummy,jnp.asarray(0.),jnp.asarray(1.)))
        out = dict(state=state,offset=offset,scale=scale,best=best,gs=gs,go=go,gscale=gscale,
                   cs=cs,co=co,cscale=cscale,has_feasible=has_feasible,prediction=preds)
        self.fits[group] = out
        return out

    def _arguments(self, fit, key):
        return (fit["best"],fit["scale"],key,fit["gs"],fit["go"],fit["gscale"],
                fit["cs"],fit["co"],fit["cscale"],fit["has_feasible"])

    def _candidates(self, key, group, center):
        cfg = self.config
        kg,kl,kc = jax.random.split(key,3)
        nlocal = int(cfg.candidates*cfg.local_fraction)
        nglobal = cfg.candidates-nlocal
        if self.mixed:
            glob = self.spec.space.sample(kg,nglobal)
            lo,hi = jnp.asarray(self.spec.space.lower),jnp.asarray(self.spec.space.upper)
            local = center+cfg.local_scale*(hi-lo)*jax.random.normal(kl,(nlocal,self.search_dim))
            for i,kind in enumerate(self.spec.space.kinds):
                if kind == "categorical":
                    local = local.at[:,i].set(jax.random.randint(jax.random.fold_in(kc,i),(nlocal,),
                                                               0,self.spec.space.categories[i]))
            local = self.spec.space.canonicalize(local)
        else:
            glob = jax.random.uniform(kg,(nglobal,self.search_dim))
            local = jnp.clip(center+cfg.local_scale*jax.random.normal(kl,(nlocal,self.search_dim)),0,1)
        return jnp.concatenate([glob,local],axis=0)

    def ask(self, remaining: float) -> Action:
        if self._pending is not None:
            raise RuntimeError("Call tell() before asking for another point")
        if remaining < self.spec.minimum_cost-1e-8:
            from .core import BudgetExhausted
            raise BudgetExhausted("No budget remains")
        group = self._next_group
        self._next_group = (group+1)%self.groups
        n = self.counts[group]
        cfg = self.config
        self.key,kc,ka = jax.random.split(self.key,3)
        if n < self.n_initial:
            z = self.initial[group][n]
        elif cfg.strategy == "random":
            z = self.spec.space.sample(kc,1)[0] if self.mixed else jax.random.uniform(kc,(self.search_dim,))
        else:
            fit = self._ensure_fit(group)
            b = self.buffers[group]
            legal = b["mask"] & (b["constraint"] <= 0)
            eligible = jnp.where(jnp.any(legal),legal,b["mask"])
            center_index = jnp.argmax(jnp.where(eligible,fit["prediction"].mean,-jnp.inf))
            points = self._candidates(kc,group,b["z"][center_index])
            args = self._arguments(fit,ka)
            values = self._values(fit["state"],self._features(points),*args)
            if not cfg.allow_repeats:
                # Compare physical points: distinct clipped latent points can map
                # to an identical controller. Padding is explicitly masked.
                physical = self._decode(points,group)
                norm = jnp.asarray(self.spec.space.upper)-jnp.asarray(self.spec.space.lower)
                dist = jnp.sum(((physical[:,None,:]-b["raw"][None,:,:])/norm)**2,axis=-1)
                duplicate = jnp.any((dist<1e-12)&b["mask"][None,:],axis=1)
                values = jnp.where(duplicate,-jnp.inf,values)
            values = jnp.where(jnp.isfinite(values),values,-jnp.inf)
            if not bool(jnp.any(jnp.isfinite(values))):
                raise FloatingPointError("No finite unvisited candidate utility; increase candidates or permit repeats")
            idx = jnp.argmax(values)
            z = points[idx]
            if cfg.optimizer != "candidates":
                refined = self._maximize(fit["state"],*args)
                refined_value = self._values(fit["state"],refined[None,:],*args)[0]
                norm = jnp.asarray(self.spec.space.upper)-jnp.asarray(self.spec.space.lower)
                dist = jnp.sum(((self._decode(refined,group)-b["raw"])/norm)**2,axis=1)
                distinct = cfg.allow_repeats | ~jnp.any((dist<1e-12)&b["mask"])
                z = jnp.where(jnp.isfinite(refined_value)&(refined_value>values[idx])&distinct,refined,z)
        x = self._decode(z,group)
        # Initialization always costs one run per design point. Replication is
        # an explicit post-initialization decision, not extra free evidence.
        r = 1 if n < self.n_initial else cfg.replicates
        if self.spec.budget_unit != "laboratory_credits":
            r = min(r,int(math.floor(remaining+1e-8)))
        self._pending = (group,z,x,r)
        return Action(x,r)

    def tell(self, obs: Observation) -> None:
        if self._pending is None:
            raise RuntimeError("tell() requires a pending ask(); use a custom policy for external warm starts")
        group,z,x,r = self._pending
        if obs.replicates != r or not bool(jnp.allclose(obs.x,x,rtol=0,atol=1e-10)):
            raise ValueError("Observation does not match the pending experiment")
        if not math.isfinite(obs.y):
            raise ValueError("Nonfinite quality observation")
        if not math.isfinite(obs.cost) or obs.cost <= 0:
            raise ValueError("Observation cost must be finite and positive")
        if obs.constraint is not None and not math.isfinite(obs.constraint):
            raise ValueError("Constraint observation must be finite")
        if self.config.known_noise and (obs.noise_variance is None or not math.isfinite(obs.noise_variance) or obs.noise_variance < 0):
            raise ValueError("A calibrated nonnegative variance of the average is required")
        n = self.counts[group]
        if n >= self.spec.capacity:
            raise RuntimeError("Policy observation capacity exceeded")
        b = self.buffers[group]
        values = dict(z=z,raw=obs.x,x=self._features(z[None,:])[0],y=obs.y,
                      variance=obs.noise_variance if obs.noise_variance is not None else 1.,
                      constraint=obs.constraint if obs.constraint is not None else -1.,
                      log_cost=math.log(obs.cost/obs.replicates),mask=True)
        for name,value in values.items():
            b[name] = b[name].at[n].set(value)
        self.counts[group] += 1
        self.fits[group] = None
        self._pending = None
        self.observations.append(obs)
        self._records.append((group,n))

    @contextmanager
    def _read_only(self):
        saved = (self.key, self.params.copy(), self.fits.copy(), self.last_hyper.copy())
        try:
            yield
        finally:
            self.key, self.params, self.fits, self.last_hyper = saved

    def recommend(self) -> jax.Array:
        # Logging a recommendation must not change the optimization trajectory,
        # hyperparameter-refit schedule, or future random-number stream.
        with self._read_only():
            return self._recommend()

    def _recommend(self) -> jax.Array:
        if not self.observations:
            raise ValueError("No experiments have completed")
        best_x,best_value = None,-math.inf
        cfg = self.config
        for group,n in enumerate(self.counts):
            if not n:
                continue
            b = self.buffers[group]
            if cfg.recommendation == "posterior" and cfg.strategy != "random":
                fit = self._ensure_fit(group)
                score = fit["offset"]+fit["scale"]*fit["prediction"].mean
            else:
                score = b["y"]
            feasible = b["mask"] & (b["constraint"] <= 0)
            if not bool(jnp.any(feasible)):
                continue
            i = int(jnp.argmax(jnp.where(feasible,score,-jnp.inf)))
            if float(score[i]) > best_value:
                best_value,best_x = float(score[i]),b["raw"][i]
        if best_x is None:
            # No feasible recommendation: return least observed violation;
            # the grading harness will mark it infeasible, not fabricate success.
            return min(self.observations,key=lambda o:o.constraint if o.constraint is not None else -1).x
        return best_x

    def predict(self, raw, *, group=0) -> NormalMarginal:
        """Posterior mean/variance in original output units, for student diagnostics.

        For an embedding supply latent normalized search coordinates instead of
        physical points; arbitrary ambient points have no unique latent inverse.
        """
        with self._read_only():
            fit = self._ensure_fit(group)
            z = jnp.atleast_2d(raw) if self.config.embedding_dimension else self._to_search(jnp.atleast_2d(raw))
            pred = self._predict(fit["state"],self._features(z))
            return NormalMarginal(fit["offset"]+fit["scale"]*pred.mean,fit["scale"]**2*pred.variance)

    def diagnostics(self) -> dict:
        return {"counts_by_embedding":list(self.counts),
                "fitted_parameters":[jax.tree_util.tree_map(lambda a:a.tolist(),p) for p in self.params],
                "pending":self._pending is not None}

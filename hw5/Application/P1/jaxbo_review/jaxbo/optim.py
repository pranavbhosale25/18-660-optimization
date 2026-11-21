"""Native-JAX optimization. Every public maximize_* routine maximizes a scalar.

Host Python selects static algorithms and dimensions; numerical iteration uses
lax loops. No scipy.optimize, jaxopt, optax, or callbacks are used.
"""
from __future__ import annotations
from typing import NamedTuple, Callable
from dataclasses import dataclass
import math
import jax
import jax.numpy as jnp
from .linalg import safe_sqrt, symmetrize


class OptimizeResult(NamedTuple):
    x: jax.Array
    value: jax.Array
    iterations: jax.Array
    converged: jax.Array


def minimize_lbfgs(fun: Callable, x0, lower=None, upper=None, *, steps=100, history=8,
                   learning_rate=1.0, tolerance=1e-6, line_search_steps=24):
    """Projected limited-memory BFGS with Armijo backtracking.

    This is NOT the full L-BFGS-B active-set algorithm. Projection handles box
    bounds; histories with nonpositive curvature are discarded.
    """
    if history < 1 or steps < 0:
        raise ValueError("history must be positive and steps nonnegative")
    x0 = jnp.asarray(x0, dtype=float)
    lo = jnp.full_like(x0,-jnp.inf) if lower is None else jnp.broadcast_to(lower,x0.shape)
    hi = jnp.full_like(x0,jnp.inf) if upper is None else jnp.broadcast_to(upper,x0.shape)
    x0 = jnp.clip(x0,lo,hi)
    vg = jax.value_and_grad(fun)
    value,grad = vg(x0)
    zeros = jnp.zeros((history,x0.size),dtype=x0.dtype)
    # x, value, gradient, s history, y history, inverse curvature, steps, converged
    state = (x0,value,grad,zeros,zeros,jnp.zeros(history,dtype=x0.dtype),jnp.array(0),jnp.array(False))
    def body(_, state):
        x,value,g,ss,ys,rhos,niter,done = state
        projected_gradient = x-jnp.clip(x-g,lo,hi)
        done = done | (jnp.max(jnp.abs(projected_gradient)) <= tolerance)
        def advance(_):
            alphas = jnp.zeros(history,dtype=x.dtype)
            def backward(k, carry):
                q,aa = carry
                idx = history-1-k
                a = rhos[idx]*jnp.dot(ss[idx],q)
                return q-a*ys[idx],aa.at[idx].set(a)
            q,alphas = jax.lax.fori_loop(0,history,backward,(g,alphas))
            denom = jnp.dot(ys[-1],ys[-1])
            scale = jnp.where(rhos[-1]>0,jnp.dot(ss[-1],ys[-1])/jnp.maximum(denom,1e-30),1.0)
            def forward(idx,r):
                beta = rhos[idx]*jnp.dot(ys[idx],r)
                return r+ss[idx]*(alphas[idx]-beta)
            direction = -jax.lax.fori_loop(0,history,forward,q*scale)
            direction = jnp.clip(x+direction,lo,hi)-x
            direction = jnp.where(jnp.dot(direction,g)<-1e-20,direction,-projected_gradient)
            def backtrack(_, carry):
                rate,accepted,chosen = carry
                trial = jnp.clip(x+rate*direction,lo,hi)
                v = fun(trial)
                ok = jnp.isfinite(v) & (v <= value+1e-4*jnp.dot(g,trial-x))
                chosen = jnp.where((~accepted)&ok,trial,chosen)
                return jnp.where(accepted|ok,rate,0.5*rate),accepted|ok,chosen
            _,accepted,newx = jax.lax.fori_loop(0,line_search_steps,backtrack,
                (jnp.asarray(learning_rate,x.dtype),jnp.array(False),x))
            newvalue,newg = vg(newx)
            s,y = newx-x,newg-g
            sy = jnp.dot(s,y)
            valid = accepted & (sy > 1e-12*jnp.linalg.norm(s)*jnp.linalg.norm(y)) & (sy>1e-30)
            newss = jnp.concatenate([ss[1:],s[None,:]],axis=0)
            newys = jnp.concatenate([ys[1:],y[None,:]],axis=0)
            newrhos = jnp.concatenate([rhos[1:],jnp.array([1/jnp.maximum(sy,1e-30)])])
            return (newx,newvalue,newg,jnp.where(valid,newss,ss),jnp.where(valid,newys,ys),
                    jnp.where(valid,newrhos,rhos),niter+1,~accepted)
        return jax.lax.cond(done,lambda _: (x,value,g,ss,ys,rhos,niter,done),advance,operand=None)
    x,value,g,ss,ys,rhos,niter,done = jax.lax.fori_loop(0,steps,body,state)
    converged = jnp.max(jnp.abs(x-jnp.clip(x-g,lo,hi)))<=tolerance
    return OptimizeResult(x,value,niter,converged)


def minimize_adam(fun,x0,lower=None,upper=None,*,steps=200,learning_rate=0.03):
    x0=jnp.asarray(x0,dtype=float)
    lo=-jnp.inf if lower is None else lower
    hi=jnp.inf if upper is None else upper
    x0=jnp.clip(x0,lo,hi)
    vg=jax.value_and_grad(fun)
    state=(x0,jnp.zeros_like(x0),jnp.zeros_like(x0),x0,fun(x0))
    def body(i,state):
        x,m,v,best,bestval=state
        _,g=vg(x)
        g=jnp.nan_to_num(g,nan=0,posinf=0,neginf=0)
        m=.9*m+.1*g; v=.999*v+.001*g*g
        xx=jnp.clip(x-learning_rate*(m/(1-.9**(i+1)))/(jnp.sqrt(v/(1-.999**(i+1)))+1e-8),lo,hi)
        val=fun(xx); improve=jnp.isfinite(val)&(val<bestval)
        return xx,m,v,jnp.where(improve,xx,best),jnp.where(improve,val,bestval)
    _,_,_,best,value=jax.lax.fori_loop(0,steps,body,state)
    return OptimizeResult(best,value,jnp.array(steps),jnp.array(False))


def maximize_discrete(fun,points):
    """Exact maximization on a specified FINITE candidate set."""
    values=jax.vmap(fun)(points)
    values=jnp.where(jnp.isfinite(values),values,-jnp.inf)
    idx=jnp.argmax(values)
    return OptimizeResult(points[idx],values[idx],jnp.array(points.shape[0]),jnp.array(True))


def maximize_random(key,fun,lower,upper,*,n=512):
    x=jax.random.uniform(key,(n,lower.size),dtype=lower.dtype,minval=lower,maxval=upper)
    return maximize_discrete(fun,x)


def maximize_multistart(key,fun,lower,upper,*,n_initial=256,restarts=8,steps=80,method="lbfgs"):
    if not 1<=restarts<=n_initial:
        raise ValueError("Require 1 <= restarts <= n_initial")
    points=jax.random.uniform(key,(n_initial,lower.size),dtype=lower.dtype,minval=lower,maxval=upper)
    values=jax.vmap(fun)(points)
    _,idx=jax.lax.top_k(jnp.nan_to_num(values,nan=-jnp.inf),restarts)
    minimize={"lbfgs":minimize_lbfgs,"adam":minimize_adam}.get(method)
    if minimize is None:
        raise ValueError("method must be 'lbfgs' or 'adam'")
    results=jax.vmap(lambda x:minimize(lambda z:-fun(z),x,lower,upper,steps=steps))(points[idx])
    best=jnp.argmin(results.value)
    return OptimizeResult(results.x[best],-results.value[best],results.iterations[best],results.converged[best])


def maximize_pattern_search(key,fun,lower,upper,*,restarts=8,steps=80,initial_step=0.2):
    """Multistart coordinate/pattern local search; useful for nondifferentiable models."""
    d=lower.size
    starts=jax.random.uniform(key,(restarts,d),dtype=lower.dtype,minval=lower,maxval=upper)
    directions=jnp.concatenate([jnp.eye(d,dtype=lower.dtype),-jnp.eye(d,dtype=lower.dtype)])
    def one(x):
        state=(x,fun(x),jnp.asarray(initial_step,dtype=x.dtype))
        def body(_,state):
            x,value,step=state
            trials=jnp.clip(x+directions*step*(upper-lower),lower,upper)
            values=jax.vmap(fun)(trials)
            idx=jnp.argmax(values); better=values[idx]>value
            return jnp.where(better,trials[idx],x),jnp.maximum(value,values[idx]),jnp.where(better,step,step*.5)
        x,v,step=jax.lax.fori_loop(0,steps,body,state)
        return OptimizeResult(x,v,jnp.array(steps),step<1e-6)
    out=jax.vmap(one)(starts); i=jnp.argmax(out.value)
    return jax.tree_util.tree_map(lambda a:a[i],out)


def maximize_cma_es(key,fun,lower,upper,*,steps=100,population=None,initial_sigma=0.25):
    """Positive-weight full-covariance CMA-ES, with clipping repair at box bounds.

    Bound repair is an explicit implementation choice, not a claim to reproduce
    any particular CMA-ES library's boundary handling.
    """
    d=lower.size
    lam=4+int(3*math.log(max(d,2))) if population is None else population
    if lam<4:
        raise ValueError("CMA-ES population must be at least 4")
    mu=lam//2
    w=jnp.log(mu+0.5)-jnp.log(jnp.arange(1,mu+1,dtype=lower.dtype)); w=w/w.sum()
    mueff=1/jnp.sum(w*w)
    cc=(4+mueff/d)/(d+4+2*mueff/d)
    cs=(mueff+2)/(d+mueff+5)
    c1=2/((d+1.3)**2+mueff)
    cmu=jnp.minimum(1-c1,2*(mueff-2+1/mueff)/((d+2)**2+mueff))
    damping=1+2*jnp.maximum(0,safe_sqrt((mueff-1)/(d+1))-1)+cs
    chi=math.sqrt(d)*(1-1/(4*d)+1/(21*d*d))
    mean=jnp.full((d,),.5,dtype=lower.dtype)
    x0=lower+mean*(upper-lower)
    state=(mean,jnp.asarray(initial_sigma,lower.dtype),jnp.eye(d,dtype=lower.dtype),
           jnp.zeros(d,dtype=lower.dtype),jnp.zeros(d,dtype=lower.dtype),x0,fun(x0),key)
    def body(i,state):
        mean,sigma,c,pc,ps,best,bestvalue,key=state
        key,k=jax.random.split(key)
        ev,b=jnp.linalg.eigh(symmetrize(c)); ev=jnp.maximum(ev,1e-14)
        root=b*safe_sqrt(ev)[None,:]
        invroot=(b*(1/safe_sqrt(ev))[None,:])@b.T
        z=jax.random.normal(k,(lam,d),dtype=lower.dtype)
        points=jnp.clip(mean+sigma*(z@root.T),0,1)
        physical=lower+points*(upper-lower)
        values=jax.vmap(fun)(physical)
        order=jnp.argsort(jnp.nan_to_num(values,nan=-jnp.inf))[::-1]
        selected=points[order[:mu]]
        newmean=w@selected; yw=(newmean-mean)/sigma
        ps=(1-cs)*ps+safe_sqrt(cs*(2-cs)*mueff)*(invroot@yw)
        h=(jnp.linalg.norm(ps)/safe_sqrt(1-(1-cs)**(2*(i+1)))) < (1.4+2/(d+1))*chi
        pc=(1-cc)*pc+h*safe_sqrt(cc*(2-cc)*mueff)*yw
        deviations=(selected-mean)/sigma
        c=(1-c1-cmu)*c+c1*(jnp.outer(pc,pc)+(1-h)*cc*(2-cc)*c)+cmu*(deviations.T*w)@deviations
        sigma=jnp.clip(sigma*jnp.exp((cs/damping)*(jnp.linalg.norm(ps)/chi-1)),1e-8,2.0)
        idx=order[0]; better=values[idx]>bestvalue
        return newmean,sigma,symmetrize(c),pc,ps,jnp.where(better,physical[idx],best),jnp.maximum(values[idx],bestvalue),key
    _,_,_,_,_,best,value,_=jax.lax.fori_loop(0,steps,body,state)
    return OptimizeResult(best,value,jnp.array(steps),jnp.array(False))


@dataclass(frozen=True)
class AcquisitionOptimizer:
    method: str = "lbfgs"
    n_initial: int = 256
    restarts: int = 8
    steps: int = 80

    def __call__(self,key,fun,lower,upper):
        if self.method in ("lbfgs","adam"):
            return maximize_multistart(key,fun,lower,upper,n_initial=self.n_initial,
                                      restarts=self.restarts,steps=self.steps,method=self.method)
        if self.method=="random":
            return maximize_random(key,fun,lower,upper,n=self.n_initial)
        if self.method=="pattern":
            return maximize_pattern_search(key,fun,lower,upper,restarts=self.restarts,steps=self.steps)
        if self.method=="cma_es":
            return maximize_cma_es(key,fun,lower,upper,steps=self.steps)
        if self.method in ("direct","adaptive_grid","soo"):
            from .global_optim import maximize_direct,maximize_adaptive_grid,maximize_soo
            f={"direct":maximize_direct,"adaptive_grid":maximize_adaptive_grid,"soo":maximize_soo}[self.method]
            return f(fun,lower,upper,splits=self.steps)
        raise ValueError(f"Unknown acquisition optimizer: {self.method}")

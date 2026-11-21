"""Gradient-free native-JAX partition optimizers.

DIRECT uses a potentially-optimal rectangle test over all current rectangles.
SOO expands at most one best leaf per depth in each sweep. Both use finite static
capacity and stop at a split budget; neither promises an exact continuous argmax.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from .optim import OptimizeResult


def maximize_direct(fun,lower,upper,*,splits=64,epsilon=1e-4):
    d=lower.size; capacity=1+2*d*splits
    centers=jnp.zeros((capacity,d),dtype=lower.dtype).at[0].set(.5)
    half=jnp.zeros_like(centers).at[0].set(.5)
    values=jnp.full(capacity,jnp.inf,dtype=lower.dtype).at[0].set(-fun((lower+upper)/2))
    active=jnp.zeros(capacity,dtype=bool).at[0].set(True)
    pending=jnp.zeros(capacity,dtype=bool)
    ids=jnp.arange(capacity)
    def potentially_optimal(half,values,active):
        radius=jnp.linalg.norm(half,axis=1)
        diff=radius[:,None]-radius[None,:]
        fv=values[:,None]-values[None,:]
        others=active[None,:]
        ratio=jnp.where(jnp.abs(diff)>1e-15,fv/jnp.where(jnp.abs(diff)>1e-15,diff,1),0)
        min_slope=jnp.max(jnp.where(others&(diff>1e-15),ratio,0),axis=1)
        max_slope=jnp.min(jnp.where(others&(diff < -1e-15),ratio,jnp.inf),axis=1)
        best=jnp.min(values)
        improvement_slope=(values-best+epsilon*jnp.abs(best))/jnp.maximum(radius,1e-30)
        min_slope=jnp.maximum(min_slope,improvement_slope)
        same_size_better=jnp.any(others&(jnp.abs(diff)<=1e-15)&(fv>1e-14),axis=1)
        return active&(~same_size_better)&(min_slope<=max_slope)
    def body(_,carry):
        centers,half,values,active,pending,count=carry
        pending=jax.lax.cond(jnp.any(pending),lambda _:pending,
                             lambda _:potentially_optimal(half,values,active),None)
        # Tie-break large rectangles first; process the saved set before recomputing.
        radius=jnp.linalg.norm(half,axis=1)
        idx=jnp.argmax(jnp.where(pending,radius,-jnp.inf))
        pending=pending.at[idx].set(False)
        center=centers[idx]; original=half[idx]; longest=jnp.max(original)
        eligible=jnp.abs(original-longest)<=1e-14
        offsets=jnp.eye(d,dtype=lower.dtype)*(2*longest/3)
        candidates=jnp.concatenate([center+offsets,center-offsets],axis=0)
        scores=-jax.vmap(fun)(lower+candidates*(upper-lower))
        order=jnp.argsort(jnp.where(eligible,jnp.minimum(scores[:d],scores[d:]),jnp.inf))
        def split_dim(k,state):
            centers,half,values,active,count,current_half=state
            dim=order[k]
            def do(s):
                centers,half,values,active,count,current_half=s
                current_half=current_half.at[dim].set(original[dim]/3)
                plus=center.at[dim].add(2*original[dim]/3)
                minus=center.at[dim].add(-2*original[dim]/3)
                centers=centers.at[count].set(plus).at[count+1].set(minus)
                half=half.at[count].set(current_half).at[count+1].set(current_half)
                values=values.at[count].set(scores[dim]).at[count+1].set(scores[d+dim])
                active=active.at[count].set(True).at[count+1].set(True)
                return centers,half,values,active,count+2,current_half
            return jax.lax.cond(eligible[dim],do,lambda s:s,state)
        centers,half,values,active,count,newhalf=jax.lax.fori_loop(0,d,split_dim,
                    (centers,half,values,active,count,original))
        half=half.at[idx].set(newhalf)
        return centers,half,values,active,pending,count
    centers,half,values,active,pending,count=jax.lax.fori_loop(0,splits,body,
                                (centers,half,values,active,pending,jnp.array(1)))
    best=jnp.argmin(values)
    return OptimizeResult(lower+centers[best]*(upper-lower),-values[best],count,jnp.array(False))


def maximize_adaptive_grid(fun,lower,upper,*,splits=64,exploration=1.0):
    """Explicit optimistic adaptive-grid heuristic; exploration is user supplied.

    Splits the longest side of the leaf with largest value + exploration*radius.
    The multiplier is NOT inferred as a valid Lipschitz bound.
    """
    d=lower.size; capacity=1+2*splits
    center=jnp.zeros((capacity,d),dtype=lower.dtype).at[0].set((lower+upper)/2)
    half=jnp.zeros_like(center).at[0].set((upper-lower)/2)
    values=jnp.full(capacity,-jnp.inf,dtype=lower.dtype).at[0].set(fun(center[0]))
    def body(i,carry):
        center,half,values=carry
        idx=jnp.argmax(values+exploration*jnp.linalg.norm(half,axis=1))
        dim=jnp.argmax(half[idx]/(upper-lower)); h=half[idx,dim]
        ch=half[idx].at[dim].set(h/3)
        plus=center[idx].at[dim].add(2*h/3); minus=center[idx].at[dim].add(-2*h/3)
        count=1+2*i
        center=center.at[count].set(plus).at[count+1].set(minus)
        half=half.at[idx].set(ch).at[count].set(ch).at[count+1].set(ch)
        values=values.at[count].set(fun(plus)).at[count+1].set(fun(minus))
        return center,half,values
    center,half,values=jax.lax.fori_loop(0,splits,body,(center,half,values))
    idx=jnp.argmax(values)
    return OptimizeResult(center[idx],values[idx],jnp.array(capacity),jnp.array(False))


def maximize_soo(fun,lower,upper,*,splits=64,max_depth=None):
    """SOO-style ternary partition, one eligible best leaf per depth per sweep.

    max_depth is explicit (default floor(sqrt(split budget))). This is a bounded
    implementation of the described SOO policy, not the proof's asymptotic schedule.
    """
    d=lower.size; capacity=1+2*splits
    hmax=max(1,int(splits**.5)) if max_depth is None else max_depth
    center=jnp.zeros((capacity,d),dtype=lower.dtype).at[0].set((lower+upper)/2)
    half=jnp.zeros_like(center).at[0].set((upper-lower)/2)
    values=jnp.full(capacity,-jnp.inf,dtype=lower.dtype).at[0].set(fun(center[0]))
    depths=jnp.full(capacity,-1,dtype=int).at[0].set(0)
    state=(center,half,values,depths,jnp.array(1),jnp.array(0),jnp.asarray(-jnp.inf,dtype=lower.dtype))
    def body(_,carry):
        center,half,values,depths,count,start_depth,record=carry
        # Search remaining levels in this sweep; when no eligible level remains,
        # start a new sweep. A fixed scan avoids a data-dependent output shape.
        def choose(carry,level):
            chosen,current_record=carry
            vals=jnp.where(depths==level,values,-jnp.inf)
            idx=jnp.argmax(vals)
            eligible=(level>=start_depth)&jnp.isfinite(vals[idx])&(vals[idx]>=current_record)&(chosen<0)
            return (jnp.where(eligible,idx,chosen),jnp.where(eligible,vals[idx],current_record)),None
        (idx,newrecord),_=jax.lax.scan(choose,(jnp.array(-1),record),jnp.arange(hmax+1))
        def restart(_):
            # Best at the shallowest populated depth (first level of a fresh sweep).
            level=jnp.min(jnp.where(depths>=0,depths,hmax+1))
            i=jnp.argmax(jnp.where(depths==level,values,-jnp.inf))
            return i,values[i]
        idx,newrecord=jax.lax.cond(idx<0,restart,lambda _:(idx,newrecord),None)
        depth=depths[idx]; dim=jnp.argmax(half[idx]/(upper-lower)); h=half[idx,dim]
        newhalf=half[idx].at[dim].set(h/3)
        plus=center[idx].at[dim].add(2*h/3); minus=center[idx].at[dim].add(-2*h/3)
        center=center.at[count].set(plus).at[count+1].set(minus)
        half=half.at[idx].set(newhalf).at[count].set(newhalf).at[count+1].set(newhalf)
        values=values.at[count].set(fun(plus)).at[count+1].set(fun(minus))
        depths=depths.at[idx].set(depth+1).at[count].set(depth+1).at[count+1].set(depth+1)
        return center,half,values,depths,count+2,depth+1,newrecord
    center,half,values,depths,count,_,_=jax.lax.fori_loop(0,splits,body,state)
    idx=jnp.argmax(values)
    return OptimizeResult(center[idx],values[idx],count,jnp.array(False))

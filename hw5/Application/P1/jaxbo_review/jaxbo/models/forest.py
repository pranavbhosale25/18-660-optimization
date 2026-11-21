"""Native-JAX bootstrap CART regression forest.

Numeric/one-hot features only. This is not a port of SMAC's specialized
categorical/conditional split rules. All candidate numeric thresholds are tested
on a random feature subset at each node; maximum depth fixes allocation shapes.
"""
from __future__ import annotations
from typing import NamedTuple
from dataclasses import dataclass
import math
import jax
import jax.numpy as jnp
from ..distributions import Gaussian,NormalMarginal,Empirical,EmpiricalMarginal


class ForestState(NamedTuple):
    feature:jax.Array
    threshold:jax.Array
    value:jax.Array


@dataclass(frozen=True)
class RandomForest:
    n_trees:int=32
    max_depth:int=5
    min_leaf:int=2
    max_features:int|None=None
    uncertainty:str="gaussian"
    requires_key:bool=True

    def fit(self,x,y,params=None,*,key=None,mask=None):
        if key is None: raise ValueError("RandomForest.fit requires an explicit key")
        if self.n_trees<1 or self.max_depth<0 or self.min_leaf<1 or x.shape[0]<1:
            raise ValueError("Invalid forest size/depth/min_leaf or empty data")
        if self.uncertainty not in ("gaussian","empirical"):
            raise ValueError("uncertainty must be gaussian or empirical")
        if mask is not None:
            # Masked bootstrap uses categorical sampling over active observations.
            sampling_logits=jnp.where(mask,0.0,-jnp.inf)
        else: sampling_logits=jnp.zeros(x.shape[0])
        n,d=x.shape; n_nodes=2**(self.max_depth+1)-1; internal=2**self.max_depth-1
        mtry=max(1,int(math.sqrt(d))) if self.max_features is None else self.max_features
        if not 1<=mtry<=d: raise ValueError("max_features must be in [1,d]")
        def train_tree(key):
            key,kboot=jax.random.split(key)
            indices=jax.random.categorical(kboot,sampling_logits,shape=(n,))
            xx,yy=x[indices],y[indices]
            bootstrap_active = jnp.arange(n) < (n if mask is None else jnp.sum(mask))
            member=jnp.zeros((n_nodes,n),dtype=bool).at[0].set(bootstrap_active)
            features=jnp.full((n_nodes,),-1,dtype=int)
            thresholds=jnp.zeros(n_nodes,dtype=x.dtype)
            bootstrap_mean = jnp.sum(jnp.where(bootstrap_active,yy,0))/jnp.maximum(bootstrap_active.sum(),1)
            values=jnp.full(n_nodes,bootstrap_mean,dtype=y.dtype)
            def node(i,carry):
                member,features,thresholds,values=carry
                active=member[i]; count=active.sum()
                mean=jnp.sum(jnp.where(active,yy,0))/jnp.maximum(count,1)
                mean=jnp.where(count>0,mean,values[i])
                values=values.at[i].set(mean)
                def split(carry):
                    member,features,thresholds,values=carry
                    kk=jax.random.fold_in(key,i)
                    dims=jax.random.permutation(kk,d)[:mtry]
                    xv=xx[:,dims].T
                    order=jnp.argsort(xv,axis=1)
                    sorted_x=jnp.take_along_axis(xv,order,axis=1)
                    sorted_y=yy[order]; sorted_member=active[order]
                    cnt=jnp.cumsum(sorted_member,axis=1)[:,:-1]
                    sums=jnp.cumsum(jnp.where(sorted_member,sorted_y,0),axis=1)[:,:-1]
                    sums2=jnp.cumsum(jnp.where(sorted_member,sorted_y**2,0),axis=1)[:,:-1]
                    total=jnp.sum(jnp.where(active,yy,0)); total2=jnp.sum(jnp.where(active,yy**2,0))
                    right_count=count-cnt
                    cost=sums2-sums*sums/jnp.maximum(cnt,1)+(total2-sums2)-(total-sums)**2/jnp.maximum(right_count,1)
                    valid=(cnt>=self.min_leaf)&(right_count>=self.min_leaf)&(sorted_x[:,:-1]<sorted_x[:,1:])
                    cost=jnp.where(valid,cost,jnp.inf)
                    flat=jnp.argmin(cost); row,col=jnp.unravel_index(flat,cost.shape)
                    f=dims[row]; threshold=(sorted_x[row,col]+sorted_x[row,col+1])/2
                    parent_cost=total2-total*total/jnp.maximum(count,1)
                    good=jnp.isfinite(cost[row,col])&(cost[row,col]<parent_cost-1e-12)
                    features=features.at[i].set(jnp.where(good,f,-1)); thresholds=thresholds.at[i].set(threshold)
                    left=active&(xx[:,f]<=threshold)&good
                    right=active&(xx[:,f]>threshold)&good
                    member=member.at[2*i+1].set(left).at[2*i+2].set(right)
                    values=values.at[2*i+1].set(mean).at[2*i+2].set(mean)
                    return member,features,thresholds,values
                # n=1 has no possible split and would otherwise create empty arrays.
                if n>1:
                    return jax.lax.cond(i<internal,split,lambda c:c,(member,features,thresholds,values))
                return member,features,thresholds,values
            _,features,thresholds,values=jax.lax.fori_loop(0,n_nodes,node,(member,features,thresholds,values))
            return ForestState(features,thresholds,values)
        return jax.vmap(train_tree)(jax.random.split(key,self.n_trees))

    def leaf_ids(self,state,x):
        def tree(feature,threshold):
            idx=jnp.zeros(x.shape[0],dtype=int)
            def body(_,idx):
                f=feature[idx]; leaf=f<0
                right=x[jnp.arange(x.shape[0]),jnp.maximum(f,0)]>threshold[idx]
                return jnp.where(leaf,idx,2*idx+1+right.astype(int))
            return jax.lax.fori_loop(0,self.max_depth,body,idx)
        return jax.vmap(tree)(state.feature,state.threshold)

    def tree_values(self,state,x):
        leaves=self.leaf_ids(state,x)
        return jnp.take_along_axis(state.value,leaves,axis=1)

    def predict(self,state,x):
        values=self.tree_values(state,x)
        if self.uncertainty=="empirical": return EmpiricalMarginal(values)
        return NormalMarginal(values.mean(axis=0),values.var(axis=0))

    def posterior(self,state,x):
        empirical=Empirical(self.tree_values(state,x))
        if self.uncertainty=="empirical": return empirical
        return Gaussian(empirical.mean,empirical.covariance)


    def thompson_function(self,state,key):
        """One fixed function draw, consistent with the chosen uncertainty model.

        Empirical mode selects a tree. Gaussian mode uses one fixed Gaussian
        vector of tree coefficients, exactly matching the finite-rank empirical
        covariance of .posterior without drawing fresh noise at each input.
        Both functions are piecewise constant; use derivative-free optimization.
        """
        if self.uncertainty=="empirical":
            index=jax.random.randint(key,(),0,self.n_trees)
            return lambda x:self.tree_values(state,x)[index]
        coefficients=jax.random.normal(key,(self.n_trees,),dtype=state.value.dtype)
        def draw(x):
            values=self.tree_values(state,x)
            mean=values.mean(axis=0)
            return mean+coefficients@(values-mean)/jnp.sqrt(self.n_trees)
        return draw

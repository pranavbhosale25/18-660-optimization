"""Box designs, mixed/conditional encodings and random embeddings."""
from __future__ import annotations
from dataclasses import dataclass
import math
import jax
import jax.numpy as jnp


def _validate_bounds(lower,upper):
    if lower.ndim!=1 or upper.shape!=lower.shape or lower.size==0:
        raise ValueError("Bounds must be nonempty vectors with matching shapes")
    if not bool(jnp.all(jnp.isfinite(lower)&jnp.isfinite(upper)&(lower<upper))):
        raise ValueError("Bounds must be finite with lower < upper")


@dataclass(frozen=True)
class Box:
    lower:object
    upper:object

    def __post_init__(self):
        lo,hi=jnp.asarray(self.lower,dtype=float),jnp.asarray(self.upper,dtype=float)
        _validate_bounds(lo,hi)
        object.__setattr__(self,"lower",lo); object.__setattr__(self,"upper",hi)

    @property
    def dimension(self): return self.lower.size

    def normalize(self,x): return (x-self.lower)/(self.upper-self.lower)
    def denormalize(self,x): return self.lower+x*(self.upper-self.lower)
    def project(self,x): return jnp.clip(x,self.lower,self.upper)

    def sample(self,key,n,method="latin"):
        return self.denormalize(initial_design(key,n,self.dimension,method))


def initial_design(key,n:int,d:int,method="latin"):
    if n<1 or d<1: raise ValueError("Design dimensions must be positive")
    if method=="random": return jax.random.uniform(key,(n,d))
    if method=="latin":
        keys=jax.random.split(key,d+1)
        permutations=jax.vmap(lambda k:jax.random.permutation(k,n))(keys[:-1]).T
        return (permutations+jax.random.uniform(keys[-1],(n,d)))/n
    if method=="halton":
        primes=[]; p=2
        while len(primes)<d:
            if all(p%q for q in range(2,int(math.sqrt(p))+1)): primes.append(p)
            p+=1
        indices=jnp.arange(1,n+1,dtype=int)
        columns=[]
        for base in primes:
            digits=int(math.ceil(math.log(n+1)/math.log(base)))+1
            def step(_,carry):
                idx,weight,result=carry
                weight=weight/base; result=result+weight*(idx%base)
                return idx//base,weight,result
            _,_,result=jax.lax.fori_loop(0,digits,step,
                            (indices,jnp.ones(n,dtype=float),jnp.zeros(n,dtype=float)))
            columns.append(result)
        # Explicit random shift; reproducible given key.
        return (jnp.stack(columns,axis=1)+jax.random.uniform(key,(d,)))%1
    raise ValueError("Design must be random, latin, or halton")


def cartesian_grid(lower,upper,points_per_axis=10,max_points=1_000_000):
    d=lower.size
    if points_per_axis<2 or points_per_axis**d>max_points:
        raise ValueError("Grid too large, or fewer than two points per axis")
    axes=[jnp.linspace(lower[i],upper[i],points_per_axis) for i in range(d)]
    return jnp.stack(jnp.meshgrid(*axes,indexing="ij"),axis=-1).reshape((-1,d))


@dataclass(frozen=True)
class Condition:
    child:int
    parent:int
    values:tuple[int,...]


@dataclass(frozen=True)
class MixedSpace:
    """Numeric encoding of real/integer/categorical variables and conditional DAGs.

    Categorical variables are integer IDs, NOT ordinal numeric distances after
    encode(). Category counts are supplied in categories (0 for noncategorical).
    Conditions are ANDed; multiple allowed values in one condition are ORed.
    """
    lower:tuple[float,...]
    upper:tuple[float,...]
    kinds:tuple[str,...]
    categories:tuple[int,...]
    conditions:tuple[Condition,...]=()

    def __post_init__(self):
        d=len(self.kinds)
        if not (len(self.lower)==len(self.upper)==len(self.categories)==d):
            raise ValueError("MixedSpace metadata lengths do not match")
        if any(k not in ("real","integer","categorical") for k in self.kinds):
            raise ValueError("Unknown variable kind")
        for i,k in enumerate(self.kinds):
            if self.lower[i]>=self.upper[i]: raise ValueError("Each variable must have at least two possible endpoints")
            if k!="real" and (int(self.lower[i])!=self.lower[i] or int(self.upper[i])!=self.upper[i]):
                raise ValueError("Discrete bounds must be integral")
            if k=="categorical" and (self.lower[i]!=0 or self.upper[i]!=self.categories[i]-1):
                raise ValueError("Categorical bounds must be 0..categories-1")
        remaining=list(self.conditions); done=set(); order=[]
        # Topological order of all dimensions, independently of supplied condition order.
        for _ in range(d):
            available=[i for i in range(d) if i not in done and all(c.parent in done for c in remaining if c.child==i)]
            if not available: break
            for i in available: done.add(i); order.append(i)
        if len(done)!=d: raise ValueError("Conditional dependency graph contains a cycle or invalid index")
        if any(not(0<=c.child<d and 0<=c.parent<d) for c in self.conditions): raise ValueError("Bad condition dimension")
        object.__setattr__(self,"_order",tuple(order))

    @property
    def dimension(self): return len(self.kinds)

    def active(self,x):
        active=jnp.ones(x.shape,dtype=bool)
        for i in self._order:
            for c in self.conditions:
                if c.child==i:
                    allowed=jnp.any(x[:,c.parent,None]==jnp.asarray(c.values)[None,:],axis=1)
                    active=active.at[:,i].set(active[:,i]&active[:,c.parent]&allowed)
        return active

    def canonicalize(self,x):
        x=jnp.clip(x,jnp.asarray(self.lower),jnp.asarray(self.upper))
        for i,k in enumerate(self.kinds):
            if k!="real": x=x.at[:,i].set(jnp.round(x[:,i]))
        active=self.active(x)
        return jnp.where(active,x,jnp.asarray(self.lower))

    def sample(self,key,n):
        columns=[]; keys=jax.random.split(key,self.dimension)
        for i,k in enumerate(self.kinds):
            if k=="real":
                col=jax.random.uniform(keys[i],(n,),minval=self.lower[i],maxval=self.upper[i])
            else:
                col=jax.random.randint(keys[i],(n,),int(self.lower[i]),int(self.upper[i])+1).astype(float)
            columns.append(col)
        return self.canonicalize(jnp.stack(columns,axis=1))

    def encode(self,x):
        """Fixed-length one-hot/arc-style embedding; inactive values are irrelevant."""
        active=self.active(x); columns=[]
        conditional={c.child for c in self.conditions}
        for i,k in enumerate(self.kinds):
            flag=active[:,i,None].astype(x.dtype)
            if k=="categorical":
                values=jax.nn.one_hot(x[:,i].astype(int),self.categories[i])*flag
            else:
                u=(x[:,i]-self.lower[i])/(self.upper[i]-self.lower[i])
                values=jnp.stack([jnp.sin(jnp.pi*u),jnp.cos(jnp.pi*u)],axis=1)*flag if i in conditional else u[:,None]
            columns.append(values)
            if i in conditional: columns.append(flag)
        return jnp.concatenate(columns,axis=1)


@dataclass(frozen=True)
class RandomEmbedding:
    """Algorithm 3 with a USER-CHOSEN latent Box. Clipping enforces physical bounds."""
    matrix:object
    physical:Box
    latent:Box

    @classmethod
    def create(cls,key,physical,latent):
        return cls(jax.random.normal(key,(physical.dimension,latent.dimension)),physical,latent)

    def project(self,z):
        normalized=jnp.clip(jnp.asarray(z)@jnp.asarray(self.matrix).T,-1,1)
        return self.physical.denormalize((normalized+1)/2)

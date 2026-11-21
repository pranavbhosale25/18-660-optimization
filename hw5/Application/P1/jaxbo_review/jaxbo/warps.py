"""Input warps from Eqs. 55-56, including derivatives w.r.t. shape parameters."""
from __future__ import annotations
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln


def _nonzero(x):
    tiny = jnp.asarray(1e-30, x.dtype)
    return jnp.where(jnp.abs(x) < tiny, jnp.where(x < 0, -tiny, tiny), x)


def _beta_fraction(a, b, x, iterations: int):
    qab, qap, qam = a+b, a+1, a-1
    d = 1.0 / _nonzero(1.0 - qab*x/qap)
    def body(m, state):
        c, d, h = state
        m2 = 2.0*m
        aa = m*(b-m)*x / ((qam+m2)*(a+m2))
        d = 1 / _nonzero(1+aa*d)
        c = _nonzero(1+aa/c)
        h = h*d*c
        aa = -(a+m)*(qab+m)*x / ((a+m2)*(qap+m2))
        d = 1 / _nonzero(1+aa*d)
        c = _nonzero(1+aa/c)
        return c, d, h*d*c
    return jax.lax.fori_loop(1, iterations+1, body, (jnp.ones_like(d), d, d))[2]


def beta_cdf(x, a, b, iterations: int = 128):
    """Regularized incomplete beta via a fixed-iteration continued fraction.

    Unlike calling JAX betainc directly, this construction differentiates a and b.
    Inputs a,b must be strictly positive; fixed iterations are an approximation.
    Tested for shapes 0.2..20. Extreme shapes require independent validation.
    """
    x, a, b = jnp.broadcast_arrays(jnp.asarray(x, dtype=float), a, b)
    eps = jnp.finfo(x.dtype).eps
    xx = jnp.clip(x, eps, 1-eps)
    flip = xx >= (a+1)/(a+b+2)
    aa, bb, zz = jnp.where(flip,b,a), jnp.where(flip,a,b), jnp.where(flip,1-xx,xx)
    logfront = gammaln(aa+bb)-gammaln(aa)-gammaln(bb)+aa*jnp.log(zz)+bb*jnp.log1p(-zz)
    v = jnp.exp(logfront)*_beta_fraction(aa,bb,zz,iterations)/aa
    result = jnp.where(flip,1-v,v)
    return jnp.where(x<=0,0.0,jnp.where(x>=1,1.0,jnp.clip(result,0,1)))


def beta_warp(p, x):
    return beta_cdf(x, jnp.exp(p["log_a"]), jnp.exp(p["log_b"]))


def beta_inverse(y, a, b, iterations: int = 60):
    """Monotone inverse by bisection; not intended for shape-parameter gradients."""
    y,a,b = jnp.broadcast_arrays(jnp.asarray(y,dtype=float),a,b)
    def body(_, interval):
        lo, hi = interval
        mid = (lo+hi)/2
        left = beta_cdf(mid,a,b) < y
        return jnp.where(left,mid,lo),jnp.where(left,hi,mid)
    lo,hi = jax.lax.fori_loop(0,iterations,body,(jnp.zeros_like(y),jnp.ones_like(y)))
    return jnp.where(y<=0,0,jnp.where(y>=1,1,(lo+hi)/2))


def kumaraswamy_warp(p, x):
    a,b = jnp.exp(p["log_a"]),jnp.exp(p["log_b"])
    eps = jnp.finfo(jnp.asarray(x).dtype).eps
    xx = jnp.clip(x,eps,1-eps)
    result = -jnp.expm1(b*jnp.log1p(-jnp.exp(a*jnp.log(xx))))
    return jnp.where(x<=0,0,jnp.where(x>=1,1,result))


def kumaraswamy_inverse(p, y):
    a,b = jnp.exp(p["log_a"]),jnp.exp(p["log_b"])
    return (-jnp.expm1(jnp.log1p(-jnp.clip(y,0,1))/b))**(1/a)

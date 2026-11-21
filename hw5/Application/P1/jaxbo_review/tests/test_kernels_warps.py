import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.special import betainc
from jaxbo import kernels,warps

@pytest.mark.parametrize("name",["rbf","matern12","matern32","matern52"])
def test_stationary_kernels_psd_jit_and_gradients(name):
    k=getattr(kernels,name);p=kernels.stationary_parameters(2,[.3,.7],1.4)
    x=jnp.array([[.1,.2],[.4,.7],[.1,.2],[.8,.6]])
    matrix=jax.jit(k)(p,x,x)
    np.testing.assert_allclose(matrix,matrix.T,atol=1e-12)
    np.testing.assert_allclose(jnp.diag(matrix),1.4**2,atol=1e-12)
    assert float(jnp.linalg.eigvalsh(matrix).min())>=-1e-10
    grad=jax.grad(lambda xx:k(p,xx,x).sum())(x)
    assert np.isfinite(grad).all()
    gradp=jax.grad(lambda pp:k(pp,x,x).sum())(p)
    assert all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(gradp))


def test_covariance_formulas():
    p=kernels.stationary_parameters(1,2.,3.)
    x,z=jnp.array([[0.]]),jnp.array([[1.]])
    r=.5
    expected=[9*np.exp(-.5*r*r),9*np.exp(-r),9*(1+np.sqrt(3)*r)*np.exp(-np.sqrt(3)*r),
              9*(1+np.sqrt(5)*r+5*r*r/3)*np.exp(-np.sqrt(5)*r)]
    for k,e in zip([kernels.rbf,kernels.matern12,kernels.matern32,kernels.matern52],expected):
        np.testing.assert_allclose(k(p,x,z),[[e]],rtol=1e-12)


def test_compositions_periodicity_and_coregionalization():
    x=jnp.array([[.1],[.4],[.9]]);p=kernels.stationary_parameters(1)
    combined=kernels.add(kernels.rbf,kernels.matern32)
    np.testing.assert_allclose(combined({'left':p,'right':p},x,x),kernels.rbf(p,x,x)+kernels.matern32(p,x,x))
    periodic=dict(p,log_period=jnp.log(1.))
    np.testing.assert_allclose(kernels.periodic(periodic,x,x+1),kernels.periodic(periodic,x,x),atol=1e-12)
    tasks=jnp.array([[.2,0],[.2,1]])
    cp={'base':p,'task_factor':jnp.array([[1.,0],[.5,1.]]),'log_task_diagonal':jnp.log(jnp.array([.1,.2]))}
    icm=kernels.coregionalized(kernels.rbf)
    np.testing.assert_allclose(icm(cp,tasks,tasks),[[1.1,.5],[.5,1.45]],atol=1e-12)


@pytest.mark.parametrize("a,b",[(.2,.3),(.5,5.),(1.,1.),(2.,3.),(8.,.4),(20.,20.)])
def test_beta_cdf_against_scipy(a,b):
    x=jnp.linspace(.001,.999,21)
    np.testing.assert_allclose(jax.jit(warps.beta_cdf)(x,a,b),betainc(a,b,np.asarray(x)),rtol=2e-10,atol=2e-10)


def test_beta_shape_derivatives_and_inverse():
    x=.4;a=2.3;b=.7;h=1e-5
    ga,gb=jax.grad(lambda aa,bb:warps.beta_cdf(x,aa,bb),argnums=(0,1))(a,b)
    refa=(betainc(a+h,b,x)-betainc(a-h,b,x))/(2*h)
    refb=(betainc(a,b+h,x)-betainc(a,b-h,x))/(2*h)
    np.testing.assert_allclose([ga,gb],[refa,refb],rtol=2e-6)
    points=jnp.array([.01,.2,.8,.99])
    np.testing.assert_allclose(warps.beta_inverse(warps.beta_cdf(points,a,b),a,b),points,atol=1e-9)
    np.testing.assert_array_equal(warps.beta_cdf(jnp.array([0.,1.]),a,b),[0.,1.])


def test_warp_identity_and_kumaraswamy_inverse():
    x=jnp.linspace(0,1,11)[:,None]
    identity={'log_a':jnp.zeros(1),'log_b':jnp.zeros(1)}
    for w in (warps.beta_warp,warps.kumaraswamy_warp):
        np.testing.assert_allclose(w(identity,x),x,atol=1e-12)
    p={'log_a':jnp.log(2.3),'log_b':jnp.log(.7)}
    np.testing.assert_allclose(warps.kumaraswamy_inverse(p,warps.kumaraswamy_warp(p,x)),x,atol=1e-10)

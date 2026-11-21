import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jaxbo.optim import minimize_lbfgs,minimize_adam,maximize_multistart,maximize_cma_es,maximize_pattern_search
from jaxbo.global_optim import maximize_direct,maximize_adaptive_grid,maximize_soo
from jaxbo.spaces import Box,Condition,MixedSpace,RandomEmbedding,initial_design,cartesian_grid
from jaxbo.tpe import ParzenEstimator


def test_lbfgs_gradient_and_box_boundary_optimum():
    loss=lambda x:jnp.sum((x-jnp.array([.3,2.]))**2)
    solve=jax.jit(lambda x:minimize_lbfgs(loss,x,jnp.zeros(2),jnp.ones(2),steps=60))
    result=solve(jnp.array([.8,.2]))
    np.testing.assert_allclose(result.x,[.3,1],atol=1e-6)
    assert bool(result.converged)


def test_adam_reduces_loss():
    loss=lambda x:jnp.sum((x-.3)**2)
    result=minimize_adam(loss,jnp.array([.9,.1]),jnp.zeros(2),jnp.ones(2),steps=100)
    assert float(result.value)<1e-4


@pytest.mark.parametrize("method",["lbfgs","cma_es","pattern"])
def test_continuous_optimizers_are_jittable_and_improve(method):
    f=lambda x:-jnp.sum((x-.37)**2);lo=jnp.zeros(2);hi=jnp.ones(2)
    if method=='lbfgs':
        solve=lambda key:maximize_multistart(key,f,lo,hi,n_initial=32,restarts=3,steps=25)
    elif method=='cma_es': solve=lambda key:maximize_cma_es(key,f,lo,hi,steps=40)
    else: solve=lambda key:maximize_pattern_search(key,f,lo,hi,steps=30,restarts=3)
    result=jax.jit(solve)(jax.random.key(3))
    assert float(result.value)>-1e-3
    assert np.all((result.x>=0)&(result.x<=1))


@pytest.mark.parametrize("optimizer",[maximize_direct,maximize_adaptive_grid,maximize_soo])
def test_partition_optimizers(optimizer):
    f=lambda x:-jnp.sum((x-.33)**2)
    result=jax.jit(lambda:optimizer(f,jnp.zeros(2),jnp.ones(2),splits=25))()
    assert float(result.value)>-.02
    assert np.all((result.x>=0)&(result.x<=1))


@pytest.mark.parametrize("method",["random","latin","halton"])
def test_initial_design_reproducible_and_in_bounds(method):
    a=jax.jit(lambda key:initial_design(key,20,3,method))(jax.random.key(2))
    b=initial_design(jax.random.key(2),20,3,method)
    np.testing.assert_allclose(a,b,rtol=1e-14,atol=1e-15)
    assert np.all((a>=0)&(a<1))
    if method=='latin':
        for i in range(3):np.testing.assert_array_equal(np.sort(np.floor(np.asarray(a[:,i])*20)),np.arange(20))


def test_box_grid_and_validation():
    box=Box([-2.,1.],[3.,2.]);x=box.sample(jax.random.key(2),8)
    np.testing.assert_allclose(box.denormalize(box.normalize(x)),x,atol=1e-12)
    assert cartesian_grid(box.lower,box.upper,3).shape==(9,2)
    with pytest.raises(ValueError):Box([1.],[0.])
    with pytest.raises(ValueError):cartesian_grid(jnp.zeros(10),jnp.ones(10),100)


def test_conditional_embedding_is_invariant_to_inactive_value():
    space=MixedSpace((0.,0.,0.),(1.,1.,3.),('categorical','real','integer'),(2,0,0),
                    (Condition(1,0,(1,)),Condition(2,0,(1,))))
    x=jnp.array([[0.,.1,0.],[0.,.9,3.],[1.,.2,2.]])
    encoded=jax.jit(space.encode)(x)
    np.testing.assert_array_equal(encoded[0],encoded[1])
    assert not np.allclose(encoded[0],encoded[2])
    sampled=space.sample(jax.random.key(2),30)
    assert np.all(np.asarray(sampled[:,2])%1==0)
    np.testing.assert_array_equal(sampled[sampled[:,0]==0,1:],0)


def test_rembo_projection_and_tpe_proposals():
    physical=Box([-2.]*5,[2.]*5);latent=Box([-1.]*2,[1.]*2)
    embedding=RandomEmbedding.create(jax.random.key(1),physical,latent)
    out=jax.jit(embedding.project)(jnp.array([[1.,-1.],[0.,0.]]))
    assert out.shape==(2,5)
    assert np.all((out>=-2)&(out<=2))
    box=Box([0.],[1.]);x=jnp.linspace(0,1,15)[:,None];y=-(x[:,0]-.7)**2
    tpe=ParzenEstimator(box,n_candidates=100);s=tpe.fit(x,y)
    point=jax.jit(tpe.ask)(jax.random.key(3),s)
    assert 0<=float(point[0])<=1
    assert abs(float(point[0])-.7)<.25

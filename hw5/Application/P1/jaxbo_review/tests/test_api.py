import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jaxbo import BayesianOptimizer,Box,AcquisitionOptimizer,scan_optimize,GeneralizedLinear,RandomForest


def test_fully_compiled_loop_optimizes_quadratic_and_is_reproducible():
    f=lambda x:-jnp.sum((x-.37)**2)
    x=jnp.array([[0.],[.7],[1.]]);y=jax.vmap(f)(x)
    opt=AcquisitionOptimizer('lbfgs',n_initial=24,restarts=2,steps=15)
    run=jax.jit(lambda key:scan_optimize(f,key,jnp.zeros(1),jnp.ones(1),x,y,n_steps=5,optimizer=opt))
    a=run(jax.random.key(3));b=run(jax.random.key(3))
    assert a.x.shape==(8,1)
    assert float(a.best_y)>-1e-3
    np.testing.assert_array_equal(a.x,b.x)
    np.testing.assert_array_equal(a.y,b.y)


@pytest.mark.parametrize("maximize",[True,False])
def test_ask_tell_direction_scaling_and_finite_candidates(maximize):
    f=lambda x:(1 if maximize else -1)*(10-100*jnp.sum((x-.4)**2))
    box=Box([0.],[1.]);bo=BayesianOptimizer(box,key=jax.random.key(7),maximize=maximize,
                     optimizer=AcquisitionOptimizer('random',n_initial=64))
    initial=jnp.array([[0.],[.7],[1.]])
    bo.tell(initial,jax.vmap(f)(initial))
    candidates=jnp.linspace(.01,.99,60)[:,None]
    for _ in range(3):
        point=bo.ask(candidates=candidates);bo.tell(point,f(point))
    x,y=bo.recommend()
    assert abs(float(x[0])-.4)<.15
    assert y==f(x)
    xp,yp=bo.recommend('posterior')
    assert np.isfinite(yp)


def test_ask_tell_validation_and_exhausted_candidates():
    bo=BayesianOptimizer(Box([0.],[1.]),key=jax.random.key(1),optimizer=AcquisitionOptimizer('random',n_initial=10))
    assert bo.ask().shape==(1,)
    with pytest.raises(ValueError):bo.tell([.2],float('nan'))
    with pytest.raises(ValueError):bo.tell([2.],1.)
    with pytest.raises(ValueError):bo.tell([[.2],[.4]],[1.])
    bo.tell([.2],1.)
    with pytest.raises(ValueError):bo.ask(candidates=jnp.array([[.2]]))
    with pytest.raises(ValueError):BayesianOptimizer(Box([0.],[1.]),key=jax.random.key(1),acquisition='es')
    with pytest.raises(ValueError):BayesianOptimizer(Box([0.],[1.]),key=jax.random.key(1),model=GeneralizedLinear())


def test_random_forest_is_interchangeable_in_host_loop():
    bo=BayesianOptimizer(Box([0.],[1.]),key=jax.random.key(1),model=RandomForest(n_trees=3,max_depth=2),
                         optimizer=AcquisitionOptimizer('random',n_initial=12))
    bo.tell(jnp.array([[0.],[.3],[.8],[1.]]),jnp.array([0.,1.,.2,0.]))
    point=bo.ask()
    assert point.shape==(1,)
    assert 0<=float(point[0])<=1


def test_first_ask_respects_finite_candidates():
    bo=BayesianOptimizer(Box([0.],[1.]),key=jax.random.key(5))
    candidates=jnp.array([[.23],[.71],[.23]])
    proposed=bo.ask(candidates=candidates,q=2)
    np.testing.assert_allclose(jnp.sort(proposed[:,0]),jnp.array([.23,.71]))
    with pytest.raises(ValueError): bo.ask(candidates=candidates,q=3)
    with pytest.raises(ValueError): bo.ask(candidates=jnp.array([[1.5]]))

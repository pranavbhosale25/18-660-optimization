import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jaxbo.models import BayesianLinear,BetaBernoulli,GeneralizedLinear,GPClassifier,RandomForest,NeuralLinear
from jaxbo.models.bandits import beta_update,beta_thompson,run_beta_bandit
from jaxbo.models.linear import identity_features,radial_features,fourier_features
from jaxbo.models.glm import link_value


def test_nig_matches_equations_9_to_12():
    x=jnp.array([[1.,.1],[1.,.5],[1.,1.] ]);y=jnp.array([.3,.8,1.2])
    model=BayesianLinear(features=identity_features,noise_variance=None,prior_variance=2,prior_shape=2.5,prior_rate=.7)
    s=jax.jit(model.fit)(x,y)
    vn=np.linalg.inv(np.eye(2)/2+np.asarray(x).T@np.asarray(x))
    wn=vn@np.asarray(x).T@np.asarray(y)
    an=2.5+1.5;bn=.7+.5*(np.asarray(y)@np.asarray(y)-wn@np.linalg.solve(vn,wn))
    np.testing.assert_allclose(s.mean,wn,atol=1e-7)
    np.testing.assert_allclose(s.covariance,vn,atol=1e-7)
    np.testing.assert_allclose([s.shape,s.rate],[an,bn],atol=1e-7)
    pred=model.predict(s,x);obs=model.predict(s,x,observation_noise=True)
    np.testing.assert_allclose(obs.scale2-pred.scale2,s.rate/s.shape)


def test_linear_known_noise_exact_and_feature_interfaces():
    x=jnp.array([[0.],[.3],[.8],[1.]]);y=1+2*x[:,0]
    m=BayesianLinear(noise_variance=.1,prior_variance=3.)
    s=m.fit(x,y);phi=np.c_[np.ones(4),np.asarray(x)]
    v=np.linalg.inv(np.eye(2)/3+phi.T@phi/.1);w=v@phi.T@np.asarray(y)/.1
    np.testing.assert_allclose(s.mean,w,atol=1e-6)
    np.testing.assert_allclose(s.covariance,v,atol=1e-6)
    for feature in [radial_features(x[::2]),fourier_features(jnp.array([[1.],[2.]]))]:
        mm=BayesianLinear(features=feature);ss=mm.fit(x,y)
        assert mm.predict(ss,x).mean.shape==(4,)


def test_beta_posterior_and_scan_bandit():
    m=BetaBernoulli(3,2,2);s=m.initialize();s=beta_update(s,1,1.);s=beta_update(s,1,0.)
    np.testing.assert_allclose(s.alpha,[2,3,2]);np.testing.assert_allclose(s.beta,[2,3,2])
    key=jax.random.key(4)
    np.testing.assert_array_equal(beta_thompson(key,s),beta_thompson(key,s))
    out,arms,rewards=jax.jit(lambda k:run_beta_bandit(k,jnp.array([.05,.1,.95]),300))(key)
    assert rewards.shape==(300,)
    assert float(rewards.mean())>.75
    assert int(jnp.argmax(out.alpha/(out.alpha+out.beta)))==2


@pytest.mark.parametrize("link",["logit","logit_negative","probit"])
def test_glm_laplace_predictions_and_fixed_thompson(link):
    x=jnp.linspace(-1,1,10)[:,None];y=(x[:,0]>0).astype(float)
    m=GeneralizedLinear(link=link,steps=50)
    s=jax.jit(m.fit)(x,y);p=m.predict(s,x)
    assert p.mean[0]<.5<p.mean[-1]
    assert np.isfinite(p.variance).all()
    draw=m.thompson_function(s,jax.random.key(2))
    np.testing.assert_array_equal(draw(x),draw(x))
    assert np.all((draw(x)>0)&(draw(x)<1))


def test_gp_classifier_probability_and_bernoulli_bounds():
    x=jnp.linspace(0,1,8)[:,None];y=(x[:,0]>.5).astype(float)
    model=GPClassifier(steps=40);s=model.fit(x,y)
    probability=model.probability(s,x)
    assert np.all((probability>=0)&(probability<=1))
    assert probability[1]<.5<probability[-2]
    samples=model.sample(s,x,jax.random.key(5),8)
    assert samples.shape==(8,8)


@pytest.mark.parametrize("inference",["hmc","mh"])
def test_glm_mcmc_is_explicit_and_finite(inference):
    x=jnp.array([[-1.],[0.],[1.]]);y=jnp.array([0.,0.,1.])
    model=GeneralizedLinear(inference=inference,steps=15,n_samples=8,burnin=8)
    with pytest.raises(ValueError):model.fit(x,y)
    state=model.fit(x,y,key=jax.random.key(2))
    assert state.samples.shape==(8,2)
    assert np.isfinite(model.predict(state,x).mean).all()
    assert 0<=float(state.acceptance_rate)<=1


@pytest.mark.parametrize("uncertainty",["gaussian","empirical"])
def test_native_forest_numeric_splits_and_uncertainty(uncertainty):
    x=jnp.linspace(0,1,40)[:,None];y=jnp.where(x[:,0]<.5,-1.,2.)
    forest=RandomForest(n_trees=8,max_depth=3,min_leaf=1,uncertainty=uncertainty)
    s=jax.jit(forest.fit)(x,y,key=jax.random.key(6))
    p=forest.predict(s,x)
    assert float(jnp.mean((p.mean-y)**2))<.1
    assert np.all(p.variance>=0)
    np.testing.assert_allclose(p.mean,forest.posterior(s,x).mean)
    assert forest.leaf_ids(s,x).shape==(8,40)


def test_neural_linear_small_training_and_bayesian_last_layer():
    x=jnp.linspace(0,1,10)[:,None];y=jnp.sin(3*x[:,0])
    model=NeuralLinear(widths=(4,),steps=15)
    state=model.fit(x,y,key=jax.random.key(4))
    assert np.isfinite(state.training_loss)
    assert model.predict(state,x).mean.shape==(10,)
    assert np.all(model.predict(state,x).variance>=0)


@pytest.mark.parametrize("uncertainty",["gaussian","empirical"])
def test_forest_thompson_is_a_fixed_coherent_function(uncertainty):
    from jaxbo.acquisitions import make_acquisition
    x=jnp.linspace(0,1,12)[:,None];y=jnp.sin(5*x[:,0])
    model=RandomForest(n_trees=5,max_depth=2,uncertainty=uncertainty)
    state=model.fit(x,y,key=jax.random.key(3))
    draw=make_acquisition("ts",model,state,y.max(),key=jax.random.key(4))
    first=draw(x)
    np.testing.assert_array_equal(first,draw(x))
    np.testing.assert_allclose(first[:4],draw(x[:4]),atol=1e-14)
    # A duplicate input must have exactly the same sampled value.
    repeated=draw(jnp.repeat(x[:1],2,axis=0))
    np.testing.assert_array_equal(repeated[0],repeated[1])

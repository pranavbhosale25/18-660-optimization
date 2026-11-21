import numpy as np
import jax
import jax.numpy as jnp
from jaxbo.inference import laplace,hmc,random_walk_metropolis,tempered_smc,gauss_hermite,gaussian_product_quadrature,quadrature_posterior
from jaxbo import ExactGP,default_params
from jaxbo.hyperparameters import fit_hyperparameters,integrate_hyperparameters,marginalized_acquisition
from jaxbo.acquisitions import expected_improvement


def test_laplace_recovers_gaussian():
    mean=jnp.array([.5,-1.]);precision=jnp.array([[2.,.2],[.2,1.]])
    logp=lambda x:-.5*(x-mean)@precision@(x-mean)
    result=jax.jit(lambda x:laplace(logp,x))(jnp.zeros(2))
    np.testing.assert_allclose(result.mean,mean,atol=2e-5)
    np.testing.assert_allclose(result.covariance,np.linalg.inv(np.asarray(precision)),atol=1e-6)


def test_hmc_gaussian_moments_and_repeatability():
    target=jnp.array([.7,-.3])
    logp=lambda x:-.5*jnp.sum((x-target)**2)
    sample=jax.jit(lambda key:hmc(key,logp,jnp.zeros(2),n_samples=2500,burnin=200,step_size=.3,leapfrog_steps=5))
    a=sample(jax.random.key(1));b=sample(jax.random.key(1))
    np.testing.assert_array_equal(a.samples,b.samples)
    np.testing.assert_allclose(a.samples.mean(axis=0),target,atol=.08)
    np.testing.assert_allclose(a.samples.var(axis=0),1.,atol=.12)
    assert float(a.acceptance_rate)>.9


def test_metropolis_gaussian_moments():
    logp=lambda x:-.5*jnp.sum(x*x)
    result=random_walk_metropolis(jax.random.key(9),logp,jnp.zeros(1),n_samples=4000,burnin=500,proposal_scale=1.5)
    assert abs(float(result.samples.mean()))<.12
    assert abs(float(result.samples.var())-1)<.2
    assert .2<float(result.acceptance_rate)<.9


def test_tempered_smc_normal_normal_posterior():
    key,kp=jax.random.split(jax.random.key(5))
    particles=jax.random.normal(kp,(1024,1))
    logprior=lambda x:-.5*jnp.sum(x*x)-.5*jnp.log(2*jnp.pi)
    loglike=lambda x:-.5*jnp.sum(((x-2.5)/.5)**2)-.5*jnp.log(2*jnp.pi*.25)
    result=jax.jit(lambda key:tempered_smc(key,loglike,logprior,particles,stages=10,mcmc_steps=4,proposal_scale=.5))(key)
    estimate=jnp.sum(result.weights*result.particles[:,0])
    np.testing.assert_allclose(estimate,2.,atol=.12)
    np.testing.assert_allclose(result.weights.sum(),1.,atol=1e-12)
    assert np.isfinite(result.log_evidence)


def test_gaussian_quadrature_and_posterior_normalization():
    nodes,weights=gauss_hermite(5)
    np.testing.assert_allclose(jnp.sum(weights),1,atol=1e-12)
    np.testing.assert_allclose(jnp.sum(weights*nodes**2),1,atol=1e-12)
    np.testing.assert_allclose(jnp.sum(weights*nodes**4),3,atol=1e-12)
    points,weights=gaussian_product_quadrature(jnp.array([1.,2.]),jnp.array([.5,1.]),order=4)
    assert points.shape==(16,2)
    np.testing.assert_allclose(weights@points,[1,2],atol=1e-12)
    posterior=quadrature_posterior(lambda x:-jnp.sum(x*x),points,weights)
    np.testing.assert_allclose(posterior.sum(),1,atol=1e-12)


def test_ml_fitting_improves_marginal_likelihood():
    x=jnp.linspace(0,1,7)[:,None];y=jnp.sin(4*x[:,0])
    model=ExactGP();p=default_params(1,lengthscale=.05,noise=.3)
    before=model.log_marginal_likelihood(model.fit(x,y,p))
    fit=jax.jit(lambda p:fit_hyperparameters(model,x,y,p,method='ml',steps=15))(p)
    after=model.log_marginal_likelihood(fit.state)
    assert float(after)>=float(before)-1e-7


def test_hyperparameter_mixture_averages_acquisitions_not_moments():
    x=jnp.array([[0.],[.5],[1.]]);y=jnp.array([0.,1.,0.]);points=jnp.array([[.3],[.7]])
    model=ExactGP();p=default_params(1,noise=.1)
    ensemble=integrate_hyperparameters(jax.random.key(7),model,x,y,p,method='mh',n_samples=4,burnin=3,step_size=.1)
    a=marginalized_acquisition('ei',model,ensemble,1.)
    manual=jax.vmap(lambda s:expected_improvement(model.predict(s,points),1.))(ensemble.states)
    np.testing.assert_allclose(a(points),ensemble.weights@manual,atol=1e-12)
    assert np.isfinite(a(points)).all()

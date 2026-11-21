import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jaxbo import ExactGP,SparseGP,SpectralGP,StudentTProcess,default_params,kernels


def data():
    x=jnp.array([[.0],[.25],[.6],[1.]])
    y=jnp.sin(5*x[:,0]);z=jnp.array([[.15],[.4],[.8]])
    return x,y,z


def test_exact_gp_against_independent_numpy_solve():
    x,y,z=data();p=default_params(1,lengthscale=.3,amplitude=1.2,noise=.07,mean=.2)
    model=ExactGP(kernel=kernels.rbf);state=jax.jit(model.fit)(x,y,p)
    k=np.asarray(kernels.rbf(p.kernel,x,x));cross=np.asarray(kernels.rbf(p.kernel,x,z))
    jitter=1e-8*max(np.diag(k).mean(),1)
    cov=k+np.eye(len(x))*(.07**2+jitter)
    alpha=np.linalg.solve(cov,np.asarray(y)-.2)
    mean=.2+cross.T@alpha
    variance=np.asarray(kernels.rbf(p.kernel,z,z))-cross.T@np.linalg.solve(cov,cross)
    posterior=model.posterior(state,z)
    np.testing.assert_allclose(posterior.mean,mean,atol=1e-10)
    np.testing.assert_allclose(posterior.covariance,variance,atol=1e-10)
    logp=-.5*((np.asarray(y)-.2)@alpha+np.linalg.slogdet(cov)[1]+len(x)*np.log(2*np.pi))
    np.testing.assert_allclose(model.log_marginal_likelihood(state),logp,atol=1e-10)


def test_latent_and_observation_uncertainty_are_distinct():
    x,y,z=data();model=ExactGP();p=default_params(1,noise=.2)
    s=model.fit(x,y,p);a=model.predict(s,z);b=model.predict(s,z,observation_noise=True)
    np.testing.assert_allclose(b.variance-a.variance,.04,atol=1e-12)
    np.testing.assert_allclose(a.variance,jnp.diag(model.posterior(s,z).covariance),atol=1e-12)
    known=jnp.array([.01,.04,.08,.02]);s2=model.fit(x,y,p,noise_variance=known)
    np.testing.assert_allclose(s2.noise_variance,known)


def test_fixed_capacity_mask_matches_unpadded_fit():
    x,y,z=data();gp=ExactGP();p=default_params(1)
    unpadded=gp.fit(x,y,p)
    xx=jnp.concatenate([x,jnp.array([[.11],[.88],[.35]])]);yy=jnp.concatenate([y,jnp.array([100.,-100.,50.])])
    padded=jax.jit(gp.fit)(xx,yy,p,mask=jnp.array([1,1,1,1,0,0,0],dtype=bool))
    a,b=gp.posterior(unpadded,z),gp.posterior(padded,z)
    np.testing.assert_allclose(a.mean,b.mean,atol=1e-10)
    np.testing.assert_allclose(a.covariance,b.covariance,atol=1e-10)
    np.testing.assert_allclose(gp.log_marginal_likelihood(unpadded),gp.log_marginal_likelihood(padded),atol=1e-10)


def test_empty_mask_is_prior_and_gradients_are_finite():
    x,y,z=data();gp=ExactGP();p=default_params(1,mean=.7)
    prior=gp.fit(x,y,p,mask=jnp.zeros(4,dtype=bool));post=gp.posterior(prior,z)
    np.testing.assert_allclose(post.mean,.7)
    np.testing.assert_allclose(post.covariance,gp.kernel(p.kernel,z,z))
    np.testing.assert_allclose(gp.log_marginal_likelihood(prior),0)
    grad=jax.grad(lambda pp:gp.log_marginal_likelihood(gp.fit(x,y,pp)))(p)
    assert all(np.isfinite(a).all() for a in jax.tree_util.tree_leaves(grad))


@pytest.mark.parametrize("method",["fitc","dtc","vfe"])
def test_sparse_with_all_inducing_points_matches_exact(method):
    x,y,z=data();p=default_params(1,noise=.08)
    exact=ExactGP();sparse=SparseGP(x,method=method)
    a=exact.fit(x,y,p);b=jax.jit(sparse.fit)(x,y,p)
    np.testing.assert_allclose(sparse.predict(b,z).mean,exact.predict(a,z).mean,atol=2e-6)
    np.testing.assert_allclose(sparse.predict(b,z).variance,exact.predict(a,z).variance,atol=2e-6)
    np.testing.assert_allclose(sparse.log_marginal_likelihood(b),exact.log_marginal_likelihood(a),atol=1e-5)


def test_vfe_is_lower_bound_and_sparse_inducing_gradients():
    x,y,z=data();p=default_params(1,noise=.1)
    exact=ExactGP();sparse=SparseGP(x[::2],method='vfe')
    s=sparse.fit(x,y,p)
    assert float(sparse.log_marginal_likelihood(s))<=float(exact.log_marginal_likelihood(exact.fit(x,y,p)))+1e-7
    gradient=jax.grad(lambda inducing:sparse.log_marginal_likelihood(sparse.fit(x,y,p,inducing=inducing)))(x[::2])
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(sparse.predict(s,z).variance,jnp.diag(sparse.posterior(s,z).covariance),atol=1e-10)


def test_spectral_matches_exact_gp_with_same_feature_kernel():
    x,y,z=data();p=default_params(1,noise=.1)
    model=SpectralGP.create(jax.random.key(2),1,12,kernel='matern52')
    kernel=lambda pp,a,b:model.features(p._replace(kernel=pp),a)@model.features(p._replace(kernel=pp),b).T
    exact=ExactGP(kernel=kernel)
    a=model.fit(x,y,p);b=exact.fit(x,y,p)
    np.testing.assert_allclose(model.predict(a,z).mean,exact.predict(b,z).mean,atol=2e-6)
    np.testing.assert_allclose(model.predict(a,z).variance,exact.predict(b,z).variance,atol=2e-6)
    draw=model.thompson_function(a,jax.random.key(3))
    np.testing.assert_array_equal(draw(z),draw(z))
    np.testing.assert_allclose(model.log_marginal_likelihood(a),exact.log_marginal_likelihood(b),atol=1e-5)


def test_student_process_has_distinct_scale_and_variance():
    x,y,z=data();model=StudentTProcess();state=model.fit(x,y)
    p=model.posterior(state,z);m=model.predict(state,z)
    np.testing.assert_allclose(m.variance,jnp.diag(p.covariance))
    np.testing.assert_allclose(m.df,2*model.prior_shape+x.shape[0])
    assert np.isfinite(p.sample(jax.random.key(7),20)).all()


def test_joint_gp_samples_preserve_duplicate_point_correlation():
    x,y,z=data();gp=ExactGP();state=gp.fit(x,y)
    samples=gp.posterior(state,jnp.array([[.3],[.3]])).sample(jax.random.key(4),1024)
    np.testing.assert_allclose(samples[:,0],samples[:,1],atol=1e-6)

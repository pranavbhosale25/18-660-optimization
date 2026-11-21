import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jaxbo import ExactGP,default_params,kernels,AcquisitionOptimizer
from jaxbo.information import entropy_search_mc,predictive_entropy_search_mc,integrated_conditional_improvement
from jaxbo.batch import fantasy_states,fantasy_acquisition,batch_greedy,batch_ucb
from jaxbo.acquisitions import expected_improvement
from jaxbo.constraints import gaussian_feasibility,constrained,cost_sensitive,augmented_lagrangian,update_multipliers
from jaxbo.distributions import NormalMarginal
from jaxbo.portfolios import initialize_hedge,hedge_choose,hedge_update,entropy_portfolio_choose
from jaxbo.extensions import HeteroscedasticGP,learning_curve_kernel,choose_training_action


def fixture():
    model=ExactGP();x=jnp.array([[.1],[.5],[.9]]);y=jnp.array([.2,.8,-.1])
    state=model.fit(x,y,default_params(1,noise=.1))
    return model,state,x,y


@pytest.mark.parametrize("builder",[entropy_search_mc,predictive_entropy_search_mc])
def test_information_zero_when_only_one_possible_optimizer(builder):
    model,state,x,y=fixture();reference=jnp.array([[.3]])
    a=builder(model,state,reference,jax.random.key(1),n_functions=24)
    np.testing.assert_allclose(jax.jit(a)(jnp.array([[.2],[.7]])),0,atol=1e-10)


@pytest.mark.parametrize("builder",[entropy_search_mc,predictive_entropy_search_mc])
def test_information_is_finite_reproducible_and_informative(builder):
    model=ExactGP(kernel=kernels.rbf);x=jnp.array([[.5]]);y=jnp.array([0.])
    state=model.fit(x,y,default_params(1,lengthscale=.1,noise=.01))
    reference=jnp.array([[0.],[1.]])
    opts={'n_fantasies':64} if builder is entropy_search_mc else {}
    a=builder(model,state,reference,jax.random.key(3),n_functions=256,**opts)
    scores=jax.jit(a)(jnp.array([[0.],[.5]]))
    assert np.isfinite(scores).all()
    assert float(scores[0])>.02
    # XLA may fuse arithmetic: eager/JIT equivalence is numerical, not bitwise.
    np.testing.assert_allclose(a(jnp.array([[0.],[.5]])),scores,rtol=1e-12,atol=1e-14)
    np.testing.assert_array_equal(jax.jit(a)(jnp.array([[0.],[.5]])),scores)


def test_ieci_mean_preserving_nonnegative():
    model,state,x,y=fixture()
    a=integrated_conditional_improvement(model,state,jnp.array([[.2],[.7]]),jnp.ones(2),.8)
    assert np.all(a(jnp.array([[.2],[.6]]))>=-1e-12)


def test_fantasy_conditioning_and_acquisition_average():
    model,state,x,y=fixture();pending=jnp.array([[.35],[.7]]);key=jax.random.key(2)
    states,values=fantasy_states(key,model,state,x,y,pending,n_fantasies=4)
    assert values.shape==(4,2)
    assert states.x.shape==(4,5,1)
    points=jnp.array([[.25],[.65]])
    a=fantasy_acquisition('ei',key,model,state,x,y,pending,n_fantasies=4)
    direct=jax.vmap(lambda s,v:expected_improvement(model.predict(s,points),jnp.maximum(y.max(),v.max())))(states,values).mean(axis=0)
    np.testing.assert_allclose(a(points),direct,atol=1e-12)
    reduced=jax.vmap(lambda s:model.predict(s,pending).variance)(states)
    assert np.all(reduced<np.asarray(model.predict(state,pending).variance)[None,:])


@pytest.mark.parametrize("strategy",["constant_liar","kriging_believer","fantasies"])
def test_batch_strategies_return_nonduplicate_points(strategy):
    model,state,x,y=fixture();optimizer=AcquisitionOptimizer('random',n_initial=20)
    points=batch_greedy(jax.random.key(4),model,state,x,y,jnp.zeros(1),jnp.ones(1),optimizer,
                       q=3,strategy=strategy,n_fantasies=3)
    assert points.shape==(3,1)
    assert np.unique(np.asarray(points[:,0])).size==3
    assert np.all((points>=0)&(points<=1))


def test_constraints_cost_and_augmented_lagrangian():
    mean=jnp.array([-1.,0.,1.]);variance=jnp.zeros(3)
    np.testing.assert_array_equal(gaussian_feasibility(mean,variance),[1,1,0])
    base=lambda x:jnp.ones(x.shape[0])*2
    feasible=lambda x:jnp.ones(x.shape[0])*.3
    np.testing.assert_allclose(constrained(base,feasible)(jnp.zeros((2,1))),.6)
    np.testing.assert_allclose(constrained(base,feasible,has_feasible=False)(jnp.zeros((2,1))),.3)
    cost=lambda x:NormalMarginal(jnp.ones(x.shape[0])*jnp.log(2),jnp.zeros(x.shape[0]))
    np.testing.assert_allclose(cost_sensitive(base,cost)(jnp.zeros((2,1))),1)
    np.testing.assert_allclose(augmented_lagrangian(1.,jnp.array([-1.,2.]),jnp.zeros(2),2.),5.)
    np.testing.assert_array_equal(update_multipliers(jnp.zeros(2),jnp.array([-1.,2.]),2.),[0.,4.])


def test_portfolio_updates_all_proposals_and_esp_selection():
    model,state,x,y=fixture();hedge=initialize_hedge(3)
    proposals=jnp.array([[.2],[.3],[.4]])
    choice,idx,probabilities=hedge_choose(jax.random.key(1),hedge,proposals)
    np.testing.assert_allclose(probabilities,jnp.ones(3)/3)
    updated=hedge_update(hedge,model,state,proposals)
    np.testing.assert_allclose(updated.gains,model.predict(state,proposals).mean)
    choice,idx,scores=entropy_portfolio_choose(proposals,lambda x:x[:,0])
    assert int(idx)==2


def test_heteroscedastic_model_and_learning_curve_kernel():
    x=jnp.linspace(0,1,6)[:,None];y=jnp.sin(5*x[:,0])
    model=HeteroscedasticGP(iterations=2);state=model.fit(x,y)
    latent=model.predict(state,x);observation=model.predict(state,x,observation_noise=True)
    assert np.all(observation.variance>=latent.variance)
    k=learning_curve_kernel()
    p={'spatial':kernels.stationary_parameters(1),'time':{'log_alpha':jnp.log(2.),'log_beta':jnp.log(1.)},'log_curve_amplitude':jnp.array(0.)}
    points=jnp.array([[.2,0.,1.],[.2,0.,3.],[.8,1.,1.]])
    gram=k(p,points,points)
    assert float(jnp.linalg.eigvalsh(gram).min())>-1e-10
    np.testing.assert_allclose(kernels.freeze_thaw(p['time'],jnp.array([[1.]]),jnp.array([[3.]])),[[1/25]])


def test_fantasies_preserve_known_heteroscedastic_training_noise():
    model,state,x,y=fixture()
    known=jnp.array([.001,.04,.2]);pending=jnp.array([[.3],[.7]])
    state=model.fit(x,y,state.params,noise_variance=known)
    states,_=fantasy_states(jax.random.key(4),model,state,x,y,pending,
                            n_fantasies=3,pending_noise=jnp.array([.1,.5]))
    expected=jnp.tile(jnp.concatenate([known,jnp.array([.1,.5])]),(3,1))
    np.testing.assert_allclose(states.noise_variance,expected)


def test_batch_imputation_preserves_noise_and_posterior_incumbent():
    model,state,x,y=fixture()
    known=jnp.array([.001,.04,.2]);state=model.fit(x,y,state.params,noise_variance=known)
    points=jax.jit(lambda key:batch_greedy(key,model,state,x,y,jnp.zeros(1),jnp.ones(1),
        AcquisitionOptimizer('random',n_initial=16),q=2,incumbent='posterior',
        noise_variance=known,pending_noise=jnp.array([.03,.08])))(jax.random.key(8))
    assert points.shape==(2,1)
    assert np.isfinite(points).all()

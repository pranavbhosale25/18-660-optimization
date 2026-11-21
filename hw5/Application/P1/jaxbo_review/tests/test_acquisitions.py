import numpy as np
import pytest
from scipy import stats,integrate
import jax
import jax.numpy as jnp
from jaxbo.distributions import NormalMarginal,StudentMarginal,BetaMarginal,EmpiricalMarginal
from jaxbo.acquisitions import expected_improvement,probability_improvement,posterior_quantile,upper_confidence_bound,student_cdf


@pytest.mark.parametrize("maximize",[True,False])
def test_normal_ei_pi_against_numerical_integral(maximize):
    pred=NormalMarginal(jnp.array([.3]),jnp.array([.49]));best=.1;xi=.02
    sign=1 if maximize else -1;threshold=best+sign*xi
    pdf=lambda x:stats.norm.pdf(x,.3,.7)
    ref=integrate.quad(lambda v:max(sign*(v-threshold),0)*pdf(v),-10,10,points=[threshold])[0]
    np.testing.assert_allclose(expected_improvement(pred,best,maximize=maximize,xi=xi),[ref],rtol=1e-8)
    refpi=1-stats.norm.cdf(threshold,.3,.7) if maximize else stats.norm.cdf(threshold,.3,.7)
    np.testing.assert_allclose(probability_improvement(pred,best,maximize=maximize,xi=xi),[refpi],rtol=1e-10)


def test_deterministic_limits_and_gradients():
    pred=NormalMarginal(jnp.array([1.,2.,3.]),jnp.zeros(3))
    np.testing.assert_array_equal(expected_improvement(pred,2.),[0.,0.,1.])
    np.testing.assert_array_equal(probability_improvement(pred,2.),[0.,0.,1.])
    for f in (expected_improvement,probability_improvement):
        g=jax.grad(lambda var:f(NormalMarginal(jnp.array([1.]),var),1.).sum())(jnp.zeros(1))
        assert np.isfinite(g).all()
    # Adversarial tails and nearly deterministic cases must not produce NaN/negative EI.
    p=NormalMarginal(jnp.array([-100.,0.,100.]),jnp.array([1.,1e-30,1.]))
    assert np.isfinite(expected_improvement(p,0.)).all()
    assert np.all(expected_improvement(p,0.)>=0)


@pytest.mark.parametrize("maximize",[True,False])
def test_student_ei_and_pi_are_student_not_gaussian(maximize):
    pred=StudentMarginal(jnp.array([.3]),jnp.array([.49]),jnp.array(5.))
    sign=1 if maximize else -1;threshold=.1
    dist=stats.t(5,loc=.3,scale=.7)
    interval=(threshold,np.inf) if maximize else (-np.inf,threshold)
    ref=integrate.quad(lambda v:sign*(v-threshold)*dist.pdf(v),*interval)[0]
    np.testing.assert_allclose(expected_improvement(pred,threshold,maximize=maximize),[ref],rtol=1e-7)
    prob=dist.sf(threshold) if maximize else dist.cdf(threshold)
    np.testing.assert_allclose(probability_improvement(pred,threshold,maximize=maximize),[prob],rtol=1e-10)
    np.testing.assert_allclose(jax.grad(lambda z:student_cdf(z,jnp.array(5.)))(jnp.array(0.)),stats.t.pdf(0,5),rtol=1e-10)


@pytest.mark.parametrize("maximize",[True,False])
@pytest.mark.parametrize("threshold",[-.2,.4,1.2])
def test_beta_improvement_support_boundaries(maximize,threshold):
    pred=BetaMarginal(jnp.array([2.]),jnp.array([3.]));sign=1 if maximize else -1
    ref=integrate.quad(lambda v:max(sign*(v-threshold),0)*stats.beta.pdf(v,2,3),0,1,points=[threshold] if 0<threshold<1 else None)[0]
    np.testing.assert_allclose(expected_improvement(pred,threshold,maximize=maximize),[ref],rtol=1e-7,atol=1e-12)


@pytest.mark.parametrize("distribution",["normal","student","beta","empirical"])
def test_quantiles(distribution):
    if distribution=='normal':
        p=NormalMarginal(jnp.array([1.]),jnp.array([4.]));ref=stats.norm.ppf(.95,1,2)
    elif distribution=='student':
        p=StudentMarginal(jnp.array([1.]),jnp.array([4.]),jnp.array(5.));ref=stats.t.ppf(.95,5,loc=1,scale=2)
    elif distribution=='beta':
        p=BetaMarginal(jnp.array([2.]),jnp.array([3.]));ref=stats.beta.ppf(.95,2,3)
    else:
        p=EmpiricalMarginal(jnp.array([[0.],[1.],[2.],[3.]]));ref=np.quantile([0,1,2,3],.95)
    np.testing.assert_allclose(posterior_quantile(p,.95),[ref],rtol=1e-8)


def test_empirical_and_ucb_direction():
    p=EmpiricalMarginal(jnp.array([[0.,2.],[1.,3.],[2.,4.]]))
    np.testing.assert_allclose(expected_improvement(p,1.),[1/3,2.])
    n=NormalMarginal(jnp.array([1.]),jnp.array([4.]))
    np.testing.assert_allclose(upper_confidence_bound(n,maximize=False,beta=3),[5.])

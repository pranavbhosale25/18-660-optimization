"""Composable constrained, cost-sensitive and augmented-Lagrangian utilities."""
from __future__ import annotations
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr
from .linalg import safe_sqrt


def gaussian_feasibility(mean,variance,lower=-jnp.inf,upper=0.0):
    """P(lower <= g <= upper) for Gaussian constraints, including deterministic limits."""
    sd=safe_sqrt(variance); denominator=jnp.where(sd>0,sd,1)
    probability=ndtr((upper-mean)/denominator)-ndtr((lower-mean)/denominator)
    return jnp.where(sd>0,jnp.clip(probability,0,1),((mean>=lower)&(mean<=upper)).astype(mean.dtype))


def constrained(acquisition,probability_feasible,*,has_feasible=True):
    """Eq. 61: weighted EI when base=EI. Independent probabilities may be multiplied.

    Before any feasible observation exists, use pure feasibility instead of an
    invalid objective incumbent. Joint/dependent constraints need a joint PoF.
    """
    return lambda x:(acquisition(x)*probability_feasible(x) if has_feasible else probability_feasible(x))


def independent_feasibility(*probability_functions):
    return lambda x:jnp.prod(jnp.stack([f(x) for f in probability_functions]),axis=0)


def known_constraint(acquisition,feasible):
    return lambda x:jnp.where(feasible(x),acquisition(x),-jnp.inf)


def cost_sensitive(acquisition,log_cost_prediction,*,mode="divide_expected_cost"):
    """GP on log(cost), Section VIII-B; distinguish two non-equivalent expectations.

    divide_expected_cost: a/E[c] = a exp(-mu - var/2), default.
    expected_inverse_cost: a E[1/c] = a exp(-mu + var/2).
    median_cost: a/median(c) = a exp(-mu).
    """
    if mode not in ("divide_expected_cost","expected_inverse_cost","median_cost"):
        raise ValueError("Unknown cost normalization")
    def utility(x):
        p=log_cost_prediction(x)
        correction={"divide_expected_cost":-.5,"expected_inverse_cost":.5,"median_cost":0}[mode]
        return acquisition(x)*jnp.exp(jnp.clip(-p.mean+correction*p.variance,-700,700))
    return utility


def augmented_lagrangian(objective,constraints,multipliers,penalty):
    """Powell-Hestenes-Rockafellar inequality AL for a MINIMIZATION objective, g<=0."""
    shifted=jnp.maximum(0,multipliers+penalty*constraints)
    return objective+jnp.sum((shifted**2-multipliers**2)/(2*penalty),axis=-1)


def update_multipliers(multipliers,constraints,penalty):
    return jnp.maximum(0,multipliers+penalty*constraints)

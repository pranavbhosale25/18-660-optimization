"""Modular native-JAX implementation of the Bayesian optimization review.

No global JAX configuration is changed on import. For GP numerical work, enable
x64 in your application BEFORE creating arrays: jax.config.update('jax_enable_x64', True).
"""
from .api import BayesianOptimizer, BOResult, scan_optimize
from .spaces import Box, MixedSpace, Condition, RandomEmbedding, initial_design, cartesian_grid
from .optim import AcquisitionOptimizer
from .models import (ExactGP, SparseGP, SpectralGP, StudentTProcess, BayesianLinear,
                     BetaBernoulli, GeneralizedLinear, GPClassifier, RandomForest,
                     NeuralLinear, GPParams, default_params)
from .acquisitions import (expected_improvement, probability_improvement,
                           upper_confidence_bound, posterior_quantile, make_acquisition,
                           thompson_on_candidates, monte_carlo_qei)
from .information import entropy_search_mc, predictive_entropy_search_mc
from .hyperparameters import fit_hyperparameters, integrate_hyperparameters, marginalized_acquisition
from .labs import (Action, Observation, TaskSpec, ExperimentSession,
                   BudgetExhausted, PolicyConfig, ConfiguredPolicy,
                   RunResult, make_problem, run_experiment, LAB_NAMES)
from . import kernels, warps

__version__ = "0.2.0"

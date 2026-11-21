"""Five configurable Bayesian-optimization application laboratories.

Importing this package does not change JAX's global precision settings.
"""
from .core import Action, Observation, TaskSpec, Problem, ExperimentSession, BudgetExhausted
from .simulators import make_problem, NAMES as LAB_NAMES
from .policies import PolicyConfig, ConfiguredPolicy
from .runner import run_experiment, RunResult

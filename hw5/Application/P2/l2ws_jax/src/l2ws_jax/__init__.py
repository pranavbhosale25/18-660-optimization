"""Pure Python/JAX teaching package for learned optimization warm starts."""

from .checkpoint import load_checkpoint, save_checkpoint
from .model import apply_model, init_params, initialize
from .problems import PROBLEM_NAMES, make_problem
from .problems.base import Problem
from .reference import ReferenceResult, solve_reference
from .runtime import Runtime, synchronize
from .solver import (
    fixed_point,
    fixed_point_residual,
    gradient_mapping_norm,
    make_baseline_solver,
    make_predictor,
    make_rollout,
    predict,
    rollout,
)
from .training import TrainConfig, TrainResult, train

__version__ = "0.2.0"
__all__ = [
    "Problem",
    "make_problem",
    "PROBLEM_NAMES",
    "init_params",
    "initialize",
    "apply_model",
    "rollout",
    "fixed_point",
    "predict",
    "make_predictor",
    "make_rollout",
    "make_baseline_solver",
    "fixed_point_residual",
    "gradient_mapping_norm",
    "TrainConfig",
    "TrainResult",
    "train",
    "save_checkpoint",
    "load_checkpoint",
    "ReferenceResult",
    "solve_reference",
    "Runtime",
    "synchronize",
]

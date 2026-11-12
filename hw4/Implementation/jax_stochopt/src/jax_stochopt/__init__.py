"""Native JAX stochastic optimization with composable functions and policies."""
from .core import Solver, make_solver
from .objectives import Reduce, compile_objective, mean_reduce, product_reduce, sum_reduce, weighted_sum
from .optimizers import Adam, Muon, NewtonCG, Optimizer, SGD
from ._tree import masked_mean, tree_dot, tree_norm
from .types import BatchStats, Result, ScheduleContext, SolverState, StepInfo, UpdateContext
from .transforms import chain_update_transforms
from . import projections, schedules

__version__ = "0.2.0"

__all__ = [
    "Adam", "BatchStats", "Muon", "NewtonCG", "Optimizer", "Reduce", "Result",
    "SGD", "ScheduleContext", "Solver", "SolverState", "StepInfo", "UpdateContext",
    "chain_update_transforms", "compile_objective", "make_solver", "masked_mean", "mean_reduce",
    "product_reduce", "projections", "schedules", "sum_reduce", "tree_dot",
    "tree_norm", "weighted_sum",
]


"""Actuation-only shooting with registered algebraic tasks and real IPOPT."""
from .geometry import ActuatorMap, Parts
from .layout import Layout, RolloutBackend, Trajectory
from .problem import ShootingProblem, Variables
from .solvers import SolveResult

__all__ = ["ActuatorMap", "Parts", "Layout", "RolloutBackend", "Trajectory", "ShootingProblem", "Variables", "SolveResult"]
__version__ = "0.2.0"

from .pyomo_task import sumsqr, dot
__all__ += ["sumsqr", "dot"]

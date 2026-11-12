"""Differentiable probabilistic geometry compensation.

64-bit arithmetic is required for probability tails and small supporting cuts.
The application should import this package before creating JAX arrays.
"""

from jax import config

config.update("jax_enable_x64", True)

from .problem import AffineGaussianBox
from .probability import GaussianBoxOracle
from .veinott import SolverOptions, VeinottResult, solve_veinott

__all__ = [
    "AffineGaussianBox",
    "GaussianBoxOracle",
    "SolverOptions",
    "VeinottResult",
    "solve_veinott",
]

__version__ = "0.2.0"

"""Optional stochastic-gradient extensions composed through generic hooks.

Importing these modules does not change any base optimizer or solver default.
"""
from .sources import EnergyGradient, ScoreGradient, compile_score
from .langevin import (
    LangevinInfo,
    LangevinNoise,
    make_energy_langevin,
    make_score_langevin,
)

__all__ = [
    "EnergyGradient",
    "ScoreGradient",
    "compile_score",
    "LangevinInfo",
    "LangevinNoise",
    "make_energy_langevin",
    "make_score_langevin",
]

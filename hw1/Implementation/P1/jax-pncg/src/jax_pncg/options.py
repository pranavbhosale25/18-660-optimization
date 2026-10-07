from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


GradientNorm = Literal["inf", "2"]
BetaMethod = Literal["pr", "pr+"]


@dataclass(frozen=True)
class PNCGOptions:
    """Static options for preconditioned Polak--Ribiere nonlinear CG.

    The options are captured by :func:`jax_pncg.make_solver`; changing one of
    them creates a separately compiled solver, which is normally what is
    wanted because loop bounds and history sizes are compile-time constants.
    """

    maxiter: int = 1000
    gtol: float = 1.0e-6
    rtol: float = 0.0
    norm: GradientNorm = "inf"

    # Nocedal--Wright Polak--Ribiere or the PR+ truncation in eq. (5.45).
    beta_method: BetaMethod = "pr+"

    # Strong-Wolfe line search.  c2=0.1 is the value suggested for nonlinear
    # CG in Numerical Optimization; in particular it is below 1/2.
    c1: float = 1.0e-4
    c2: float = 0.1
    initial_step: float = 1.0
    max_step: float = 1.0e6
    step_growth: float = 2.0
    max_line_search_iterations: int = 20
    max_zoom_iterations: int = 40
    zoom_tolerance: float = 1.0e-12

    # Restarts.  A zero value disables the corresponding strategy.
    restart_interval: int = 0
    orthogonality_restart: float = 0.0
    descent_restart: bool = True
    descent_tolerance: float = 0.0
    retry_on_line_search_failure: bool = True

    # Stores only scalar diagnostics, not the full iterate trajectory.
    record_history: bool = True

    def __post_init__(self) -> None:
        if self.maxiter < 0:
            raise ValueError("maxiter must be nonnegative")
        if self.gtol < 0.0 or self.rtol < 0.0:
            raise ValueError("gtol and rtol must be nonnegative")
        if self.norm not in ("inf", "2"):
            raise ValueError("norm must be 'inf' or '2'")
        if self.beta_method not in ("pr", "pr+"):
            raise ValueError("beta_method must be 'pr' or 'pr+'")
        if not (0.0 < self.c1 < self.c2 < 1.0):
            raise ValueError("line-search constants must satisfy 0 < c1 < c2 < 1")
        if self.initial_step <= 0.0:
            raise ValueError("initial_step must be positive")
        if self.max_step <= 0.0 or self.initial_step > self.max_step:
            raise ValueError("require 0 < initial_step <= max_step")
        if self.step_growth <= 1.0:
            raise ValueError("step_growth must exceed one")
        if self.max_line_search_iterations <= 0:
            raise ValueError("max_line_search_iterations must be positive")
        if self.max_zoom_iterations <= 0:
            raise ValueError("max_zoom_iterations must be positive")
        if self.zoom_tolerance < 0.0:
            raise ValueError("zoom_tolerance must be nonnegative")
        if self.restart_interval < 0:
            raise ValueError("restart_interval must be nonnegative")
        if not (0.0 <= self.orthogonality_restart < 1.0):
            raise ValueError("orthogonality_restart must lie in [0, 1)")
        if self.descent_tolerance < 0.0:
            raise ValueError("descent_tolerance must be nonnegative")

"""Pure-JAX sequential quadratic programming."""

from .merit import violation
from .qp import QPOptions
from .runtime import Runtime, configure
from .solver import HISTORY_COLUMNS, KKT, SQP, STATUS_MESSAGES, Options, Result, Status

__all__ = [
    "configure",
    "Runtime",
    "SQP",
    "Options",
    "Result",
    "KKT",
    "Status",
    "STATUS_MESSAGES",
    "HISTORY_COLUMNS",
    "QPOptions",
    "violation",
]
__version__ = "0.1.0"

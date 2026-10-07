from __future__ import annotations

import pytest

from jax_pncg import PNCGOptions


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maxiter": -1},
        {"gtol": -1.0},
        {"norm": "bad"},
        {"beta_method": "fr"},
        {"c1": 0.2, "c2": 0.1},
        {"initial_step": 2.0, "max_step": 1.0},
        {"step_growth": 1.0},
        {"restart_interval": -1},
        {"orthogonality_restart": 1.0},
    ],
)
def test_invalid_options_raise(kwargs) -> None:
    with pytest.raises(ValueError):
        PNCGOptions(**kwargs)

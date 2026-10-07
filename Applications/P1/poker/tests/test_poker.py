"""Public Poker checks; the private grading suite is more extensive."""

import jax.numpy as jnp
import numpy as np

from games import solve_matrix_game
from poker import PokerRules, build_poker


def test_supplied_poker_environment_is_zero_sum():
    data = build_poker(PokerRules(bids=(0, 1)))
    np.testing.assert_allclose(
        data.kernels,
        -data.kernels.transpose(0, 2, 1),
        atol=1e-12,
    )


def test_matrix_game_has_valid_mixed_solution(oracle):
    result = solve_matrix_game(jnp.array([[1.0, -1.0], [-1.0, 1.0]]), oracle)
    assert np.all(np.asarray(result.status) == 1)
    np.testing.assert_allclose(result.row, [[0.5, 0.5]], atol=1e-6)
    assert float(result.gap) < 1e-6


def test_negative_game_values_are_allowed(oracle):
    result = solve_matrix_game(jnp.full((2, 2), -3.0), oracle)
    assert abs(float(result.payoff) + 3.0) < 1e-6

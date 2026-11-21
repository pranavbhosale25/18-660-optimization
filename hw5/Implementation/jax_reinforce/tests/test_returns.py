import numpy as np
import pytest
import jax
import jax.numpy as jnp

from reinforce_jax import GAEEstimator, ReturnConfig, n_step_returns

from .helpers import batch


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_finite_two_step_returns_are_jittable(dtype):
    data = batch([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    next_values = jnp.asarray([[10.0], [20.0], [30.0], [40.0]], dtype)
    config = ReturnConfig(gamma=0.5, horizon=2, missing="drop")

    result = jax.jit(lambda trajectory, values: n_step_returns(
        trajectory, values, config))(data, next_values)

    np.testing.assert_allclose(
        result.returns[:, 0], [7.0, 11.0, 15.0, 24.0],
        rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(
        result.mask[:, 0], [True, True, True, False])
    np.testing.assert_array_equal(result.steps[:, 0], [2, 2, 2, 1])
    assert result.returns.dtype == dtype


def test_full_returns_stop_at_a_true_terminal_state():
    data = batch(
        [1.0, 2.0, 3.0],
        terminated=[False, False, True],
    )
    next_values = jnp.full_like(data.rewards, 1000.0)

    result = n_step_returns(
        data, next_values, ReturnConfig(gamma=0.5, horizon=None, missing="drop"))

    np.testing.assert_allclose(result.returns[:, 0], [2.75, 3.5, 3.0])
    np.testing.assert_array_equal(result.mask[:, 0], [True, True, True])
    np.testing.assert_array_equal(result.steps[:, 0], [3, 2, 1])


def test_terminal_and_truncation_have_different_tail_behavior():
    rewards = [[1.0, 1.0], [2.0, 2.0], [100.0, 100.0]]
    data = batch(
        rewards,
        terminated=[[False, False], [True, False], [False, False]],
        truncated=[[False, False], [False, True], [False, False]],
    )
    next_values = jnp.asarray(
        [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    config = ReturnConfig(gamma=0.5, horizon=3, missing="bootstrap")

    result = jax.jit(lambda trajectory, values: n_step_returns(
        trajectory, values, config))(data, next_values)

    # The terminal lane has no tail. The truncated lane uses the value at the
    # actual post-step observation, and neither lane crosses into the reset.
    np.testing.assert_allclose(result.returns[:2, 0], [2.0, 2.0])
    np.testing.assert_allclose(result.returns[:2, 1], [7.0, 12.0])

    dropped = n_step_returns(
        data, next_values,
        ReturnConfig(gamma=0.5, horizon=3, missing="drop"))
    np.testing.assert_array_equal(dropped.mask[:2, 0], [True, True])
    np.testing.assert_array_equal(dropped.mask[:2, 1], [False, False])


def test_truncation_at_exact_finite_horizon_is_complete():
    data = batch(
        [1.0, 2.0],
        truncated=[False, True],
    )
    next_values = jnp.asarray([[10.0], [20.0]])

    result = n_step_returns(
        data,
        next_values,
        ReturnConfig(gamma=0.5, horizon=2, missing="drop"),
    )

    np.testing.assert_allclose(result.returns[:, 0], [7.0, 12.0])
    np.testing.assert_array_equal(result.mask[:, 0], [True, False])
    np.testing.assert_array_equal(result.steps[:, 0], [2, 1])


def test_invalid_and_incomplete_entries_are_masked_in_drop_mode():
    data = batch(
        [1.0, 2.0, 3.0, 4.0],
        valid=[True, True, False, False],
    )
    result = n_step_returns(
        data,
        jnp.asarray([[10.0], [20.0], [30.0], [40.0]]),
        ReturnConfig(gamma=0.5, horizon=2, missing="drop"),
    )

    np.testing.assert_array_equal(
        result.mask[:, 0], [True, False, False, False])
    np.testing.assert_array_equal(result.steps[:, 0], [2, 1, 0, 0])
    np.testing.assert_allclose(result.returns[:, 0], [7.0, 12.0, 0.0, 0.0])


def test_horizon_beyond_stored_axis_keeps_numeric_targets_but_drops_entries():
    data = batch([1.0, 2.0])
    next_values = jnp.asarray([[10.0], [20.0]])

    result = n_step_returns(
        data,
        next_values,
        ReturnConfig(gamma=0.5, horizon=5, missing="drop"),
    )

    np.testing.assert_allclose(result.returns[:, 0], [7.0, 12.0])
    np.testing.assert_array_equal(result.mask[:, 0], [False, False])
    np.testing.assert_array_equal(result.steps[:, 0], [2, 1])


@pytest.mark.parametrize(
    ("gamma", "expected"),
    [(0.0, [1.0, 2.0, 3.0]), (1.0, [6.0, 5.0, 3.0])],
)
def test_discount_endpoints(gamma, expected):
    data = batch([1.0, 2.0, 3.0], terminated=[False, False, True])

    result = n_step_returns(
        data,
        jnp.zeros_like(data.rewards),
        ReturnConfig(gamma=gamma, horizon=None),
    )

    np.testing.assert_allclose(result.returns[:, 0], expected)
    np.testing.assert_array_equal(result.steps[:, 0], [3, 2, 1])


def test_returns_remain_differentiable_in_rewards_and_bootstrap_values():
    data = batch([1.0, 2.0, 3.0])
    config = ReturnConfig(gamma=0.5, horizon=2)

    def first_target(rewards, next_values):
        changed = data._replace(rewards=rewards)
        return n_step_returns(changed, next_values, config).returns[0, 0]

    reward_gradient, value_gradient = jax.grad(
        first_target, argnums=(0, 1))(
            data.rewards, jnp.zeros_like(data.rewards))

    np.testing.assert_allclose(reward_gradient[:, 0], [1.0, 0.5, 0.0])
    np.testing.assert_allclose(value_gradient[:, 0], [0.0, 0.25, 0.0])


def test_full_returns_remain_differentiable():
    data = batch([1.0, 2.0])
    config = ReturnConfig(gamma=0.5, horizon=None)

    def first_target(rewards, next_values):
        changed = data._replace(rewards=rewards)
        return n_step_returns(changed, next_values, config).returns[0, 0]

    reward_gradient, value_gradient = jax.grad(
        first_target,
        argnums=(0, 1),
    )(data.rewards, jnp.zeros_like(data.rewards))

    np.testing.assert_allclose(reward_gradient[:, 0], [1.0, 0.5])
    np.testing.assert_allclose(value_gradient[:, 0], [0.0, 0.25])


def test_termination_takes_precedence_when_both_flags_are_true():
    data = batch([3.0], terminated=[True], truncated=[True])
    result = n_step_returns(
        data,
        jnp.asarray([[100.0]]),
        ReturnConfig(gamma=0.9, horizon=None, missing="drop"),
    )

    np.testing.assert_allclose(result.returns, [[3.0]])
    np.testing.assert_array_equal(result.mask, [[True]])


def test_supplied_gae_estimator_remains_compatible_with_return_targets():
    data = batch(
        [1.0, 2.0, 3.0, 4.0],
        terminated=[False, True, False, False],
    )
    state_values = jnp.asarray([[0.1], [0.2], [0.3], [0.4]])
    next_values = jnp.asarray([[0.2], [99.0], [0.4], [0.5]])

    gae = jax.jit(GAEEstimator(gamma=0.9, lam=1.0))(
        data, state_values, next_values)
    full = n_step_returns(data, next_values, ReturnConfig(gamma=0.9))

    np.testing.assert_allclose(gae.targets, full.returns)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": -0.1},
        {"gamma": 1.1},
        {"horizon": 0},
        {"horizon": -1},
        {"missing": "unsupported"},
    ],
)
def test_return_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        ReturnConfig(**kwargs)

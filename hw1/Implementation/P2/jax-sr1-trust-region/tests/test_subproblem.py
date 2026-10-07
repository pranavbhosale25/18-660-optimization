from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from trsr1.subproblem import solve_quadratic_subproblem

jax.config.update("jax_enable_x64", True)


def _solve(gradient, matrix, radius):
    function = jax.jit(solve_quadratic_subproblem)
    return function(
        jnp.asarray(gradient, dtype=jnp.float64),
        jnp.asarray(matrix, dtype=jnp.float64),
        jnp.asarray(radius, dtype=jnp.float64),
    )


def _model_reduction(step, gradient, matrix) -> float:
    step = np.asarray(step)
    gradient = np.asarray(gradient)
    matrix = np.asarray(matrix)
    return float(-(step @ gradient + 0.5 * step @ matrix @ step))


def test_positive_definite_interior_step_is_full_quasi_newton_step() -> None:
    matrix = np.asarray([[4.0, 1.0], [1.0, 3.0]])
    gradient = np.asarray([1.0, -2.0])
    result = _solve(gradient, matrix, 10.0)

    expected = -np.linalg.solve(matrix, gradient)
    np.testing.assert_allclose(result.step, expected, atol=1.0e-12)
    assert not bool(result.hit_boundary)
    assert bool(result.valid)


def test_positive_definite_boundary_step_lies_on_dogleg_path() -> None:
    matrix = np.diag([4.0, 1.0])
    gradient = np.asarray([1.0, 1.0])
    radius = 0.8
    result = _solve(gradient, matrix, radius)

    cauchy = -(gradient @ gradient) / (gradient @ matrix @ gradient) * gradient
    full_step = -np.linalg.solve(matrix, gradient)
    segment = full_step - cauchy
    fraction = np.dot(np.asarray(result.step) - cauchy, segment) / np.dot(
        segment, segment
    )

    np.testing.assert_allclose(np.linalg.norm(result.step), radius, atol=1.0e-12)
    assert 0.0 <= fraction <= 1.0
    np.testing.assert_allclose(
        result.step,
        cauchy + fraction * segment,
        atol=1.0e-12,
    )
    assert _model_reduction(result.step, gradient, matrix) >= _model_reduction(
        cauchy, gradient, matrix
    )
    assert bool(result.hit_boundary)
    assert bool(result.valid)


def test_indefinite_model_falls_back_to_boundary_cauchy_point() -> None:
    matrix = np.diag([-2.0, 1.0])
    gradient = np.asarray([1.0, 1.0])
    radius = 0.5
    result = _solve(gradient, matrix, radius)

    expected = -radius * gradient / np.linalg.norm(gradient)
    np.testing.assert_allclose(result.step, expected, atol=1.0e-12)
    assert bool(result.hit_boundary)
    assert bool(result.valid)


def test_indefinite_model_can_use_an_interior_cauchy_point() -> None:
    matrix = np.diag([-1.0, 4.0])
    gradient = np.asarray([0.1, 1.0])
    radius = 1.0
    result = _solve(gradient, matrix, radius)

    scale = (gradient @ gradient) / (gradient @ matrix @ gradient)
    expected = -scale * gradient
    np.testing.assert_allclose(result.step, expected, atol=1.0e-12)
    assert not bool(result.hit_boundary)
    assert bool(result.valid)


def test_zero_gradient_returns_zero_step() -> None:
    result = _solve(np.zeros(2), np.diag([-2.0, 1.0]), 1.0)

    np.testing.assert_array_equal(result.step, np.zeros(2))
    assert float(result.predicted_reduction) == 0.0
    assert not bool(result.hit_boundary)
    assert bool(result.valid)


def test_invalid_radius_is_reported_without_host_control_flow() -> None:
    result = _solve(np.ones(2), np.eye(2), -1.0)
    assert not bool(result.valid)


def test_subproblem_step_vectorizes_over_independent_models() -> None:
    gradients = jnp.asarray([[1.0, -2.0], [1.0, 1.0]], dtype=jnp.float64)
    matrices = jnp.asarray(
        [
            [[4.0, 1.0], [1.0, 3.0]],
            [[-2.0, 0.0], [0.0, 1.0]],
        ],
        dtype=jnp.float64,
    )
    radii = jnp.asarray([10.0, 0.5], dtype=jnp.float64)
    mapped = jax.jit(jax.vmap(solve_quadratic_subproblem))
    result = mapped(gradients, matrices, radii)

    assert result.step.shape == (2, 2)
    assert np.all(np.asarray(result.valid))
    np.testing.assert_array_equal(
        result.hit_boundary,
        np.asarray([False, True]),
    )

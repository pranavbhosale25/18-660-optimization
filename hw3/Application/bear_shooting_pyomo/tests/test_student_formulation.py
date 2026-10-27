"""Check the student formulation on an analytic fixture, not a bear surrogate."""

import numpy as np
import pytest

from bear_shooting import Layout, ShootingProblem
from bear_shooting.ipopt_c import find_library
from examples.bear_run import register_constraints, register_objectives, solve_with_ipopt


class LinearObservationFixture:
    """A cheap exact observation map for checking Pyomo algebra and IPOPT."""

    def __init__(self, horizon):
        self.layout = Layout(horizon, ("a", "b", "c"), ("com",), 0.1)

    def rollout(self, controls):
        observations = np.zeros(self.layout.observation_shape)
        observations[:, 0, 0] = 0.2
        observations[-1, 0, 0] += np.sum(controls)
        return observations.ravel()

    def vjp(self, seed):
        terminal_x = np.asarray(seed).reshape(self.layout.observation_shape)[-1, 0, 0]
        return np.full(self.layout.control_shape, terminal_x)


@pytest.mark.parametrize("horizon", [1, 3])
def test_objective_values_and_gradients_against_numpy_and_finite_differences(horizon):
    simulator = LinearObservationFixture(horizon)
    problem = ShootingProblem(simulator)
    center = np.linspace(-0.1, 0.1, simulator.layout.n_controls).reshape(-1, 3)
    controls = center + np.linspace(0.01, 0.04, center.size).reshape(center.shape)
    radius = 0.2
    register_objectives(problem, control_center=center, search_radius=radius)
    register_constraints(
        problem, start_com=np.array([0.2, 0.0, 0.0]),
        target_distance=0.3, target_window=0.25,
    )

    def numpy_objective(u):
        correction = (u.reshape(center.shape) - center) / radius
        value = np.mean(correction**2)
        if horizon > 1:
            value += 0.05 * np.mean(np.diff(correction, axis=0)**2)
        return value

    values = problem.oracle.values(controls.ravel())
    np.testing.assert_allclose(values[0], numpy_objective(controls), atol=1e-14)
    np.testing.assert_allclose(values[1], np.sum(controls) / 0.3, atol=1e-14)
    np.testing.assert_allclose(problem.compile().lower, [1.0])
    np.testing.assert_allclose(problem.compile().upper, [0.55 / 0.3])
    terms = [term.name for term in problem.compile().objectives]
    assert terms == (["gait_deviation", "correction_smoothness"] if horizon > 1 else ["gait_deviation"])

    step = 1e-6
    eye = np.eye(controls.size)
    numerical = np.array([
        (numpy_objective(controls.ravel() + step*d)
         - numpy_objective(controls.ravel() - step*d)) / (2*step)
        for d in eye
    ])
    jacobian = problem.oracle.total_jacobian(controls.ravel())
    np.testing.assert_allclose(jacobian[0], numerical, atol=1e-9)
    np.testing.assert_allclose(jacobian[1], 1 / 0.3)


@pytest.mark.ipopt
def test_student_solver_reaches_known_minimum_through_native_ipopt():
    try:
        find_library()
    except ImportError:
        pytest.skip("Native IPOPT library is not installed.")
    simulator = LinearObservationFixture(2)
    problem = ShootingProblem(simulator)
    center = np.zeros(simulator.layout.control_shape)
    register_objectives(problem, control_center=center, search_radius=0.5)
    register_constraints(
        problem, start_com=np.array([0.2, 0.0, 0.0]),
        target_distance=0.3, target_window=0.25,
    )
    result = solve_with_ipopt(problem, center, max_iter=60, device="cpu")
    assert result.success
    assert result.engine == "ipopt"
    # Six identical controls give the least squared effort for sum(u) >= .3.
    np.testing.assert_allclose(result.controls, 0.05, atol=5e-4)
    np.testing.assert_allclose(result.objective, 0.01, atol=2e-4)
    assert result.history["iterations"]
    last = result.history["iterations"][-1]
    if "controls" in last:  # Native IPOPT >= 3.14 supplies accepted iterates.
        np.testing.assert_allclose(last["controls"], result.controls.ravel(), atol=1e-10)
        np.testing.assert_allclose(
            last["physical_constraint_values"], np.sum(result.controls), atol=1e-10,
        )
        assert last["task_violation_inf"] <= 5e-4

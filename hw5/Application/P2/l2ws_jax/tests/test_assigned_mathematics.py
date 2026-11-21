"""Independent checks of the assigned formulas and curvature bounds."""

import jax
import numpy as np
import pytest
from l2ws_jax.problems.kinematics import InverseKinematics
from l2ws_jax.problems.phase import PhaseRetrieval


def test_kinematics_matches_cartesian_formula():
    problem = InverseKinematics(regularization=0.17, lengths=(0.6, 1.1, 0.4))
    q = problem.array([0.2, -0.5, 0.8])
    context = problem.array([0.7, -0.1, -0.2, 0.4, 0.3])
    angles = np.cumsum(np.asarray(q))
    position = np.array(
        [
            np.dot(np.asarray(problem.lengths), np.cos(angles)),
            np.dot(np.asarray(problem.lengths), np.sin(angles)),
        ]
    )
    expected = (
        np.linalg.norm(position - np.asarray(context[:2])) ** 2
        + 0.17 * np.linalg.norm(np.asarray(q - context[2:])) ** 2
    ) / 2
    np.testing.assert_allclose(problem.objective_one(q, context), expected)


def test_phase_matches_quartic_formula_and_analytic_gradient():
    problem = PhaseRetrieval(dimension=4, measurements=12, radius=0.9)
    y = problem.array([0.2, -0.6, 0.3, 0.7])
    context = problem.array(np.linspace(0.01, 1.2, 12))
    matrix = np.asarray(problem.A)
    z = matrix @ np.asarray(y)
    residual = z * z - np.asarray(context)
    expected = np.dot(residual, residual) / 48
    gradient = matrix.T @ (residual * z) / 12
    np.testing.assert_allclose(problem.objective_one(y, context), expected)
    np.testing.assert_allclose(jax.grad(problem.objective_one)(y, context), gradient)


@pytest.mark.parametrize("name", ["kinematics", "phase"])
def test_hessian_bound_at_box_corners_and_interior(name):
    problem = InverseKinematics() if name == "kinematics" else PhaseRetrieval()
    rng = np.random.default_rng(83)
    contexts = problem.sample(4, problem.key(83), shift=True)
    points = rng.uniform(
        np.asarray(problem.lower),
        np.asarray(problem.upper),
        size=(20, problem.decision_dim),
    )
    points = np.concatenate(
        (points, np.asarray(problem.lower)[None], np.asarray(problem.upper)[None])
    )
    hessian = jax.jit(jax.hessian(problem.objective_one))
    for context in contexts:
        bound = 1 / float(problem.step_size_one(context))
        for point in points:
            eigenvalues = np.linalg.eigvalsh(hessian(problem.array(point), context))
            assert np.max(np.abs(eigenvalues)) <= bound * (1 + 1e-12)

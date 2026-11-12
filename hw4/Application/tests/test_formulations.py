"""Compare assembled matrices against the assignment's physical formulas."""
import numpy as np
import pytest
from geocomp.processes import formulate_dlp_problem, formulate_cnc_problem
from geocomp.embedded import build_embedded_instance


@pytest.mark.parametrize("formulate,upper", [(formulate_dlp_problem, 4.), (formulate_cnc_problem, 0.)])
def test_surface_objective_and_response(formulate, upper):
    rng = np.random.default_rng(12)
    weights = rng.dirichlet(np.ones(3), size=7)
    areas = rng.dirichlet(np.ones(7))
    response = rng.normal(size=(3, 3))
    bias = np.array([.1, .2, .3])
    p = formulate(name="test", bias_mm=bias, command_response=response,
                  covariance_mm2=np.eye(3) * .01, tolerance_mm=.2,
                  basis_weights=weights, vertex_area_weights=areas, alpha=.9)
    x = rng.normal(size=3)
    expected = .5 * (.85 * np.sum(areas * (weights @ x)**2) + .15 / 3 * (x @ x))
    np.testing.assert_allclose(p.objective(x), expected)
    np.testing.assert_allclose(p.mean(x), bias + response @ (.2 * x))
    np.testing.assert_array_equal(p.bounds, np.tile([-4., upper], (3, 1)))
    np.testing.assert_array_equal(p.lower, np.full(3, -.2))
    assert p.alpha == .9 and p.name == "test"


@pytest.mark.parametrize("controls", [2, 3, 5, 6])
def test_embedded_objective_scaling_and_bath_constraints(controls):
    instance = build_embedded_instance(n_controls=controls, n_sections=17)
    p, basis, par = instance.problem, instance.basis, instance.parameters
    x = np.random.default_rng(4).normal(size=4 * controls)
    X = x.reshape(4, controls).T
    direct = (.85 / 17 * np.sum((basis.weights @ X)**2)
              + .08 / controls * np.sum(X**2)
              + .07 / controls * np.sum(np.diff(X, n=2, axis=0)**2)) / 8
    np.testing.assert_allclose(p.objective(x), direct)
    scales = np.r_[np.full(3 * controls, par.centerline_budget_mm / np.sqrt(3)),
                   np.full(controls, par.radius_budget_mm)]
    np.testing.assert_allclose(p.upper, scales)
    np.testing.assert_allclose(p.lower, -scales)
    np.testing.assert_allclose(instance.command_coefficients(x), scales * x)
    center = basis.centerline + par.centerline_budget_mm / np.sqrt(3) * basis.weights @ X[:, :3]
    lower = instance.bath_bounds_mm[0] + 1.45 * par.radius_mm
    upper = instance.bath_bounds_mm[1] - 1.45 * par.radius_mm
    expected_slack = np.r_[(upper - center).T.ravel(), (center - lower).T.ravel()]
    np.testing.assert_allclose(p.b_ub - p.A_ub @ x, expected_slack)
    np.testing.assert_allclose(p.bounds[-controls:, 1], .45 * par.radius_mm / par.radius_budget_mm)
    assert p.linear_feasible(np.zeros(4 * controls))

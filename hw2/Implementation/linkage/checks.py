"""Local verification harness; the distributed folder did not include checks.py.

Run: python checks.py --part all
These checks supplement, rather than claim to replace, instructor/hidden tests.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import jax.numpy as jnp

import linkage_core as core
import optimize


def finite_difference(function, lengths, epsilon=1e-6):
    """Central differences re-solve the linkage for each perturbed length."""
    columns = []
    for index in range(4):
        delta = jnp.zeros(4).at[index].set(epsilon)
        columns.append((np.asarray(function(lengths + delta))
                        - np.asarray(function(lengths - delta))) / (2 * epsilon))
    return np.stack(columns, axis=-1)


def check_closure(example):
    mech = core.mechanism_with_lengths(example.template_mech, example.initial_lengths)
    angle = example.driver_angles[0]
    pose = core.driver_pose_from_angle(mech, angle)
    closed = core.closed_form_initial_guess(mech, angle)
    np.testing.assert_allclose(core.closure_energy(closed, mech, pose), 0, atol=1e-24)
    # Translating only the coupler opens its two joints by the same displacement.
    shift = jnp.array([.17, -.11])
    opened = closed.at[:2].add(shift)
    np.testing.assert_allclose(core.closure_energy(opened, mech, pose),
                               jnp.dot(shift, shift), rtol=1e-12, atol=1e-14)
    simulation = core.checked_simulation_from_lengths(
        example.initial_lengths, example.template_mech, example.driver_angles)
    diagnostics = core.simulation_diagnostics(
        example.initial_lengths, example.template_mech, example.driver_angles,
        simulation, include_condition=True)
    assert diagnostics['max_closure_norm'] < 1e-7
    assert diagnostics['max_stationarity_norm'] < 1e-7
    return diagnostics


def check_sensitivities(example):
    lengths = example.initial_lengths
    def solved_states(ell):
        return core.checked_simulation_from_lengths(
            ell, example.template_mech, example.driver_angles).state_trajectory
    states = solved_states(lengths)
    sensitivities = np.asarray(optimize.batched_state_sensitivities(
        lengths, example.template_mech, example.driver_angles, states))
    assert sensitivities.shape == (len(example.driver_angles), 6, 4)
    numerical = finite_difference(solved_states, lengths)
    np.testing.assert_allclose(sensitivities, numerical, rtol=2e-5, atol=2e-6)
    # Damped sensitivities should satisfy the regularized stationarity system.
    damping = 1e-5
    regularized = optimize.batched_state_sensitivities(
        lengths, example.template_mech, example.driver_angles, states, damping)
    residuals = []
    for t in (0, len(states)//2, len(states)-1):
        H = core.frame_hessian_from_lengths(states[t], lengths,
                    example.template_mech, example.driver_angles[t])
        B = core.frame_mixed_from_lengths(states[t], lengths,
                    example.template_mech, example.driver_angles[t])
        residuals.append(np.max(np.abs((.5*(H+H.T)+damping*jnp.eye(6)) @ regularized[t]+B)))
    assert max(residuals) < 1e-10
    return {'max_finite_difference_error': float(np.max(np.abs(sensitivities-numerical))),
            'max_damped_system_residual': float(max(residuals))}


def check_gradient(example):
    errors = []
    direct_norm = 0.0
    # The endpoint case also exercises explicit dependence on the coupler length.
    specifications = [example.point_spec,
        core.make_point_on_link_spec(2, anchor='right_endpoint', offset_xyz=(.35,.02,0))]
    for spec in specifications:
        def solved_loss(ell):
            simulation = core.checked_simulation_from_lengths(
                ell, example.template_mech, example.driver_angles)
            return core.curve_loss_from_states_and_lengths(simulation.state_trajectory,
                ell, example.template_mech, example.driver_angles, spec, example.target_curve)
        result = optimize.design_objective_and_gradient(example.initial_lengths,
            example.template_mech, example.driver_angles, spec, example.target_curve)
        numerical = finite_difference(solved_loss, example.initial_lengths)
        np.testing.assert_allclose(result.gradient_lengths, numerical, rtol=2e-5, atol=2e-6)
        errors.append(float(np.max(np.abs(np.asarray(result.gradient_lengths)-numerical))))
        _, (_, direct) = core.curve_loss_value_and_grads(result.simulation.state_trajectory,
            example.initial_lengths, example.template_mech, example.driver_angles,
            spec, example.target_curve)
        if spec is specifications[-1]:
            direct_norm = float(jnp.linalg.norm(direct))
    assert direct_norm > 1e-6, 'The endpoint case must exercise the direct length derivative.'
    return {'max_finite_difference_error': max(errors), 'endpoint_direct_gradient_norm': direct_norm}


def check_update(example):
    ell = jnp.array([.0005, .02, 2., 3.])
    gradient = jnp.array([100., 100., 2., -1.])
    mask = jnp.array([False, True, True, False])
    actual = optimize.gradient_descent_update(ell, gradient, .1, mask, .001)
    np.testing.assert_allclose(actual, [.0005, .001, 1.8, 3.], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(np.asarray(actual)[[0,3]], np.asarray(ell)[[0,3]])
    np.testing.assert_allclose(optimize.gradient_descent_update(ell, gradient, .1),
                               [.001, .001, 1.8, 3.1], rtol=0, atol=1e-15)
    for steps in (0, 2):
        history = optimize.optimize_link_lengths(example.initial_lengths,
            example.template_mech, example.driver_angles, example.point_spec,
            example.target_curve, num_steps=steps, design_mask=[False,True,True,True])
        assert history.length_history.shape == (steps+1, 4)
        assert history.loss_history.shape == (steps+1,)
        assert history.gradient_history.shape == (steps+1, 4)
        np.testing.assert_array_equal(history.length_history[:,0],
                                     np.full(steps+1, float(example.initial_lengths[0])))
        final = optimize.design_objective_and_gradient(history.length_history[-1],
            example.template_mech, example.driver_angles, example.point_spec, example.target_curve)
        np.testing.assert_allclose(history.loss_history[-1], final.loss, rtol=1e-12)
        np.testing.assert_allclose(history.gradient_history[-1], final.gradient_lengths, rtol=1e-12)
        np.testing.assert_allclose(history.best_loss, np.min(history.loss_history), rtol=1e-12)
        np.testing.assert_array_equal(history.best_lengths,
                                     history.length_history[int(np.argmin(history.loss_history))])
    return {'masked_projection': 'passed', 'zero_and_two_step_histories': 'passed'}


def main():
    checks = {'closure': check_closure, 'sensitivities': check_sensitivities,
              'gradient': check_gradient, 'update': check_update}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=['all', *checks], default='all')
    args = parser.parse_args()
    example = core.make_on_link_coupler_synthesis_example()
    results = {}
    for name, check in checks.items():
        if args.part in ('all', name):
            results[name] = check(example)
            print(f'PASS: {name}', flush=True)
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

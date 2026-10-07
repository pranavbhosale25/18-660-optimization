"""Public Physics checks; the private grading suite is more extensive."""

import jax.numpy as jnp
import numpy as np

from physics import resolve_contact, step, time_to_collision
from worlds import make_ball, make_world


def test_isolated_frictionless_impact(oracle):
    world = make_world([[0, 0, 1]], [0], friction=0, restitution=0.5)
    ball = make_ball([0, 0, 0], [1, 0, -2])
    result = resolve_contact(world, ball, jnp.array([True]), oracle)
    assert bool(result.converged) and int(result.qp_solves) == 1
    np.testing.assert_allclose(result.u, [1, 0, 1], atol=2e-6)


def test_prescribed_friction_budget(oracle):
    world = make_world([[0, 0, 1]], [0], friction=0, restitution=0.5)
    ball = make_ball([0, 0, 0], [4, 0, -2])
    result = resolve_contact(
        world,
        ball,
        jnp.array([True]),
        oracle,
        budget=jnp.array([0.7]),
    )
    assert bool(result.converged)
    np.testing.assert_allclose(result.u, [3.3, 0, 1], atol=2e-6)


def test_collision_time_for_approaching_and_separating_point():
    world = make_world([[0, 0, 1]], [0])
    ball = make_ball([0, 0, 2], [1, 0, -4])
    time, hit = time_to_collision(world, ball, 1.0, 1e-7)
    np.testing.assert_allclose(time, 0.5)
    assert bool(hit)

    time, hit = time_to_collision(
        world,
        ball._replace(velocity=-ball.velocity),
        1.0,
        1e-7,
    )
    assert not bool(hit) and np.isinf(float(time))


def test_event_driver_handles_multiple_hits(oracle):
    world = make_world(
        np.concatenate((np.eye(3), -np.eye(3))),
        [-1] * 6,
        0,
        1,
    )
    ball = make_ball([0, 0, 0], [20, 0, 0])
    result = step(world, ball, 0.3, jnp.zeros(3), oracle)
    assert bool(result.success)
    assert int(result.qp_status) == 1
    assert int(result.events) == int(result.qp_solves) == 3
    np.testing.assert_allclose(result.ball.position, [0, 0, 0], atol=2e-6)
    np.testing.assert_allclose(result.ball.velocity, [-20, 0, 0], atol=2e-6)

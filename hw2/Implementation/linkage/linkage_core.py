"""Four-bar geometry, penalty/Newton simulation, and Frechet objective.

Importing this module does not simulate, optimize, create figures, or write files.
Internal numeric kernels use JAX; checked_* wrappers validate inputs/results on the
host and are intentionally not jittable.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
Array = jax.Array

STATE_DIM = 6
LENGTH_DIM = 4
ANCHOR_CENTER = 0
ANCHOR_LEFT = 1
ANCHOR_RIGHT = 2


class LinkGeometry(NamedTuple):
    length: Array
    width: Array
    thickness: Array


class FourBarMechanism(NamedTuple):
    ground_left: Array
    ground_right: Array
    ground: LinkGeometry
    driver: LinkGeometry
    coupler: LinkGeometry
    follower: LinkGeometry
    branch_sign: Array


class PointOnLinkSpec(NamedTuple):
    link_index: Array
    anchor_mode: Array
    offset_xyz: Array


class PenaltyResult(NamedTuple):
    state_trajectory: Array
    driver_poses: Array
    energies: Array


class DesignResult(NamedTuple):
    """Scalar loss, total (4,) gradient, (T,6,4) sensitivities, forward solve."""
    loss: Array
    gradient_lengths: Array
    state_sensitivities: Array
    simulation: PenaltyResult


class DesignHistory(NamedTuple):
    """Histories contain num_steps + 1 EVALUATED designs, including the final one.

    Shapes: lengths (K+1, 4), losses (K+1,), gradients (K+1, 4).
    best_simulation must correspond to best_lengths and best_loss.
    """
    length_history: Array
    loss_history: Array
    gradient_history: Array
    best_lengths: Array
    best_loss: Array
    best_simulation: PenaltyResult


class CurveSynthesisExample(NamedTuple):
    template_mech: FourBarMechanism
    nominal_lengths: Array
    initial_lengths: Array
    driver_angles: Array
    point_spec: PointOnLinkSpec
    target_curve: Array


# -----------------------------------------------------------------------------
# Geometry.
# -----------------------------------------------------------------------------


def make_link_geometry(length: float, width: float, thickness: float = 0.0) -> LinkGeometry:
    return LinkGeometry(
        length=jnp.asarray(length, dtype=jnp.float64),
        width=jnp.asarray(width, dtype=jnp.float64),
        thickness=jnp.asarray(thickness, dtype=jnp.float64),
    )


def make_four_bar(
    ground_length: float,
    driver_length: float,
    coupler_length: float,
    follower_length: float,
    moving_link_width: float = 0.22,
    ground_width: float = 0.18,
    thickness: float = 0.0,
    branch_sign: float = 1.0,
) -> FourBarMechanism:
    return FourBarMechanism(
        ground_left=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64),
        ground_right=jnp.array([ground_length, 0.0, 0.0], dtype=jnp.float64),
        ground=make_link_geometry(ground_length, ground_width, thickness),
        driver=make_link_geometry(driver_length, moving_link_width, thickness),
        coupler=make_link_geometry(coupler_length, moving_link_width, thickness),
        follower=make_link_geometry(follower_length, moving_link_width, thickness),
        branch_sign=jnp.asarray(branch_sign, dtype=jnp.float64),
    )


def mechanism_lengths(mech: FourBarMechanism) -> Array:
    return jnp.array(
        [mech.ground.length, mech.driver.length, mech.coupler.length, mech.follower.length],
        dtype=jnp.float64,
    )


def mechanism_widths(mech: FourBarMechanism) -> Array:
    return jnp.array(
        [mech.ground.width, mech.driver.width, mech.coupler.width, mech.follower.width],
        dtype=jnp.float64,
    )


def mechanism_with_lengths(template: FourBarMechanism, lengths: Array) -> FourBarMechanism:
    ground_length, driver_length, coupler_length, follower_length = lengths
    return FourBarMechanism(
        ground_left=template.ground_left,
        ground_right=template.ground_left + jnp.array([ground_length, 0.0, 0.0], dtype=jnp.float64),
        ground=LinkGeometry(ground_length, template.ground.width, template.ground.thickness),
        driver=LinkGeometry(driver_length, template.driver.width, template.driver.thickness),
        coupler=LinkGeometry(coupler_length, template.coupler.width, template.coupler.thickness),
        follower=LinkGeometry(follower_length, template.follower.width, template.follower.thickness),
        branch_sign=template.branch_sign,
    )


def planar_transform(pose: Array) -> Array:
    x, y, theta = pose
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array(
        [[c, -s, 0.0, x], [s, c, 0.0, y], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=jnp.float64,
    )


def world_point_from_local(pose: Array, point_local_h: Array) -> Array:
    return planar_transform(pose) @ point_local_h


world_points_from_local = jax.vmap(world_point_from_local, in_axes=(None, 0))


def local_endpoints(length: Array) -> Array:
    return jnp.array(
        [[-0.5 * length, 0.0, 0.0, 1.0], [0.5 * length, 0.0, 0.0, 1.0]],
        dtype=jnp.float64,
    )


def local_rectangle(length: Array, width: Array) -> Array:
    lx = 0.5 * length
    ly = 0.5 * width
    return jnp.array(
        [[-lx, -ly, 0.0, 1.0], [lx, -ly, 0.0, 1.0], [lx, ly, 0.0, 1.0], [-lx, ly, 0.0, 1.0]],
        dtype=jnp.float64,
    )


def pose_from_left_endpoint(anchor_xyz: Array, theta: Array, length: Array) -> Array:
    return jnp.array(
        [anchor_xyz[0] + 0.5 * length * jnp.cos(theta), anchor_xyz[1] + 0.5 * length * jnp.sin(theta), theta],
        dtype=jnp.float64,
    )


def ground_pose(mech: FourBarMechanism) -> Array:
    delta = mech.ground_right - mech.ground_left
    angle = jnp.arctan2(delta[1], delta[0])
    return pose_from_left_endpoint(mech.ground_left, angle, mech.ground.length)


def driver_pose_from_angle(mech: FourBarMechanism, driver_angle: Array) -> Array:
    return pose_from_left_endpoint(mech.ground_left, driver_angle, mech.driver.length)


def split_state(state: Array) -> tuple[Array, Array]:
    return state[:3], state[3:]


def link_endpoint_world(pose: Array, link: LinkGeometry, endpoint_index: int) -> Array:
    return world_points_from_local(pose, local_endpoints(link.length))[endpoint_index, :3]


def rectangle_vertices_xy(pose: Array, length: Array, width: Array) -> Array:
    return world_points_from_local(pose, local_rectangle(length, width))[:, :2]


rectangle_vertices_one_frame = jax.vmap(rectangle_vertices_xy, in_axes=(0, 0, 0))
rectangle_vertices_all_frames = jax.vmap(rectangle_vertices_one_frame, in_axes=(0, None, None))


def build_link_pose_stack(mech: FourBarMechanism, state_trajectory: Array, driver_poses: Array) -> Array:
    n = state_trajectory.shape[0]
    ground_poses = jnp.broadcast_to(ground_pose(mech)[None, :], (n, 3))
    return jnp.stack([ground_poses, driver_poses, state_trajectory[:, :3], state_trajectory[:, 3:]], axis=1)


def driver_poses_from_angles(mech: FourBarMechanism, driver_angles: Array) -> Array:
    return jax.vmap(lambda a: driver_pose_from_angle(mech, a))(driver_angles)


# -----------------------------------------------------------------------------
# Closed-form initial guess.
# -----------------------------------------------------------------------------


def closed_form_initial_guess(mech: FourBarMechanism, driver_angle: Array) -> Array:
    anchor_a = mech.ground_left[:2]
    anchor_d = mech.ground_right[:2]
    point_b = anchor_a + mech.driver.length * jnp.array([jnp.cos(driver_angle), jnp.sin(driver_angle)])

    chord = anchor_d - point_b
    d = jnp.maximum(jnp.linalg.norm(chord), 1e-12)
    direction = chord / d
    a = (mech.coupler.length**2 - mech.follower.length**2 + d**2) / (2.0 * d)
    h = jnp.sqrt(jnp.maximum(mech.coupler.length**2 - a**2, 0.0))

    midpoint = point_b + a * direction
    normal = jnp.array([-direction[1], direction[0]], dtype=jnp.float64)
    point_c = midpoint + mech.branch_sign * h * normal

    coupler_angle = jnp.arctan2(point_c[1] - point_b[1], point_c[0] - point_b[0])
    follower_angle = jnp.arctan2(point_c[1] - anchor_d[1], point_c[0] - anchor_d[0])

    coupler_pose = pose_from_left_endpoint(jnp.array([point_b[0], point_b[1], 0.0]), coupler_angle, mech.coupler.length)
    follower_pose = pose_from_left_endpoint(jnp.array([anchor_d[0], anchor_d[1], 0.0]), follower_angle, mech.follower.length)
    return jnp.concatenate([coupler_pose, follower_pose], axis=0)


# -----------------------------------------------------------------------------
# Original penalty + Newton solver.
# -----------------------------------------------------------------------------


def contact_pairs(state: Array, mech: FourBarMechanism, driver_pose: Array) -> tuple[Array, Array]:
    coupler_pose, follower_pose = split_state(state)
    points_a = jnp.stack(
        [
            link_endpoint_world(coupler_pose, mech.coupler, 0),
            link_endpoint_world(coupler_pose, mech.coupler, 1),
            link_endpoint_world(follower_pose, mech.follower, 0),
        ],
        axis=0,
    )
    points_b = jnp.stack(
        [
            link_endpoint_world(driver_pose, mech.driver, 1),
            link_endpoint_world(follower_pose, mech.follower, 1),
            mech.ground_right,
        ],
        axis=0,
    )
    return points_a, points_b


def closure_energy(state: Array, mech: FourBarMechanism, driver_pose: Array) -> Array:
    """Return the scalar closure penalty for the supplied linkage state."""
    # STUDENT TODO: Implement closure_energy.
    # raise NotImplementedError(
    #     "STUDENT TODO: implement closure_energy in linkage_core.py"
    # )

    points_a, points_b = contact_pairs(state, mech, driver_pose)
    residuals = points_a - points_b
    energy = 0.5 * jnp.sum(residuals**2)
    return energy


penalty_grad = jax.grad(closure_energy, argnums=0)
penalty_hessian = jax.jacfwd(penalty_grad, argnums=0)


def solve_penalty_frame(
    state0: Array,
    mech: FourBarMechanism,
    driver_pose: Array,
    newton_iters: int = 8,
    damping: float = 1e-8,
) -> tuple[Array, Array]:
    damping_array = jnp.asarray(damping, dtype=jnp.float64)

    def body(_: int, state: Array) -> Array:
        g = penalty_grad(state, mech, driver_pose)
        h = penalty_hessian(state, mech, driver_pose)
        h = 0.5 * (h + h.T) + damping_array * jnp.eye(STATE_DIM, dtype=jnp.float64)
        return state - jnp.linalg.solve(h, g)

    state_star = jax.lax.fori_loop(0, newton_iters, body, state0)
    return state_star, closure_energy(state_star, mech, driver_pose)


def simulate_penalty(
    mech: FourBarMechanism,
    driver_angles: Array,
    initial_state: Array,
    newton_iters: int = 8,
    damping: float = 1e-8,
) -> PenaltyResult:
    def scan_body(prev_state: Array, angle: Array) -> tuple[Array, tuple[Array, Array, Array]]:
        driver_pose = driver_pose_from_angle(mech, angle)
        state_star, energy = solve_penalty_frame(prev_state, mech, driver_pose, newton_iters=newton_iters, damping=damping)
        return state_star, (state_star, driver_pose, energy)

    _, (states, driver_poses, energies) = jax.lax.scan(scan_body, initial_state, driver_angles)
    return PenaltyResult(states, driver_poses, energies)


simulate_penalty = jax.jit(simulate_penalty, static_argnames=("newton_iters",))


def frame_energy_from_lengths(
    state: Array,
    lengths: Array,
    template_mech: FourBarMechanism,
    driver_angle: Array,
) -> Array:
    mech = mechanism_with_lengths(template_mech, lengths)
    driver_pose = driver_pose_from_angle(mech, driver_angle)
    return closure_energy(state, mech, driver_pose)


frame_stationarity_from_lengths = jax.grad(frame_energy_from_lengths, argnums=0)
frame_hessian_from_lengths = jax.jacfwd(frame_stationarity_from_lengths, argnums=0)
frame_mixed_from_lengths = jax.jacfwd(frame_stationarity_from_lengths, argnums=1)


def simulate_penalty_from_lengths(
    lengths: Array,
    template_mech: FourBarMechanism,
    driver_angles: Array,
    newton_iters: int = 8,
    damping: float = 1e-8,
) -> PenaltyResult:
    mech = mechanism_with_lengths(template_mech, lengths)
    initial_state = closed_form_initial_guess(mech, driver_angles[0])
    return simulate_penalty(
        mech,
        driver_angles,
        initial_state,
        newton_iters=newton_iters,
        damping=damping,
    )


# -----------------------------------------------------------------------------
# End-effector path and curve objective.
# -----------------------------------------------------------------------------


def make_point_on_link_spec(
    link_index: int,
    anchor: str = "center",
    offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PointOnLinkSpec:
    anchor_map = {
        "center": ANCHOR_CENTER,
        "left_endpoint": ANCHOR_LEFT,
        "right_endpoint": ANCHOR_RIGHT,
    }
    if anchor not in anchor_map:
        raise ValueError(f"Unknown anchor mode '{anchor}'.")
    return PointOnLinkSpec(
        link_index=jnp.asarray(link_index, dtype=jnp.int32),
        anchor_mode=jnp.asarray(anchor_map[anchor], dtype=jnp.int32),
        offset_xyz=jnp.asarray(offset_xyz, dtype=jnp.float64),
    )


def local_point_h_from_spec(
    mech: FourBarMechanism,
    point_spec: PointOnLinkSpec,
) -> Array:
    lengths = mechanism_lengths(mech)
    link_length = jax.lax.dynamic_index_in_dim(
        lengths,
        point_spec.link_index,
        axis=0,
        keepdims=False,
    )
    anchor_x = jnp.where(
        point_spec.anchor_mode == ANCHOR_CENTER,
        0.0,
        jnp.where(
            point_spec.anchor_mode == ANCHOR_LEFT,
            -0.5 * link_length,
            0.5 * link_length,
        ),
    )
    point_xyz = jnp.array([anchor_x, 0.0, 0.0], dtype=jnp.float64) + point_spec.offset_xyz
    return jnp.concatenate([point_xyz, jnp.array([1.0], dtype=jnp.float64)], axis=0)


def end_effector_path(
    mech: FourBarMechanism,
    state_trajectory: Array,
    driver_poses: Array,
    point_spec: PointOnLinkSpec,
) -> Array:
    link_poses = build_link_pose_stack(mech, state_trajectory, driver_poses)
    point_local_h = local_point_h_from_spec(mech, point_spec)

    def one_frame(poses_t: Array) -> Array:
        pose = jax.lax.dynamic_index_in_dim(
            poses_t,
            point_spec.link_index,
            axis=0,
            keepdims=False,
        )
        return world_point_from_local(pose, point_local_h)[:2]

    return jax.vmap(one_frame)(link_poses)


def rectangles_from_trajectory(
    mech: FourBarMechanism,
    state_trajectory: Array,
    driver_poses: Array,
) -> Array:
    link_poses = build_link_pose_stack(mech, state_trajectory, driver_poses)
    return rectangle_vertices_all_frames(
        link_poses,
        mechanism_lengths(mech),
        mechanism_widths(mech),
    )


def discrete_frechet_distance(path_a: Array, path_b: Array) -> Array:
    # Same distance values as the original norm. Choose a finite zero
    # subgradient at coincident points; this does NOT smooth Frechet's min/max.
    if path_a.ndim != 2 or path_b.ndim != 2 or path_a.shape[1] != path_b.shape[1]:
        raise ValueError("Paths must be rank-two arrays with matching point dimensions.")
    if path_a.shape[0] == 0 or path_b.shape[0] == 0:
        raise ValueError("Paths must be nonempty.")
    delta = path_a[:, None, :] - path_b[None, :, :]
    squared_distances = jnp.sum(delta * delta, axis=-1)
    positive = squared_distances > 0.0
    roots = jnp.sqrt(jnp.where(positive, squared_distances, 1.0))
    distances = jnp.where(positive, roots, 0.0)
    n, m = distances.shape
    table = jnp.full((n + 1, m + 1), jnp.inf, dtype=distances.dtype).at[0, 0].set(0.0)

    def row_body(i: int, table_i: Array) -> Array:
        def col_body(j: int, table_ij: Array) -> Array:
            previous = jnp.minimum(
                jnp.minimum(table_ij[i - 1, j], table_ij[i - 1, j - 1]),
                table_ij[i, j - 1],
            )
            return table_ij.at[i, j].set(jnp.maximum(previous, distances[i - 1, j - 1]))

        return jax.lax.fori_loop(1, m + 1, col_body, table_i)

    table = jax.lax.fori_loop(1, n + 1, row_body, table)
    return table[n, m]


def curve_loss_from_states_and_lengths(
    state_trajectory: Array,
    lengths: Array,
    template_mech: FourBarMechanism,
    driver_angles: Array,
    point_spec: PointOnLinkSpec,
    target_curve: Array,
) -> Array:
    mech = mechanism_with_lengths(template_mech, lengths)
    driver_poses = driver_poses_from_angles(mech, driver_angles)
    path = end_effector_path(mech, state_trajectory, driver_poses, point_spec)
    return discrete_frechet_distance(path, target_curve)


curve_loss_value_and_grads = jax.jit(jax.value_and_grad(
    curve_loss_from_states_and_lengths,
    argnums=(0, 1),
))


def curve_loss_with_simulation(
    lengths: Array,
    template_mech: FourBarMechanism,
    driver_angles: Array,
    point_spec: PointOnLinkSpec,
    target_curve: Array,
    newton_iters: int = 8,
    damping: float = 1e-8,
) -> tuple[Array, PenaltyResult]:
    simulation = simulate_penalty_from_lengths(
        lengths=lengths,
        template_mech=template_mech,
        driver_angles=driver_angles,
        newton_iters=newton_iters,
        damping=damping,
    )
    mech = mechanism_with_lengths(template_mech, lengths)
    path = end_effector_path(mech, simulation.state_trajectory, simulation.driver_poses, point_spec)
    loss = discrete_frechet_distance(path, target_curve)
    return loss, simulation


# Implicit differentiation and the outer optimizer belong in optimize.py.


# -----------------------------------------------------------------------------
# Diagnostics and example problem.
# -----------------------------------------------------------------------------


def closure_constraint_norms(
    mech: FourBarMechanism,
    result: PenaltyResult,
) -> Array:
    return jnp.sqrt(result.energies)


def make_on_link_coupler_synthesis_example(
    num_frames: int = 48,
    angle_start: float = -1.0,
    angle_end: float = 1.0,
    perturbation: tuple[float, float, float, float] = (0.14, -0.09, 0.12, -0.08),
    point_offset_xy: tuple[float, float] = (0.35, 0.02),
    newton_iters: int = 8,
    damping: float = 1e-8,
) -> CurveSynthesisExample:
    template_mech = make_four_bar(
        ground_length=2.0,
        driver_length=1.0,
        coupler_length=2.5,
        follower_length=2.5,
        moving_link_width=0.12,
        ground_width=0.09,
        branch_sign=1.0,
    )

    nominal_lengths = mechanism_lengths(template_mech)
    driver_angles = jnp.linspace(angle_start, angle_end, num_frames)

    point_spec = make_point_on_link_spec(
        2,
        anchor="center",
        offset_xyz=(point_offset_xy[0], point_offset_xy[1], 0.0),
    )

    nominal_result = simulate_penalty_from_lengths(
        lengths=nominal_lengths,
        template_mech=template_mech,
        driver_angles=driver_angles,
        newton_iters=newton_iters,
        damping=damping,
    )

    target_curve = end_effector_path(
        template_mech,
        nominal_result.state_trajectory,
        nominal_result.driver_poses,
        point_spec,
    )

    initial_lengths = nominal_lengths * (1.0 + jnp.asarray(perturbation, dtype=jnp.float64))

    return CurveSynthesisExample(
        template_mech=template_mech,
        nominal_lengths=nominal_lengths,
        initial_lengths=initial_lengths,
        driver_angles=driver_angles,
        point_spec=point_spec,
        target_curve=target_curve,
    )


# Plotting and animation are in linkage_viz.py.




# Compile reusable numerical kernels once rather than inside optimization loops.
simulate_penalty_from_lengths = jax.jit(
    simulate_penalty_from_lengths, static_argnames=("newton_iters",)
)
curve_loss_with_simulation = jax.jit(
    curve_loss_with_simulation, static_argnames=("newton_iters",)
)
closure_constraint_norms = jax.jit(closure_constraint_norms)


@jax.jit
def stationarity_norms(
    lengths: Array, template_mech: FourBarMechanism,
    driver_angles: Array, simulation: PenaltyResult,
) -> Array:
    gradients = jax.vmap(frame_stationarity_from_lengths, in_axes=(0, None, None, 0))(
        simulation.state_trajectory, lengths, template_mech, driver_angles
    )
    return jnp.linalg.norm(gradients, axis=1)


def validate_lengths_and_angles(lengths: Array, driver_angles: Array) -> None:
    ell = np.asarray(lengths, dtype=float)
    angles = np.asarray(driver_angles, dtype=float)
    if ell.shape != (LENGTH_DIM,) or not np.all(np.isfinite(ell)) or np.any(ell <= 0):
        raise ValueError("lengths must contain four finite, strictly positive values.")
    if angles.ndim != 1 or len(angles) == 0 or not np.all(np.isfinite(angles)):
        raise ValueError("driver_angles must be a nonempty finite vector.")


def sampled_feasibility_margin(lengths: Array, driver_angles: Array) -> float:
    """Smallest strict triangle-inequality slack at the SAMPLED input angles.

    The coupler and follower are two intersecting circles about the driver tip
    and right ground pin. Positivity alone does not guarantee their intersection.
    This is a sampled check, not a guarantee for all intervening input angles.
    """
    validate_lengths_and_angles(lengths, driver_angles)
    ground, driver, coupler, follower = np.asarray(lengths, dtype=float)
    angles = np.asarray(driver_angles, dtype=float)
    # This stable expression also avoids cancellation close to angle zero.
    d = np.hypot(ground - driver * np.cos(angles), driver * np.sin(angles))
    return float(min(np.min(d), np.min(d - abs(coupler - follower)),
                     np.min(coupler + follower - d)))


def simulation_diagnostics(
    lengths: Array, template_mech: FourBarMechanism, driver_angles: Array,
    simulation: PenaltyResult, *, include_condition: bool = False,
) -> dict[str, float]:
    mech = mechanism_with_lengths(template_mech, lengths)
    values = {
        "max_closure_norm": float(jnp.max(closure_constraint_norms(mech, simulation))),
        "max_stationarity_norm": float(jnp.max(stationarity_norms(
            lengths, template_mech, driver_angles, simulation))),
        "sampled_feasibility_margin": sampled_feasibility_margin(lengths, driver_angles),
    }
    if include_condition:
        hessians = jax.vmap(frame_hessian_from_lengths, in_axes=(0, None, None, 0))(
            simulation.state_trajectory, lengths, template_mech, driver_angles
        )
        values["max_hessian_condition"] = float(np.max(np.linalg.cond(np.asarray(hessians))))
    return values


def checked_simulation_from_lengths(
    lengths: Array, template_mech: FourBarMechanism, driver_angles: Array,
    newton_iters: int = 8, damping: float = 1e-8, *,
    closure_tol: float = 1e-7, stationarity_tol: float = 1e-7,
    feasibility_margin: float = 1e-7,
) -> PenaltyResult:
    """Run the original solver and reject invalid or unconverged results.

    The low-level, differentiable simulation is unchanged. This host wrapper
    is for demonstrations, finite-difference checks, and the outer loop.
    It verifies that the original assembly branch is retained in every frame.
    """
    validate_lengths_and_angles(lengths, driver_angles)
    if isinstance(newton_iters, bool) or not isinstance(newton_iters, (int, np.integer)) or newton_iters < 1:
        raise ValueError("newton_iters must be a positive integer.")
    if not np.isfinite(damping) or damping < 0:
        raise ValueError("Newton damping must be finite and nonnegative.")
    for name, value in (("closure_tol", closure_tol), ("stationarity_tol", stationarity_tol),
                        ("feasibility_margin", feasibility_margin)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if float(template_mech.branch_sign) not in (-1.0, 1.0):
        raise ValueError("branch_sign must be +1 or -1.")
    if sampled_feasibility_margin(lengths, driver_angles) <= feasibility_margin:
        raise ValueError(
            "Infeasible or near-toggle design at a sampled angle. Reduce the outer "
            "step size or choose a safer starting design; clipping lengths is not enough."
        )
    simulation = simulate_penalty_from_lengths(
        jnp.asarray(lengths, dtype=jnp.float64), template_mech,
        jnp.asarray(driver_angles, dtype=jnp.float64), newton_iters=newton_iters, damping=damping
    )
    if not all(np.all(np.isfinite(np.asarray(a))) for a in simulation):
        raise RuntimeError("Newton produced nonfinite values. Reduce the step or use a safer design.")
    if np.any(np.asarray(simulation.energies) < 0):
        raise RuntimeError("closure_energy must return nonnegative values.")
    diagnostics = simulation_diagnostics(lengths, template_mech, driver_angles, simulation)
    if (diagnostics["max_closure_norm"] > closure_tol or
            diagnostics["max_stationarity_norm"] > stationarity_tol):
        raise RuntimeError(
            f"Newton did not converge: closure={diagnostics['max_closure_norm']:.3e}, "
            f"stationarity={diagnostics['max_stationarity_norm']:.3e}. "
            "Increase newton_iters, sample angles more densely, or reduce the design step."
        )
    ground, driver, coupler, _ = np.asarray(lengths, dtype=float)
    angles = np.asarray(driver_angles)
    anchor = np.asarray(template_mech.ground_left[:2])
    b = anchor + driver * np.stack((np.cos(angles), np.sin(angles)), axis=1)
    q = np.asarray(simulation.state_trajectory[:, :3])
    c = q[:, :2] + 0.5 * coupler * np.stack((np.cos(q[:, 2]), np.sin(q[:, 2])), axis=1)
    chord = anchor + np.array([ground, 0.0]) - b
    bc = c - b
    signed_area = chord[:, 0] * bc[:, 1] - chord[:, 1] * bc[:, 0]
    if np.any(float(template_mech.branch_sign) * signed_area <= 0):
        raise RuntimeError("Assembly branch changed. Use denser driver-angle samples or a safer step.")
    return simulation


def validated_design_mask(design_mask: Array | None, lengths: Array) -> Array:
    """Return a Boolean (4,) mask; reject ambiguous weights and broadcasting."""
    if np.shape(lengths) != (LENGTH_DIM,):
        raise ValueError("lengths must have shape (4,).")
    if design_mask is None:
        return jnp.ones((LENGTH_DIM,), dtype=bool)
    mask = np.asarray(design_mask)
    if mask.shape != (LENGTH_DIM,) or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("design_mask must have shape (4,) and contain only 0/1 or bool values.")
    return jnp.asarray(mask, dtype=bool)


def validate_update_inputs(
    lengths: Array, gradient_lengths: Array, step_size: float,
    design_mask: Array | None, min_length: float,
) -> Array:
    """Validate host-side update arguments and return the Boolean mask."""
    validate_lengths_and_angles(lengths, jnp.array([0.0]))
    gradient = np.asarray(gradient_lengths)
    if gradient.shape != (LENGTH_DIM,) or not np.all(np.isfinite(gradient)):
        raise ValueError("gradient_lengths must have shape (4,) and be finite.")
    if not np.isfinite(step_size) or step_size < 0:
        raise ValueError("step_size must be finite and nonnegative.")
    if not np.isfinite(min_length) or min_length <= 0:
        raise ValueError("min_length must be finite and positive.")
    return validated_design_mask(design_mask, lengths)


def validate_optimization_inputs(
    initial_lengths: Array, num_steps: int, step_size: float,
    design_mask: Array | None, min_length: float,
) -> Array:
    if isinstance(num_steps, bool) or not isinstance(num_steps, (int, np.integer)) or num_steps < 0:
        raise ValueError("num_steps must be a nonnegative integer.")
    mask = validate_update_inputs(initial_lengths, jnp.zeros(4), step_size, design_mask, min_length)
    if np.any(np.asarray(initial_lengths) < min_length):
        raise ValueError("All initial lengths must be at least min_length.")
    return mask


def check_design_result(result: DesignResult, num_frames: int) -> None:
    """Catch unfinished/invalid numerical answers before they drive an update."""
    for name, value, shape in (
        ("loss", result.loss, ()),
        ("gradient_lengths", result.gradient_lengths, (4,)),
        ("state_sensitivities", result.state_sensitivities, (num_frames, 6, 4)),
    ):
        data = np.asarray(value)
        if data.shape != shape or not np.all(np.isfinite(data)):
            raise RuntimeError(f"{name} must be finite and have shape {shape}; got {data.shape}.")

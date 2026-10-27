"""Optimize the soft bear to reach a forward terminal target.

This is a constrained, reduced-space shooting problem.  IPOPT chooses bounded
piecewise-constant spatial activations.  Newton/Warp supplies the trajectory
and its reverse-mode derivatives.  The quick, small, and full presets provide
increasingly long versions of the same formulation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pyomo.environ as pyo

from bear_shooting import ActuatorMap, ShootingProblem
from bear_shooting.newton_bear import BearPhysics, BearScene


SIZE_DEFAULTS = {
    "quick": {
        "output": Path("results/bear_run_quick.npz"),
        "horizon": 15,
        "distance_body_lengths": 0.075,
        "continuation": (1.0,),
    },
    "small": {
        "output": Path("results/bear_run_small.npz"),
        "horizon": 75,
        "distance_body_lengths": 0.55,
        "continuation": (0.95, 1.0),
    },
    "full": {
        "output": Path("results/bear_run.npz"),
        "horizon": 150,
        "distance_body_lengths": 1.0,
        "continuation": (0.95, 1.0),
    },
}

STAGE_FEASIBILITY_TOLERANCE = 5e-4


def _positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def create_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--asset", type=Path, default=Path("assets/bear.usd"))
    parser.add_argument("--device", default="cpu", help="Warp device")
    parser.add_argument(
        "--size",
        choices=tuple(SIZE_DEFAULTS),
        default="quick",
        help="problem-size preset; individual command-line values override it",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=argparse.SUPPRESS,
        help="result path selected from --size unless supplied explicitly",
    )
    parser.add_argument(
        "--horizon",
        type=_positive_int,
        default=argparse.SUPPRESS,
        help="number of control intervals selected from --size unless supplied",
    )
    parser.add_argument("--control-dt", type=_positive_float, default=1.0 / 15.0)
    parser.add_argument("--substeps", type=_positive_int, default=320)
    parser.add_argument("--bins", type=_positive_int, nargs=3, default=(4, 1, 1))
    parser.add_argument("--initial-height", type=float, default=0.11)
    parser.add_argument(
        "--distance-body-lengths",
        type=_positive_float,
        default=argparse.SUPPRESS,
        help="required terminal COM displacement divided by reference x-span",
    )
    parser.add_argument(
        "--continuation",
        type=_positive_float,
        nargs="+",
        default=argparse.SUPPRESS,
        help="increasing fractions of the requested distance",
    )
    parser.add_argument(
        "--target-window",
        type=_positive_float,
        default=0.25,
        help="allowed overshoot beyond each terminal target in metres",
    )
    parser.add_argument("--activation-limit", type=_positive_float, default=0.3)
    parser.add_argument(
        "--search-radius",
        type=_positive_float,
        default=0.05,
        help=(
            "per-stage absolute control-change limit around the incoming gait; "
            "use a larger value to permit more substantial gait changes"
        ),
    )
    parser.add_argument("--initial-amplitude", type=float, default=0.3)
    parser.add_argument("--initial-frequency", type=_positive_float, default=0.5)
    parser.add_argument("--max-iter", type=_positive_int, default=60)
    parser.add_argument("--max-state-memory-mb", type=_positive_float, default=10240)
    parser.add_argument("--max-jacobian-mb", type=_positive_float, default=512)
    parser.add_argument(
        "--capture-graph",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help=(
            "capture the long forward/reverse launch sequences; by default this "
            "is enabled on CUDA and disabled on CPU"
        ),
    )
    parser.add_argument(
        "--evaluate-initial",
        action="store_true",
        help="roll out the deterministic initial controls without invoking IPOPT",
    )
    return parser


def resolve_size_defaults(args):
    """Fill size-dependent values without overriding explicit CLI arguments."""
    defaults = SIZE_DEFAULTS[args.size]
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    return args


def traveling_wave_controls(layout, amplitude: float, frequency: float) -> np.ndarray:
    """Return a deterministic, smooth phase-shifted initial guess."""
    if not math.isfinite(amplitude):
        raise ValueError("Initial amplitude must be finite.")
    times = np.arange(layout.horizon, dtype=float) * layout.control_dt
    controls = np.empty(layout.control_shape, dtype=float)
    for channel, name in enumerate(layout.actuator_names):
        try:
            x_index = int(name.rsplit("_", 3)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                "The symmetric traveling wave requires ActuatorMap.grid names."
            ) from exc
        # Drive each lateral/vertical pair identically and advance phase along
        # x.  This produces forward deformation without deliberately steering.
        phase = 0.5 * np.pi * x_index
        controls[:, channel] = -amplitude * np.sin(
            2.0 * np.pi * frequency * times + phase
        )
    return controls


def resolve_graph_capture(device, requested: bool | None) -> bool:
    """Resolve the CLI's automatic CUDA-only graph-capture setting."""
    if requested is not None:
        return bool(requested)
    name = str(device).lower()
    return name == "cuda" or name.startswith("cuda:")


def register_objectives(problem, *, control_center, search_radius):
    """Register the assignment's scalar objective term(s) for one stage.

    ``problem`` is an initialized :class:`ShootingProblem`. Its public
    variables are available through ``problem.variables``. ``control_center``
    is a finite array with the same ``(interval, actuator)`` shape as the
    controls, and ``search_radius`` is a positive dimensionless scalar.

    Add the objective term(s) to ``problem`` using its public registration API.
    The assignment PDF defines the required expressions and constants.
    """
    # STUDENT TODO: Register the objective term(s) for this continuation stage.
    controls = problem.variables.u
    intervals, actuators = controls.shape
    # Keep these as symbolic Pyomo expressions so the supplied differentiation
    # layer can compute exact derivatives of the control penalties.
    correction = controls - control_center
    deviation = pyo.quicksum(
        (correction[k, a] / search_radius) ** 2
        for k in range(intervals)
        for a in range(actuators)
    ) / (intervals * actuators)
    problem.add_objective("gait_deviation", deviation)

    # Smooth the correction to the incoming gait, rather than the gait itself.
    # A single interval has no neighboring pair and needs no smoothing term.
    if intervals > 1:
        smoothness = pyo.quicksum(
            ((correction[k + 1, a] - correction[k, a]) / search_radius) ** 2
            for k in range(intervals - 1)
            for a in range(actuators)
        ) / ((intervals - 1) * actuators)
        problem.add_objective("correction_smoothness", smoothness, weight=0.05)


def register_constraints(
    problem, *, start_com, target_distance, target_window
):
    """Register the assignment's task constraint(s) for one stage.

    ``start_com`` is the initial three-dimensional center of mass.
    ``target_distance`` and ``target_window`` are positive distances in metres.
    Simulated terminal quantities are available through ``problem.variables``.

    Add the required constraint(s) with the public registration API. The
    assignment PDF defines the expressions, bounds, and numerical scaling.
    """
    # STUDENT TODO: Register the constraint(s) for this continuation stage.
    forward_displacement = problem.variables.position("com")[0] - start_com[0]
    # Bounds are in metres. Scaling by the requested displacement makes the
    # solver's feasibility tolerance relative to this stage's target distance.
    problem.add_constraint(
        "forward_target",
        forward_displacement,
        lower=target_distance,
        upper=target_distance + target_window,
        scale=target_distance,
    )


def build_problem(
    simulator,
    *,
    start_com,
    target_distance,
    target_window,
    activation_limit,
    max_jacobian_mb,
    control_center,
    search_radius,
):
    """Construct one continuation stage of the constrained locomotion NLP."""
    center = np.asarray(control_center, dtype=float)
    if center.shape != simulator.layout.control_shape or not np.all(
        np.isfinite(center)
    ):
        raise ValueError("control_center must be finite and match the control shape.")
    if np.any(np.abs(center) > activation_limit + 1.0e-12):
        raise ValueError("control_center exceeds the physical activation limit.")
    if not math.isfinite(search_radius) or search_radius <= 0:
        raise ValueError("search_radius must be finite and positive.")
    lower = np.maximum(-activation_limit, center - search_radius)
    upper = np.minimum(activation_limit, center + search_radius)
    problem = ShootingProblem(
        simulator,
        lower=lower,
        upper=upper,
        max_jacobian_mb=max_jacobian_mb,
    )
    register_objectives(
        problem,
        control_center=center,
        search_radius=search_radius,
    )
    register_constraints(
        problem,
        start_com=start_com,
        target_distance=target_distance,
        target_window=target_window,
    )
    return problem


def stage_ipopt_options(max_iter, *, device):
    """Return the supplied IPOPT settings for a locomotion stage."""
    options = {
        "max_iter": max_iter,
        "print_level": 5,
        # The deterministic gait touches the physical activation bounds.
        # Preserve that starting point to numerical precision.
        "bound_push": 1.0e-8,
        "bound_frac": 1.0e-8,
        "mu_strategy": "monotone",
    }
    device_name = str(device).lower()
    if device_name == "cpu" or device_name.startswith("cpu:"):
        # Contact dynamics make the final few derivative evaluations slightly
        # noisy on CPU. Stop after three consistently acceptable iterates
        # rather than asking IPOPT to refine that numerical noise.
        options.update(acceptable_tol=5.0e-4, acceptable_iter=3)
    return options


def solve_with_ipopt(problem, initial_controls, *, max_iter, device):
    """Invoke IPOPT for one stage and return its complete solve result.

    Use ``initial_controls`` as the initial point. The supplied solver options
    come from :func:`stage_ipopt_options` for ``device``; the independent
    result check uses :data:`STAGE_FEASIBILITY_TOLERANCE`.
    """
    # STUDENT TODO: Call ShootingProblem.solve with the supplied IPOPT settings.
    # The native bridge calls the differentiable rollout and checks the final
    # controls and task constraints independently of IPOPT's return status.
    return problem.solve(
        initial_controls,
        engine="ipopt",
        ipopt_options=stage_ipopt_options(max_iter, device=device),
        feasibility_tolerance=STAGE_FEASIBILITY_TOLERANCE,
    )


def run_metrics(trajectory, start_com, body_length):
    final_com = trajectory.position("com")
    displacement = final_com - start_com
    com_index = trajectory.layout.part_names.index("com")
    com_path = trajectory.positions[:, com_index, :]
    return {
        "initial_com_m": start_com.tolist(),
        "final_com_m": final_com.tolist(),
        "com_displacement_m": displacement.tolist(),
        "forward_body_lengths": float(displacement[0] / body_length),
        "mean_forward_speed_m_per_s": float(
            displacement[0] / trajectory.layout.times[-1]
        ),
        "fraction_of_control_intervals_with_forward_com_progress": float(
            np.mean(np.diff(com_path[:, 0]) > 0.0)
        ),
        "terminal_com_velocity_m_per_s": trajectory.velocity("com").tolist(),
        "maximum_absolute_lateral_com_m": float(
            np.max(np.abs(com_path[:, 1] - start_com[1]))
        ),
        "com_height_range_m": [
            float(np.min(com_path[:, 2])),
            float(np.max(com_path[:, 2])),
        ],
    }


def validate_continuation(fractions, requested_distance):
    fractions = tuple(float(value) for value in fractions)
    if not fractions:
        raise ValueError("Supply at least one continuation fraction.")
    if not math.isfinite(requested_distance) or requested_distance <= 0:
        raise ValueError("Requested distance must be finite and positive.")
    if any(not math.isfinite(value) or value <= 0 or value > 1 for value in fractions):
        raise ValueError("Continuation fractions must lie in (0, 1].")
    if any(right <= left for left, right in zip(fractions, fractions[1:])):
        raise ValueError("Continuation fractions must be strictly increasing.")
    if not np.isclose(fractions[-1], 1.0):
        raise ValueError("The final continuation fraction must be exactly 1.0.")
    return tuple(value * requested_distance for value in fractions)


def main():
    wall_start = time.perf_counter()
    args = resolve_size_defaults(create_parser().parse_args())
    physics = BearPhysics(translation=(0.0, 0.0, args.initial_height))
    scene = BearScene.load(args.asset, device=args.device, physics=physics)
    actuators = ActuatorMap.grid(
        scene.vertices, scene.tetrahedra, bins=tuple(args.bins)
    )
    simulator = scene.make_rollout(
        horizon=args.horizon,
        control_dt=args.control_dt,
        substeps=args.substeps,
        actuators=actuators,
        max_state_memory_mb=args.max_state_memory_mb,
        capture_graph=resolve_graph_capture(
            args.device, getattr(args, "capture_graph", None)
        ),
    )

    body_length = float(np.ptp(scene.vertices[:, 0]))
    requested_distance = args.distance_body_lengths * body_length
    start_com = scene.parts.reference("com")
    continuation_distances = validate_continuation(
        args.continuation, requested_distance
    )
    zero_controls = np.zeros(simulator.layout.control_shape)
    # Warm all forward kernels once. The following initial-gait rollout is then
    # captured as the reusable CUDA graph when capture is enabled.
    simulator.rollout(zero_controls)
    initial_controls = traveling_wave_controls(
        simulator.layout, args.initial_amplitude, args.initial_frequency
    )
    if np.max(np.abs(initial_controls)) > args.activation_limit:
        raise ValueError("Initial amplitude exceeds the activation limit.")

    initial_trajectory = simulator.layout.trajectory(
        simulator.rollout(initial_controls)
    )
    print(
        json.dumps(
            {
                "reference_body_length_m": body_length,
                "size": args.size,
                "device": args.device,
                "requested_distance_m": requested_distance,
                "duration_s": args.horizon * args.control_dt,
                "control_shape": simulator.layout.control_shape,
                "physics_substeps": args.horizon * args.substeps,
                "state_memory_lower_bound_mib": simulator.state_memory_lower_bound_mb,
                "initial_guess": run_metrics(
                    initial_trajectory, start_com, body_length
                ),
                "initial_contact": simulator.contact_diagnostics(),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.evaluate_initial:
        print(
            json.dumps(
                {"end_to_end_wall_seconds": time.perf_counter() - wall_start},
                indent=2,
            ),
            flush=True,
        )
        return

    controls = initial_controls
    stages = []
    final_problem = final_result = None
    for stage, distance in enumerate(continuation_distances, start=1):
        print(
            f"Continuation stage {stage}/{len(continuation_distances)}: "
            f"target {distance / body_length:.3f} body lengths",
            flush=True,
        )
        problem = build_problem(
            simulator,
            start_com=start_com,
            target_distance=distance,
            target_window=args.target_window,
            activation_limit=args.activation_limit,
            control_center=controls,
            search_radius=args.search_radius,
            max_jacobian_mb=args.max_jacobian_mb,
        )
        result = solve_with_ipopt(
            problem,
            controls,
            max_iter=args.max_iter,
            device=args.device,
        )
        metrics = run_metrics(result.trajectory, start_com, body_length)
        stage_summary = {
            "fraction": float(distance / requested_distance),
            "distance_m": float(distance),
            "solve": result.summary(),
            "motion": metrics,
        }
        stages.append(stage_summary)
        print(json.dumps(stage_summary, indent=2), flush=True)
        result.require_success()
        controls = result.controls
        final_problem, final_result = problem, result

    assert final_problem is not None and final_result is not None
    output_path = final_result.save(args.output)
    simulator.save_replay(
        output_path.with_name(output_path.stem + "_particles.npz")
    )
    contact = simulator.contact_diagnostics()
    final_metrics = run_metrics(final_result.trajectory, start_com, body_length)
    minimum_progress = requested_distance - 5e-4 * requested_distance
    if final_metrics["com_displacement_m"][0] < minimum_progress:
        raise RuntimeError("Accepted result did not independently reach the target.")

    task = {
        "kind": "forward_com_target",
        "forward_target": {
            "part": "com",
            "axis": "x",
            "minimum_displacement_m": requested_distance,
            "maximum_displacement_m": requested_distance + args.target_window,
            "start_m": start_com.tolist(),
            "target_x_m": float(start_com[0] + requested_distance),
            "body_length_m": body_length,
        },
        "parts": simulator.part_definitions,
        "control_bounds": {
            "lower": final_problem.lower.tolist(),
            "upper": final_problem.upper.tolist(),
        },
        "physics": {
            "size": args.size,
            "device": args.device,
            "initial_height_m": args.initial_height,
            "duration_s": args.horizon * args.control_dt,
            "substeps": args.substeps,
            "bins": list(args.bins),
            "activation_limit": args.activation_limit,
            "stage_search_radius": args.search_radius,
        },
        "warm_start": {
            "kind": "longitudinal_traveling_wave",
            "amplitude": args.initial_amplitude,
            "frequency_hz": args.initial_frequency,
            "time_sample": "control_interval_start",
            "phase_advance_radians_per_x_bin": 0.5 * math.pi,
        },
        "continuation": stages,
        "final_motion": final_metrics,
        "contact_diagnostics": contact,
    }
    task_path = output_path.with_name(output_path.stem + "_task.json")
    task_path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
    end_to_end_wall_seconds = time.perf_counter() - wall_start
    report_path = output_path.with_suffix(".json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["end_to_end_wall_seconds"] = end_to_end_wall_seconds
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "end_to_end_wall_seconds": end_to_end_wall_seconds,
                **final_metrics,
                "contact": contact,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

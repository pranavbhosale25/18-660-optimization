"""Independently replay and verify a saved constrained bear locomotion solve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bear_shooting import ActuatorMap
from bear_shooting.newton_bear import BearPhysics, BearScene


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "result",
        nargs="?",
        type=Path,
        default=Path("results/bear_run_quick.npz"),
        help="saved locomotion result to replay independently",
    )
    parser.add_argument("--asset", type=Path, default=Path("assets/bear.usd"))
    parser.add_argument("--device", default="cpu", help="Warp replay device")
    parser.add_argument(
        "--output", type=Path, help="verification JSON path selected from the result"
    )
    parser.add_argument("--max-state-memory-mb", type=float, default=10240.0)
    return parser


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def main() -> None:
    args = create_parser().parse_args()
    result_path = args.result.expanduser().resolve()
    task_path = result_path.with_name(result_path.stem + "_task.json")
    task = _load_json(task_path)
    if task.get("kind") != "forward_com_target":
        raise ValueError(f"{task_path} does not describe a forward-COM target.")

    with np.load(result_path, allow_pickle=False) as saved:
        controls = np.asarray(saved["controls"], dtype=float)
        expected_observations = np.asarray(saved["observations"], dtype=float)
    if controls.ndim != 2 or not np.all(np.isfinite(controls)):
        raise ValueError("Saved controls must be a finite two-dimensional array.")

    physics_data = task["physics"]
    duration = float(physics_data["duration_s"])
    horizon = controls.shape[0]
    scene = BearScene.load(
        args.asset,
        device=args.device,
        physics=BearPhysics(
            translation=(0.0, 0.0, float(physics_data["initial_height_m"]))
        ),
    )
    actuators = ActuatorMap.grid(
        scene.vertices,
        scene.tetrahedra,
        bins=tuple(int(value) for value in physics_data["bins"]),
    )
    simulator = scene.make_rollout(
        horizon=horizon,
        control_dt=duration / horizon,
        substeps=int(physics_data["substeps"]),
        actuators=actuators,
        max_state_memory_mb=args.max_state_memory_mb,
        capture_graph=False,
    )
    replayed = simulator.layout.trajectory(simulator.rollout(controls))
    start_com = scene.parts.reference("com")
    final_com = replayed.position("com")
    displacement = final_com - start_com
    com_index = replayed.layout.part_names.index("com")
    com_path = replayed.positions[:, com_index, :]
    forward_increments = np.diff(com_path[:, 0])

    target = task["forward_target"]
    lower = float(target["minimum_displacement_m"])
    upper = float(target["maximum_displacement_m"])
    # Match the post-solve acceptance threshold used by bear_run.py.  The
    # tolerance is relative because the registered IPOPT row is distance-scaled.
    tolerance = 5.0e-4 * lower
    reaches_target = bool(lower - tolerance <= displacement[0] <= upper + tolerance)
    expected_flat = expected_observations.reshape(-1)
    replayed_flat = replayed.values
    if expected_flat.shape != replayed_flat.shape:
        raise ValueError("Saved and replayed observation layouts do not match.")

    contact = simulator.contact_diagnostics()
    report = {
        "independent_scene_and_rollout": True,
        "device": str(simulator.device),
        "horizon": horizon,
        "duration_s": duration,
        "control_shape": list(controls.shape),
        "initial_com_m": start_com.tolist(),
        "final_com_m": final_com.tolist(),
        "com_displacement_m": displacement.tolist(),
        "forward_body_lengths": float(
            displacement[0] / float(target["body_length_m"])
        ),
        "mean_forward_speed_m_per_s": float(displacement[0] / duration),
        "terminal_com_velocity_m_per_s": replayed.velocity("com").tolist(),
        "fraction_of_control_intervals_with_forward_com_progress": float(
            np.mean(forward_increments > 0.0)
        ),
        "maximum_absolute_lateral_com_displacement_m": float(
            np.max(np.abs(com_path[:, 1] - start_com[1]))
        ),
        "com_height_range_m": [
            float(np.min(com_path[:, 2])),
            float(np.max(com_path[:, 2])),
        ],
        "maximum_absolute_activation": float(np.max(np.abs(controls))),
        "target_displacement_bounds_m": [lower, upper],
        "target_acceptance_tolerance_m": tolerance,
        "reaches_target": reaches_target,
        "maximum_saved_observation_difference": float(
            np.max(np.abs(replayed_flat - expected_flat), initial=0.0)
        ),
        "contact_diagnostics": contact,
        "fraction_of_physics_substeps_with_contact": float(
            contact["substeps_with_contact"] / (horizon * int(physics_data["substeps"]))
        ),
    }
    output = (
        args.output
        if args.output is not None
        else result_path.with_name(result_path.stem + "_verification.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not reaches_target:
        raise RuntimeError("Independent replay did not satisfy the forward target.")


if __name__ == "__main__":
    main()

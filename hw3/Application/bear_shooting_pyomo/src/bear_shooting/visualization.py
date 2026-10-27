"""Validated loading and geometry helpers for saved bear shooting replays.

This module deliberately has no Newton or OpenGL imports. It turns the files
written by a supplied bear optimization script into one checked data object
that can be consumed by an interactive viewer, a USD exporter, or unit tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .export import boundary_triangles

_PARTICLE_SUFFIX = "_particles.npz"
_TASK_SUFFIX = "_task.json"


@dataclass(frozen=True)
class SolutionPaths:
    """The replay and optional companion files belonging to one solve."""

    replay: Path
    result: Path | None
    report: Path | None
    task: Path | None

    @classmethod
    def infer(
        cls,
        source: str | Path,
        *,
        replay: str | Path | None = None,
        report: str | Path | None = None,
        task: str | Path | None = None,
    ) -> SolutionPaths:
        """Infer companion filenames from a result or ``*_particles.npz`` file."""
        source = Path(source).expanduser()
        if source.name.endswith(_PARTICLE_SUFFIX):
            stem = source.name[: -len(_PARTICLE_SUFFIX)]
            base = source.with_name(stem)
            inferred_replay = source
            inferred_result = base.with_suffix(".npz")
        else:
            base = source.with_suffix("")
            inferred_replay = base.with_name(base.name + _PARTICLE_SUFFIX)
            inferred_result = base.with_suffix(".npz")

        replay_path = (
            Path(replay).expanduser() if replay is not None else inferred_replay
        )
        if not replay_path.is_file():
            raise FileNotFoundError(
                f"Particle replay not found: {replay_path}. Run a supplied solve "
                "script so it writes the companion *_particles.npz file, or pass --replay."
            )

        result_path = inferred_result if inferred_result.is_file() else None
        report_path = (
            Path(report).expanduser()
            if report is not None
            else base.with_suffix(".json")
        )
        task_path = (
            Path(task).expanduser()
            if task is not None
            else base.with_name(base.name + _TASK_SUFFIX)
        )
        if report is not None and not report_path.is_file():
            raise FileNotFoundError(f"Result report not found: {report_path}")
        if task is not None and not task_path.is_file():
            raise FileNotFoundError(f"Task description not found: {task_path}")
        return cls(
            replay=replay_path.resolve(),
            result=None if result_path is None else result_path.resolve(),
            report=report_path.resolve() if report_path.is_file() else None,
            task=task_path.resolve() if task_path.is_file() else None,
        )


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _finite_array(value, shape, label):
    array = np.asarray(value, dtype=float)
    try:
        array = np.broadcast_to(array, shape).copy()
    except ValueError as exc:
        raise ValueError(f"{label} must be broadcastable to shape {shape}.") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values.")
    return array


def physical_violation(values, lower, upper) -> np.ndarray:
    """Return componentwise physical bound violation.

    JSON ``null`` bounds are interpreted as unbounded sides.
    """
    values = np.asarray(values, dtype=float).reshape(-1)

    def bounds(raw, default):
        if raw is None:
            return np.full(values.shape, default, dtype=float)
        raw = list(raw)
        if len(raw) != len(values):
            raise ValueError("Constraint values and bounds must have equal lengths.")
        return np.asarray(
            [default if value is None else value for value in raw], dtype=float
        )

    lower_array = bounds(lower, -np.inf)
    upper_array = bounds(upper, np.inf)
    if np.any(lower_array > upper_array):
        raise ValueError("Constraint lower bounds exceed upper bounds.")
    return np.maximum(np.maximum(lower_array - values, values - upper_array), 0.0)


def tolerance_box_lines(center, half_extent, *, display_scale: float = 1.0):
    """Return starts/ends for the twelve edges of an axis-aligned box.

    ``display_scale=1`` is the exact feasible box.  Other values are useful
    for a deliberately labeled visual aid but must not be described as the
    true constraint region.
    """
    center = _finite_array(center, (3,), "box center")
    half_extent = _finite_array(half_extent, (3,), "box half extent")
    if np.any(half_extent < 0):
        raise ValueError("Box half extents cannot be negative.")
    if not np.isfinite(display_scale) or display_scale <= 0:
        raise ValueError("display_scale must be finite and positive.")
    extent = half_extent * float(display_scale)
    corners = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        dtype=float,
    )
    corners = center + corners * extent
    edge_indices = []
    for first in range(8):
        for axis_mask in (4, 2, 1):
            second = first ^ axis_mask
            if first < second:
                edge_indices.append((first, second))
    edges = np.asarray(edge_indices, dtype=int)
    if edges.shape != (12, 2):
        raise AssertionError("A box must have exactly twelve edges.")
    return corners[edges[:, 0]], corners[edges[:, 1]]


def marker_cross_lines(center, radius):
    """Return a three-axis target glyph; its radius is not a tolerance."""
    center = _finite_array(center, (3,), "marker center")
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Marker radius must be finite and positive.")
    offsets = np.eye(3) * radius
    return center - offsets, center + offsets


def x_target_slab_lines(x_lower, x_upper, y_bounds, z_bounds):
    """Return the wireframe edges of a displayed terminal-x target slab.

    The two x coordinates are physical bounds.  The y/z bounds only choose a
    useful finite drawing extent for planes that are mathematically unbounded
    in those directions.
    """
    x_lower = float(x_lower)
    x_upper = float(x_upper)
    y_bounds = _finite_array(y_bounds, (2,), "target-plane y bounds")
    z_bounds = _finite_array(z_bounds, (2,), "target-plane z bounds")
    if not np.isfinite(x_lower) or not np.isfinite(x_upper):
        raise ValueError("Target-slab x bounds must be finite.")
    if x_lower > x_upper:
        raise ValueError("Target-slab lower x bound cannot exceed its upper bound.")
    if y_bounds[0] >= y_bounds[1] or z_bounds[0] >= z_bounds[1]:
        raise ValueError("Target-plane display bounds must have positive width.")

    def rectangle(x):
        return np.array(
            [
                [x, y_bounds[0], z_bounds[0]],
                [x, y_bounds[1], z_bounds[0]],
                [x, y_bounds[1], z_bounds[1]],
                [x, y_bounds[0], z_bounds[1]],
            ],
            dtype=float,
        )

    lower = rectangle(x_lower)
    perimeter = np.array(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=int)
    starts = [lower[perimeter[:, 0]]]
    ends = [lower[perimeter[:, 1]]]
    if x_upper > x_lower:
        upper = rectangle(x_upper)
        starts.extend((upper[perimeter[:, 0]], lower))
        ends.extend((upper[perimeter[:, 1]], upper))
    return np.vstack(starts), np.vstack(ends)


def x_interval_error_segment(point, x_lower, x_upper):
    """Return the shortest x-only segment from a point to an x interval.

    A point already inside the interval has zero physical error, represented
    by empty arrays rather than a zero-length line with undefined orientation.
    """
    point = _finite_array(point, (3,), "x-interval point")
    x_lower = float(x_lower)
    x_upper = float(x_upper)
    if not np.isfinite(x_lower) or not np.isfinite(x_upper) or x_lower > x_upper:
        raise ValueError("The x interval must have finite, ordered bounds.")
    nearest_x = float(np.clip(point[0], x_lower, x_upper))
    if nearest_x == point[0]:
        return np.empty((0, 3)), np.empty((0, 3))
    target = point.copy()
    target[0] = nearest_x
    return point.reshape(1, 3), target.reshape(1, 3)


def polyline_segments(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("Polyline points must be a finite (N,3) array.")
    if len(points) < 2:
        return np.empty((0, 3)), np.empty((0, 3))
    return points[:-1].copy(), points[1:].copy()


@dataclass(frozen=True)
class ForwardTarget:
    """A terminal bound on one landmark's x displacement."""

    part: str
    start: np.ndarray
    minimum_displacement: float
    maximum_displacement: float
    body_length: float | None

    @property
    def lower_x(self):
        return float(self.start[0] + self.minimum_displacement)

    @property
    def upper_x(self):
        return float(self.start[0] + self.maximum_displacement)


@dataclass
class ReplayData:
    """A checked particle replay with its task and solver diagnostics."""

    paths: SolutionPaths
    particles: np.ndarray
    tetrahedra: np.ndarray
    surface_triangles: np.ndarray
    times: np.ndarray
    tet_to_channel: np.ndarray | None
    metadata: dict[str, Any]
    task: dict[str, Any]
    report: dict[str, Any]
    controls: np.ndarray | None
    actuator_names: tuple[str, ...]
    parts: dict[str, dict[str, np.ndarray]]
    targets: dict[str, np.ndarray]
    target_tolerances: dict[str, np.ndarray]
    landmark_trajectories: dict[str, np.ndarray]
    forward_target: ForwardTarget | None
    control_lower: np.ndarray | None
    control_upper: np.ndarray | None

    @classmethod
    def load(cls, paths: SolutionPaths) -> ReplayData:
        with np.load(paths.replay, allow_pickle=False) as replay:
            required = {"particles", "tetrahedra", "times"}
            missing = sorted(required - set(replay.files))
            if missing:
                raise ValueError(
                    f"Replay {paths.replay} is missing: {', '.join(missing)}"
                )
            particles = np.asarray(replay["particles"], dtype=float)
            tetrahedra = np.asarray(replay["tetrahedra"])
            times = np.asarray(replay["times"], dtype=float).reshape(-1)
            tet_to_channel = (
                np.asarray(replay["tet_to_channel"], dtype=np.int32).reshape(-1)
                if "tet_to_channel" in replay.files
                else None
            )
            metadata = {}
            if "metadata" in replay.files:
                try:
                    metadata = json.loads(str(np.asarray(replay["metadata"]).item()))
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Replay metadata in {paths.replay} is invalid JSON."
                    ) from exc

        if particles.ndim != 3 or particles.shape[2] != 3 or particles.shape[0] < 2:
            raise ValueError("Replay particles must have shape (K,N,3) with K >= 2.")
        if not np.all(np.isfinite(particles)):
            raise ValueError("Replay particles contain nonfinite values.")
        if (
            len(times) != len(particles)
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(
                "Replay times must be finite, strictly increasing, and match particle knots."
            )
        if (
            tetrahedra.ndim != 2
            or tetrahedra.shape[1] != 4
            or tetrahedra.dtype.kind not in "iu"
        ):
            raise ValueError("Replay tetrahedra must be an integer (T,4) array.")
        tetrahedra = tetrahedra.astype(np.int32, copy=False)
        if np.any(tetrahedra < 0) or np.any(tetrahedra >= particles.shape[1]):
            raise ValueError(
                "Replay tetrahedra contain an out-of-range particle index."
            )
        if tet_to_channel is not None and len(tet_to_channel) != len(tetrahedra):
            raise ValueError("tet_to_channel must contain one entry per tetrahedron.")

        task_data = _read_json(paths.task)
        report_data = _read_json(paths.report)
        raw_parts = task_data.get("parts") or metadata.get("parts") or {}
        if not isinstance(raw_parts, dict):
            raise TypeError("Task/replay parts metadata must be a mapping.")
        parts = {}
        for name, definition in raw_parts.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(definition, dict)
            ):
                raise ValueError(
                    "Every part needs a nonempty name and a metadata object."
                )
            indices = np.asarray(definition.get("indices"), dtype=np.int64).reshape(-1)
            if (
                not len(indices)
                or np.any(indices < 0)
                or np.any(indices >= particles.shape[1])
            ):
                raise ValueError(f"Part {name!r} has invalid particle indices.")
            weights = np.asarray(
                definition.get("weights", np.ones(len(indices))), dtype=float
            ).reshape(-1)
            if (
                len(weights) != len(indices)
                or not np.all(np.isfinite(weights))
                or np.any(weights < 0)
            ):
                raise ValueError(f"Part {name!r} has invalid weights.")
            total = float(np.sum(weights))
            if total <= 0:
                raise ValueError(f"Part {name!r} weights must have positive sum.")
            parts[name] = {
                "indices": indices.astype(np.int32),
                "weights": weights / total,
            }

        raw_targets = task_data.get("targets", {})
        if not isinstance(raw_targets, dict):
            raise TypeError("Task targets must be a mapping from part names to XYZ.")
        targets = {}
        for name, target in raw_targets.items():
            if name not in parts:
                raise ValueError(
                    f"Target {name!r} has no matching material-part definition."
                )
            targets[name] = _finite_array(target, (3,), f"target {name!r}")

        raw_tolerances = task_data.get(
            "target_tolerances", task_data.get("position_tolerance", 0.0)
        )
        target_tolerances = {}
        for name in targets:
            value = (
                raw_tolerances.get(name, 0.0)
                if isinstance(raw_tolerances, dict)
                else raw_tolerances
            )
            tolerance = _finite_array(value, (3,), f"target tolerance {name!r}")
            if np.any(tolerance < 0):
                raise ValueError(f"Target tolerance {name!r} cannot be negative.")
            target_tolerances[name] = tolerance

        landmark_trajectories = {}
        for name, definition in parts.items():
            selected = particles[:, definition["indices"], :]
            landmark_trajectories[name] = np.einsum(
                "kij,i->kj", selected, definition["weights"]
            )

        forward_target = None
        raw_forward_target = task_data.get("forward_target")
        if raw_forward_target is not None:
            if not isinstance(raw_forward_target, dict):
                raise TypeError("Task forward_target must be a metadata object.")
            part = raw_forward_target.get("part")
            if not isinstance(part, str) or part not in parts:
                raise ValueError(
                    "forward_target.part must name a registered material part."
                )
            if raw_forward_target.get("axis") != "x":
                raise ValueError("The replay viewer supports only an x-axis forward target.")
            start = _finite_array(
                raw_forward_target.get("start_m"), (3,), "forward-target start_m"
            )
            minimum = float(raw_forward_target.get("minimum_displacement_m"))
            maximum = float(raw_forward_target.get("maximum_displacement_m"))
            if (
                not np.isfinite(minimum)
                or not np.isfinite(maximum)
                or minimum > maximum
            ):
                raise ValueError(
                    "Forward-target displacement bounds must be finite and ordered."
                )
            initial_part = landmark_trajectories[part][0]
            if not np.allclose(start, initial_part, rtol=1e-5, atol=1e-5):
                raise ValueError(
                    "forward_target.start_m does not match the saved part position."
                )
            expected_target_x = float(start[0] + minimum)
            target_x = float(raw_forward_target.get("target_x_m", expected_target_x))
            if not np.isfinite(target_x) or not np.isclose(
                target_x, expected_target_x, rtol=1e-10, atol=1e-10
            ):
                raise ValueError(
                    "forward_target.target_x_m must equal start_x plus the minimum "
                    "displacement."
                )
            raw_body_length = raw_forward_target.get("body_length_m")
            body_length = None if raw_body_length is None else float(raw_body_length)
            if body_length is not None and (
                not np.isfinite(body_length) or body_length <= 0
            ):
                raise ValueError("forward_target.body_length_m must be positive.")
            forward_target = ForwardTarget(
                part=part,
                start=start,
                minimum_displacement=minimum,
                maximum_displacement=maximum,
                body_length=body_length,
            )

        controls = None
        actuator_names = tuple(metadata.get("actuators", ()))
        if paths.result is not None:
            with np.load(paths.result, allow_pickle=False) as result:
                if "times" in result.files:
                    result_times = np.asarray(result["times"], dtype=float).reshape(-1)
                    if result_times.shape != times.shape or not np.allclose(
                        result_times, times, rtol=0, atol=1e-12
                    ):
                        raise ValueError(
                            "Result and particle replay use different knot times."
                        )
                if "controls" in result.files:
                    controls = np.asarray(result["controls"], dtype=float)
                    if (
                        controls.ndim != 2
                        or controls.shape[0] != len(times) - 1
                        or not np.all(np.isfinite(controls))
                    ):
                        raise ValueError(
                            "Result controls must be finite with shape (K-1,A)."
                        )
                if "actuator_names" in result.files:
                    actuator_names = tuple(
                        str(value) for value in result["actuator_names"].tolist()
                    )
        if (
            controls is not None
            and actuator_names
            and controls.shape[1] != len(actuator_names)
        ):
            raise ValueError("Actuator names do not match the control array width.")
        if tet_to_channel is not None and controls is not None:
            active = tet_to_channel[tet_to_channel >= 0]
            if len(active) and int(np.max(active)) >= controls.shape[1]:
                raise ValueError("tet_to_channel references a missing control channel.")

        control_lower = control_upper = None
        raw_control_bounds = task_data.get("control_bounds")
        if raw_control_bounds is not None:
            if controls is None or not isinstance(raw_control_bounds, dict):
                raise ValueError(
                    "Saved control bounds require result controls and lower/upper mappings."
                )
            control_lower = _finite_array(
                raw_control_bounds.get("lower"), controls.shape, "control lower bounds"
            )
            control_upper = _finite_array(
                raw_control_bounds.get("upper"), controls.shape, "control upper bounds"
            )
            if np.any(control_lower > control_upper):
                raise ValueError("Saved control lower bounds exceed upper bounds.")

        return cls(
            paths=paths,
            particles=particles,
            tetrahedra=tetrahedra,
            surface_triangles=boundary_triangles(particles[0], tetrahedra),
            times=times,
            tet_to_channel=tet_to_channel,
            metadata=metadata,
            task=task_data,
            report=report_data,
            controls=controls,
            actuator_names=actuator_names,
            parts=parts,
            targets=targets,
            target_tolerances=target_tolerances,
            landmark_trajectories=landmark_trajectories,
            forward_target=forward_target,
            control_lower=control_lower,
            control_upper=control_upper,
        )

    @property
    def knot_count(self):
        return len(self.times)

    @property
    def scene_diagonal(self):
        extent = np.ptp(self.particles.reshape(-1, 3), axis=0)
        return float(np.linalg.norm(extent))

    def target_error(self, part, knot=-1):
        if part not in self.targets:
            raise KeyError(f"Unknown target part {part!r}.")
        return self.landmark_trajectories[part][knot] - self.targets[part]

    def target_violation(self, part, knot=-1):
        error = self.target_error(part, knot)
        tolerance = self.target_tolerances[part]
        return np.maximum(np.abs(error) - tolerance, 0.0)

    def forward_displacement(self, knot=-1):
        """Return the selected part's x displacement from its saved start."""
        if self.forward_target is None:
            raise ValueError("This replay has no forward target.")
        current = self.landmark_trajectories[self.forward_target.part][knot]
        return float(current[0] - self.forward_target.start[0])

    def forward_violation(self, knot=-1):
        """Return physical violation of the terminal x-displacement interval."""
        if self.forward_target is None:
            raise ValueError("This replay has no forward target.")
        displacement = self.forward_displacement(knot)
        return float(
            max(
                self.forward_target.minimum_displacement - displacement,
                displacement - self.forward_target.maximum_displacement,
                0.0,
            )
        )

    def report_constraint_arrays(self):
        """Return ``value/lower/upper`` rows for Newton heatmaps."""
        result = {}
        constraints = self.report.get("constraints", {})
        if not isinstance(constraints, dict):
            return result
        for name, entry in constraints.items():
            if not isinstance(entry, dict) or "value" not in entry:
                continue
            values = np.asarray(entry["value"], dtype=float).reshape(-1)
            if not np.all(np.isfinite(values)):
                continue

            def row(raw, expected=values):
                if raw is None or len(raw) != len(expected):
                    return np.full(expected.shape, np.nan)
                return np.asarray(
                    [np.nan if value is None else value for value in raw], dtype=float
                )

            result[str(name)] = np.vstack(
                (values, row(entry.get("lower")), row(entry.get("upper")))
            )
        return result

    def normalized_controls(self):
        """Map saved controls to [-1,1] at their bounds, when bounds are available."""
        if (
            self.controls is None
            or self.control_lower is None
            or self.control_upper is None
        ):
            return None
        width = self.control_upper - self.control_lower
        normalized = np.zeros_like(self.controls)
        nonzero = width > 0
        normalized[nonzero] = (
            2.0
            * (self.controls[nonzero] - self.control_lower[nonzero])
            / width[nonzero]
            - 1.0
        )
        return normalized

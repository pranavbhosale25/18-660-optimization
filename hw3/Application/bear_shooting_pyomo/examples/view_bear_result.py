"""Replay a saved locomotion solve in Newton's viewer and show its constraints.

The viewer consumes files written by ``bear_run.py``. It does not rerun the
simulation or optimization, so the displayed particle states are the exact
saved control-knot states.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from bear_shooting.visualization import (
    ReplayData,
    SolutionPaths,
    marker_cross_lines,
    polyline_segments,
    tolerance_box_lines,
    x_interval_error_segment,
    x_target_slab_lines,
)

_SURFACE_COLOR = (0.45, 0.24, 0.10)
_GROUND_COLOR = (0.28, 0.30, 0.33)
_TARGET_COLOR = (0.10, 0.75, 1.00)
_INTERMEDIATE_COLOR = (1.00, 0.68, 0.12)
_EXACT_COLOR = (0.15, 0.90, 0.30)
_ACCEPTED_COLOR = (1.00, 0.72, 0.05)
_FAILED_COLOR = (1.00, 0.18, 0.12)
_AID_COLOR = (0.85, 0.25, 1.00)
_START_COLOR = (0.95, 0.95, 0.95)


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


def _nonnegative_float(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return result


def create_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "result",
        nargs="?",
        type=Path,
        default=Path("results/bear_run_quick.npz"),
        help="result .npz or companion *_particles.npz",
    )
    parser.add_argument(
        "--replay", type=Path, help="override the inferred *_particles.npz"
    )
    parser.add_argument("--task", type=Path, help="override the inferred *_task.json")
    parser.add_argument("--report", type=Path, help="override the inferred result JSON")
    parser.add_argument("--viewer", choices=("gl", "usd", "null"), default="gl")
    parser.add_argument(
        "--device", default="cpu", help="Warp device used to feed the viewer"
    )
    parser.add_argument(
        "--output-path", type=Path, default=Path("results/bear_run_replay.usda")
    )
    parser.add_argument("--width", type=_positive_int, default=1600)
    parser.add_argument("--height", type=_positive_int, default=900)
    parser.add_argument("--render-fps", type=_positive_float, default=60.0)
    parser.add_argument(
        "--num-frames",
        type=_positive_int,
        help="stop after this many rendered frames; defaults to one pass for USD/null and unlimited for GL",
    )
    parser.add_argument(
        "--frames-per-knot",
        type=_positive_int,
        help=(
            "rendered frames for each saved control knot; by default this is "
            "derived from the saved knot times and --render-fps"
        ),
    )
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="loop after the last saved knot; defaults on for GL and off for USD/null",
    )
    parser.add_argument(
        "--paused", action="store_true", help="start the interactive viewer paused"
    )
    parser.add_argument(
        "--headless", action="store_true", help="create a headless GL context"
    )
    parser.add_argument(
        "--marker-radius",
        type=_positive_float,
        help="visible target glyph radius in metres",
    )
    parser.add_argument(
        "--tolerance-aid-scale",
        type=_nonnegative_float,
        default=50.0,
        help="also draw a labeled enlarged tolerance box; zero or one disables it",
    )
    parser.add_argument(
        "--show-ground", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--show-patches", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--show-trails", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--test", action="store_true", help=argparse.SUPPRESS)
    return parser


def resolve_frames_per_knot(times, render_fps, requested=None):
    """Choose a knot hold count that reproduces physical time by default."""
    if requested is not None:
        return int(requested)
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2:
        return 1
    steps = np.diff(times)
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0.0):
        raise ValueError("Saved replay times must be finite and strictly increasing.")
    return max(1, int(round(float(np.median(steps)) * float(render_fps))))


def _warp_arrays(wp, viewer, values, dtype):
    return wp.array(np.ascontiguousarray(values), dtype=dtype, device=viewer.device)


def _nice_grid(data: ReplayData):
    points = data.particles.reshape(-1, 3)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    if data.forward_target is not None:
        minimum[0] = min(minimum[0], data.forward_target.lower_x)
        maximum[0] = max(maximum[0], data.forward_target.upper_x)
    span = max(float(maximum[0] - minimum[0]), float(maximum[1] - minimum[1]), 1.0)
    raw_step = span / 10.0
    exponent = 10.0 ** math.floor(math.log10(raw_step))
    fraction = raw_step / exponent
    multiplier = (
        1.0
        if fraction <= 1.0
        else 2.0
        if fraction <= 2.0
        else 5.0
        if fraction <= 5.0
        else 10.0
    )
    step = multiplier * exponent
    padding = max(step, 0.1 * span)
    x0 = math.floor((minimum[0] - padding) / step) * step
    x1 = math.ceil((maximum[0] + padding) / step) * step
    y0 = math.floor((minimum[1] - padding) / step) * step
    y1 = math.ceil((maximum[1] + padding) / step) * step
    xs = np.arange(x0, x1 + 0.5 * step, step)
    ys = np.arange(y0, y1 + 0.5 * step, step)
    starts = np.vstack(
        (
            np.column_stack((xs, np.full_like(xs, y0), np.zeros_like(xs))),
            np.column_stack((np.full_like(ys, x0), ys, np.zeros_like(ys))),
        )
    )
    ends = np.vstack(
        (
            np.column_stack((xs, np.full_like(xs, y1), np.zeros_like(xs))),
            np.column_stack((np.full_like(ys, x1), ys, np.zeros_like(ys))),
        )
    )
    return starts.astype(np.float32), ends.astype(np.float32)


class BearReplayViewer:
    """Newton-viewer adapter for exact saved control-knot states."""

    def __init__(self, viewer, data: ReplayData, args, wp):
        self.viewer = viewer
        self.data = data
        self.args = args
        self.wp = wp
        self.frames_per_knot = args.frames_per_knot
        self.loop = (args.viewer == "gl") if args.loop is None else args.loop
        self.tick = 0
        self.rendered_frames = 0
        self.last_scalar_knot = None
        self.show_ground = args.show_ground
        self.show_patches = args.show_patches
        self.show_trails = args.show_trails
        self.marker_radius = args.marker_radius or max(
            0.01, 0.008 * data.scene_diagonal
        )
        self.tolerance_aid_scale = args.tolerance_aid_scale
        self.total_ticks = data.knot_count * self.frames_per_knot
        self.forward_plane_yz_bounds = self._forward_plane_bounds()
        self.faces_wp = _warp_arrays(
            wp, viewer, data.surface_triangles.reshape(-1), wp.int32
        )
        ground_starts, ground_ends = _nice_grid(data)
        self.ground_starts_wp = _warp_arrays(wp, viewer, ground_starts, wp.vec3)
        self.ground_ends_wp = _warp_arrays(wp, viewer, ground_ends, wp.vec3)
        self.ground_colors_wp = _warp_arrays(
            wp,
            viewer,
            np.broadcast_to(_GROUND_COLOR, ground_starts.shape).astype(np.float32),
            wp.vec3,
        )
        self._diagnostics_logged = False

        if hasattr(viewer, "set_camera"):
            # This is the camera used by Newton's own differentiable bear example.
            viewer.set_camera(pos=wp.vec3(25.0, -20.0, 10.0), pitch=-20.0, yaw=130.0)

    @property
    def knot(self):
        return min(self.tick // self.frames_per_knot, self.data.knot_count - 1)

    def step(self):
        if self.tick + 1 < self.total_ticks:
            self.tick += 1
        elif self.loop:
            self.tick = 0

    def _constraint_color(self, part, knot):
        if knot != self.data.knot_count - 1:
            return _INTERMEDIATE_COLOR
        maximum = float(np.max(self.data.target_violation(part, knot), initial=0.0))
        if maximum <= 1e-12:
            return _EXACT_COLOR
        if bool(self.data.report.get("success", False)):
            return _ACCEPTED_COLOR
        return _FAILED_COLOR

    def _forward_plane_bounds(self):
        """Choose finite y/z drawing extents for an x-only target."""
        points = self.data.particles.reshape(-1, 3)
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)
        span = np.maximum(maximum - minimum, 0.1)
        margin = max(0.05 * self.data.scene_diagonal, 0.1)
        y_bounds = np.array(
            [minimum[1] - max(margin, 0.1 * span[1]),
             maximum[1] + max(margin, 0.1 * span[1])]
        )
        z_bounds = np.array(
            [min(0.0, minimum[2]), maximum[2] + max(margin, 0.1 * span[2])]
        )
        return y_bounds, z_bounds

    def _forward_constraint_color(self, knot):
        if knot != self.data.knot_count - 1:
            return _INTERMEDIATE_COLOR
        if self.data.forward_violation(knot) <= 1e-12:
            return _EXACT_COLOR
        if bool(self.data.report.get("success", False)):
            return _ACCEPTED_COLOR
        return _FAILED_COLOR

    def _log_diagnostics_once(self):
        if self._diagnostics_logged:
            return
        self._diagnostics_logged = True
        if self.data.controls is not None:
            self.viewer.log_array("controls/raw actuation", self.data.controls)
        normalized = self.data.normalized_controls()
        if normalized is not None:
            self.viewer.log_array("controls/bound-normalized (-1 to 1)", normalized)
        for name, array in self.data.report_constraint_arrays().items():
            self.viewer.log_array(f"constraints/{name} [value; lower; upper]", array)
        iterations = self.data.report.get("history", {}).get("iterations", [])
        if isinstance(iterations, list):
            for item in iterations:
                if not isinstance(item, dict):
                    continue
                for key in ("objective", "primal_infeasibility", "dual_infeasibility"):
                    value = item.get(key)
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        self.viewer.log_scalar(f"IPOPT/{key}", value)

    def _log_target_geometry(self, points, knot):
        wp = self.wp
        target_points = []
        current_points = []
        current_colors = []
        box_starts = []
        box_ends = []
        box_colors = []
        aid_starts = []
        aid_ends = []
        glyph_starts = []
        glyph_ends = []
        error_starts = []
        error_ends = []
        error_colors = []
        trail_starts = []
        trail_ends = []
        patch_points = []
        patch_colors = []

        for part, target in self.data.targets.items():
            current = self.data.landmark_trajectories[part][knot]
            color = self._constraint_color(part, knot)
            target_points.append(target)
            current_points.append(current)
            current_colors.append(color)
            starts, ends = tolerance_box_lines(
                target, self.data.target_tolerances[part]
            )
            box_starts.extend(starts)
            box_ends.extend(ends)
            box_colors.extend([color] * len(starts))
            glyph_start, glyph_end = marker_cross_lines(target, self.marker_radius)
            glyph_starts.extend(glyph_start)
            glyph_ends.extend(glyph_end)
            error_starts.append(current)
            error_ends.append(target)
            error_colors.append(color)
            trail_start, trail_end = polyline_segments(
                self.data.landmark_trajectories[part]
            )
            trail_starts.extend(trail_start)
            trail_ends.extend(trail_end)
            definition = self.data.parts[part]
            selected = points[definition["indices"]]
            patch_points.extend(selected)
            patch_colors.extend([color] * len(selected))
            if self.tolerance_aid_scale > 1.0:
                starts, ends = tolerance_box_lines(
                    target,
                    self.data.target_tolerances[part],
                    display_scale=self.tolerance_aid_scale,
                )
                aid_starts.extend(starts)
                aid_ends.extend(ends)

        def vec3(values):
            return _warp_arrays(
                wp,
                self.viewer,
                np.asarray(values, dtype=np.float32).reshape(-1, 3),
                wp.vec3,
            )

        def solid_color(color, count):
            return vec3(np.broadcast_to(color, (count, 3)))

        if target_points:
            self.viewer.log_points(
                "/constraints/target_markers",
                vec3(target_points),
                radii=0.55 * self.marker_radius,
                colors=solid_color(_TARGET_COLOR, len(target_points)),
            )
            self.viewer.log_points(
                "/constraints/current_landmarks",
                vec3(current_points),
                radii=0.38 * self.marker_radius,
                colors=vec3(current_colors),
            )
            self.viewer.log_lines(
                "/constraints/exact_tolerance_boxes",
                vec3(box_starts),
                vec3(box_ends),
                vec3(box_colors),
            )
            self.viewer.log_lines(
                "/constraints/target_glyphs",
                vec3(glyph_starts),
                vec3(glyph_ends),
                solid_color(_TARGET_COLOR, len(glyph_starts)),
            )
            self.viewer.log_lines(
                "/constraints/current_to_target",
                vec3(error_starts),
                vec3(error_ends),
                vec3(error_colors),
            )
            self.viewer.log_lines(
                "/constraints/trajectory_trails",
                vec3(trail_starts) if trail_starts else None,
                vec3(trail_ends) if trail_ends else None,
                solid_color(_TARGET_COLOR, len(trail_starts)) if trail_starts else None,
                hidden=not self.show_trails,
            )
            self.viewer.log_points(
                "/constraints/material_patch_vertices",
                vec3(patch_points),
                radii=0.12 * self.marker_radius,
                colors=vec3(patch_colors),
                hidden=not self.show_patches,
            )
            if aid_starts:
                self.viewer.log_lines(
                    "/constraints/enlarged_tolerance_aid",
                    vec3(aid_starts),
                    vec3(aid_ends),
                    solid_color(_AID_COLOR, len(aid_starts)),
                )

    def _log_forward_target_geometry(self, knot):
        target = self.data.forward_target
        if target is None:
            return

        wp = self.wp

        def vec3(values):
            return _warp_arrays(
                wp,
                self.viewer,
                np.asarray(values, dtype=np.float32).reshape(-1, 3),
                wp.vec3,
            )

        def solid_color(color, count):
            return vec3(np.broadcast_to(color, (count, 3)))

        y_bounds, z_bounds = self.forward_plane_yz_bounds
        slab_starts, slab_ends = x_target_slab_lines(
            target.lower_x, target.upper_x, y_bounds, z_bounds
        )
        self.viewer.log_lines(
            "/constraints/forward_x_target_slab",
            vec3(slab_starts),
            vec3(slab_ends),
            solid_color(_TARGET_COLOR, len(slab_starts)),
        )

        start_glyph_starts, start_glyph_ends = marker_cross_lines(
            target.start, self.marker_radius
        )
        self.viewer.log_lines(
            "/constraints/forward_start_com_glyph",
            vec3(start_glyph_starts),
            vec3(start_glyph_ends),
            solid_color(_START_COLOR, len(start_glyph_starts)),
        )
        self.viewer.log_points(
            "/constraints/forward_start_com",
            vec3([target.start]),
            radii=0.55 * self.marker_radius,
            colors=solid_color(_START_COLOR, 1),
        )

        trajectory = self.data.landmark_trajectories[target.part]
        current = trajectory[knot]
        color = self._forward_constraint_color(knot)
        self.viewer.log_points(
            "/constraints/forward_current_com",
            vec3([current]),
            radii=0.45 * self.marker_radius,
            colors=solid_color(color, 1),
        )
        trail_starts, trail_ends = polyline_segments(trajectory)
        self.viewer.log_lines(
            "/constraints/forward_com_trajectory",
            vec3(trail_starts),
            vec3(trail_ends),
            solid_color(_TARGET_COLOR, len(trail_starts)),
            hidden=not self.show_trails,
        )

        segment_starts, segment_ends = x_interval_error_segment(
            current, target.lower_x, target.upper_x
        )
        distance_path = "/constraints/forward_com_to_x_interval"
        if not len(segment_starts):
            # Newton's USD line backend represents segments as oriented
            # capsules; a zero-length capsule has an undefined orientation.
            # Hide the batch at feasible knots instead of exporting a NaN.
            self.viewer.log_lines(
                distance_path, None, None, None, hidden=True
            )
        else:
            self.viewer.log_lines(
                distance_path,
                vec3(segment_starts),
                vec3(segment_ends),
                solid_color(color, 1),
            )

    def _log_knot_scalars(self, knot):
        if knot == self.last_scalar_knot:
            return
        clear = self.last_scalar_knot is not None and knot < self.last_scalar_knot
        self.last_scalar_knot = knot
        for part in self.data.targets:
            error = self.data.target_error(part, knot)
            violation = self.data.target_violation(part, knot)
            self.viewer.log_scalar(
                f"targets/{part}/distance_to_terminal_target_m",
                float(np.linalg.norm(error)),
                clear=clear,
            )
            # This is a true constraint violation only at the terminal knot.
            if knot == self.data.knot_count - 1:
                self.viewer.log_scalar(
                    f"targets/{part}/terminal_box_violation_m",
                    float(np.max(violation, initial=0.0)),
                    clear=clear,
                )
        if self.data.forward_target is not None:
            self.viewer.log_scalar(
                "forward_target/com_x_displacement_m",
                self.data.forward_displacement(knot),
                clear=clear,
            )
            if knot == self.data.knot_count - 1:
                self.viewer.log_scalar(
                    "forward_target/terminal_x_interval_violation_m",
                    self.data.forward_violation(knot),
                    clear=clear,
                )

    def render(self):
        knot = self.knot
        points = self.data.particles[knot]
        wp = self.wp
        self.viewer.begin_frame(float(self.data.times[knot]))
        self.viewer.log_mesh(
            "/bear/surface",
            _warp_arrays(wp, self.viewer, points.astype(np.float32), wp.vec3),
            self.faces_wp,
            color=_SURFACE_COLOR,
            backface_culling=False,
        )
        self.viewer.log_lines(
            "/world/ground_grid",
            self.ground_starts_wp,
            self.ground_ends_wp,
            self.ground_colors_wp,
            hidden=not self.show_ground,
        )
        self._log_target_geometry(points, knot)
        self._log_forward_target_geometry(knot)
        self._log_diagnostics_once()
        self._log_knot_scalars(knot)
        self.viewer.end_frame()
        self.rendered_frames += 1

    def gui(self, ui):
        ui.text("Saved control-knot replay (no re-simulation)")
        changed, selected = ui.slider_int(
            "Control knot", self.knot, 0, self.data.knot_count - 1
        )
        if changed:
            self.tick = int(selected) * self.frames_per_knot
        changed, value = ui.checkbox("Loop", self.loop)
        if changed:
            self.loop = value
        changed, value = ui.checkbox("Ground grid", self.show_ground)
        if changed:
            self.show_ground = value
        changed, value = ui.checkbox("Material patch vertices", self.show_patches)
        if changed:
            self.show_patches = value
        changed, value = ui.checkbox("Landmark trails", self.show_trails)
        if changed:
            self.show_trails = value
        ui.text(f"Saved time: {self.data.times[self.knot]:.6g} s")
        if self.data.targets:
            ui.text("Cyan cross/sphere: visible target marker (not tolerance)")
            ui.text("Small colored wireframe: exact terminal tolerance box")
            if self.tolerance_aid_scale > 1.0:
                ui.text(
                    f"Magenta wireframe: {self.tolerance_aid_scale:g}x visual aid only"
                )
            ui.text(
                "Gold marker: intermediate state; terminal constraint not active yet"
            )
            ui.text("Final green = exact, amber = accepted roundoff, red = failed")

        if self.data.forward_target is not None:
            target = self.data.forward_target
            ui.separator()
            ui.text("Forward terminal target (x displacement only)")
            ui.text(
                "Cyan wireframe slab: allowed terminal COM x interval; its y/z "
                "drawing extent is not a bound"
            )
            ui.text("White marker: starting COM; colored marker: current COM")
            displacement = self.data.forward_displacement(self.knot)
            ui.text(f"Current COM x displacement: {displacement:.6g} m")
            ui.text(
                "Allowed terminal displacement: "
                f"[{target.minimum_displacement:.6g}, "
                f"{target.maximum_displacement:.6g}] m"
            )
            if self.knot == self.data.knot_count - 1:
                ui.text(
                    "Terminal x-interval violation: "
                    f"{self.data.forward_violation(self.knot):.6g} m"
                )

        if self.data.targets:
            ui.separator()
            ui.text("Terminal target diagnostics")
            for part in self.data.targets:
                error = self.data.target_error(part, self.knot)
                maximum = float(np.max(np.abs(error), initial=0.0))
                label = (
                    "terminal error"
                    if self.knot == self.data.knot_count - 1
                    else "distance-to-target"
                )
                ui.text(f"{part}: {label} max |axis| = {maximum:.6g} m")

        constraints = self.data.report.get("constraints", {})
        if isinstance(constraints, dict) and constraints:
            ui.separator()
            ui.text("Final numerical constraint report")
            for name, entry in constraints.items():
                if isinstance(entry, dict):
                    violation = entry.get("max_physical_violation")
                    if isinstance(violation, (int, float)):
                        ui.text(f"{name}: max violation = {violation:.6g}")

        if self.data.report:
            ui.separator()
            ui.text(f"Solver: {self.data.report.get('status', 'unknown')}")
            ui.text(f"Accepted result: {bool(self.data.report.get('success', False))}")
            objective = self.data.report.get("objective")
            if isinstance(objective, (int, float)):
                ui.text(f"Objective: {objective:.8g}")

    def test_final(self):
        if self.rendered_frames < 1:
            raise AssertionError("The replay viewer rendered no frames.")
        if not np.all(np.isfinite(self.data.particles)):
            raise AssertionError("The replay contains nonfinite particle data.")


def _make_viewer(args, data, wp):
    try:
        from newton.viewer import ViewerGL, ViewerNull, ViewerUSD
    except ImportError as exc:
        raise ImportError(
            "The bear viewer needs Newton 1.5.2. Install the project's viewer extra."
        ) from exc

    wp.init()
    wp.set_device(args.device)
    one_pass = data.knot_count * args.frames_per_knot
    frame_limit = args.num_frames
    if args.viewer in ("usd", "null") and frame_limit is None:
        frame_limit = one_pass
    if args.viewer == "gl":
        viewer = ViewerGL(
            width=args.width,
            height=args.height,
            vsync=True,
            headless=args.headless,
            paused=args.paused,
        )
    elif args.viewer == "usd":
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        viewer = ViewerUSD(
            output_path=str(args.output_path),
            fps=args.render_fps,
            up_axis="Z",
            num_frames=frame_limit,
        )
    else:
        viewer = ViewerNull(num_frames=frame_limit)
    return viewer, frame_limit


def main():
    args = create_parser().parse_args()
    paths = SolutionPaths.infer(
        args.result,
        replay=args.replay,
        report=args.report,
        task=args.task,
    )
    data = ReplayData.load(paths)
    args.frames_per_knot = resolve_frames_per_knot(
        data.times, args.render_fps, args.frames_per_knot
    )

    try:
        import warp as wp
    except ImportError as exc:
        raise ImportError("The bear viewer needs warp-lang 1.16.0.") from exc

    viewer, frame_limit = _make_viewer(args, data, wp)
    example = BearReplayViewer(viewer, data, args, wp)
    if hasattr(viewer, "register_ui_callback"):
        viewer.register_ui_callback(example.gui, position="side")
    try:
        while viewer.is_running() and (
            frame_limit is None or example.rendered_frames < frame_limit
        ):
            frame_start = time.perf_counter()
            if viewer.should_step():
                example.step()
            example.render()
            delay = 1.0 / args.render_fps - (time.perf_counter() - frame_start)
            if delay > 0 and args.viewer == "gl":
                time.sleep(delay)
    finally:
        viewer.close()
    if args.test:
        example.test_final()
    if args.viewer == "usd":
        print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()

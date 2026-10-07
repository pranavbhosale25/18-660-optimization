"""Provided plotting/animation helpers. No student implementation is needed."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

import linkage_core as core


def _endpoint(pose: np.ndarray, length: float, right: bool = True) -> np.ndarray:
    local_x = (0.5 if right else -0.5) * length
    return pose[:2] + local_x * np.array([np.cos(pose[2]), np.sin(pose[2])])


def _bounds(rectangles: np.ndarray, *curves) -> tuple[np.ndarray, np.ndarray]:
    arrays = [rectangles.reshape(-1, 2)]
    arrays.extend(np.asarray(curve).reshape(-1, 2) for curve in curves if curve is not None)
    points = np.concatenate(arrays)
    if not np.all(np.isfinite(points)):
        raise ValueError("Cannot plot nonfinite geometry. Check the simulation first.")
    return points.min(axis=0) - 0.35, points.max(axis=0) + 0.35


def animate_linkage(
    mech: core.FourBarMechanism,
    state_trajectory: core.Array,
    driver_poses: core.Array,
    title: str = "Four-bar linkage",
    effector_path_xy: core.Array | None = None,
    target_curve_xy: core.Array | None = None,
    status_values: core.Array | None = None,
    status_label: str = "value",
    interval_ms: int = 60,
) -> animation.FuncAnimation:
    """Return a live animation; retain it in a variable before displaying/saving.

    Call plt.show() for a live desktop window, or use save_animation_gif /
    save_animation_html to export. The caller is responsible for closing the
    figure. GIF export uses Pillow and needs no external video encoder.
    """
    states = np.asarray(state_trajectory)
    poses = np.asarray(driver_poses)
    if states.ndim != 2 or states.shape[1] != 6 or len(states) == 0:
        raise ValueError("state_trajectory must have nonempty shape (T, 6).")
    if poses.shape != (len(states), 3):
        raise ValueError("driver_poses must have shape (T, 3).")
    rectangles = np.asarray(core.rectangles_from_trajectory(mech, state_trajectory, driver_poses))
    effector = None if effector_path_xy is None else np.asarray(effector_path_xy)
    target = None if target_curve_xy is None else np.asarray(target_curve_xy)
    status = None if status_values is None else np.asarray(status_values)
    if effector is not None and effector.shape != (len(states), 2):
        raise ValueError("effector_path_xy must have shape (T, 2).")
    if status is not None and status.shape != (len(states),):
        raise ValueError("status_values must have shape (T,).")
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive.")
    lo, hi = _bounds(rectangles, effector, target)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), title=title, xlabel="x", ylabel="y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    patches = []
    for i, label in enumerate(("Ground", "Driver", "Coupler", "Follower")):
        patch, = ax.fill(rectangles[0, i, :, 0], rectangles[0, i, :, 1], alpha=0.5, label=label)
        patches.append(patch)
    if target is not None:
        ax.plot(target[:, 0], target[:, 1], linestyle="--", linewidth=1.6, label="Target")
    trace = dot = None
    if effector is not None:
        trace, = ax.plot([], [], linewidth=1.8, label="Traced point")
        dot, = ax.plot([], [], marker="o", linestyle="none", markersize=5)
    joint_line, = ax.plot([], [], marker="o", markersize=4, linewidth=1)
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    ax.legend(loc="lower right", fontsize=8, ncols=2)
    left = np.asarray(mech.ground_left[:2])
    right = np.asarray(mech.ground_right[:2])

    def update(frame: int):
        for i, patch in enumerate(patches):
            patch.set_xy(rectangles[frame, i])
        joints = np.stack((left, _endpoint(poses[frame], float(mech.driver.length)),
                           _endpoint(states[frame, :3], float(mech.coupler.length)), right))
        joint_line.set_data(joints[:, 0], joints[:, 1])
        message = f"frame {frame + 1}/{len(states)}"
        if status is not None:
            message += f"\n{status_label} = {float(status[frame]):.2e}"
        text.set_text(message)
        artists = [*patches, joint_line, text]
        if trace is not None:
            trace.set_data(effector[:frame + 1, 0], effector[:frame + 1, 1])
            dot.set_data([effector[frame, 0]], [effector[frame, 1]])
            artists.extend((trace, dot))
        return tuple(artists)

    anim = animation.FuncAnimation(fig, update, frames=len(states), interval=interval_ms,
                                   blit=False, repeat=True)
    return anim


def plot_linkage_snapshots(
    mech: core.FourBarMechanism,
    state_trajectory: core.Array,
    driver_poses: core.Array,
    title: str = "Linkage snapshots",
    effector_path_xy: core.Array | None = None,
    target_curve_xy: core.Array | None = None,
    num_snapshots: int = 6,
) -> plt.Figure:
    if num_snapshots < 1:
        raise ValueError("num_snapshots must be positive.")
    rectangles = np.asarray(core.rectangles_from_trajectory(mech, state_trajectory, driver_poses))
    states, poses = np.asarray(state_trajectory), np.asarray(driver_poses)
    lo, hi = _bounds(rectangles, effector_path_xy, target_curve_xy)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), title=title, xlabel="x", ylabel="y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.fill(rectangles[0, 0, :, 0], rectangles[0, 0, :, 1], alpha=0.4, label="Ground")
    for frame in np.unique(np.linspace(0, len(states) - 1, num_snapshots, dtype=int)):
        joints = np.stack((np.asarray(mech.ground_left[:2]),
                           _endpoint(poses[frame], float(mech.driver.length)),
                           _endpoint(states[frame, :3], float(mech.coupler.length)),
                           np.asarray(mech.ground_right[:2])))
        ax.plot(joints[:, 0], joints[:, 1], marker="o", markersize=3, linewidth=1, alpha=0.4)
    for path, label, linestyle in ((target_curve_xy, "Target", "--"),
                                   (effector_path_xy, "Current path", "-")):
        if path is not None:
            path = np.asarray(path)
            ax.plot(path[:, 0], path[:, 1], linewidth=2, linestyle=linestyle, label=label)
    ax.legend()
    return fig


def plot_loss_history(history: core.DesignHistory) -> plt.Figure:
    losses = np.asarray(history.loss_history)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    ax.plot(np.arange(len(losses)), losses, label="Evaluated loss")
    ax.plot(np.arange(len(losses)), np.minimum.accumulate(losses), linestyle="--", label="Best so far")
    ax.set(xlabel="Number of design updates", ylabel="Discrete Fréchet distance",
           title="Outer optimization (initial and final designs included)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig


def plot_curve_comparison(target, before, after) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    for path, label, style in ((target, "Target", "--"), (before, "Initial", "-"),
                                (after, "Best evaluated design", "-")):
        path = np.asarray(path)
        ax.plot(path[:, 0], path[:, 1], linestyle=style, linewidth=2, label=label)
    ax.set(xlabel="x", ylabel="y", title="Coupler-point curve synthesis")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig


def save_animation_html(anim: animation.FuncAnimation, filepath: str | Path) -> str:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = anim.to_jshtml()
    path.write_text('<!doctype html><html><head><meta charset="utf-8">'
                    '<title>Four-bar simulation</title></head><body>' + content + '</body></html>',
                    encoding="utf-8")
    return str(path)


def save_animation_gif(anim: animation.FuncAnimation, filepath: str | Path,
                       fps: int = 12, dpi: int = 100) -> str:
    if fps <= 0 or dpi <= 0:
        raise ValueError("fps and dpi must be positive.")
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), writer=animation.PillowWriter(fps=fps), dpi=dpi)
    return str(path)

"""Render convergence plots, saved trajectory snapshots, and a real-time GIF.

No interactive window is needed. All mesh frames come from the saved Newton
particle replay; no physics is rerun or replaced by an illustrative animation.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from bear_shooting.visualization import ReplayData, SolutionPaths


def convergence(data, output):
    rows = data.report.get("history", {}).get("iterations", [])
    if not rows:
        raise ValueError("The native IPOPT report has no iteration history.")
    with (output / "ipopt_iterations.csv").open("w", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    iteration = [row["iteration"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(iteration, [r["objective"] for r in rows], "o-", color="#2563eb", markersize=3)
    axes[0].set(title="Composite loss", ylabel="Jdev + 0.05 Jsmooth")
    axes[1].plot(iteration, [r["primal_infeasibility"] for r in rows], "o-", label="Primal infeasibility", markersize=3)
    axes[1].plot(iteration, [r["dual_infeasibility"] for r in rows], "o-", label="Dual infeasibility", markersize=3)
    axes[1].axhline(5e-4, color="gray", linestyle=":", label="Stage acceptance tolerance")
    axes[1].set(title="IPOPT constraint / optimality residuals", ylabel="IPOPT residual (scaled)")
    axes[1].set_yscale("symlog", linthresh=1e-6)
    largest_residual = max(
        max(row["primal_infeasibility"], row["dual_infeasibility"])
        for row in rows
    )
    axes[1].set_ylim(0.0, max(1e-5, 2.0 * largest_residual))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("IPOPT iteration")
        ax.grid(True, alpha=0.22)
    fig.suptitle("Exercise 3.1 - quick CPU optimization")
    fig.savefig(output / "convergence.png", dpi=180)
    plt.close(fig)

    if all("physical_constraint_values" in row for row in rows):
        # This is the actual terminal motion constraint evaluated at each
        # accepted optimizer iterate, not a simulation-time trajectory.
        target = data.forward_target
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.axhspan(target.minimum_displacement, target.maximum_displacement,
                   color="#bbf7d0", alpha=0.5, label="Allowed terminal COM interval")
        ax.plot(iteration, [r["physical_constraint_values"][0] for r in rows],
                "o-", color="#0891b2", markersize=4, label="Terminal COM displacement")
        ax.set(xlabel="IPOPT iteration", ylabel="Terminal COM displacement (m)",
               title="Forward-target constraint over optimization iterates")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.22)
        fig.savefig(output / "constraint_values.png", dpi=180)
        plt.close(fig)


def motion_plot(data, output):
    target = data.forward_target
    displacement = data.landmark_trajectories[target.part][:, 0] - target.start[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].axhspan(target.minimum_displacement, target.maximum_displacement, color="#bbf7d0", alpha=0.5, label="Terminal target interval")
    axes[0].plot(data.times, displacement, "o-", color="#2563eb", markersize=3, label="Saved COM displacement")
    axes[0].set(xlabel="Simulation time (s)", ylabel="Forward COM displacement (m)", title="Motion under final controls")
    axes[0].legend(fontsize=8)
    for a, name in enumerate(data.actuator_names):
        axes[1].step(data.times, np.r_[data.controls[:, a], data.controls[-1, a]], where="post", label=name)
    axes[1].set(xlabel="Simulation time (s)", ylabel="Activation", title="Optimized piecewise-constant controls")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(True, alpha=0.22)
    fig.suptitle("The COM target applies only at the terminal time")
    fig.savefig(output / "motion_and_controls.png", dpi=180)
    plt.close(fig)


def setup_scene(ax, data):
    # One fixed camera and one set of limits prevent apparent motion caused
    # by camera tracking or per-frame autoscaling.
    points = data.particles.reshape(-1, 3)
    low, high = points.min(axis=0), points.max(axis=0)
    span = np.maximum(high - low, 0.1)
    pad = 0.08 * np.max(span)
    ax.set_xlim(low[0] - pad, high[0] + pad)
    ax.set_ylim(low[1] - pad, high[1] + pad)
    ax.set_zlim(min(0.0, low[2]) - 0.02, high[2] + pad)
    ax.set_box_aspect((span[0] + 2*pad, span[1] + 2*pad, high[2] + pad))
    ax.view_init(elev=18, azim=-65)
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    xx, yy = np.meshgrid([low[0] - pad, high[0] + pad], [low[1] - pad, high[1] + pad])
    ax.plot_surface(xx, yy, np.zeros_like(xx), color="#d1d5db", alpha=0.24, shade=False)
    target = data.forward_target
    # The cyan plane marks the lower terminal COM x bound. It is not a
    # restriction on every bear vertex, nor a constraint at intermediate times.
    ax.plot_surface(
        np.full((2, 2), target.lower_x),
        np.array([[low[1], high[1]], [low[1], high[1]]]),
        np.array([[0.0, 0.0], [high[2], high[2]]]),
        color="#06b6d4", alpha=0.12, shade=False,
    )
    path = data.landmark_trajectories[target.part]
    ax.plot(*path.T, color="#0891b2", linewidth=1.3)


def mesh_artist(ax, data, knot):
    artist = Poly3DCollection(
        data.particles[knot][data.surface_triangles],
        facecolors="#a8794c", edgecolors="#654321", linewidth=0.08, alpha=1.0,
        shade=True, lightsource=matplotlib.colors.LightSource(azdeg=300, altdeg=40),
    )
    ax.add_collection3d(artist)
    return artist


def render_replay(data, output):
    selected = np.unique(np.linspace(0, data.knot_count - 1, 4).astype(int))
    fig = plt.figure(figsize=(15, 4.7), constrained_layout=True)
    for i, knot in enumerate(selected):
        ax = fig.add_subplot(1, len(selected), i + 1, projection="3d")
        setup_scene(ax, data)
        mesh_artist(ax, data, knot)
        ax.set_title(f"t = {data.times[knot]:.2f} s")
    fig.suptitle("Saved Newton bear trajectory - cyan plane: terminal COM minimum x")
    fig.savefig(output / "trajectory_snapshots.png", dpi=170)
    plt.close(fig)

    fig = plt.figure(figsize=(8.4, 6.2), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    setup_scene(ax, data)
    mesh = mesh_artist(ax, data, 0)
    com_path = data.landmark_trajectories[data.forward_target.part]
    com, = ax.plot(*com_path[:1].T, "o", color="#0891b2", markersize=4)
    title = ax.set_title("")

    def update(knot):
        mesh.set_verts(data.particles[knot][data.surface_triangles])
        com.set_data_3d(*com_path[knot:knot+1].T)
        displacement = com_path[knot, 0] - data.forward_target.start[0]
        title.set_text(f"Quick CPU bear | t = {data.times[knot]:.2f} s | COM dx = {displacement:.3f} m")
        return mesh, com, title

    # Preserve saved control-knot timing. A GIF's 10 ms timing resolution
    # introduces a small quantization error at the default 15 fps.
    fps = 1.0 / float(np.median(np.diff(data.times)))
    animation = FuncAnimation(fig, update, frames=data.knot_count, blit=False)
    animation.save(output / "bear_walk.gif", writer=PillowWriter(fps=fps), dpi=110)
    update(data.knot_count - 1)
    fig.savefig(output / "bear_final.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=Path("results/bear_run_quick.npz"))
    parser.add_argument("--output", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = ReplayData.load(SolutionPaths.infer(args.result))
    if data.forward_target is None or data.controls is None:
        raise ValueError("A forward-target result with saved controls is required.")
    convergence(data, args.output)
    motion_plot(data, args.output)
    render_replay(data, args.output)
    print(f"Wrote convergence, motion, snapshots, final frame, GIF, and iteration CSV to {args.output}")


if __name__ == "__main__":
    main()

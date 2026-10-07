"""Simulate and animate the supplied linkage, without running optimization code.

Examples:
    python simulate.py --show
    python simulate.py --example synthesis
    python simulate.py --format html --output-dir outputs/preview
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import linkage_core as core


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--example", choices=("rotation", "synthesis"), default="rotation",
                        help="Full rotation, or the initial design with its target curve")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--newton-iters", type=int, default=8)
    parser.add_argument("--damping", type=float, default=1e-8)
    parser.add_argument("--format", choices=("gif", "html"), default="gif")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: outputs/rotation or outputs/initial")
    parser.add_argument("--show", action="store_true", help="Also open a live Matplotlib window")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frames < 2:
        parser.error("--frames must be at least 2; sparse sampling may fail Newton convergence")
    if args.newton_iters < 1:
        parser.error("--newton-iters must be positive")
    if not np.isfinite(args.damping) or args.damping < 0:
        parser.error("--damping must be finite and nonnegative")

    # Headless export is the default. Do not override the desktop backend for --show.
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import linkage_viz as viz

    output = args.output_dir or Path("outputs/rotation" if args.example == "rotation"
                                     else "outputs/initial")
    target = point_spec = path = None
    if args.example == "rotation":
        mech = core.make_four_bar(4.0, 1.5, 4.2, 3.0)
        lengths = core.mechanism_lengths(mech)
        angles = jnp.linspace(1.8, 1.8 + 2.0 * jnp.pi, args.frames)
        title = "Pure penalty + Newton simulation"
    else:
        example = core.make_on_link_coupler_synthesis_example(
            num_frames=args.frames, newton_iters=args.newton_iters, damping=args.damping)
        mech = core.mechanism_with_lengths(example.template_mech, example.initial_lengths)
        lengths, angles = example.initial_lengths, example.driver_angles
        target, point_spec = example.target_curve, example.point_spec
        title = "Initial design and target curve"

    print(f"Simulating {args.example}: {args.frames} frames ...", flush=True)
    simulation = core.checked_simulation_from_lengths(
        lengths, mech, angles, newton_iters=args.newton_iters, damping=args.damping)
    diagnostics = core.simulation_diagnostics(
        lengths, mech, angles, simulation, include_condition=True)
    if point_spec is not None:
        path = core.end_effector_path(mech, simulation.state_trajectory,
                                     simulation.driver_poses, point_spec)
        diagnostics["frechet_loss"] = float(core.discrete_frechet_distance(path, target))
    print(json.dumps(diagnostics, indent=2), flush=True)

    output.mkdir(parents=True, exist_ok=True)
    report = {"example": args.example, "lengths": np.asarray(lengths).tolist(),
              "frames": args.frames, "newton_iters": args.newton_iters,
              "newton_damping": args.damping, **diagnostics}
    (output / "diagnostics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    arrays = dict(lengths=lengths, driver_angles=angles,
                  states=simulation.state_trajectory, driver_poses=simulation.driver_poses,
                  energies=simulation.energies)
    if path is not None:
        arrays.update(path=path, target_curve=target)
    np.savez_compressed(output / "simulation.npz", **{k: np.asarray(v) for k, v in arrays.items()})

    try:
        fig = viz.plot_linkage_snapshots(
            mech, simulation.state_trajectory, simulation.driver_poses, title=title,
            effector_path_xy=path, target_curve_xy=target)
        fig.savefig(output / "snapshots.png", dpi=140)
        # The live display is the animation, not a second static window.
        plt.close(fig)
        anim = viz.animate_linkage(
            mech, simulation.state_trajectory, simulation.driver_poses, title=title,
            effector_path_xy=path, target_curve_xy=target,
            status_values=core.closure_constraint_norms(mech, simulation),
            status_label="closure norm")
        save = viz.save_animation_gif if args.format == "gif" else viz.save_animation_html
        animation_path = output / f"animation.{args.format}"
        save(anim, animation_path)
        print(f"Saved animation, snapshots, arrays, and diagnostics in {output.resolve()}", flush=True)
        if args.show:
            # Retain `anim` until the blocking GUI event loop returns.
            plt.show()
    finally:
        plt.close("all")


if __name__ == "__main__":
    main()

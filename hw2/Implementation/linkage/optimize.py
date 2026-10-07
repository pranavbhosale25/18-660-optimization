"""Complete and run implicit four-bar design optimization.

Examples (after completing the exercises):
    python optimize.py
    python optimize.py --show
    python optimize.py --steps 10 --freeze-ground --output-dir outputs/frozen_ground
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import linkage_core as core

Array = jax.Array


@jax.jit
def batched_state_sensitivities(
    lengths: Array,
    template_mech: core.FourBarMechanism,
    driver_angles: Array,
    state_trajectory: Array,
    sensitivity_damping: float = 0.0,
) -> Array:
    """Return dx*/d(lengths) with shape ``(T, 6, 4)``.

    ``lengths`` has shape ``(4,)``, ``driver_angles`` has shape ``(T,)``, and
    ``state_trajectory`` has shape ``(T, 6)``. ``sensitivity_damping`` is a
    nonnegative scalar.
    """
    # STUDENT TODO 1: Implement batched_state_sensitivities.
    # raise NotImplementedError(
    #     "STUDENT TODO 1: implement the IFT state sensitivities in optimize.py"
    # )
    def frame_sensitivity(state, driver_angle):
        # Compute the frame Hessian and mixed derivative for a single frame.
        H = core.frame_hessian_from_lengths(
            state, lengths, template_mech, driver_angle
        )

        # Compute the mixed derivative of the frame energy with respect to lengths.
        B = core.frame_mixed_from_lengths(
            state, lengths, template_mech, driver_angle
        )

        # Symmetrize the Hessian and add sensitivity damping to improve conditioning
        H = 0.5 * (H + H.T)
        H = H + sensitivity_damping * jnp.eye(
            core.STATE_DIM, dtype=H.dtype
        )

        # solve H * dx/dlengths = -B for dx/dlengths where S = dx/dlengths
        return jnp.linalg.solve(H, -B)

    # apply frame_sensitivity to each frame in the trajectory using vmap
    return jax.vmap(frame_sensitivity)(
        state_trajectory, driver_angles
    )



def design_objective_and_gradient(
    lengths: Array,
    template_mech: core.FourBarMechanism,
    driver_angles: Array,
    point_spec: core.PointOnLinkSpec,
    target_curve: Array,
    newton_iters: int = 8,
    damping: float = 1e-8,
    sensitivity_damping: float = 0.0,
) -> core.DesignResult:
    """STUDENT TODO 2: assemble the TOTAL outer gradient, including direct dependence.

    grad_states has shape (T, 6); grad_lengths_direct has shape (4,).
    state_sensitivities has shape (T, 6, 4). Return a total gradient of shape (4,).
    The scalar loss is the original discrete Frechet distance, not a squared
    pointwise loss. Its gradient check is local and may fail at nonsmooth ties.
    """
    if not np.isfinite(sensitivity_damping) or sensitivity_damping < 0:
        raise ValueError("sensitivity_damping must be finite and nonnegative.")
    simulation = core.checked_simulation_from_lengths(
        lengths, template_mech, driver_angles,
        newton_iters=newton_iters, damping=damping,
    )
    # Evaluate partial derivatives with solved states held independent of lengths.
    states = jax.lax.stop_gradient(simulation.state_trajectory)
    loss, (grad_states, grad_lengths_direct) = core.curve_loss_value_and_grads(
        states, lengths, template_mech, driver_angles, point_spec, target_curve,
    )
    state_sensitivities = batched_state_sensitivities(
        lengths, template_mech, driver_angles, states,
        sensitivity_damping=sensitivity_damping,
    )
    # STUDENT TODO 2: Combine the partials and sensitivities into
    # gradient_lengths, then allow the supplied result checks below to run.
    # raise NotImplementedError(
        # "STUDENT TODO 2: assemble the total design gradient in optimize.py"
    # )

    gradient_lengths = (grad_lengths_direct
                        + jnp.einsum("ti,tij->j", grad_states, state_sensitivities))

    result = core.DesignResult(loss, gradient_lengths, state_sensitivities, simulation)
    core.check_design_result(result, len(driver_angles))
    return result


def gradient_descent_update(
    lengths: Array,
    gradient_lengths: Array,
    step_size: float,
    design_mask: Array | None = None,
    min_length: float = 1e-3,
) -> Array:
    """STUDENT TODO 3: take one masked, lower-bound-projected gradient step.

    True/1 means a length is allowed to change. Clip only those active lengths
    to min_length. Inactive lengths must remain EXACTLY unchanged, even if
    one is below min_length. No mask means all four lengths are active.
    Positivity is not a complete linkage-feasibility constraint.
    """
    mask = core.validate_update_inputs(
        lengths, gradient_lengths, step_size, design_mask, min_length
    )
    lengths = jnp.asarray(lengths, dtype=jnp.float64)
    gradient_lengths = jnp.asarray(gradient_lengths, dtype=jnp.float64)
    # # STUDENT TODO 3: Project active coordinates and preserve inactive ones.
    # raise NotImplementedError(
    #     "STUDENT TODO 3: implement the projected design update in optimize.py"
    # )

    candidate_lengths = lengths - step_size * gradient_lengths
    projected_lengths = jnp.maximum(candidate_lengths, min_length)
    return jnp.where(mask, projected_lengths, lengths)




def optimize_link_lengths(
    initial_lengths: Array,
    template_mech: core.FourBarMechanism,
    driver_angles: Array,
    point_spec: core.PointOnLinkSpec,
    target_curve: Array,
    num_steps: int = 40,
    step_size: float = 1e-2,
    design_mask: Array | None = None,
    min_length: float = 1e-3,
    newton_iters: int = 8,
    damping: float = 1e-8,
    sensitivity_damping: float = 0.0,
) -> core.DesignHistory:
    """Run the supplied outer optimization loop using the three routines above.

    Perform exactly num_steps updates. Evaluate and store the INITIAL design,
    every updated design, and the FINAL design: histories have num_steps+1 rows.
    num_steps=0 must return one evaluated initial design, not empty histories.

    At every evaluated design, re-simulate and recompute the total gradient.
    Track the best EVALUATED loss, lengths, and matching simulation. Return
    core.DesignHistory with (length_history, loss_history, gradient_history,
    best_lengths, best_loss, best_simulation). Use jnp.stack for the histories.

    The reference task uses a fixed step, so the loss need not decrease on
    every update. If a candidate is invalid, let the checked solver raise a
    clear error; rerun with a smaller step. Line search is an optional extension.
    """
    mask = core.validate_optimization_inputs(
        initial_lengths, num_steps, step_size, design_mask, min_length
    )
    lengths = jnp.asarray(initial_lengths, dtype=jnp.float64)
    length_history = []
    loss_history = []
    gradient_history = []
    best_lengths = lengths
    best_loss = jnp.asarray(jnp.inf)
    best_simulation = None

    for iteration in range(num_steps + 1):
        design = design_objective_and_gradient(
            lengths, template_mech, driver_angles, point_spec, target_curve,
            newton_iters=newton_iters, damping=damping,
            sensitivity_damping=sensitivity_damping,
        )
        length_history.append(lengths)
        loss_history.append(design.loss)
        gradient_history.append(design.gradient_lengths)
        if best_simulation is None or float(design.loss) < float(best_loss):
            best_lengths = lengths
            best_loss = design.loss
            best_simulation = design.simulation
        if iteration < num_steps:
            lengths = gradient_descent_update(
                lengths, design.gradient_lengths, step_size,
                design_mask=mask, min_length=min_length,
            )

    return core.DesignHistory(
        jnp.stack(length_history), jnp.stack(loss_history), jnp.stack(gradient_history),
        best_lengths, best_loss, best_simulation,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--freeze-ground", action="store_true")
    parser.add_argument("--min-length", type=float, default=1e-3)
    parser.add_argument("--newton-iters", type=int, default=8)
    parser.add_argument("--damping", type=float, default=1e-8, help="Forward Newton damping")
    parser.add_argument("--sensitivity-damping", type=float, default=0.0,
                        help="Nonnegative sensitivity regularization")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/optimization"))
    parser.add_argument("--format", choices=("gif", "html"), default="gif")
    parser.add_argument("--no-animation", action="store_true",
                        help="Save plots and numerical results only")
    parser.add_argument("--show", action="store_true", help="Also show plots/animation in desktop windows")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    if args.frames < 2 or args.newton_iters < 1:
        parser.error("--frames must be at least 2 and --newton-iters must be positive")
    for name in ("step_size", "damping", "sensitivity_damping"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if not np.isfinite(args.min_length) or args.min_length <= 0:
        parser.error("--min-length must be finite and positive")

    example = core.make_on_link_coupler_synthesis_example(
        num_frames=args.frames, newton_iters=args.newton_iters, damping=args.damping)
    mask = np.array([not args.freeze_ground, True, True, True])
    print(f"Running {args.steps} updates using optimize.py ...", flush=True)
    try:
        history = optimize_link_lengths(
            example.initial_lengths, example.template_mech, example.driver_angles,
            example.point_spec, example.target_curve, num_steps=args.steps,
            step_size=args.step_size, design_mask=mask, min_length=args.min_length,
            newton_iters=args.newton_iters, damping=args.damping,
            sensitivity_damping=args.sensitivity_damping)
    except NotImplementedError as error:
        raise SystemExit(f"Unfinished exercise: {error}\n"
                         "Complete the four STUDENT TODO sections before optimizing.") from None

    # Presentation/serialization only. optimize_link_lengths owns all outer updates.
    lengths = np.asarray(history.length_history)
    losses = np.asarray(history.loss_history)
    gradients = np.asarray(history.gradient_history)
    if (lengths.shape != (args.steps + 1, 4) or losses.shape != (args.steps + 1,)
            or gradients.shape != (args.steps + 1, 4)):
        raise ValueError("Histories must include all num_steps + 1 evaluated designs.")
    if not all(np.all(np.isfinite(v)) for v in (lengths, losses, gradients)):
        raise ValueError("Optimization returned nonfinite history values.")
    if args.freeze_ground:
        np.testing.assert_array_equal(lengths[:, 0], np.full(args.steps + 1, lengths[0, 0]))

    before_mech = core.mechanism_with_lengths(example.template_mech, example.initial_lengths)
    before_simulation = core.checked_simulation_from_lengths(
        example.initial_lengths, example.template_mech, example.driver_angles,
        newton_iters=args.newton_iters, damping=args.damping)
    best_mech = core.mechanism_with_lengths(example.template_mech, history.best_lengths)
    best_simulation = history.best_simulation
    before_path = core.end_effector_path(before_mech, before_simulation.state_trajectory,
                                        before_simulation.driver_poses, example.point_spec)
    best_path = core.end_effector_path(best_mech, best_simulation.state_trajectory,
                                      best_simulation.driver_poses, example.point_spec)
    report = {
        "implementation": "optimize", "num_updates": args.steps,
        "num_evaluated_designs": len(losses), "step_size": args.step_size,
        "frames": args.frames, "newton_iters": args.newton_iters,
        "newton_damping": args.damping, "sensitivity_damping": args.sensitivity_damping,
        "min_length": args.min_length, "design_mask": mask.astype(int).tolist(),
        "initial_loss": float(losses[0]), "final_loss": float(losses[-1]),
        "best_loss": float(history.best_loss), "best_index": int(np.argmin(losses)),
        "initial_lengths": lengths[0].tolist(), "final_lengths": lengths[-1].tolist(),
        "best_lengths": np.asarray(history.best_lengths).tolist(),
        "best_simulation_diagnostics": core.simulation_diagnostics(
            history.best_lengths, example.template_mech, example.driver_angles,
            best_simulation, include_condition=True),
    }
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    names = ("ground", "driver", "coupler", "follower")
    header = ",".join(["updates", "loss", *names, *("grad_" + name for name in names)])
    np.savetxt(output / "history.csv", np.column_stack((np.arange(len(losses)), losses,
                                                       lengths, gradients)),
               delimiter=",", header=header, comments="", fmt=["%d"] + ["%.16g"] * 9)
    np.savez_compressed(
        output / "results.npz", length_history=lengths, loss_history=losses,
        gradient_history=gradients, best_lengths=np.asarray(history.best_lengths),
        best_loss=np.asarray(history.best_loss), driver_angles=np.asarray(example.driver_angles),
        target_curve=np.asarray(example.target_curve), before_path=np.asarray(before_path),
        best_path=np.asarray(best_path), before_states=np.asarray(before_simulation.state_trajectory),
        before_driver_poses=np.asarray(before_simulation.driver_poses),
        before_energies=np.asarray(before_simulation.energies),
        best_states=np.asarray(best_simulation.state_trajectory),
        best_driver_poses=np.asarray(best_simulation.driver_poses),
        best_energies=np.asarray(best_simulation.energies))

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import linkage_viz as viz

    animations = []  # Keep live objects alive until plt.show() returns.
    try:
        for filename, fig in (
            ("loss_history.png", viz.plot_loss_history(history)),
            ("curve_comparison.png", viz.plot_curve_comparison(example.target_curve, before_path, best_path)),
        ):
            fig.savefig(output / filename, dpi=140)
            if not args.show:
                plt.close(fig)
        if not args.no_animation:
            save = viz.save_animation_gif if args.format == "gif" else viz.save_animation_html
            for label, mech, simulation, path in (
                ("before", before_mech, before_simulation, before_path),
                ("best", best_mech, best_simulation, best_path),
            ):
                anim = viz.animate_linkage(
                    mech, simulation.state_trajectory, simulation.driver_poses,
                    title="Initial design" if label == "before" else "Best evaluated design",
                    effector_path_xy=path, target_curve_xy=example.target_curve,
                    status_values=core.closure_constraint_norms(mech, simulation),
                    status_label="closure norm")
                animations.append(anim)
                save(anim, output / f"{label}.{args.format}")
        print(f"Saved results in {output.resolve()}", flush=True)
        if args.show:
            plt.show()
    finally:
        plt.close("all")


if __name__ == "__main__":
    main()

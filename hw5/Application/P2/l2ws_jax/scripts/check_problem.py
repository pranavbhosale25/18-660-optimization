"""Public numerical checks for a completed homework module, without answer data.

Run from the installed project: python scripts/check_problem.py --problem toy.
These sampled self-consistency checks do not verify that a submitted objective
matches the mathematical model in the separate homework PDF.
"""

from __future__ import annotations

import argparse
import json
import numbers
import sys
from pathlib import Path

# A student copy must never resolve an instructor editable installation first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if (PROJECT_ROOT / "src").is_dir():
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from l2ws_jax.model import init_params, initialize
from l2ws_jax.problems import PROBLEM_NAMES, make_problem
from l2ws_jax.solver import rollout
from l2ws_jax.training import TrainConfig, train

_CHECK_BATCH_SIZE = 8
_CHECK_SOLVER_STEPS = 20
_CHECK_MODEL_WIDTH = 16
_MODEL_INITIALIZATION_SEED = 2026
_CONTEXT_SEEDS = {False: 2027, True: 2028}
_DIRECTION_SEED = 2030
_SHORT_TRAINING_UNROLL = 2
_SHORT_TRAINING_SIZE = 32
_SHORT_VALIDATION_SIZE = 8
_SHORT_TRAINING_SEED = 2718
_DESCENT_EPSILON_MULTIPLIER = 64
_TOLERANCES = {
    "float32": {
        "comparison": (3e-5, 3e-6),
        "gradient": (8e-3, 5e-4),
        "feasibility": 2e-6,
    },
    "float64": {
        "comparison": (2e-10, 2e-11),
        "gradient": (3e-5, 2e-8),
        "feasibility": 5e-12,
    },
}


def _array(problem, value, shape, name, *, finite=True):
    if not isinstance(value, jax.Array):
        raise AssertionError(f"{name} must return a JAX array")
    if value.shape != shape:
        raise AssertionError(
            f"{name} has shape {value.shape}; expected {shape}"
        )
    if value.dtype != problem.dtype:
        raise AssertionError(
            f"{name} has dtype {value.dtype}; expected {problem.dtype}"
        )
    if value.devices() != {problem.device}:
        raise AssertionError(f"{name} must be on {problem.device}")
    host = np.asarray(jax.device_get(value))
    if finite and not np.isfinite(host).all():
        raise AssertionError(f"{name} must be finite")
    return host


def _feasible(problem, values, contexts, tolerance, name):
    """Check the supplied projection and advertised constraint set."""
    host = np.asarray(jax.device_get(values))
    residual = np.asarray(jax.device_get(problem.feasibility(values, contexts)))
    if not np.isfinite(residual).all() or np.max(residual) > tolerance:
        raise AssertionError(f"{name} fails projection-based feasibility")
    violation = float(np.max(residual))
    if problem.constraint == "box":
        lower = np.asarray(jax.device_get(problem.lower))
        upper = np.asarray(jax.device_get(problem.upper))
        violation = max(
            violation,
            float(np.maximum(lower - host, 0).max()),
            float(np.maximum(host - upper, 0).max()),
        )
    if violation > tolerance:
        raise AssertionError(
            f"{name} violates {problem.constraint} constraints by {violation:g}"
        )
    np.testing.assert_allclose(
        problem.project(values, contexts),
        values,
        rtol=0,
        atol=tolerance,
        err_msg=f"{name} projection idempotence",
    )
    return violation


def check(
    name="toy", train_steps=10, *, dtype="float64", device="auto", jit=True
):
    """Check shapes, precision, derivatives, feasibility, and training.

    A directional central difference uses machine_epsilon**(1/3), scaled by
    each starting point's norm. Float32 checks allow larger roundoff/truncation
    errors. With ``jit=False`` the compiled comparison is explicitly skipped;
    the same checks and short training run execute with JIT disabled.
    """
    if (
        isinstance(train_steps, bool)
        or not isinstance(train_steps, numbers.Integral)
        or train_steps < 1
    ):
        raise ValueError("train_steps must be a positive integer")
    if not isinstance(jit, bool):
        raise ValueError("jit must be boolean")
    problem = make_problem(name, dtype=dtype, device=device)
    machine_epsilon = np.finfo(problem.dtype).eps
    finite_difference_epsilon = float(np.cbrt(machine_epsilon))
    tolerances = _TOLERANCES[problem.dtype.name]
    comparison_rtol, comparison_atol = tolerances["comparison"]
    gradient_rtol, gradient_atol = tolerances["gradient"]
    feasibility_tolerance = tolerances["feasibility"]
    descent_tolerance = _DESCENT_EPSILON_MULTIPLIER * machine_epsilon
    batch_size = _CHECK_BATCH_SIZE
    solver_steps = _CHECK_SOLVER_STEPS
    params = init_params(
        problem,
        problem.key(_MODEL_INITIALIZATION_SEED),
        width=_CHECK_MODEL_WIDTH,
    )

    for field, value in problem.data.items():
        if value.devices() != {problem.device}:
            raise AssertionError(
                f"Fixed problem array {field} is on the wrong device"
            )
        if (
            np.issubdtype(value.dtype, np.floating)
            and value.dtype != problem.dtype
        ):
            raise AssertionError(
                f"Fixed problem array {field} has the wrong floating dtype"
            )
        if not np.isfinite(np.asarray(jax.device_get(value))).all():
            raise AssertionError(f"Fixed problem array {field} must be finite")

    def diagnostics(parameters, contexts):
        start = problem.baseline(contexts)
        path = rollout(problem, start, contexts, solver_steps, return_path=True)
        return {
            "features": problem.features(contexts),
            "baseline": start,
            "initial_prediction": initialize(problem, parameters, contexts),
            "objective": problem.objective(start, contexts),
            "gradient": problem.gradient(start, contexts),
            "step_size": problem.step_size(contexts),
            "path": path,
            "path_objectives": jax.vmap(
                lambda y: problem.objective(y, contexts)
            )(path),
            "certificate": problem.certificate(path[-1], contexts),
        }

    compiled_diagnostics = None
    reports = []
    for shifted in (False, True):
        with jax.default_device(problem.device), jax.disable_jit(not jit):
            contexts = problem.sample(
                batch_size,
                problem.key(_CONTEXT_SEEDS[shifted]),
                shift=shifted,
            )
            _array(
                problem, contexts, (batch_size, problem.context_dim), "sample"
            )
            # This remains eager when the comparison below uses JIT.
            with jax.disable_jit(True):
                eager = diagnostics(params, contexts)
            jax.block_until_ready(eager)
            if jit:
                if compiled_diagnostics is None:
                    compiled_diagnostics = (
                        jax.jit(diagnostics).lower(params, contexts).compile()
                    )
                compiled = compiled_diagnostics(params, contexts)
                jax.block_until_ready(compiled)
                for field in eager:
                    np.testing.assert_allclose(
                        compiled[field],
                        eager[field],
                        rtol=comparison_rtol,
                        atol=comparison_atol,
                        equal_nan=(field == "certificate"),
                        err_msg=f"Compiled/eager disagreement in {field}",
                    )
            output = compiled if jit else eager
            _array(
                problem,
                output["features"],
                (batch_size, problem.context_dim),
                "features",
            )
            start = output["baseline"]
            _array(
                problem, start, (batch_size, problem.decision_dim), "baseline"
            )
            _array(
                problem,
                output["initial_prediction"],
                start.shape,
                "initial prediction",
            )
            np.testing.assert_allclose(
                output["initial_prediction"],
                start,
                rtol=comparison_rtol,
                atol=comparison_atol,
                err_msg=(
                    "Zero output layer must decode to the classical baseline"
                ),
            )
            baseline_violation = _feasible(
                problem, start, contexts, feasibility_tolerance, "baseline"
            )
            values = _array(
                problem, output["objective"], (batch_size,), "objective"
            )
            gradient = output["gradient"]
            _array(problem, gradient, start.shape, "gradient")
            alpha = _array(
                problem, output["step_size"], (batch_size, 1), "step_size"
            )
            if not (alpha > 0).all():
                raise AssertionError("step_size must be strictly positive")

            direction = jax.random.normal(
                problem.key(_DIRECTION_SEED),
                start.shape,
                dtype=problem.dtype,
            )
            direction = direction / jnp.linalg.norm(
                direction, axis=-1, keepdims=True
            )
            scale = jnp.maximum(
                jnp.linalg.norm(start, axis=-1),
                jnp.asarray(1.0, dtype=problem.dtype),
            )
            epsilon = (
                jnp.asarray(finite_difference_epsilon, dtype=problem.dtype)
                * scale
            )
            perturbation = epsilon[:, None] * direction
            numerical = (
                problem.objective(start + perturbation, contexts)
                - problem.objective(start - perturbation, contexts)
            ) / (2 * epsilon)
            analytic = jnp.sum(gradient * direction, axis=-1)
            np.testing.assert_allclose(
                numerical,
                analytic,
                rtol=gradient_rtol,
                atol=gradient_atol,
                err_msg="Directional finite difference disagrees with autodiff",
            )

            probe = problem.project(start + direction, contexts)
            _array(problem, probe, start.shape, "project")
            _feasible(
                problem,
                probe,
                contexts,
                feasibility_tolerance,
                "projected probe",
            )
            path = output["path"]
            _array(
                problem,
                path,
                (solver_steps + 1, batch_size, problem.decision_dim),
                "rollout",
            )
            trajectory_values = _array(
                problem,
                output["path_objectives"],
                (solver_steps + 1, batch_size),
                "rollout objectives",
            )
            increments = np.diff(trajectory_values, axis=0)
            allowed = descent_tolerance * (1.0 + np.abs(trajectory_values[:-1]))
            if not np.all(increments <= allowed):
                raise AssertionError(
                    "The sampled solver trajectory is not monotone within "
                    "roundoff"
                )
            maximum_violation = baseline_violation
            for decision in path[1:]:
                maximum_violation = max(
                    maximum_violation,
                    _feasible(
                        problem,
                        decision,
                        contexts,
                        feasibility_tolerance,
                        "rollout iterate",
                    ),
                )
            certificate = _array(
                problem,
                output["certificate"],
                (batch_size,),
                "certificate",
                finite=problem.convex,
            )
            if problem.convex and np.any(certificate < -feasibility_tolerance):
                raise AssertionError("Convex certificate must be nonnegative")
            if (
                not problem.convex
                and not (np.isfinite(certificate) | np.isnan(certificate)).all()
            ):
                raise AssertionError(
                    "A missing general certificate must be represented by "
                    "NaN, not infinity"
                )
            reports.append(
                {
                    "shifted": shifted,
                    "max_directional_gradient_error": float(
                        np.max(np.abs(np.asarray(numerical - analytic)))
                    ),
                    "max_feasibility_violation": maximum_violation,
                    "largest_objective_increase": float(
                        max(0.0, np.max(increments))
                    ),
                    "mean_initial_objective": float(np.mean(values)),
                    "mean_refined_objective": float(
                        np.mean(trajectory_values[-1])
                    ),
                }
            )

    configuration = TrainConfig(
        steps=int(train_steps),
        unroll=_SHORT_TRAINING_UNROLL,
        width=_CHECK_MODEL_WIDTH,
        train_size=_SHORT_TRAINING_SIZE,
        validation_size=_SHORT_VALIDATION_SIZE,
        batch_size=_CHECK_BATCH_SIZE,
        seed=_SHORT_TRAINING_SEED,
        validate_every=max(1, int(train_steps) // 2),
        jit=jit,
    )
    result = train(problem, configuration)
    return {
        "problem": name,
        "checks": "passed",
        "runtime": problem.runtime.info(),
        "jit": jit,
        "compiled_eager_agreement": "passed" if jit else "skipped (--no-jit)",
        "finite_difference_base_epsilon": finite_difference_epsilon,
        "sampled_checks": reports,
        "short_training_updates": int(train_steps),
        "training_unroll": configuration.unroll,
        "selected_update": result.best_step,
        "initial_validation_loss": result.history[0]["validation_loss"],
        "final_logged_validation_loss": result.history[-1]["validation_loss"],
        "note": (
            "These self-consistency checks do not compare the submitted "
            "objective with the assigned mathematical formula, establish "
            "semantic correctness, prove a global curvature bound, certify "
            "optimality, or establish generalization or a speedup."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem", choices=("toy",) + PROBLEM_NAMES, default="toy"
    )
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, gpu, or an indexed device such as gpu:0",
    )
    parser.add_argument(
        "--no-jit",
        dest="jit",
        action="store_false",
        help="debug eagerly; skip compiled comparison",
    )
    parser.add_argument("--train-steps", type=int, default=10)
    args = parser.parse_args()
    if args.train_steps < 1:
        parser.error("--train-steps must be positive")
    print(
        json.dumps(
            check(
                args.problem,
                args.train_steps,
                dtype=args.dtype,
                device=args.device,
                jit=args.jit,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

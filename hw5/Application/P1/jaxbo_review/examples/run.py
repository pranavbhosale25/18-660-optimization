"""Run one student-selected configuration or a complete controlled comparison."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import jax

# Application-level precision selection occurs before application arrays exist.
jax.config.update("jax_enable_x64", True)

from jaxbo import LAB_NAMES, PolicyConfig, make_problem, run_experiment

from .configurations import ExperimentPlan, PLAN_BUILDERS


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_WINDOWS_RESERVED_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


def validate_plan(problem: str, plan: ExperimentPlan) -> ExperimentPlan:
    """Validate the public shape of a student plan without judging its quality."""
    if not isinstance(plan, ExperimentPlan):
        raise TypeError(f"{problem}_plan() must return ExperimentPlan")
    if not isinstance(plan.primary, PolicyConfig):
        raise TypeError("ExperimentPlan.primary must be a PolicyConfig")
    if not isinstance(plan.ablations, dict) or len(plan.ablations) != 2:
        raise ValueError("ExperimentPlan.ablations must contain exactly two entries")

    serialized = [plan.primary.to_dict()]
    normalized_names = set()
    for name, config in plan.ablations.items():
        if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
            raise ValueError(
                "Ablation names must be nonempty and contain only letters, "
                "numbers, underscores, and hyphens"
            )
        normalized_name = name.casefold()
        if normalized_name in ("primary", "all"):
            raise ValueError("'primary' and 'all' are reserved variant names")
        if normalized_name in _WINDOWS_RESERVED_NAMES:
            raise ValueError(f"{name!r} is a reserved filename on Windows")
        if normalized_name in normalized_names:
            raise ValueError("Ablation names must also be unique ignoring case")
        normalized_names.add(normalized_name)
        if not isinstance(config, PolicyConfig):
            raise TypeError(f"Ablation {name!r} must be a PolicyConfig")
        serialized.append(config.to_dict())

    if serialized[1] == serialized[0] or serialized[2] == serialized[0]:
        raise ValueError("Each ablation must differ from the primary configuration")
    if serialized[1] == serialized[2]:
        raise ValueError("The two ablations must differ from each other")
    return plan


def load_plan(problem: str) -> ExperimentPlan:
    """Call and validate the student plan builder for one public problem."""
    try:
        builder = PLAN_BUILDERS[problem]
    except KeyError as error:
        raise ValueError(f"Unknown problem {problem!r}; choose from {LAB_NAMES}") from error
    return validate_plan(problem, builder())


def _variant_configs(plan: ExperimentPlan, variant: str) -> dict[str, PolicyConfig]:
    if variant == "primary":
        return {"primary": plan.primary}
    if variant == "all":
        return {"primary": plan.primary, **plan.ablations}
    if variant not in plan.ablations:
        available = ("primary", "all", *plan.ablations)
        raise ValueError(f"Unknown variant {variant!r}; choose from {available}")
    return {variant: plan.ablations[variant]}


def run_variants(*, problem_name: str, instance: int, seed: int, variant: str,
                 output_dir: Path, trace_every: int) -> list[Path]:
    """Execute selected variants as independent, fully charged episodes."""
    plan = load_plan(problem_name)
    configs = _variant_configs(plan, variant)

    submission_dir = output_dir / problem_name
    submission_dir.mkdir(parents=True, exist_ok=True)
    (submission_dir / "submission.json").write_text(
        json.dumps(plan.primary.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    saved = []
    for variant_name, config in configs.items():
        problem = make_problem(problem_name, instance=instance)
        destination = (
            output_dir
            / problem_name
            / f"instance-{instance}"
            / f"seed-{seed}"
            / variant_name
        )

        def progress(step, observation):
            if step == 1 or step % 10 == 0:
                print(
                    f"[{variant_name}] {step} observations; "
                    f"{observation.cumulative_cost:.2f}/{problem.spec.budget:g} "
                    f"{problem.spec.budget_unit}",
                    flush=True,
                )

        result = run_experiment(
            problem,
            config,
            seed=seed,
            trace_every=trace_every,
            callback=progress,
        )
        result.save(destination)
        saved.append(destination)
        print(json.dumps(result.score, indent=2, allow_nan=False))
        print(f"Saved {variant_name!r} results in {destination}")
    return saved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=LAB_NAMES, required=True)
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variant",
        default="primary",
        help="primary, all, or an ablation name returned by the task plan",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--trace-every",
        type=int,
        default=10,
        help="save a recommendation snapshot every N observations",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_variants(
        problem_name=args.problem,
        instance=args.instance,
        seed=args.seed,
        variant=args.variant,
        output_dir=args.output_dir,
        trace_every=args.trace_every,
    )


if __name__ == "__main__":
    main()

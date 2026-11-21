"""Public interface checks for the five student experiment plans."""
import jax
import pytest

from examples.configurations import ExperimentPlan, PLAN_BUILDERS
from examples.run import _parser, load_plan, validate_plan
from jaxbo import ConfiguredPolicy, LAB_NAMES, PolicyConfig, make_problem
from jaxbo.labs.core import validate_point


def test_student_plan_registry_matches_public_problem_names():
    assert tuple(PLAN_BUILDERS) == LAB_NAMES


@pytest.mark.parametrize("problem_name", LAB_NAMES)
def test_student_plan_is_valid_and_can_propose_legal_initial_actions(problem_name):
    plan = load_plan(problem_name)
    problem = make_problem(problem_name, instance=0)

    configurations = (plan.primary, *plan.ablations.values())
    for index, configuration in enumerate(configurations):
        assert PolicyConfig.from_dict(configuration.to_dict()) == configuration
        policy = ConfiguredPolicy(
            problem.spec,
            configuration,
            key=jax.random.fold_in(jax.random.key(0), index),
        )
        action = policy.ask(problem.spec.budget)
        validated = validate_point(problem.spec, action.x)
        assert validated.shape == (problem.spec.dimension,)
        assert isinstance(action.replicates, int)
        assert action.replicates >= 1


def test_plan_validation_checks_structure_without_prescribing_choices():
    primary = PolicyConfig(strategy="random")
    plan = ExperimentPlan(
        primary,
        {
            "first-comparison": PolicyConfig(strategy="random", replicates=2),
            "second_comparison": PolicyConfig(strategy="random", allow_repeats=True),
        },
    )
    assert validate_plan("example", plan) is plan


@pytest.mark.parametrize("name", ("primary", "ALL", "con", "LPT1"))
def test_plan_validation_rejects_reserved_output_names(name):
    plan = ExperimentPlan(
        PolicyConfig(strategy="random"),
        {
            name: PolicyConfig(strategy="random", replicates=2),
            "comparison": PolicyConfig(strategy="random", allow_repeats=True),
        },
    )
    with pytest.raises(ValueError, match="reserved"):
        validate_plan("example", plan)


def test_plan_validation_rejects_names_that_differ_only_by_case():
    plan = ExperimentPlan(
        PolicyConfig(strategy="random"),
        {
            "comparison": PolicyConfig(strategy="random", replicates=2),
            "COMPARISON": PolicyConfig(strategy="random", allow_repeats=True),
        },
    )
    with pytest.raises(ValueError, match="unique ignoring case"):
        validate_plan("example", plan)


def test_known_observation_variances_require_a_gp_surrogate():
    with pytest.raises(ValueError, match="require a GP surrogate"):
        PolicyConfig(
            surrogate="forest",
            hyper_method="fixed",
            known_noise=True,
        )


def test_unknown_plan_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown problem"):
        load_plan("not-a-problem")


def test_student_distribution_rejects_heldout_instances():
    with pytest.raises(ValueError, match="development instances only"):
        make_problem("coating", instance=0, split="heldout")


def test_runner_help_lists_public_controls():
    help_text = _parser().format_help()
    for flag in ("--problem", "--instance", "--seed", "--variant", "--output-dir"):
        assert flag in help_text

"""Pure experiment-plan functions completed by students.

These functions select optimization components. Experiment execution, random
keys, budgets, observations, and file output are handled by ``examples.run``.
"""
from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

from jaxbo import PolicyConfig


class ExperimentPlan(NamedTuple):
    """The complete configuration output for one application problem.

    Fields:
        primary: The ``PolicyConfig`` selected as the student's main method.
            The driver runs it for ``--variant primary`` and writes it to
            ``submission.json``.
        ablations: A built-in ``dict`` containing exactly two controlled
            comparison configurations. Every key is the human-readable name
            of one comparison and every value is a ``PolicyConfig``. The keys
            become directory names in the saved results.

    The three configurations must have pairwise-different ``to_dict()``
    representations. A difference in serialized fields does not guarantee a
    meaningful experiment: students remain responsible for choosing ablations
    that alter behavior relevant to the selected strategy.
    """

    primary: PolicyConfig
    ablations: dict[str, PolicyConfig]


def coating_plan() -> ExperimentPlan:
    """Build the complete experiment plan for ``coating``.

    Args:
        None. In particular, the function is not passed a problem instance,
        random seed, observations, budget, or output path. Its signature must
        remain argument-free so the supplied driver can call it.

    Returns:
        An ``ExperimentPlan`` whose ``primary`` field is one ``PolicyConfig``
        and whose ``ablations`` field is a built-in
        ``dict[str, PolicyConfig]`` with exactly two entries. Each dictionary
        key must match ``[A-Za-z0-9][A-Za-z0-9_-]*``, be unique ignoring case,
        and must not be ``primary``, ``all``, or a Windows device name such as
        ``CON``, ``PRN``, ``AUX``, ``NUL``, ``COM1``, or ``LPT1``. Names are
        used as output-directory components. The primary and both ablations
        must have pairwise-different ``PolicyConfig.to_dict()`` values. Every
        configuration must also be compatible with the public ``coating``
        ``TaskSpec`` so that ``ConfiguredPolicy`` can construct it.

        The primary is the submitted configuration for this problem. The two
        alternatives are controlled comparisons. The driver runs all three as
        independent episodes with the same requested instance and seed, and
        each episode receives the complete problem budget.

    Side effects:
        None. Construct configurations only; do not run an experiment, inspect
        a simulator, use random state, read observations, or write files.

    Raises:
        A completed implementation should not intentionally raise. The starter
        raises ``NotImplementedError`` only to mark unfinished work.
    """
    # STUDENT TODO: return the coating ExperimentPlan described above.
    primary = PolicyConfig(kernel="matern52", acquisition="ei", noise=0.01,
                           candidates=512, hyper_steps=30)
    return ExperimentPlan(primary, {
        "random_search": replace(primary, strategy="random"),
        "rbf_kernel": replace(primary, kernel="rbf"),
    })


def pipeline_plan() -> ExperimentPlan:
    """Build the complete experiment plan for ``pipeline``.

    Args:
        None. In particular, the function is not passed a problem instance,
        random seed, observations, budget, or output path. Its signature must
        remain argument-free so the supplied driver can call it.

    Returns:
        An ``ExperimentPlan`` whose ``primary`` field is one ``PolicyConfig``
        and whose ``ablations`` field is a built-in
        ``dict[str, PolicyConfig]`` with exactly two entries. Each dictionary
        key must match ``[A-Za-z0-9][A-Za-z0-9_-]*``, be unique ignoring case,
        and must not be ``primary``, ``all``, or a Windows device name such as
        ``CON``, ``PRN``, ``AUX``, ``NUL``, ``COM1``, or ``LPT1``. Names are
        used as output-directory components. The primary and both ablations
        must have pairwise-different ``PolicyConfig.to_dict()`` values. Every
        configuration must also be compatible with the public ``pipeline``
        ``TaskSpec`` so that ``ConfiguredPolicy`` can construct it.

        The primary is the submitted configuration for this problem. The two
        alternatives are controlled comparisons. The driver runs all three as
        independent episodes with the same requested instance and seed, and
        each episode receives the complete problem budget.

    Side effects:
        None. Construct configurations only; do not run an experiment, inspect
        a simulator, use random state, read observations, or write files.

    Raises:
        A completed implementation should not intentionally raise. The starter
        raises ``NotImplementedError`` only to mark unfinished work.
    """
    # STUDENT TODO: return the pipeline ExperimentPlan described above.
    primary = PolicyConfig(mixed_encoding="activity", kernel="matern52",
                           acquisition="ei", candidates=768, hyper_steps=30)
    return ExperimentPlan(primary, {
        "onehot_encoding": replace(primary, mixed_encoding="onehot"),
        "forest_surrogate": replace(primary, surrogate="forest",
                                    hyper_method="fixed"),
    })


def production_plan() -> ExperimentPlan:
    """Build the complete experiment plan for ``production``.

    Args:
        None. In particular, the function is not passed a problem instance,
        random seed, observations, budget, or output path. Its signature must
        remain argument-free so the supplied driver can call it.

    Returns:
        An ``ExperimentPlan`` whose ``primary`` field is one ``PolicyConfig``
        and whose ``ablations`` field is a built-in
        ``dict[str, PolicyConfig]`` with exactly two entries. Each dictionary
        key must match ``[A-Za-z0-9][A-Za-z0-9_-]*``, be unique ignoring case,
        and must not be ``primary``, ``all``, or a Windows device name such as
        ``CON``, ``PRN``, ``AUX``, ``NUL``, ``COM1``, or ``LPT1``. Names are
        used as output-directory components. The primary and both ablations
        must have pairwise-different ``PolicyConfig.to_dict()`` values. Every
        configuration must also be compatible with the public ``production``
        ``TaskSpec`` so that ``ConfiguredPolicy`` can construct it.

        The primary is the submitted configuration for this problem. The two
        alternatives are controlled comparisons. The driver runs all three as
        independent episodes with the same requested instance and seed, and
        each episode receives the complete problem budget.

    Side effects:
        None. Construct configurations only; do not run an experiment, inspect
        a simulator, use random state, read observations, or write files.

    Raises:
        A completed implementation should not intentionally raise. The starter
        raises ``NotImplementedError`` only to mark unfinished work.
    """
    # STUDENT TODO: return the production ExperimentPlan described above.
    primary = PolicyConfig(known_noise=True, replicates=3,
                           incumbent="posterior", recommendation="posterior",
                           candidates=512, hyper_steps=30)
    return ExperimentPlan(primary, {
        "single_run": replace(primary, replicates=1),
        "ignore_reported_variance": replace(primary, known_noise=False),
    })


def materials_plan() -> ExperimentPlan:
    """Build the complete experiment plan for ``materials``.

    Args:
        None. In particular, the function is not passed a problem instance,
        random seed, observations, budget, or output path. Its signature must
        remain argument-free so the supplied driver can call it.

    Returns:
        An ``ExperimentPlan`` whose ``primary`` field is one ``PolicyConfig``
        and whose ``ablations`` field is a built-in
        ``dict[str, PolicyConfig]`` with exactly two entries. Each dictionary
        key must match ``[A-Za-z0-9][A-Za-z0-9_-]*``, be unique ignoring case,
        and must not be ``primary``, ``all``, or a Windows device name such as
        ``CON``, ``PRN``, ``AUX``, ``NUL``, ``COM1``, or ``LPT1``. Names are
        used as output-directory components. The primary and both ablations
        must have pairwise-different ``PolicyConfig.to_dict()`` values. Every
        configuration must also be compatible with the public ``materials``
        ``TaskSpec`` so that ``ConfiguredPolicy`` can construct it.

        The primary is the submitted configuration for this problem. The two
        alternatives are controlled comparisons. The driver runs all three as
        independent episodes with the same requested instance and seed, and
        each episode receives the complete problem budget.

    Side effects:
        None. Construct configurations only; do not run an experiment, inspect
        a simulator, use random state, read observations, or write files.

    Raises:
        A completed implementation should not intentionally raise. The starter
        raises ``NotImplementedError`` only to mark unfinished work.
    """
    # STUDENT TODO: return the materials ExperimentPlan described above.
    primary = PolicyConfig(feasibility=True, cost_aware=True,
                           cost_mode="divide_expected_cost", noise=0.02,
                           candidates=512, hyper_steps=30)
    return ExperimentPlan(primary, {
        "ignore_cost": replace(primary, cost_aware=False),
        "ignore_feasibility": replace(primary, feasibility=False),
    })


def controller_plan() -> ExperimentPlan:
    """Build the complete experiment plan for ``controller``.

    Args:
        None. In particular, the function is not passed a problem instance,
        random seed, observations, budget, or output path. Its signature must
        remain argument-free so the supplied driver can call it.

    Returns:
        An ``ExperimentPlan`` whose ``primary`` field is one ``PolicyConfig``
        and whose ``ablations`` field is a built-in
        ``dict[str, PolicyConfig]`` with exactly two entries. Each dictionary
        key must match ``[A-Za-z0-9][A-Za-z0-9_-]*``, be unique ignoring case,
        and must not be ``primary``, ``all``, or a Windows device name such as
        ``CON``, ``PRN``, ``AUX``, ``NUL``, ``COM1``, or ``LPT1``. Names are
        used as output-directory components. The primary and both ablations
        must have pairwise-different ``PolicyConfig.to_dict()`` values. Every
        configuration must also be compatible with the public ``controller``
        ``TaskSpec`` so that ``ConfiguredPolicy`` can construct it.

        The primary is the submitted configuration for this problem. The two
        alternatives are controlled comparisons. The driver runs all three as
        independent episodes with the same requested instance and seed, and
        each episode receives the complete problem budget.

    Side effects:
        None. Construct configurations only; do not run an experiment, inspect
        a simulator, use random state, read observations, or write files.

    Raises:
        A completed implementation should not intentionally raise. The starter
        raises ``NotImplementedError`` only to mark unfinished work.
    """
    # STUDENT TODO: return the controller ExperimentPlan described above.
    primary = PolicyConfig(embedding_dimension=5, embeddings=1,
                           acquisition="ei", candidates=768, hyper_steps=30)
    return ExperimentPlan(primary, {
        "ambient_gp": replace(primary, embedding_dimension=0),
        "random_search": replace(primary, strategy="random",
                                 embedding_dimension=0),
    })


PLAN_BUILDERS = {
    "coating": coating_plan,
    "pipeline": pipeline_plan,
    "production": production_plan,
    "materials": materials_plan,
    "controller": controller_plan,
}

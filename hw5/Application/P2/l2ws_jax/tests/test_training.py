"""Training invariants, exact-zero gradients, and checkpoint round trips."""

import json
from dataclasses import replace

import jax
import numpy as np
import pytest

from l2ws_jax.checkpoint import load_checkpoint, save_checkpoint
from l2ws_jax.model import apply_model, init_params, initialize
from l2ws_jax.problems import make_problem
from l2ws_jax.solver import batch_loss, predict, rollout
from l2ws_jax.training import TrainConfig, train


def short_config(**updates):
    config = TrainConfig(
        steps=5,
        validate_every=3,
        train_size=16,
        validation_size=8,
        batch_size=8,
        width=7,
        unroll=2,
        learning_rate=0.01,
        seed=19,
    )
    return replace(config, **updates)


def leaves(tree):
    return jax.tree_util.tree_leaves(tree)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_initial_model_is_baseline_and_preserves_precision(dtype):
    problem = make_problem("toy", dtype=dtype, device="cpu")
    contexts = problem.sample(5, problem.key(5))
    params = init_params(problem, problem.key(2), width=7)
    np.testing.assert_array_equal(
        apply_model(params, problem.features(contexts)),
        np.zeros((5, problem.decision_dim)),
    )
    np.testing.assert_array_equal(
        initialize(problem, params, contexts), problem.baseline(contexts)
    )
    assert all(
        array.dtype == np.dtype(dtype) and array.device == problem.device
        for array in leaves(params)
    )


@pytest.mark.parametrize("kind", ["objective", "fixed_point", "regression"])
def test_all_losses_train_without_nan_at_exact_solution(kind):
    # In one dimension this toy has curvature=1 and takes the exact minimizer
    # in one step. Both unsquared norm losses must use a finite zero
    # subgradient.
    problem = make_problem("toy", dimension=1, dtype="float64", device="cpu")
    contexts = problem.sample(6, problem.key(11))
    validation = problem.sample(4, problem.key(12))
    params = init_params(problem, problem.key(13), width=4)

    def loss(parameters):
        output = rollout(
            problem, initialize(problem, parameters, contexts), contexts, 1
        )
        return batch_loss(
            problem,
            output,
            contexts,
            kind,
            contexts if kind == "regression" else None,
        )

    value, gradient = jax.jit(jax.value_and_grad(loss))(params)
    assert float(value) == 0.0
    assert all(np.isfinite(array).all() for array in leaves(gradient))
    assert all(np.all(np.asarray(array) == 0) for array in leaves(gradient))
    arguments = (
        {"targets": contexts, "validation_targets": validation}
        if kind == "regression"
        else {}
    )
    result = train(
        problem,
        short_config(loss=kind, unroll=1, steps=2, width=4),
        contexts=contexts,
        validation=validation,
        **arguments,
    )
    assert result.best_step == 0
    assert all(row["validation_loss"] == 0.0 for row in result.history)
    assert all(np.isfinite(array).all() for array in leaves(result.params))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_seed_determinism_and_eager_jit_agreement(dtype):
    problem = make_problem("toy", dtype=dtype, device="cpu")
    config = short_config()
    compiled = train(problem, config)
    repeated = train(problem, config)
    eager = train(problem, replace(config, jit=False))
    assert [row["step"] for row in compiled.history] == [0, 3, 5]
    assert compiled.best_step == repeated.best_step == eager.best_step
    assert (
        compiled.history[-1]["validation_objective"]
        < compiled.history[0]["validation_objective"]
    )
    tolerance = 2e-6 if dtype == "float32" else 2e-12
    for first, second, third in zip(
        leaves(compiled.params), leaves(repeated.params), leaves(eager.params)
    ):
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, third, atol=tolerance, rtol=tolerance)
        assert first.dtype == np.dtype(dtype)
    for first, second in zip(compiled.history, eager.history):
        np.testing.assert_allclose(
            first["validation_loss"],
            second["validation_loss"],
            atol=tolerance,
            rtol=tolerance,
        )
    assert compiled.compilation_seconds > 0
    assert eager.compilation_seconds == 0
    assert compiled.execution_seconds > 0
    assert (
        compiled.elapsed_seconds
        >= compiled.compilation_seconds + compiled.execution_seconds
    )


@pytest.mark.parametrize(
    "source_dtype,target_dtype",
    [("float64", "float32"), ("float32", "float64")],
)
def test_checkpoint_preserves_nondefault_problem_and_supports_precision_move(
    tmp_path, source_dtype, target_dtype
):
    original = make_problem(
        "toy", dimension=5, dtype=source_dtype, device="cpu"
    )
    fixed = original.state_dict()
    fixed["curvature"] *= 1.25
    problem = original.with_state(fixed)
    result = train(problem, short_config(steps=2, width=6))
    path = tmp_path / "portable.npz"
    save_checkpoint(result, problem, path)
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind != "O" for key in archive.files)
        metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
        assert metadata["problem_config"]["dimension"] == 5
        assert metadata["dtype"] == source_dtype
    restored_problem, restored = load_checkpoint(path, device="cpu")
    assert restored_problem.context_dim == restored_problem.decision_dim == 5
    assert restored_problem.config == {"dimension": 5}
    np.testing.assert_array_equal(restored_problem.curvature, problem.curvature)
    for first, second in zip(leaves(result.params), leaves(restored.params)):
        np.testing.assert_array_equal(first, second)
    assert restored.history == result.history
    assert restored.config == result.config
    assert restored.best_step == result.best_step
    assert restored.compilation_seconds == result.compilation_seconds
    assert restored.execution_seconds == result.execution_seconds
    contexts = problem.sample(4, problem.key(7))
    np.testing.assert_array_equal(
        predict(problem, result.params, contexts),
        predict(restored_problem, restored.params, contexts),
    )
    moved_problem, moved = load_checkpoint(
        path, dtype=target_dtype, device="cpu"
    )
    assert moved_problem.dtype == np.dtype(target_dtype)
    assert all(
        array.dtype == np.dtype(target_dtype)
        and array.device == moved_problem.device
        for array in leaves(moved.params)
    )
    np.testing.assert_array_equal(
        moved_problem.curvature,
        np.asarray(problem.curvature).astype(target_dtype),
    )
    np.testing.assert_array_equal(
        original.curvature, np.logspace(0, 1, 5).astype(source_dtype)
    )
    with pytest.raises(AttributeError, match="immutable"):
        moved_problem.curvature = moved_problem.curvature * 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("steps", 0),
        ("unroll", -1),
        ("width", True),
        ("seed", -1),
        ("learning_rate", float("nan")),
        ("grad_clip", 0),
        ("jit", "yes"),
    ],
)
def test_invalid_config_fails_early(field, value):
    with pytest.raises(ValueError, match=field):
        replace(short_config(), **{field: value}).validate()


def test_supplied_data_shapes_precision_and_finiteness_are_checked():
    problem = make_problem("toy", dtype="float32", device="cpu")
    good = problem.sample(4, problem.key(1))
    config = short_config(steps=1)
    with pytest.raises(ValueError, match="shape"):
        train(problem, config, contexts=good[:, :1])
    with pytest.raises(ValueError, match="dtype"):
        train(problem, config, contexts=np.asarray(good, dtype=np.float64))
    bad = np.array(good, copy=True)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        train(problem, config, contexts=bad)
    with pytest.raises(ValueError, match="same batch size"):
        train(
            problem,
            replace(config, loss="regression"),
            contexts=good,
            validation=good,
            targets=good[:2],
            validation_targets=good,
        )


def test_checkpoint_rejects_object_arrays(tmp_path):
    path = tmp_path / "not_a_checkpoint.npz"
    np.savez(path, metadata=np.array([{"unsafe": "object"}], dtype=object))
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_checkpoint(path)


@pytest.mark.parametrize("kind", ["fixed_point", "regression"])
def test_norm_losses_learn_on_nontrivial_finite_horizon(kind):
    problem = make_problem("toy", dtype="float64", device="cpu")
    contexts = problem.sample(16, problem.key(3))
    validation = problem.sample(8, problem.key(4))
    labels = (
        {"targets": contexts, "validation_targets": validation}
        if kind == "regression"
        else {}
    )
    result = train(
        problem,
        short_config(loss=kind, steps=4, validate_every=2),
        contexts=contexts,
        validation=validation,
        **labels,
    )
    assert result.best_step > 0
    assert (
        min(row["validation_loss"] for row in result.history[1:])
        < result.history[0]["validation_loss"]
    )


def test_reference_diagnostics_warn_and_survive_checkpoint(
    monkeypatch, tmp_path
):
    import sys
    from types import SimpleNamespace

    problem = make_problem("toy", dtype="float32", device="cpu")
    contexts = problem.sample(4, problem.key(8))

    def fake_reference(_problem, data):
        return SimpleNamespace(
            solution=np.asarray(data, dtype=np.float64),
            certificates=np.array([0.0, 2e-7, 0.0, 2e-7]),
            globally_certified=np.array([True, False, True, False]),
            solver_success=np.array([True, True, True, True]),
            stationarity=np.array([0.0, 1e-3, 0.0, 1e-3]),
            label="numerical reference with convex certificate",
            elapsed_seconds=0.25,
            compilation_seconds=0.05,
            tolerance=1e-7,
        )

    # Patch the module boundary, so this test checks label handling rather than
    # relying on whether a particular external solver reaches a given tolerance.
    monkeypatch.setitem(
        sys.modules,
        "l2ws_jax.reference",
        SimpleNamespace(solve_reference=fake_reference),
    )
    with pytest.warns(RuntimeWarning, match="CPU float64 1e-7") as caught:
        result = train(
            problem,
            short_config(loss="regression", steps=1),
            contexts=contexts,
            validation=contexts,
        )
    assert len(caught) == 2
    for split in ("training", "validation"):
        information = result.reference_info[split]
        assert information["certified_fraction_at_1e-7"] == 0.5
        assert information["solver_success_fraction"] == 1.0
        assert information["max_certificate"] == 2e-7
        assert information["certificate_dtype"] == "float64"
    assert all(
        array.dtype == np.dtype("float32") for array in leaves(result.params)
    )
    path = tmp_path / "with_diagnostics.npz"
    save_checkpoint(result, problem, path)
    _, restored = load_checkpoint(path, device="cpu")
    assert restored.reference_info == result.reference_info

"""Public common-infrastructure tests.

The tests do not require homework application solutions.
"""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from l2ws_jax import fixed_point, make_problem, make_rollout, rollout
from l2ws_jax.benchmark import (
    benchmark_problem,
    evaluation_seed,
    run_benchmarks,
)
from l2ws_jax.numerics import euclidean_norm
from l2ws_jax.runtime import resolve_device
from l2ws_jax.training import RANDOM_SEED_UPPER_BOUND


def test_benchmark_refuses_nonempty_output_directory(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "old-result.txt"
    sentinel.write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="new or empty"):
        run_benchmarks(output, names=(), make_plot=False)
    assert sentinel.read_text(encoding="utf-8") == "old"


@pytest.mark.parametrize("names", [(), ("toy", "toy"), ("missing",)])
def test_benchmark_rejects_invalid_problem_lists_before_creating_output(
    tmp_path, names
):
    output = tmp_path / "unused"
    with pytest.raises(ValueError):
        run_benchmarks(output, names=names, make_plot=False)
    assert not output.exists()


def test_invalid_benchmark_configuration_creates_no_problem_directory(
    tmp_path,
):
    with pytest.raises(ValueError, match="steps"):
        benchmark_problem("toy", tmp_path, train_steps=0)
    assert not (tmp_path / "toy").exists()


def test_evaluation_seed_pairs_are_bounded_and_can_be_held_fixed():
    first = {
        evaluation_seed(7, False),
        evaluation_seed(7, True),
    }
    second = {
        evaluation_seed(8, False),
        evaluation_seed(8, True),
    }
    assert first.isdisjoint(second)
    assert evaluation_seed(7, False, base_seed=107) == 107
    assert evaluation_seed(7, True, base_seed=107) == 108
    assert evaluation_seed(8, False, base_seed=107) == 107
    assert evaluation_seed(8, True, base_seed=107) == 108
    wrapped = evaluation_seed(RANDOM_SEED_UPPER_BOUND - 1, True)
    assert 0 <= wrapped < RANDOM_SEED_UPPER_BOUND


@pytest.mark.parametrize(
    ("training_seed", "shifted", "base_seed"),
    [
        (-1, False, None),
        (RANDOM_SEED_UPPER_BOUND, False, None),
        (7, 1, None),
        (7, False, -1),
        (7, False, RANDOM_SEED_UPPER_BOUND),
    ],
)
def test_evaluation_seed_rejects_invalid_inputs(
    training_seed, shifted, base_seed
):
    with pytest.raises(ValueError):
        evaluation_seed(
            training_seed,
            shifted,
            base_seed=base_seed,
        )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_unrolled_gradient_matches_quadratic_closed_form(dtype):
    p = make_problem("toy", dtype=dtype, device="cpu")
    context = p.array([[0.3, -0.2, 0.5], [-0.4, 0.8, -0.1]])
    start = p.array([[0.2, 0.6, -0.1], [0.1, -0.2, 0.7]])
    steps = 4

    def value(y):
        return jnp.sum(p.objective(rollout(p, y, context, steps), context))

    observed = jax.jit(jax.grad(value))(start)
    expected = (
        p.curvature
        * (1.0 - 0.1 * p.curvature) ** (2 * steps)
        * (start - context)
    )
    tolerance = 2e-6 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(
        observed, expected, rtol=tolerance, atol=tolerance
    )
    assert observed.dtype == p.dtype


def test_batch_size_does_not_rescale_solver():
    p = make_problem("toy", dtype="float64")
    context = p.array([[0.3, -0.1, 0.7]])
    start = p.array([[0.8, 0.4, -0.2]])
    single = fixed_point(p, start, context)
    repeated = fixed_point(
        p, jnp.repeat(start, 7, axis=0), jnp.repeat(context, 7, axis=0)
    )
    np.testing.assert_allclose(
        repeated, np.repeat(np.asarray(single), 7, axis=0), atol=0, rtol=0
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_zero_norm_has_finite_first_and_second_derivatives(dtype):
    p = make_problem("toy", dtype=dtype)
    zero = p.array([0.0, 0.0, 0.0])
    assert float(euclidean_norm(zero)) == 0
    np.testing.assert_array_equal(jax.grad(euclidean_norm)(zero), np.zeros(3))
    assert np.isfinite(jax.jacfwd(jax.grad(euclidean_norm))(zero)).all()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_jit_eager_and_trajectory_agree(dtype):
    p = make_problem("toy", dtype=dtype)
    context = p.sample(8, p.key(82))
    start = p.baseline(context)
    compiled = make_rollout(p, 5)(start, context)
    with jax.disable_jit():
        eager = rollout(p, start, context, 5)
        identity = rollout(p, start, context, 0)
    path = make_rollout(p, 5, return_path=True)(start, context)
    assert path.shape == (6, 8, p.decision_dim)
    tolerance = 2e-6 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(compiled, eager, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(
        compiled, path[-1], atol=tolerance, rtol=tolerance
    )
    np.testing.assert_array_equal(identity, start)


def test_precision_coexists_and_problem_state_is_immutable():
    p64 = make_problem("toy", dtype="float64")
    p32 = p64.to(dtype="float32")
    assert p32.curvature.dtype == np.dtype("float32")
    assert p64.curvature.dtype == np.dtype("float64")
    assert jax.config.x64_enabled
    with pytest.raises(AttributeError, match="immutable"):
        p32.curvature = p32.curvature * 2
    with pytest.raises(TypeError):
        p32.data["curvature"] = p32.curvature
    before = p32.state_dict()["curvature"]
    with pytest.raises(AttributeError, match="immutable"):
        p32.register("curvature", p32.curvature * 2)
    np.testing.assert_array_equal(p32.state_dict()["curvature"], before)
    np.testing.assert_array_equal(p32.curvature, before)
    changed = p64.config
    changed["dimension"] = 77
    assert p64.config["dimension"] == 3


@pytest.mark.parametrize("device", ["banana", "gpu:-1", "cpu:abc", "cpu:0:1"])
def test_invalid_device_is_rejected(device):
    with pytest.raises(ValueError):
        resolve_device(device)


def test_standalone_constant_outputs_respect_selected_device():
    # A fresh process exposes two actual logical CPU devices. This tests device
    # selection without pretending that a CPU execution is GPU coverage.
    code = """
import jax
import jax.numpy as jnp
from l2ws_jax import make_problem, init_params, predict
from l2ws_jax.problems.toy import QuadraticDemo
class ConstantStepDemo(QuadraticDemo):
    def step_size_one(self, context):
        return jnp.asarray(.1, dtype=self.dtype)
p = ConstantStepDemo(dtype='float32', device='cpu:1')
x = p.sample(3, p.key(8))
assert x.device == p.device
assert p.step_size(x).device == p.device
assert p.baseline(x).device == p.device
assert (p.baseline(x) + 1).device == p.device
params = init_params(p, p.key(9), width=6)
y = jax.block_until_ready(predict(p, params, x, steps=2))
assert y.device == p.device
"""
    environment = dict(
        os.environ,
        JAX_NUM_CPU_DEVICES="2",
        JAX_PLATFORMS="cpu",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )

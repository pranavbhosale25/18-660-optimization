from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_pncg import (
    PNCGOptions,
    STATUS_CONVERGED,
    STATUS_MAXITER,
    diagonal_preconditioner,
    make_solver,
)
from jax_pncg.test_problems import (
    BEALE_2,
    POWELL_SINGULAR_4,
    ROSENBROCK_2,
    WOOD_4,
    diagonal_quadratic,
    ill_conditioned_quadratic_diagonal,
)


@pytest.mark.parametrize("beta_method", ["pr", "pr+"])
def test_rosenbrock_converges_for_pr_variants(beta_method: str) -> None:
    options = PNCGOptions(
        maxiter=1000,
        gtol=1.0e-8,
        beta_method=beta_method,
        record_history=True,
    )
    result = make_solver(ROSENBROCK_2.objective, options=options)(
        jnp.asarray(ROSENBROCK_2.start, dtype=jnp.float64)
    )

    np.testing.assert_allclose(result.x, np.ones(2), rtol=0.0, atol=2.0e-7)
    assert int(result.status) == STATUS_CONVERGED
    assert bool(result.success)
    assert float(result.grad_norm) <= options.gtol
    assert int(result.history.size) == int(result.nit) + 1

    values = np.asarray(result.history.value[: int(result.history.size)])
    # Armijo sufficient decrease makes accepted objective values monotone.
    assert np.all(np.diff(values) <= 5.0e-13)


@pytest.mark.parametrize(
    "problem, atol, maxiter",
    [
        (BEALE_2, 2.0e-6, 1000),
        (WOOD_4, 2.0e-5, 2000),
        # Powell is singular at the solution, so a small gradient does not
        # imply the same coordinate accuracy as on a strongly convex problem.
        (POWELL_SINGULAR_4, 3.0e-3, 3000),
    ],
)
def test_standard_test_problems(problem, atol: float, maxiter: int) -> None:
    options = PNCGOptions(maxiter=maxiter, gtol=1.0e-7)
    result = make_solver(problem.objective, options=options)(
        jnp.asarray(problem.start, dtype=jnp.float64)
    )

    assert int(result.status) == STATUS_CONVERGED
    np.testing.assert_allclose(
        result.x,
        np.asarray(problem.minimizer),
        rtol=0.0,
        atol=atol,
    )
    assert float(result.fun) < 2.0e-10


def test_preconditioned_pr_coefficient_uses_new_z_and_gradient_difference() -> None:
    def objective(x: jax.Array) -> jax.Array:
        diagonal = jnp.asarray([1.0, 4.0], dtype=x.dtype)
        return 0.5 * jnp.vdot(x, diagonal * x).real

    def variable_preconditioner(x: jax.Array, gradient: jax.Array) -> jax.Array:
        return gradient / (1.0 + x**2)

    x0 = jnp.asarray([2.0, 1.0], dtype=jnp.float64)
    options = PNCGOptions(
        maxiter=1,
        gtol=0.0,
        beta_method="pr",
        descent_restart=False,
        record_history=True,
    )
    result = make_solver(
        objective, preconditioner=variable_preconditioner, options=options
    )(x0)

    grad0 = np.asarray([2.0, 4.0])
    z0 = grad0 / (1.0 + np.asarray(x0) ** 2)
    grad1 = np.asarray(result.grad)
    z1 = np.asarray(result.preconditioned_grad)
    expected = np.vdot(z1, grad1 - grad0).real / np.vdot(grad0, z0).real
    alternative = np.vdot(grad1, z1 - z0).real / np.vdot(grad0, z0).real

    assert not np.isclose(expected, alternative)
    np.testing.assert_allclose(
        result.history.beta[1], expected, rtol=2.0e-14, atol=2.0e-14
    )


def test_fixed_diagonal_preconditioner_solves_scaled_quadratic_in_one_step() -> None:
    n = 64
    diagonal_np = ill_conditioned_quadratic_diagonal(n, condition_number=1.0e8)
    diagonal = jnp.asarray(diagonal_np, dtype=jnp.float64)
    x0 = jnp.ones((n,), dtype=jnp.float64)
    options = PNCGOptions(maxiter=10, gtol=1.0e-8)

    solver = make_solver(
        diagonal_quadratic,
        preconditioner=diagonal_preconditioner(diagonal_np),
        options=options,
    )
    result = solver(x0, diagonal)

    assert int(result.status) == STATUS_CONVERGED
    assert int(result.nit) == 1
    assert int(result.nfev) == 2
    np.testing.assert_allclose(result.x, np.zeros(n), rtol=0.0, atol=1.0e-14)


def test_maxiter_status_is_reported_without_host_control_flow() -> None:
    options = PNCGOptions(maxiter=1, gtol=1.0e-14)
    result = make_solver(ROSENBROCK_2.objective, options=options)(
        jnp.asarray(ROSENBROCK_2.start, dtype=jnp.float64)
    )

    assert int(result.status) == STATUS_MAXITER
    assert not bool(result.success)
    assert int(result.nit) == 1


def test_solver_is_jittable_as_a_component() -> None:
    solver = make_solver(
        ROSENBROCK_2.objective,
        options=PNCGOptions(maxiter=1000, gtol=1.0e-7),
        jit_compile=False,
    )

    @jax.jit
    def final_value(x0: jax.Array) -> jax.Array:
        return solver(x0).fun

    value = final_value(jnp.asarray(ROSENBROCK_2.start, dtype=jnp.float64))
    assert float(value) < 1.0e-12


def test_input_must_be_real_vector() -> None:
    solver = make_solver(ROSENBROCK_2.objective)
    with pytest.raises(TypeError):
        solver(jnp.asarray([-1.2 + 0.0j, 1.0 + 0.0j]))


def test_input_must_be_a_nonempty_vector() -> None:
    solver = make_solver(ROSENBROCK_2.objective)
    with pytest.raises(ValueError):
        solver(jnp.ones((2, 2), dtype=jnp.float64))
    with pytest.raises(ValueError):
        solver(jnp.empty((0,), dtype=jnp.float64))


def test_invalid_preconditioner_falls_back_to_the_gradient() -> None:
    def objective(x: jax.Array) -> jax.Array:
        return 0.5 * jnp.vdot(x, x).real

    def invalid_preconditioner(
        x: jax.Array, gradient: jax.Array
    ) -> jax.Array:
        del x
        return -gradient

    result = make_solver(
        objective,
        preconditioner=invalid_preconditioner,
        options=PNCGOptions(maxiter=10, gtol=1.0e-12),
    )(jnp.asarray([1.0, -2.0], dtype=jnp.float64))

    assert bool(result.success)
    assert int(result.nit) == 1
    assert int(result.restarts) == 1


def test_stationary_point_ignores_a_nonfinite_preconditioner() -> None:
    def objective(x: jax.Array) -> jax.Array:
        return 0.5 * jnp.vdot(x, x).real

    def invalid_preconditioner(
        x: jax.Array, gradient: jax.Array
    ) -> jax.Array:
        del x, gradient
        return jnp.asarray([jnp.nan, jnp.nan])

    result = make_solver(
        objective,
        preconditioner=invalid_preconditioner,
        options=PNCGOptions(gtol=0.0),
    )(jnp.zeros((2,), dtype=jnp.float64))

    assert bool(result.success)
    assert int(result.nit) == 0
    assert int(result.restarts) == 0
    np.testing.assert_array_equal(result.preconditioned_grad, np.zeros(2))


def test_preconditioner_output_shape_is_checked() -> None:
    def bad_shape_preconditioner(
        x: jax.Array, gradient: jax.Array
    ) -> jax.Array:
        del x, gradient
        return jnp.ones((1,), dtype=jnp.float64)

    solver = make_solver(
        ROSENBROCK_2.objective,
        preconditioner=bad_shape_preconditioner,
    )
    with pytest.raises(ValueError, match="same shape"):
        solver(jnp.asarray(ROSENBROCK_2.start, dtype=jnp.float64))

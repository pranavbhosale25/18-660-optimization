from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from trsr1.updates import safeguarded_sr1_update

jax.config.update("jax_enable_x64", True)


def test_sr1_update_is_symmetric_and_satisfies_secant_equation() -> None:
    matrix = jnp.eye(2, dtype=jnp.float64)
    step = jnp.asarray([1.0, 2.0])
    gradient_difference = jnp.asarray([2.0, 4.0])

    update = jax.jit(
        lambda b, s, y: safeguarded_sr1_update(
            b, s, y, skip_tolerance=1.0e-8
        )
    )(matrix, step, gradient_difference)

    assert bool(update.applied)
    np.testing.assert_allclose(update.matrix, update.matrix.T, atol=1.0e-14)
    np.testing.assert_allclose(
        update.matrix @ step,
        gradient_difference,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert float(update.secant_residual_norm) < 1.0e-12


def test_sr1_update_skips_degenerate_denominator() -> None:
    matrix = jnp.eye(2, dtype=jnp.float64)
    step = jnp.asarray([1.0, 0.0])
    gradient_difference = jnp.asarray([1.0, 1.0])
    # u = y - B s = [0, 1], so u.T s = 0.
    update = safeguarded_sr1_update(
        matrix,
        step,
        gradient_difference,
        skip_tolerance=1.0e-8,
    )

    assert not bool(update.applied)
    np.testing.assert_array_equal(update.matrix, matrix)


def test_sr1_update_skips_when_model_already_satisfies_secant() -> None:
    matrix = jnp.asarray([[3.0, 1.0], [1.0, 2.0]])
    step = jnp.asarray([0.5, -1.0])
    gradient_difference = matrix @ step
    update = safeguarded_sr1_update(
        matrix,
        step,
        gradient_difference,
        skip_tolerance=1.0e-8,
    )

    assert not bool(update.applied)
    assert float(update.residual_norm) == 0.0
    np.testing.assert_array_equal(update.matrix, matrix)

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_pncg import PNCGOptions
from jax_pncg.line_search import (
    LS_NOT_DESCENT,
    LS_SUCCESS,
    strong_wolfe_line_search,
)


def _quadratic(x: jax.Array) -> jax.Array:
    diagonal = jnp.asarray([1.0, 25.0], dtype=x.dtype)
    return 0.5 * jnp.vdot(x, diagonal * x).real


def test_zoom_returns_a_step_satisfying_the_strong_wolfe_conditions() -> None:
    options = PNCGOptions(c1=1.0e-4, c2=0.1, initial_step=1.0)
    value_and_grad = jax.value_and_grad(_quadratic)
    x = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    value0, grad0 = value_and_grad(x)
    direction = -grad0
    slope0 = jnp.vdot(grad0, direction).real

    result = strong_wolfe_line_search(
        value_and_grad,
        x,
        value0,
        grad0,
        direction,
        jnp.asarray(options.initial_step, dtype=x.dtype),
        (),
        options,
    )

    assert bool(result.success)
    assert int(result.status) == LS_SUCCESS
    assert int(result.nfev) > 1  # The unit trial step is bracketed, then zoomed.
    assert float(result.value) <= float(
        value0 + options.c1 * result.alpha * slope0
    )
    assert abs(float(result.directional_derivative)) <= float(
        options.c2 * abs(slope0)
    )


def test_non_descent_direction_is_rejected_without_an_evaluation() -> None:
    options = PNCGOptions()
    value_and_grad = jax.value_and_grad(_quadratic)
    x = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    value0, grad0 = value_and_grad(x)

    result = strong_wolfe_line_search(
        value_and_grad,
        x,
        value0,
        grad0,
        grad0,
        jnp.asarray(1.0, dtype=x.dtype),
        (),
        options,
    )

    assert not bool(result.success)
    assert int(result.status) == LS_NOT_DESCENT
    assert int(result.nfev) == 0

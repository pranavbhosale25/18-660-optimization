"""Batched projected gradient, differentiable scan, and JIT factories."""

from functools import lru_cache

import jax
import jax.numpy as jnp

from .numerics import euclidean_norm


def _step(problem, y, context, alpha):
    return problem.project(y - alpha * problem.gradient(y, context), context)


def fixed_point(problem, y, context):
    return _step(problem, y, context, problem.step_size(context))


def rollout(problem, y0, context, steps, *, return_path=False):
    """Run a pure solve with a loop length that is static under JIT.

    Exact autodiff through this function retains Hessian-vector dependencies.
    The context-dependent step bound is computed once per solve. No host data
    conversion, Python iteration over instances, or mutable solver state occurs.
    """
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative Python integer")
    if steps == 0:
        return y0[None] if return_path else y0
    alpha = problem.step_size(context)
    if return_path:

        def body(y, _):
            next_y = _step(problem, y, context, alpha)
            return next_y, next_y

        _, path = jax.lax.scan(body, y0, xs=None, length=steps)
        return jnp.concatenate((y0[None], path), axis=0)

    def body(y, _):
        return _step(problem, y, context, alpha), None

    result, _ = jax.lax.scan(body, y0, xs=None, length=steps)
    return result


def fixed_point_residual(problem, y, context):
    return euclidean_norm(fixed_point(problem, y, context) - y)


def gradient_mapping_norm(problem, y, context):
    alpha = problem.step_size(context)
    return euclidean_norm((y - _step(problem, y, context, alpha)) / alpha)


def batch_loss(problem, y, context, kind, reference=None):
    if kind == "objective":
        values = problem.objective(y, context)
        if values.shape != (len(y),):
            raise ValueError("Batched objective must have shape [batch]")
        return jnp.mean(values)
    if kind == "fixed_point":
        return jnp.mean(fixed_point_residual(problem, y, context))
    if kind == "regression":
        if reference is None:
            raise ValueError("Regression loss requires reference solutions")
        return jnp.mean(problem.solution_distance(y, reference))
    raise ValueError("loss must be objective, fixed_point, or regression")


@lru_cache(maxsize=128)
def make_rollout(problem, steps, *, return_path=False, jit=True):
    """Cache a solve; dtype, shape, or horizon changes may recompile."""

    def solve(y0, context):
        return rollout(problem, y0, context, steps, return_path=return_path)

    return jax.jit(solve) if jit else solve


@lru_cache(maxsize=128)
def make_predictor(problem, steps=0, *, jit=True):
    """Compile the network, decoder, solver, and baseline together."""
    from .model import initialize

    def inference(params, context):
        return rollout(
            problem, initialize(problem, params, context), context, steps
        )

    return jax.jit(inference) if jit else inference


def predict(problem, params, context, steps=0, *, jit=True):
    return make_predictor(problem, steps, jit=jit)(params, context)


@lru_cache(maxsize=128)
def make_baseline_solver(problem, steps=0, *, jit=True):
    """Use the same compilation scope for the classical comparison."""

    def inference(context):
        return rollout(problem, problem.baseline(context), context, steps)

    return jax.jit(inference) if jit else inference

"""Pure functional, jitted score-function and direct-gradient learning stages."""
from typing import Callable, NamedTuple
import jax
import jax.numpy as jnp
from .types import (TrainState, PreparedBatch, DirectState, ObjectiveResult,
                    Policy, Optimizer, Bootstrap, CriticApply)
from .optim import apply_updates, global_norm
from .objectives import critic_loss, masked_mean


class Learner(NamedTuple):
    init: Callable
    prepare: Callable
    update: Callable


def make_learner(*, policy: Policy, bootstrap: Bootstrap, estimator: Callable,
                 objective: Callable, actor_optimizer: Optimizer,
                 critic_apply: CriticApply | None = None,
                 critic_optimizer: Optimizer | None = None,
                 baseline: Bootstrap | None = None) -> Learner:
    """Construct compiled functions. Models, losses, estimators and optimizers swap independently.

    prepare() freezes behavior data, targets and advantages. Calling update()
    repeatedly with the same PreparedBatch is appropriate for PPO, but not for
    uncorrected REINFORCE/A2C. Actor and critic use separate parameter PyTrees.
    baseline optionally overrides the state-only control variate independently
    of the return tail (e.g. Monte Carlo returns with a learned V baseline).
    """
    if (critic_apply is None) != (critic_optimizer is None):
        raise ValueError("Supply both critic_apply and critic_optimizer, or neither")

    objective_value_and_grad = getattr(objective, "value_and_grad", None)

    def init(actor_params, critic_params=None):
        if critic_params is None:
            critic_params = {}
        return TrainState(actor_params, critic_params, actor_optimizer.init(actor_params),
                          () if critic_optimizer is None else critic_optimizer.init(critic_params),
                          jnp.asarray(0, jnp.int32))

    def prepare(actor_params, critic_params, trajectory, key):
        batch = jax.tree.map(jax.lax.stop_gradient, trajectory)
        key_s, key_n = jax.random.split(key)
        state_keys = jax.random.split(key_s, batch.rewards.shape)
        next_keys = jax.random.split(key_n, batch.rewards.shape)
        state_fn = bootstrap if baseline is None else baseline
        evaluate_state = jax.vmap(jax.vmap(lambda obs, k: state_fn(actor_params, critic_params, obs, k)))
        evaluate_next = jax.vmap(jax.vmap(lambda obs, k: bootstrap(actor_params, critic_params, obs, k)))
        # Independent keys: a Q-based baseline must not reuse the actual action sample.
        state_values = evaluate_state(batch.observations, state_keys)
        next_values = evaluate_next(batch.next_observations, next_keys)
        estimates = estimator(batch, state_values, next_values)
        return jax.tree.map(jax.lax.stop_gradient, PreparedBatch(
            batch, estimates.targets, estimates.advantages, estimates.mask))

    def update(state, prepared, key):
        data = jax.tree.map(jax.lax.stop_gradient, prepared)
        if objective_value_and_grad is None:
            (actor_loss, stats), actor_grads = jax.value_and_grad(
                lambda p: objective(policy, p, data, key),
                has_aux=True,
            )(state.actor_params)
        else:
            (actor_loss, stats), actor_grads = objective_value_and_grad(
                policy, state.actor_params, data, key)
        active = jnp.any(data.mask)

        def actor_step(_):
            updates, opt_state = actor_optimizer.update(actor_grads, state.actor_opt_state, state.actor_params)
            return apply_updates(state.actor_params, updates), opt_state

        actor_params, actor_opt_state = jax.lax.cond(
            active, actor_step, lambda _: (state.actor_params, state.actor_opt_state), operand=None)
        if critic_apply is not None:
            critic_value, critic_grads = jax.value_and_grad(
                lambda params: critic_loss(critic_apply, params, data)
            )(state.critic_params)

            def critic_step(_):
                updates, opt_state = critic_optimizer.update(
                    critic_grads, state.critic_opt_state, state.critic_params)
                return apply_updates(state.critic_params, updates), opt_state

            critic_params, critic_opt_state = jax.lax.cond(
                active, critic_step, lambda _: (state.critic_params, state.critic_opt_state), operand=None)
            critic_grad_norm = global_norm(critic_grads)
        else:
            critic_params, critic_opt_state = state.critic_params, state.critic_opt_state
            critic_value = jnp.zeros((), actor_loss.dtype)
            critic_grad_norm = jnp.zeros((), actor_loss.dtype)
        new_state = TrainState(actor_params, critic_params, actor_opt_state, critic_opt_state,
                               state.updates + active.astype(jnp.int32))
        stats = dict(stats, actor_loss=actor_loss, critic_loss=critic_value,
                     actor_grad_norm=global_norm(actor_grads), critic_grad_norm=critic_grad_norm,
                     mean_target=masked_mean(data.targets, data.mask), valid_count=jnp.sum(data.mask))
        return new_state, stats

    return Learner(init, jax.jit(prepare), jax.jit(update))


class DirectLearner(NamedTuple):
    init: Callable
    update: Callable


def make_direct_learner(return_fn: Callable, optimizer: Optimizer, *,
                        analytic_grad: Callable | None = None) -> DirectLearner:
    """MAXIMIZE a differentiable expected/sample return, or use its supplied gradient.

    return_fn(params, context, key) -> ObjectiveResult(value, valid_count).
    analytic_grad(params, context, key) -> gradient PyTree of POSITIVE value.
    In the absence of analytic_grad, JAX reverse-mode AD computes the pathwise
    derivative. A discrete exact-enumeration objective may also use this API.
    """
    def init(params):
        return DirectState(params, optimizer.init(params), jnp.asarray(0, jnp.int32))

    def update(state, context, key):
        if analytic_grad is None:
            def loss_fn(params):
                result = return_fn(params, context, key)
                return -result.value, result.valid_count
            (loss, valid_count), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            value = -loss
        else:
            result = return_fn(state.params, context, key)
            value, valid_count = result
            grads = jax.tree.map(lambda g: -g, analytic_grad(state.params, context, key))
        active = valid_count > 0

        def do_update(_):
            updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
            return apply_updates(state.params, updates), opt_state

        params, opt_state = jax.lax.cond(
            active, do_update, lambda _: (state.params, state.opt_state), operand=None)
        return DirectState(params, opt_state, state.updates + active.astype(jnp.int32)), {
            "return": value, "loss": -value, "grad_norm": global_norm(grads), "valid_count": valid_count}

    return DirectLearner(init, jax.jit(update))

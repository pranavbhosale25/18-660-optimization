"""Interchangeable actor and critic objectives.

Prepared trajectories, return targets, and advantages are treated as fixed data
by these functions.  Arrays describing trajectories use leading ``[time,
batch]`` axes; model parameters and gradients may be arbitrary array PyTrees.
"""
from dataclasses import dataclass
from typing import Literal
import jax
import jax.numpy as jnp
from .types import Array, CriticApply, Policy, PreparedBatch, PyTree


def masked_mean(x: Array, mask: Array) -> Array:
    return jnp.sum(jnp.where(mask, x, 0.)) / jnp.maximum(jnp.sum(mask), 1)


def weighted_log_probability_gradient(
        policy: Policy,
        params: PyTree,
        observations: Array,
        raw_actions: Array,
        coefficients: Array,
) -> PyTree:
    """Differentiate a weighted sum of recorded-action log probabilities.

    Parameters
    ----------
    policy:
        Static policy object. Its
        ``log_prob(params, observation, raw_action)`` method must return one
        rank-zero scalar log probability for one recorded transition. Do not
        sample a new action in this function.
    params:
        Actor parameter PyTree with differentiable floating-point array leaves.
        The returned gradient must have exactly the same container structure,
        leaf shapes, and corresponding floating dtypes.
    observations:
        Fixed recorded observations with shape ``[T, B, *observation_shape]``.
    raw_actions:
        Fixed recorded distribution-space actions with the same leading
        ``[T, B]`` axes as ``observations``. Categorical raw actions are integer
        indices with shape ``[T, B]``. Continuous raw actions are floating-point
        arrays with shape ``[T, B, *action_shape]``. For a bounded Gaussian,
        these are pre-transform samples, not the transformed actions sent to the
        environment.
    coefficients:
        Fixed floating-point array with shape ``[T, B]``. Entry ``[t, b]``
        multiplies the parameter gradient of the corresponding scalar log
        probability. Masking, discounting, advantages, signs, and loss-reduction
        factors are already included in these coefficients. Apply no additional
        sign, averaging, normalization, discount, or mask. For finite policy
        evaluations and derivatives, a zero coefficient contributes exactly
        zero.

    Returns
    -------
    PyTree
        One aggregate parameter-gradient PyTree. It is the sum, over all
        ``[t, b]`` records, of ``coefficients[t, b]`` times the gradient with
        respect to ``params`` of that record's log probability. Do not return a
        separate gradient for every record. If every coefficient is zero, every
        returned gradient leaf must be zero.

    Notes
    -----
    All ``T`` and ``B`` axes must agree. ``policy.log_prob`` and its parameter
    derivative are expected to be finite for every supplied record, including
    padded records whose coefficient is zero. Differentiate only with respect
    to ``params``; observations, raw actions, and coefficients are recorded data.
    The implementation must support arbitrary parameter PyTrees and remain
    compatible with ``jax.jit`` without converting arrays to NumPy/Python
    values or creating data-dependent output shapes.
    """
    # STUDENT TODO: construct and return the weighted log-probability gradient.
    observations, raw_actions, coefficients = jax.tree.map(
        jax.lax.stop_gradient, (observations, raw_actions, coefficients))

    def weighted_log_probability(actor_params):
        log_probabilities = jax.vmap(jax.vmap(
            lambda observation, action: policy.log_prob(
                actor_params, observation, action)))(observations, raw_actions)
        return jnp.sum(coefficients * log_probabilities)

    return jax.grad(weighted_log_probability)(params)


def _policy_terms(policy: Policy, params: PyTree, prepared: PreparedBatch,
                  key: Array, entropy_coef: float, normalize: bool):
    data = jax.tree.map(jax.lax.stop_gradient, prepared)
    batch = data.trajectory
    logp = jax.vmap(jax.vmap(lambda o, a: policy.log_prob(params, o, a)))(
        batch.observations, batch.raw_actions)
    advantage = data.advantages
    if normalize:
        mean = masked_mean(advantage, data.mask)
        variance = masked_mean((advantage - mean)**2, data.mask)
        advantage = (advantage - mean) / jnp.sqrt(variance + 1e-8)
    if entropy_coef:
        keys = jax.random.split(key, batch.rewards.shape)
        entropy = jax.vmap(jax.vmap(lambda o, k: policy.entropy(params, o, k)))(
            batch.observations, keys)
    else:
        entropy = jnp.zeros_like(logp)
    return logp, advantage, entropy, data


def _reduction_coefficients(data: PreparedBatch, gamma: float,
                            discount_weights: bool,
                            reduction: Literal["episodes", "transitions"]) -> Array:
    weights = jnp.ones_like(data.advantages)
    if discount_weights:
        weights = jnp.asarray(gamma, weights.dtype) ** data.trajectory.episode_steps
    if reduction == "episodes":
        denominator = jnp.asarray(data.mask.shape[1], weights.dtype)
    else:
        denominator = jnp.maximum(jnp.sum(data.mask), 1).astype(weights.dtype)
    return jnp.where(data.mask, weights / denominator, 0.)


def _reduce(values: Array, data: PreparedBatch, gamma: float,
            discount_weights: bool,
            reduction: Literal["episodes", "transitions"]) -> Array:
    coefficients = _reduction_coefficients(
        data, gamma, discount_weights, reduction)
    # Select masked records away rather than relying only on a zero
    # coefficient: IEEE arithmetic would otherwise allow ``0 * NaN`` from
    # padded records to contaminate the result.
    return jnp.sum(jnp.where(data.mask, coefficients * values, 0.))


def reinforce_actor_loss(
        log_probabilities: Array,
        advantages: Array,
        entropy: Array,
        data: PreparedBatch,
        *,
        gamma: float,
        entropy_coef: float,
        discount_weights: bool,
        reduction: Literal["episodes", "transitions"],
) -> Array:
    """Combine recorded REINFORCE terms into one scalar loss to minimize.

    Parameters
    ----------
    log_probabilities:
        Floating-point array with shape ``[T, B]``. Entry ``[t, b]`` is the
        current policy's scalar log probability of the recorded raw action.
        Preserve autodiff through this input.
    advantages:
        Floating-point array with shape ``[T, B]``. Use this argument directly,
        rather than ``data.advantages``: the caller may already have normalized
        it. In the learner this is fixed, stopped-gradient training data.
    entropy:
        Floating-point array with shape ``[T, B]`` containing the policy entropy
        associated with each record. Preserve autodiff through this input. A
        positive entropy coefficient makes entropy a bonus, so it lowers the
        minimized loss.
    data:
        Prepared batch providing the Boolean ``data.mask`` and integer
        ``data.trajectory.episode_steps``, both with shape ``[T, B]``. False mask
        entries contribute exactly zero. Other fields are not needed to reduce
        this loss. The learner has already detached prepared training data.
    gamma:
        Discount in ``[0, 1]``. When ``discount_weights`` is true, multiply each
        per-record loss by ``gamma ** episode_steps[t, b]``. Otherwise use a
        weight of one.
    entropy_coef:
        Nonnegative scalar multiplying the entropy bonus. Zero disables the
        entropy contribution.
    discount_weights:
        Whether to apply the outer within-episode discount described above.
    reduction:
        Either ``"episodes"`` or ``"transitions"``. Episode reduction sums all
        valid weighted records and divides by the total batch-lane count ``B``;
        empty lanes contribute zero and do not change that denominator.
        Transition reduction divides by the number of true mask entries, with a
        minimum denominator of one. Neither reduction divides by the sum of
        discount weights.

    Returns
    -------
    Array
        One rank-zero floating-point JAX array. Before masking and reduction,
        each record's minimized loss is the negative log probability times its
        supplied advantage, minus ``entropy_coef`` times its entropy. If every
        mask entry is false, return scalar zero. Do not mutate ``data`` or break
        gradients through ``log_probabilities`` and ``entropy``.

    Notes
    -----
    Configuration values are validated by ``ReinforceObjective`` before normal
    use and are static while its update is jitted. All array inputs must have
    matching ``[T, B]`` shapes and compatible floating dtypes.
    """
    # STUDENT TODO: combine the supplied actor data into one scalar loss.
    losses = -log_probabilities * advantages
    if entropy_coef:
        losses = losses - entropy_coef * entropy
    return _reduce(losses, data, gamma, discount_weights, reduction)


def critic_loss(critic_apply: CriticApply, params: PyTree,
                data: PreparedBatch) -> Array:
    """Return the half masked mean-squared error for a learned critic.

    Parameters
    ----------
    critic_apply:
        Static callable with signature
        ``critic_apply(params, observation, action) -> scalar``. It must return
        one rank-zero floating-point prediction for one record. A state-value
        critic may ignore ``action``. An action-value critic consumes it.
    params:
        Critic parameter PyTree with differentiable floating-point array leaves.
        Gradients of the returned scalar must flow with respect to this PyTree.
    data:
        Prepared batch. ``data.trajectory.observations`` and
        ``data.trajectory.actions`` have leading ``[T, B]`` axes;
        observations and continuous actions may have additional trailing axes.
        Use the environment action in ``actions``, not the distribution-space
        ``raw_actions``. Categorical actions are normally scalar integer indices.
        ``data.targets`` is floating point and ``data.mask`` is Boolean, both
        with shape ``[T, B]``.

    Returns
    -------
    Array
        One rank-zero floating-point JAX array. For every true mask entry,
        compute the squared difference between the critic prediction and target;
        return one half of their mean. The denominator is the number of true
        mask entries, with a minimum of one. False entries contribute exactly
        zero, and an all-false mask returns scalar zero.

    Notes
    -----
    Treat the entire ``PreparedBatch`` as fixed data inside this function. In
    particular, differentiating the result with respect to ``data.targets`` must
    produce zero, even when a target contains a bootstrap prediction. Only the
    derivative with respect to ``params`` is used for critic training.

    Evaluate every ``[t, b]`` record while preserving arbitrary trailing action
    and observation axes. The implementation must support arbitrary parameter
    PyTrees and remain compatible with nested ``jax.vmap``/``jax.jit`` without
    converting data to NumPy/Python values or creating dynamic shapes.
    """
    # STUDENT TODO: evaluate the critic and return its scalar training loss.
    data = jax.tree.map(jax.lax.stop_gradient, data)
    predictions = jax.vmap(jax.vmap(
        lambda observation, action: critic_apply(params, observation, action)))(
            data.trajectory.observations, data.trajectory.actions)
    # Mask before squaring so unavailable targets cannot contaminate gradients.
    residuals = jnp.where(data.mask, predictions - data.targets, 0.)
    return 0.5 * masked_mean(jnp.square(residuals), data.mask)


@dataclass(frozen=True)
class ReinforceObjective:
    """Likelihood-ratio policy gradient; includes n-step A2C with critic advantages.

    Default loss is a mean over lanes of SUMS over episode time, with gamma**t
    weights. With complete episodes and no normalization this matches the
    discounted episodic objective, rather than a length-normalized surrogate.
    """
    gamma: float = 0.99
    entropy_coef: float = 0.0
    normalize_advantages: bool = False
    discount_weights: bool = True
    reduction: Literal["episodes", "transitions"] = "episodes"

    def __post_init__(self):
        if not 0 <= self.gamma <= 1 or self.entropy_coef < 0:
            raise ValueError("Invalid discount or entropy coefficient")
        if self.reduction not in ("episodes", "transitions"):
            raise ValueError("reduction must be episodes or transitions")

    def __call__(self, policy: Policy, params: PyTree, data: PreparedBatch, key: Array):
        logp, advantage, entropy, data = _policy_terms(
            policy, params, data, key, self.entropy_coef, self.normalize_advantages)
        loss = reinforce_actor_loss(
            logp,
            advantage,
            entropy,
            data,
            gamma=self.gamma,
            entropy_coef=self.entropy_coef,
            discount_weights=self.discount_weights,
            reduction=self.reduction,
        )
        return loss, {"entropy": masked_mean(entropy, data.mask),
                      "approx_kl": masked_mean(data.trajectory.behavior_log_probs - logp, data.mask),
                      "clip_fraction": jnp.zeros((), logp.dtype)}

    def value_and_grad(self, policy: Policy, params: PyTree,
                       data: PreparedBatch, key: Array):
        """Evaluate the loss and explicitly assemble its actor gradient."""
        logp, advantage, entropy, fixed_data = _policy_terms(
            policy, params, data, key, self.entropy_coef,
            self.normalize_advantages)
        loss = reinforce_actor_loss(
            logp,
            advantage,
            entropy,
            fixed_data,
            gamma=self.gamma,
            entropy_coef=self.entropy_coef,
            discount_weights=self.discount_weights,
            reduction=self.reduction,
        )
        metrics = {
            "entropy": masked_mean(entropy, fixed_data.mask),
            "approx_kl": masked_mean(
                fixed_data.trajectory.behavior_log_probs - logp,
                fixed_data.mask,
            ),
            "clip_fraction": jnp.zeros((), logp.dtype),
        }

        def loss_from_log_probabilities(candidate_logp):
            return reinforce_actor_loss(
                candidate_logp,
                advantage,
                entropy,
                fixed_data,
                gamma=self.gamma,
                entropy_coef=self.entropy_coef,
                discount_weights=self.discount_weights,
                reduction=self.reduction,
            )

        coefficients = jax.grad(loss_from_log_probabilities)(logp)
        gradient = weighted_log_probability_gradient(
            policy,
            params,
            fixed_data.trajectory.observations,
            fixed_data.trajectory.raw_actions,
            coefficients,
        )

        if self.entropy_coef:
            def entropy_loss(candidate_params):
                keys = jax.random.split(
                    key, fixed_data.trajectory.rewards.shape)
                candidate_entropy = jax.vmap(jax.vmap(
                    lambda observation, entropy_key: policy.entropy(
                        candidate_params,
                        observation,
                        entropy_key,
                    )
                ))(fixed_data.trajectory.observations, keys)
                return reinforce_actor_loss(
                    logp,
                    advantage,
                    candidate_entropy,
                    fixed_data,
                    gamma=self.gamma,
                    entropy_coef=self.entropy_coef,
                    discount_weights=self.discount_weights,
                    reduction=self.reduction,
                )

            entropy_gradient = jax.grad(entropy_loss)(params)
            gradient = jax.tree.map(
                lambda score, entropy: score + entropy,
                gradient,
                entropy_gradient,
            )

        return (loss, metrics), gradient


@dataclass(frozen=True)
class PPOObjective:
    """PPO clipped objective. Reuse ONE prepared batch across the PPO epochs.

    This is a swappable loss component, not an entire production PPO trainer:
    minibatching, target-KL early stopping, schedules and value clipping are
    optional caller-level extensions.
    """
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    entropy_coef: float = 0.0
    normalize_advantages: bool = True
    discount_weights: bool = False
    reduction: Literal["episodes", "transitions"] = "transitions"

    def __post_init__(self):
        ReinforceObjective(self.gamma, self.entropy_coef, self.normalize_advantages,
                           self.discount_weights, self.reduction)
        if not 0 < self.clip_epsilon < 1:
            raise ValueError("clip_epsilon must be in (0, 1)")

    def __call__(self, policy: Policy, params: PyTree, data: PreparedBatch, key: Array):
        logp, advantage, entropy, data = _policy_terms(
            policy, params, data, key, self.entropy_coef, self.normalize_advantages)
        log_ratio = logp - data.trajectory.behavior_log_probs
        ratio = jnp.exp(log_ratio)
        clipped = jnp.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        surrogate = jnp.minimum(ratio * advantage, clipped * advantage)
        loss = _reduce(-surrogate - self.entropy_coef * entropy, data,
                       self.gamma, self.discount_weights, self.reduction)
        return loss, {"entropy": masked_mean(entropy, data.mask),
                      "approx_kl": masked_mean((ratio - 1) - log_ratio, data.mask),
                      "clip_fraction": masked_mean(
                          (jnp.abs(ratio - 1) > self.clip_epsilon).astype(logp.dtype), data.mask)}

    def value_and_grad(self, policy: Policy, params: PyTree,
                       data: PreparedBatch, key: Array):
        """Evaluate PPO using ordinary reverse-mode differentiation."""
        return jax.value_and_grad(
            lambda p: self(policy, p, data, key), has_aux=True)(params)

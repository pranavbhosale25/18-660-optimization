import numpy as np
import pytest
import jax
import jax.numpy as jnp

from reinforce_jax import (
    CategoricalPolicy,
    GaussianPolicy,
    NStepEstimator,
    PPOObjective,
    PreparedBatch,
    ReinforceObjective,
    ReturnConfig,
    adam,
    critic_loss,
    make_bootstrap,
    make_learner,
    sgd,
    weighted_log_probability_gradient,
)
from reinforce_jax.objectives import reinforce_actor_loss
from reinforce_jax.optim import apply_updates

from .helpers import assert_trees_equal, batch


def categorical_policy(n_actions=2):
    return CategoricalPolicy(lambda params, observation: params, n_actions)


def linear_value(params, observation, action):
    del action
    return params * observation[0]


def make_test_learner(optimizer=None, critic=False):
    policy = categorical_policy()
    critic_apply = linear_value if critic else None
    return make_learner(
        policy=policy,
        bootstrap=make_bootstrap(
            policy,
            "v" if critic else "zero",
            critic_apply=critic_apply,
        ),
        estimator=NStepEstimator(
            ReturnConfig(gamma=0.5, horizon=1),
            use_baseline=critic,
        ),
        objective=ReinforceObjective(gamma=0.5),
        actor_optimizer=optimizer or adam(0.01),
        critic_apply=critic_apply,
        critic_optimizer=sgd(0.1) if critic else None,
    )


def actor_fixture():
    policy = categorical_policy()
    params = jnp.log(jnp.asarray([0.25, 0.75]))
    trajectory = batch(
        [1.0, 1.0],
        actions=[0, 1],
        terminated=[False, True],
    )
    advantages = jnp.asarray([[2.0], [4.0]])
    prepared = PreparedBatch(
        trajectory,
        targets=advantages,
        advantages=advantages,
        mask=trajectory.valid,
    )
    return policy, params, prepared


def test_weighted_categorical_log_probability_gradient():
    policy = categorical_policy(3)
    params = jnp.log(jnp.asarray([0.2, 0.3, 0.5]))
    observations = jnp.zeros((1, 1, 1))
    raw_actions = jnp.asarray([[1]], jnp.int32)
    coefficients = jnp.asarray([[2.0]])

    gradient = jax.jit(lambda p: weighted_log_probability_gradient(
        policy, p, observations, raw_actions, coefficients))(params)

    np.testing.assert_allclose(
        gradient, [-0.4, 1.4, -1.0], rtol=1e-12, atol=1e-12)


def test_weighted_gaussian_log_probability_gradient_supports_pytrees():
    policy = GaussianPolicy(
        lambda params, observation: jnp.concatenate(
            (params["mean"], params["log_std"])),
        action_dim=1,
    )
    params = {
        "mean": jnp.asarray([0.2]),
        "log_std": jnp.asarray([-0.4]),
    }
    observations = jnp.zeros((1, 1, 1))
    raw_actions = jnp.asarray([[[0.7]]])
    coefficients = jnp.ones((1, 1))

    gradient = jax.jit(lambda p: weighted_log_probability_gradient(
        policy, p, observations, raw_actions, coefficients))(params)

    assert jax.tree.structure(gradient) == jax.tree.structure(params)
    np.testing.assert_allclose(
        gradient["mean"], [1.112770464246234], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        gradient["log_std"], [-0.443614767876883],
        rtol=1e-12, atol=1e-12)


def test_weighted_log_probability_gradient_uses_all_records_and_observations():
    policy = CategoricalPolicy(
        lambda params, observation:
            params["weight"] * observation[0] + params["bias"],
        n_actions=2,
    )
    params = {
        "weight": jnp.zeros(2),
        "bias": jnp.zeros(2),
    }
    observations = jnp.asarray(
        [[[1.0], [2.0]], [[-1.0], [0.5]]])
    raw_actions = jnp.asarray([[0, 1], [1, 0]], jnp.int32)
    coefficients = jnp.asarray([[1.0, 2.0], [-0.5, 0.0]])

    gradient = jax.jit(lambda p, c: weighted_log_probability_gradient(
        policy, p, observations, raw_actions, c))(params, coefficients)
    zero_gradient = jax.jit(lambda p, c: weighted_log_probability_gradient(
        policy, p, observations, raw_actions, c))(
            params, jnp.zeros_like(coefficients))

    np.testing.assert_allclose(gradient["weight"], [-1.75, 1.75])
    np.testing.assert_allclose(gradient["bias"], [-0.25, 0.25])
    np.testing.assert_array_equal(
        zero_gradient["weight"], jnp.zeros_like(params["weight"]))
    np.testing.assert_array_equal(
        zero_gradient["bias"], jnp.zeros_like(params["bias"]))


def test_reinforce_actor_loss_uses_masks_discounts_and_entropy():
    _, _, prepared = actor_fixture()
    log_probabilities = jnp.log(jnp.asarray([[0.25], [0.75]]))
    entropy = jnp.full_like(
        log_probabilities, 0.5623351446188083)

    loss = jax.jit(lambda logp, advantages, ent, data: reinforce_actor_loss(
        logp,
        advantages,
        ent,
        data,
        gamma=0.5,
        entropy_coef=0.2,
        discount_weights=True,
        reduction="episodes",
    ))(log_probabilities, prepared.advantages, entropy, prepared)

    np.testing.assert_allclose(
        loss, 3.1792523237577006, rtol=1e-12, atol=1e-12)

    masked_data = prepared._replace(
        mask=jnp.asarray([[True], [False]]))
    masked_loss = reinforce_actor_loss(
        log_probabilities,
        prepared.advantages,
        entropy,
        masked_data,
        gamma=0.5,
        entropy_coef=0.2,
        discount_weights=True,
        reduction="episodes",
    )
    np.testing.assert_allclose(
        masked_loss, 2.660121693316019, rtol=1e-12, atol=1e-12)


def test_actor_transition_reduction_without_discount_and_all_masked_result():
    _, _, prepared = actor_fixture()
    log_probabilities = jnp.log(jnp.asarray([[0.25], [0.75]]))
    entropy = jnp.zeros_like(log_probabilities)

    transition_loss = reinforce_actor_loss(
        log_probabilities,
        prepared.advantages,
        entropy,
        prepared,
        gamma=0.1,
        entropy_coef=0.0,
        discount_weights=False,
        reduction="transitions",
    )
    all_masked_loss = reinforce_actor_loss(
        log_probabilities,
        prepared.advantages,
        entropy,
        prepared._replace(mask=jnp.zeros_like(prepared.mask)),
        gamma=0.1,
        entropy_coef=0.0,
        discount_weights=False,
        reduction="transitions",
    )

    np.testing.assert_allclose(
        transition_loss, 1.9616585060234524, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(all_masked_loss, 0.0)


def test_actor_loss_uses_supplied_advantages_and_total_lane_denominator():
    trajectory = batch(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        valid=[[True, False, False], [True, False, False]],
    )
    stored_advantages = jnp.full((2, 3), 100.0)
    prepared = PreparedBatch(
        trajectory,
        targets=jnp.zeros((2, 3)),
        advantages=stored_advantages,
        mask=trajectory.valid,
    )
    supplied_advantages = jnp.asarray(
        [[2.0, -7.0, -7.0], [4.0, -7.0, -7.0]])
    log_probabilities = jnp.full((2, 3), jnp.log(0.5))

    loss = reinforce_actor_loss(
        log_probabilities,
        supplied_advantages,
        jnp.zeros((2, 3)),
        prepared,
        gamma=0.5,
        entropy_coef=0.0,
        discount_weights=False,
        reduction="episodes",
    )

    # Only lane zero contributes, but episode reduction divides by all B=3
    # lanes. The supplied advantages, not prepared.advantages, are used.
    np.testing.assert_allclose(loss, 2.0 * jnp.log(2.0))


def test_actor_loss_ignores_nonfinite_values_in_masked_records():
    _, _, prepared = actor_fixture()
    masked_data = prepared._replace(mask=jnp.asarray([[True], [False]]))
    log_probabilities = jnp.asarray([[jnp.log(0.25)], [jnp.nan]])
    advantages = jnp.asarray([[2.0], [jnp.nan]])
    entropy = jnp.asarray([[0.5], [jnp.nan]])

    loss = reinforce_actor_loss(
        log_probabilities,
        advantages,
        entropy,
        masked_data,
        gamma=0.5,
        entropy_coef=0.2,
        discount_weights=True,
        reduction="episodes",
    )

    np.testing.assert_allclose(loss, -2.0 * jnp.log(0.25) - 0.1)


def test_explicit_actor_gradient_matches_loss_autodiff_with_entropy():
    policy, params, prepared = actor_fixture()
    objective = ReinforceObjective(gamma=0.5, entropy_coef=0.2)
    key = jax.random.key(7)

    (explicit_loss, explicit_metrics), explicit_gradient = (
        objective.value_and_grad(policy, params, prepared, key))
    (automatic_loss, automatic_metrics), automatic_gradient = jax.value_and_grad(
        lambda p: objective(policy, p, prepared, key),
        has_aux=True,
    )(params)

    np.testing.assert_allclose(explicit_loss, automatic_loss, atol=1e-12)
    np.testing.assert_allclose(
        explicit_metrics["entropy"], 0.5623351446188083,
        rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        explicit_metrics["entropy"], automatic_metrics["entropy"], atol=1e-12)
    np.testing.assert_allclose(explicit_gradient, automatic_gradient, atol=1e-12)


def test_critic_loss_masks_entries_and_treats_targets_as_fixed_data():
    trajectory = batch(
        [0.0, 0.0, 0.0],
        observations=[1.0, 2.0, 3.0],
        valid=[True, False, True],
    )
    targets = jnp.asarray([[1.0], [100.0], [7.0]])
    prepared = PreparedBatch(
        trajectory,
        targets=targets,
        advantages=jnp.zeros_like(targets),
        mask=trajectory.valid,
    )

    value = jax.jit(lambda p: critic_loss(
        linear_value, p, prepared))(jnp.asarray(2.0))
    parameter_gradient = jax.grad(lambda p: critic_loss(
        linear_value, p, prepared))(jnp.asarray(2.0))
    target_gradient = jax.grad(lambda candidate_targets: critic_loss(
        linear_value,
        jnp.asarray(2.0),
        prepared._replace(targets=candidate_targets),
    ))(targets)

    np.testing.assert_allclose(value, 0.5)
    np.testing.assert_allclose(parameter_gradient, -1.0)
    np.testing.assert_array_equal(target_gradient, jnp.zeros_like(targets))


def test_action_value_critic_supports_vector_actions_pytrees_and_empty_masks():
    def action_value(params, observation, action):
        return (params["observation"] * observation[0]
                + jnp.dot(params["action"], action))

    trajectory = batch(
        [0.0, 0.0],
        observations=[1.0, 2.0],
        actions=[[[1.0, 2.0]], [[3.0, 4.0]]],
    )
    targets = jnp.asarray([[4.0], [10.0]])
    prepared = PreparedBatch(
        trajectory,
        targets=targets,
        advantages=jnp.zeros_like(targets),
        mask=trajectory.valid,
    )
    params = {
        "observation": jnp.asarray(1.0),
        "action": jnp.ones(2),
    }

    value, gradient = jax.value_and_grad(
        lambda p: critic_loss(action_value, p, prepared))(params)
    empty_data = prepared._replace(mask=jnp.zeros_like(prepared.mask))
    empty_value, empty_gradient = jax.value_and_grad(
        lambda p: critic_loss(action_value, p, empty_data))(params)

    np.testing.assert_allclose(value, 0.25)
    np.testing.assert_allclose(gradient["observation"], -1.0)
    np.testing.assert_allclose(gradient["action"], [-1.5, -2.0])
    np.testing.assert_allclose(empty_value, 0.0)
    np.testing.assert_array_equal(
        empty_gradient["observation"], jnp.asarray(0.0))
    np.testing.assert_array_equal(
        empty_gradient["action"], jnp.zeros_like(params["action"]))


def test_critic_uses_environment_actions_instead_of_raw_actions():
    def action_value(params, observation, action):
        del observation
        return params * action

    trajectory = batch(
        [0.0, 0.0],
        actions=[2.0, 4.0],
        raw_actions=[20.0, 40.0],
    )
    targets = jnp.asarray([[2.0], [4.0]])
    prepared = PreparedBatch(
        trajectory,
        targets=targets,
        advantages=jnp.zeros_like(targets),
        mask=trajectory.valid,
    )

    loss = critic_loss(action_value, jnp.asarray(1.0), prepared)

    np.testing.assert_allclose(loss, 0.0)


def test_one_actor_critic_update_exercises_student_components():
    policy = categorical_policy()
    learner = make_learner(
        policy=policy,
        bootstrap=make_bootstrap(policy, "v", critic_apply=linear_value),
        estimator=NStepEstimator(
            ReturnConfig(gamma=1.0, horizon=1),
            use_baseline=True,
        ),
        objective=ReinforceObjective(gamma=1.0),
        actor_optimizer=sgd(0.1),
        critic_apply=linear_value,
        critic_optimizer=sgd(0.1),
    )
    trajectory = batch(
        [[1.0, 3.0]],
        observations=[[1.0, 1.0]],
        next_observations=[[2.0, 2.0]],
        actions=[[0, 1]],
        terminated=[[True, True]],
    )
    key = jax.random.key(0)
    state = learner.init(jnp.zeros(2), jnp.asarray(0.0))

    prepared = learner.prepare(
        state.actor_params, state.critic_params, trajectory, key)
    new_state, metrics = learner.update(state, prepared, key)

    np.testing.assert_allclose(prepared.targets, [[1.0, 3.0]])
    np.testing.assert_allclose(prepared.advantages, [[1.0, 3.0]])
    np.testing.assert_allclose(new_state.actor_params, [-0.05, 0.05])
    np.testing.assert_allclose(new_state.critic_params, 0.2)
    np.testing.assert_allclose(metrics["actor_loss"], np.log(4.0))
    np.testing.assert_allclose(metrics["critic_loss"], 2.5)
    assert int(new_state.updates) == 1


def test_all_dropped_batch_preserves_parameters_and_optimizer_state():
    learner = make_test_learner(critic=True)
    trajectory = batch([2.0], terminated=[True])
    state = learner.init(jnp.zeros(2), jnp.asarray(1.0))
    key = jax.random.key(0)
    prepared = learner.prepare(
        state.actor_params, state.critic_params, trajectory, key)
    state, _ = learner.update(state, prepared, key)
    dropped = prepared._replace(mask=jnp.zeros_like(prepared.mask))

    new_state, metrics = learner.update(state, dropped, key)

    assert_trees_equal(state, new_state)
    assert int(metrics["valid_count"]) == 0


def test_supplied_ppo_objective_remains_usable():
    policy = categorical_policy()
    params = jnp.asarray([0.1, -0.1])
    trajectory = batch(
        [[1.0, 2.0]], terminated=[[True, True]], actions=[[0, 1]])
    old_log_probabilities = jax.nn.log_softmax(params)[
        jnp.asarray([[0, 1]])]
    trajectory = trajectory._replace(
        behavior_log_probs=old_log_probabilities)
    prepared = PreparedBatch(
        trajectory, trajectory.rewards, trajectory.rewards, trajectory.valid)
    objective = PPOObjective(normalize_advantages=False)

    loss, metrics = objective(
        policy, params, prepared, jax.random.key(0))

    np.testing.assert_allclose(loss, -1.5)
    np.testing.assert_allclose(metrics["approx_kl"], 0.0)
    np.testing.assert_allclose(metrics["clip_fraction"], 0.0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_supplied_adam_optimizer_preserves_dtype(dtype):
    params = {"w": jnp.asarray([1.0, -1.0], dtype)}
    gradients = {"w": jnp.asarray([2.0, -3.0], dtype)}
    optimizer = adam(0.1)

    updates, state = jax.jit(optimizer.update)(
        gradients, optimizer.init(params), params)
    new_params = apply_updates(params, updates)

    np.testing.assert_allclose(new_params["w"], [0.9, -0.9], atol=1e-6)
    assert new_params["w"].dtype == dtype
    assert int(state.count) == 1

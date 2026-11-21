"""Array-only PyTrees and public functional interfaces.

Trajectories are time-major: [time, batch, ...]. Observations are encoded
floating-point feature vectors. Discrete actions are zero-based int32 indices.
The host adapter alone converts these indices to Gymnasium's space.start.
"""
from typing import Any, Callable, NamedTuple, Protocol
import jax

Array = jax.Array
PyTree = Any
ModelApply = Callable[[PyTree, Array], Array]
CriticApply = Callable[[PyTree, Array, Array], Array]
Bootstrap = Callable[[PyTree, PyTree, Array, Array], Array]


class ActionSample(NamedTuple):
    action: Array
    raw_action: Array  # Pre-tanh sample for a bounded Gaussian; index for categorical.
    log_prob: Array


class Trajectory(NamedTuple):
    observations: Array
    actions: Array
    raw_actions: Array
    rewards: Array
    next_observations: Array  # Actual post-step observation, NEVER the reset observation.
    terminated: Array
    truncated: Array
    valid: Array
    episode_steps: Array
    behavior_log_probs: Array


class ReturnTargets(NamedTuple):
    returns: Array
    mask: Array
    steps: Array


class Estimates(NamedTuple):
    targets: Array
    advantages: Array
    mask: Array


class PreparedBatch(NamedTuple):
    trajectory: Trajectory
    targets: Array
    advantages: Array
    mask: Array


class TrainState(NamedTuple):
    actor_params: PyTree
    critic_params: PyTree
    actor_opt_state: PyTree
    critic_opt_state: PyTree
    updates: Array


class DirectState(NamedTuple):
    params: PyTree
    opt_state: PyTree
    updates: Array


class ObjectiveResult(NamedTuple):
    value: Array  # Return to MAXIMIZE.
    valid_count: Array  # Zero skips the entire optimizer update, including momentum.


class Policy(Protocol):
    is_discrete: bool
    reparameterizable: bool

    def sample(self, params: PyTree, obs: Array, key: Array) -> ActionSample: ...
    def log_prob(self, params: PyTree, obs: Array, raw_action: Array) -> Array: ...
    def entropy(self, params: PyTree, obs: Array, key: Array) -> Array: ...
    def mode(self, params: PyTree, obs: Array) -> Array: ...


class Optimizer(Protocol):
    """Compatible with ordinary Optax GradientTransformation objects."""
    def init(self, params: PyTree) -> PyTree: ...
    def update(self, grads: PyTree, state: PyTree,
               params: PyTree | None = None) -> tuple[PyTree, PyTree]: ...


class FunctionalEnv(Protocol):
    """Single-environment JAX API. State may be an arbitrary array PyTree."""
    def reset(self, key: Array) -> PyTree: ...
    def observe(self, state: PyTree) -> Array: ...
    def step(self, state: PyTree, action: Array, key: Array
             ) -> tuple[PyTree, Array, Array, Array]:
        """Return next_state, scalar reward, terminated, truncated."""
        ...

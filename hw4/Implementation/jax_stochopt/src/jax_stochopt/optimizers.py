"""Functional update rules for native JAX stochastic optimization.

An update is a *direction*: the solver applies ``params - learning_rate *
direction`` and then its projection. Rules neither draw random numbers nor
choose a learning rate. All state is a JAX PyTree; configuration is static.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Callable, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from .types import PyTree, ScheduleContext


ValueFunction = Callable[[PyTree], jax.Array]


@runtime_checkable
class Optimizer(Protocol):
    """Extension protocol for first- or second-order update rules.

    ``value_fn`` evaluates the same frozen minibatch that produced ``grads``.
    A rule may differentiate it again to obtain Hessian-vector products.
    ``context`` exposes the step, parameters, previous optimizer state,
    current batch statistics, and user arguments. Rules must return a state
    with the same PyTree structure, array shapes, and dtypes as ``init``.
    """

    def init(self, params: PyTree) -> PyTree: ...

    def update(
        self,
        params: PyTree,
        grads: PyTree,
        state: PyTree,
        *,
        value_fn: ValueFunction,
        context: Any,
    ) -> tuple[PyTree, PyTree]: ...


def _zeros(tree: PyTree) -> PyTree:
    return jax.tree.map(jnp.zeros_like, tree)


def _dot(left: PyTree, right: PyTree) -> jax.Array:
    products = jax.tree.leaves(jax.tree.map(lambda x, y: jnp.vdot(x, y).real, left, right))
    return sum(products)


def _scale(tree: PyTree, scale: jax.Array) -> PyTree:
    return jax.tree.map(lambda x: jnp.asarray(scale, dtype=x.dtype) * x, tree)


def _add(left: PyTree, right: PyTree) -> PyTree:
    return jax.tree.map(jnp.add, left, right)


def _sub(left: PyTree, right: PyTree) -> PyTree:
    return jax.tree.map(jnp.subtract, left, right)


def _select(condition: jax.Array, yes: PyTree, no: PyTree) -> PyTree:
    return jax.tree.map(lambda x, y: jnp.where(condition, x, y), yes, no)


def _all_finite(tree: PyTree) -> jax.Array:
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)]))


def _safe_sqrt(value: jax.Array) -> jax.Array:
    """Exact nonnegative square root with finite zero subgradient."""
    positive = value > 0
    root = jnp.sqrt(jnp.where(positive, value, jnp.ones_like(value)))
    return jnp.where(positive, root, jnp.zeros_like(value))


def _check_decay(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError(f"{name} must be finite and in [0, 1).")


def _check_positive(name: str, value: float, *, allow_zero: bool = False) -> None:
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        relation = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}.")


def _check_iterations(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


class SGDState(NamedTuple):
    """Unnormalized momentum accumulator, initially zero."""

    velocity: PyTree


@dataclass(frozen=True)
class SGD:
    """Minibatch gradient descent with optional classical momentum.

    The solver applies the learning rate and subtracts the direction returned
    by :meth:`update`. ``momentum=0`` must reduce to ordinary minibatch
    gradient descent. There is no implicit weight decay.
    """

    momentum: float = 0.0

    def __post_init__(self) -> None:
        _check_decay("momentum", self.momentum)

    def init(self, params: PyTree) -> SGDState:
        return SGDState(_zeros(params))

    def update(
        self,
        params: PyTree,
        grads: PyTree,
        state: SGDState,
        *,
        value_fn: ValueFunction,
        context: ScheduleContext,
    ) -> tuple[PyTree, SGDState]:
        """Produce the optimizer direction and state for one minibatch.

        The surrounding solver has already sampled and evaluated the current
        minibatch before calling this method. This method updates the SGD
        optimizer state; it does not update the parameters itself.

        Parameters
        ----------
        params : PyTree[jax.Array]
            Current parameters before this solver update. Every leaf is a
            float32 or float64 array. Scalar leaves and leaves of any rank are
            allowed.
        grads : PyTree[jax.Array]
            Gradient estimate from the current minibatch at ``params``. Its
            tree structure, leaf shapes, and leaf dtypes exactly match
            ``params``.
        state : SGDState
            State carried from the preceding update. ``state.velocity`` is a
            parameter-like PyTree containing the momentum information before
            the current gradient is processed. ``SGD.init`` creates an
            all-zero velocity for the first update.
        value_fn : Callable[[PyTree], jax.Array]
            Evaluates a rank-zero objective on the same frozen minibatch for
            another parameter PyTree. It is part of the common optimizer
            interface; classical SGD requires no extra objective evaluation.
        context : ScheduleContext
            Read-only data for this update. ``context.step`` is the zero-based
            update number; ``context.params`` is the same pre-update parameter
            tree; ``context.optimizer_state`` is the incoming ``SGDState``;
            ``context.stats`` describes the freshly evaluated current
            minibatch; and ``context.args`` is the run-time problem data. The
            learning rate has not yet been selected.

        Returns
        -------
        direction : PyTree[jax.Array]
            Direction consumed by the outer solver. It must exactly match the
            structure, leaf shapes, and leaf dtypes of ``params``. It does not
            include the learning rate and is not a new parameter value; the
            outer solver applies the learning rate and parameter change.
        new_state : SGDState
            State carried into the next step or a resumed run.
            ``new_state.velocity`` must have exactly the same PyTree
            structure, leaf shapes, and leaf dtypes as ``params``.

        Notes
        -----
        The returned state is visible to the learning-rate schedule for this
        same step. This method does not draw randomness, apply constraints,
        choose a learning rate, or mutate its inputs. It must remain pure and
        compatible with JAX JIT and automatic differentiation.
        """
        # STUDENT TODO (Problem 1 - SGD): implement one classical-momentum
        # update. Return both the direction consumed by the outer solver and
        # the next SGDState. Preserve the structure, shapes, and dtypes of
        # every PyTree leaf. See the assignment PDF for the required recurrence.
        del params, value_fn, context
        velocity = jax.tree.map(
            lambda old, grad: jnp.asarray(self.momentum, dtype=grad.dtype) * old + grad,
            state.velocity, grads,
        )
        return velocity, SGDState(velocity)


class AdamState(NamedTuple):
    """Step count, first moment ``m``, and second moment ``v``."""

    count: jax.Array
    m: PyTree
    v: PyTree


def _adam_direction(
    m: jax.Array,
    v: jax.Array,
    count: jax.Array,
    beta1: float,
    beta2: float,
    eps: float,
    eps_root: float,
) -> jax.Array:
    # -expm1(t*log(beta)) avoids subtracting nearly equal float32 numbers
    # when beta is close to one (and avoids eager/JIT power discrepancies).
    correction1 = 1.0 if beta1 == 0 else -jnp.expm1(count.astype(m.dtype) * math.log(beta1))
    correction2 = 1.0 if beta2 == 0 else -jnp.expm1(count.astype(v.dtype) * math.log(beta2))
    m_hat = m / correction1
    v_hat = v / correction2
    return m_hat / (jnp.sqrt(v_hat + eps_root) + eps)


@dataclass(frozen=True)
class Adam:
    """Bias-corrected Adam with a smooth, zero-safe denominator.

    The direction is ``m_hat / (sqrt(v_hat + eps_root) + eps)``. The small
    strictly positive ``eps_root`` is intentional: ordinary ``sqrt(v_hat)``
    can produce NaN hypergradients at a zero gradient, even with an epsilon
    outside the square root. This slightly smooths conventional Adam near
    zero. This is Adam, without decoupled weight decay or AMSGrad.
    """

    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    eps_root: float = 1e-16

    def __post_init__(self) -> None:
        _check_decay("beta1", self.beta1)
        _check_decay("beta2", self.beta2)
        _check_positive("eps", self.eps, allow_zero=True)
        _check_positive("eps_root", self.eps_root)

    def init(self, params: PyTree) -> AdamState:
        return AdamState(jnp.asarray(0, dtype=jnp.int32), _zeros(params), _zeros(params))

    def update(
        self,
        params: PyTree,
        grads: PyTree,
        state: AdamState,
        *,
        value_fn: ValueFunction,
        context: Any,
    ) -> tuple[PyTree, AdamState]:
        del params, value_fn, context
        count = state.count + 1
        m = jax.tree.map(lambda old, g: self.beta1 * old + (1 - self.beta1) * g, state.m, grads)
        v = jax.tree.map(
            lambda old, g: self.beta2 * old + (1 - self.beta2) * jnp.square(g), state.v, grads
        )
        direction = jax.tree.map(
            lambda first, second: _adam_direction(
                first, second, count, self.beta1, self.beta2, self.eps, self.eps_root
            ),
            m,
            v,
        )
        return direction, AdamState(count, m, v)


class NewtonCGState(NamedTuple):
    """Diagnostics from the latest damped Newton direction.

    ``residual_norm`` is ``||(H + damping I) direction - gradient||`` for
    the returned direction. ``negative_curvature`` also flags nonfinite or
    insufficient curvature rejected by the configured threshold.
    """

    count: jax.Array
    cg_iterations: jax.Array
    residual_norm: jax.Array
    negative_curvature: jax.Array


class _CGState(NamedTuple):
    solution: PyTree
    residual: PyTree
    search: PyTree
    residual_squared: jax.Array
    active: jax.Array
    iterations: jax.Array
    rejected_curvature: jax.Array


@dataclass(frozen=True)
class NewtonCG:
    """Stochastic damped Newton using only minibatch Hessian-vector products.

    Approximately solves ``(H + damping I) direction = gradient`` using
    at most ``maxiter`` conjugate-gradient iterations. The Hessian is of
    ``value_fn`` on the *same* frozen minibatch used for the gradient.

    The damped Hessian must be symmetric positive definite for the Newton
    solve and its implicit derivative to be valid. Positive damping alone
    does not guarantee this for an arbitrary nonconvex objective. Encountered
    nonpositive/nonfinite curvature or a non-descent result triggers a pure
    gradient fallback; this is a safeguard, not a nonconvex Newton guarantee.

    ``jax.lax.custom_linear_solve`` provides implicit differentiation of
    the linear system, including at zero gradient. The primal and derivative
    solves are approximate to the configured iteration/tolerance limits;
    gradients are not the derivative of a finite, truncated CG program.
    Inspect ``residual_norm`` and increase ``maxiter`` or damping if needed.
    No dense Hessian is constructed. Learning rate selection is external;
    this rule does not perform a line search or trust-region optimization.
    """

    damping: float = 1e-3
    maxiter: int = 10
    tol: float = 1e-6
    curvature_tol: float = 0.0

    def __post_init__(self) -> None:
        _check_positive("damping", self.damping, allow_zero=True)
        _check_iterations("maxiter", self.maxiter)
        _check_positive("tol", self.tol, allow_zero=True)
        _check_positive("curvature_tol", self.curvature_tol, allow_zero=True)

    def init(self, params: PyTree) -> NewtonCGState:
        scalar_dtype = jnp.result_type(*[x.dtype for x in jax.tree.leaves(params)])
        return NewtonCGState(
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=scalar_dtype),
            jnp.asarray(False),
        )

    def _solve(self, matvec: Callable[[PyTree], PyTree], rhs: PyTree):
        """Fixed-shape, bounded CG; guards also protect vmapped zero solves."""
        initial_squared = _dot(rhs, rhs)
        threshold_squared = (self.tol * self.tol) * initial_squared
        carry = _CGState(
            _zeros(rhs), rhs, rhs, initial_squared,
            initial_squared > threshold_squared,
            jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False),
        )

        def iteration(_, current):
            def active_step(current):
                image = matvec(current.search)
                curvature = _dot(current.search, image)
                admissible = (
                    jnp.isfinite(curvature)
                    & (curvature > self.curvature_tol * _dot(current.search, current.search))
                )
                # Safe denominators matter because vmap may turn cond into
                # select, evaluating both branches even for inactive solves.
                alpha = current.residual_squared / jnp.where(admissible, curvature, 1)
                solution = _add(current.solution, _scale(current.search, alpha))
                residual = _sub(current.residual, _scale(image, alpha))
                residual_squared = _dot(residual, residual)
                safe_previous = jnp.where(current.residual_squared > 0, current.residual_squared, 1)
                beta = residual_squared / safe_previous
                search = _add(residual, _scale(current.search, beta))
                finite = _all_finite(solution) & jnp.isfinite(residual_squared)
                accepted = admissible & finite
                return _CGState(
                    _select(accepted, solution, current.solution),
                    _select(accepted, residual, current.residual),
                    _select(accepted, search, current.search),
                    jnp.where(accepted, residual_squared, current.residual_squared),
                    accepted & (residual_squared > threshold_squared),
                    current.iterations + accepted.astype(jnp.int32),
                    current.rejected_curvature | ~accepted,
                )

            return jax.lax.cond(current.active, active_step, lambda x: x, current)

        final = jax.lax.fori_loop(0, self.maxiter, iteration, carry)
        return final.solution, (final.iterations, final.rejected_curvature)

    def update(
        self,
        params: PyTree,
        grads: PyTree,
        state: NewtonCGState,
        *,
        value_fn: ValueFunction,
        context: Any,
    ) -> tuple[PyTree, NewtonCGState]:
        del context
        grad_fn = jax.grad(value_fn)

        def matvec(vector):
            hvp = jax.jvp(grad_fn, (params,), (vector,))[1]
            return _add(hvp, _scale(vector, self.damping))

        candidate, (iterations, rejected) = jax.lax.custom_linear_solve(
            matvec, grads, solve=self._solve, symmetric=True, has_aux=True
        )
        stationary = _dot(grads, grads) == 0
        usable = (~rejected) & _all_finite(candidate) & ((_dot(grads, candidate) > 0) | stationary)
        # Accept the linear-solve branch at g=0: its implicit derivative is
        # A^{-1}, unlike differentiating an early-exit zero CG program.
        direction = _select(usable, candidate, grads)
        residual = _sub(matvec(direction), grads)
        return direction, NewtonCGState(
            state.count + 1, iterations, _safe_sqrt(_dot(residual, residual)), rejected
        )


class MuonState(NamedTuple):
    """Step, first moments, and scalar/vector Adam second moments.

    ``m`` contains Muon EMA momentum for matrix leaves and Adam first moments
    for scalar/vector leaves. ``v`` has ``None`` at matrix leaves, so matrix
    parameters do not allocate unused Adam second-moment buffers.
    """

    count: jax.Array
    m: PyTree
    v: PyTree


@dataclass(frozen=True)
class Muon:
    """Matrix Muon with Adam on scalar/vector leaves, entirely in JAX.

    For a matrix, ``m_new = momentum*m + (1-momentum)*g`` and the Nesterov
    input is ``(1-momentum)*g + momentum*m_new``. Newton-Schulz uses the
    published quintic coefficients ``(3.4445, -4.7750, 2.0315)`` and a fixed
    iteration count. It is an approximate spectral transformation, not an
    exact polar factor or a constraint projection.

    Arrays with more than two dimensions are explicitly interpreted as
    matrices of shape ``(shape[0], prod(shape[1:]))`` and reshaped back.
    Scalar/vector leaves use the same smoothed Adam as :class:`Adam`.
    Optional shape scaling multiplies the matrix result by
    ``sqrt(max(1, rows/cols))``. A single external learning rate applies to
    both parameter groups, so tuning may differ from published ML settings.

    Float32/float64 are preserved; there is no bfloat16 cast. Normalization
    is ``G/sqrt(sum(G**2) + eps**2)`` so zero updates have finite derivatives.
    Matrix multiplication requests highest available arithmetic precision.
    Reference: https://github.com/KellerJordan/Muon/blob/master/muon.py .
    """

    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    eps: float = 1e-7
    scale_by_shape: bool = True
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    adam_eps_root: float = 1e-16

    def __post_init__(self) -> None:
        _check_decay("momentum", self.momentum)
        _check_iterations("ns_steps", self.ns_steps)
        _check_positive("eps", self.eps)
        _check_decay("beta1", self.beta1)
        _check_decay("beta2", self.beta2)
        _check_positive("adam_eps", self.adam_eps, allow_zero=True)
        _check_positive("adam_eps_root", self.adam_eps_root)

    def init(self, params: PyTree) -> MuonState:
        return MuonState(
            jnp.asarray(0, dtype=jnp.int32),
            _zeros(params),
            jax.tree.map(lambda x: jnp.zeros_like(x) if x.ndim < 2 else None, params),
        )

    def _orthogonalize(self, update: jax.Array) -> jax.Array:
        shape = update.shape
        matrix = update.reshape(shape[0], -1)
        rows, cols = matrix.shape
        transposed = rows > cols
        x = matrix.T if transposed else matrix
        x = x / jnp.sqrt(jnp.sum(jnp.square(x)) + self.eps * self.eps)

        def mm(left, right):
            return jnp.matmul(left, right, precision=jax.lax.Precision.HIGHEST)

        def iteration(_, x):
            gram = mm(x, x.T)
            polynomial = -4.7750 * gram + 2.0315 * mm(gram, gram)
            return 3.4445 * x + mm(polynomial, x)

        x = jax.lax.fori_loop(0, self.ns_steps, iteration, x)
        x = x.T if transposed else x
        if self.scale_by_shape:
            x = x * math.sqrt(max(1.0, rows / cols))
        return x.reshape(shape)

    def update(
        self,
        params: PyTree,
        grads: PyTree,
        state: MuonState,
        *,
        value_fn: ValueFunction,
        context: Any,
    ) -> tuple[PyTree, MuonState]:
        del params, value_fn, context
        leaves, structure = jax.tree.flatten(grads)
        first_moments = structure.flatten_up_to(state.m)
        second_moments = structure.flatten_up_to(state.v)
        directions, new_first, new_second = [], [], []
        count = state.count + 1
        for g, old_m, old_v in zip(leaves, first_moments, second_moments):
            if g.ndim >= 2:
                m = self.momentum * old_m + (1 - self.momentum) * g
                update = (1 - self.momentum) * g + self.momentum * m if self.nesterov else m
                direction = self._orthogonalize(update)
                v = None
            else:
                m = self.beta1 * old_m + (1 - self.beta1) * g
                v = self.beta2 * old_v + (1 - self.beta2) * jnp.square(g)
                direction = _adam_direction(
                    m, v, count, self.beta1, self.beta2, self.adam_eps, self.adam_eps_root
                )
            directions.append(direction)
            new_first.append(m)
            new_second.append(v)
        return structure.unflatten(directions), MuonState(
            count, structure.unflatten(new_first), structure.unflatten(new_second)
        )

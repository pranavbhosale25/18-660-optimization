"""Fixed-budget contact simulation using one velocity QP per contact event.

World normals point INWARD: normals @ x >= offsets. A positive optional
radius is geometric clearance only; default radius=0 gives a point mass.
"""
from dataclasses import dataclass
from typing import NamedTuple
import jax
import jax.numpy as jnp
from qp import QP, Solver


class World(NamedTuple):
    normals: jax.Array   # (n, 3), unit inward normals
    offsets: jax.Array   # (n,), geometric surface offsets, before radius erosion
    friction: jax.Array  # (n,), coefficient mu >= 0
    restitution: jax.Array  # (n,), normal velocity coefficient in [0, 1]


class Ball(NamedTuple):
    position: jax.Array
    velocity: jax.Array
    radius: jax.Array  # optional geometric clearance only; zero for a point
    mass: jax.Array


@dataclass(frozen=True)
class PhysicsConfig:
    contact_tolerance: float = 1e-7
    velocity_tolerance: float = 2e-6
    complementarity_tolerance: float = 2e-6
    bounce_threshold: float = 0.2
    max_events: int = 24
    energy_tolerance: float = 2e-5

    def __post_init__(self):
        if self.max_events < 1:
            raise ValueError("Iteration limits must be positive")
        if min(self.contact_tolerance, self.velocity_tolerance,
               self.complementarity_tolerance, self.energy_tolerance) <= 0:
            raise ValueError("Tolerances must be positive")
        if self.bounce_threshold < 0:
            raise ValueError("bounce_threshold must be nonnegative")


class ContactResult(NamedTuple):
    u: jax.Array                  # outgoing point velocity (3,)
    normal_impulse: jax.Array      # (n,), recovered from normal constraint duals
    tangent_impulse: jax.Array     # (2*n,), recovered from epigraph duals
    budget: jax.Array              # (n,), fixed BEFORE optimization
    target: jax.Array              # (n,), normal restitution targets
    slip_bound: jax.Array          # (n,), epigraph variables, not impulses
    normal_residual: jax.Array
    friction_residual: jax.Array
    budget_violation: jax.Array
    momentum_residual: jax.Array
    qp_residual: jax.Array
    energy_change: jax.Array
    qp_solves: jax.Array
    qp_iterations: jax.Array
    qp_status: jax.Array
    converged: jax.Array


class StepResult(NamedTuple):
    ball: Ball
    remaining_time: jax.Array
    events: jax.Array
    success: jax.Array
    min_gap: jax.Array
    contact_residual: jax.Array
    max_energy_increase: jax.Array
    qp_solves: jax.Array
    qp_iterations: jax.Array
    qp_status: jax.Array          # 1 if every QP attempted in this step succeeded


def gaps(world: World, position, radius):
    return world.normals @ position - world.offsets - radius


def tangent_frames(normals):
    """Deterministic orthonormal frame; L1 friction depends on this choice."""
    ref = jax.nn.one_hot(jnp.argmin(jnp.abs(normals), axis=1), 3,
                         dtype=normals.dtype)
    t1 = jnp.cross(normals, ref)
    t1 = t1 / jnp.linalg.norm(t1, axis=1, keepdims=True)
    t2 = jnp.cross(normals, t1)
    return jnp.stack((t1, t2), axis=1)


def mass_diagonal(ball):
    """Translational inertia only; no angular state or torque."""
    return jnp.full(3, ball.mass)


def contact_jacobians(world, ball):
    """u=v in R^3; N v is normal speed, T v is tangential slip.

    The optional clearance radius changes only geometry, not inertia.
    """
    N = world.normals
    T = tangent_frames(world.normals).reshape(-1, 3)
    return N, T


def contact_parameters(world, ball, active, config=PhysicsConfig()):
    """Provided isolated-wall predictor; no optimization is used here.

    beta=mu*m*(1+e_eff)*max(-incoming,0) is a friction IMPULSE budget.
    It is not a bound on the normal impulse. At corners it is only a model
    approximation to an actual-load Coulomb limit.
    """
    N, T = contact_jacobians(world, ball)
    incoming = N @ ball.velocity
    approach = jnp.maximum(-incoming, 0.)
    e = jnp.where(incoming < -config.bounce_threshold, world.restitution, 0.)
    target = jnp.where(active, e*approach, 0.)
    budget = jnp.where(active, world.friction*ball.mass*(approach+target), 0.)
    return N, T, target, budget


def contact_qp(velocity, mass, N, T, target, budget, active) -> QP:
    """Return the contact QP for the supplied data.

    ``velocity`` has shape ``(3,)`` and ``mass`` is scalar. For ``n`` plane
    slots, ``N`` has shape ``(n, 3)``, ``T`` has shape ``(2*n, 3)``, and
    ``target``, ``budget``, and ``active`` each have shape ``(n,)``.

    The supplied ``resolve_contact`` routine expects a fixed return layout:
    the primal vector is ``[u(3), s(n)]``. Constraint rows must follow the
    four constraint families from the assignment in this order: normal
    (``n`` rows), positive-tangent (``2*n``), negative-tangent (``2*n``),
    and slip-bound (``n``). Keep all ``n`` contact slots, including inactive
    ones.

    """
    # STUDENT TODO: Implement contact_qp.
    n = N.shape[0]
    # Promote integer velocities too: a fractional mass must not be truncated
    # when assembling the Hessian, and constraint bounds need floating point.
    dtype = jnp.result_type(velocity, mass, N, T, target, budget, 1.0)
    velocity = jnp.asarray(velocity, dtype=dtype)
    # One epigraph variable per plane: s[i] >= both absolute tangent speeds.
    # Expanding m/2 * ||u-v||^2 gives Q=diag(m I_3, 0), c=[-m v, beta];
    # the omitted constant m/2 * ||v||^2 does not affect the minimizer.
    Q = jnp.zeros((3 + n, 3 + n), dtype=dtype)
    Q = Q.at[:3, :3].set(mass * jnp.eye(3, dtype=dtype))
    c = jnp.concatenate((-mass * velocity, jnp.where(active, budget, 0.)))
    identity = jnp.eye(n, dtype=dtype)
    repeated_identity = jnp.repeat(identity, 2, axis=0)

    # Preserve this row order: resolve_contact recovers impulses from its duals.
    normal_rows = jnp.concatenate((N, jnp.zeros((n, n), dtype=dtype)), axis=1)
    positive_tangent = jnp.concatenate((T, -repeated_identity), axis=1)
    negative_tangent = jnp.concatenate((-T, -repeated_identity), axis=1)
    slip_rows = jnp.concatenate((jnp.zeros((n, 3), dtype=dtype), identity), axis=1)
    A = jnp.concatenate((normal_rows, positive_tangent, negative_tangent, slip_rows))

    # Active planes enforce N u >= target and +/- T u <= repeated(s).
    # Inactive normal/tangent rows are unbounded, so they exert no impulse.
    tangent_active = jnp.repeat(active, 2)
    lower = jnp.concatenate((jnp.where(active, target, -jnp.inf),
                             jnp.full((4 * n,), -jnp.inf, dtype=dtype),
                             jnp.zeros(n, dtype=dtype)))
    upper = jnp.concatenate((jnp.full((n,), jnp.inf, dtype=dtype),
                             jnp.where(tangent_active, 0., jnp.inf),
                             jnp.where(tangent_active, 0., jnp.inf),
                             jnp.where(active, jnp.inf, 0.)))
    # Active s >= 0; inactive s = 0 keeps unused slots well-defined.
    return QP(Q, c, A, lower, upper)


def contact_residuals(N, T, velocity, target, normal, tangent, budget, active):
    """Normal complementarity and fixed-budget maximum dissipation.

    min_{||tau||_1 <= beta} tau.slip = -beta*||slip||_infinity.
    This certificate uses beta, NEVER the solved normal impulse times mu.
    """
    w = N @ velocity-target
    normal_res = jnp.maximum(
        jnp.max(jnp.where(active, jnp.maximum(-w, 0.), 0.)),
        jnp.max(jnp.abs(normal*w)))
    normal_res = jnp.maximum(normal_res, jnp.max(jnp.maximum(-normal, 0.)))
    normal_res = jnp.maximum(normal_res,
                            jnp.max(jnp.where(active, 0., jnp.abs(normal))))
    tau, slip = tangent.reshape(-1, 2), (T @ velocity).reshape(-1, 2)
    cap_error = jnp.max(jnp.maximum(jnp.abs(tau).sum(axis=1)-budget, 0.))
    friction_res = jnp.max(jnp.abs((tau*slip).sum(axis=1) +
                                    budget*jnp.max(jnp.abs(slip), axis=1)))
    return normal_res, friction_res, cap_error


def contact_error(out):
    return jnp.max(jnp.stack((out.normal_residual, out.friction_residual,
                              out.budget_violation, out.momentum_residual,
                              out.qp_residual)))


def resolve_contact(world, ball, active, solver: Solver,
                    config=PhysicsConfig(), *, budget=None) -> ContactResult:
    """Exactly ONE QP call for all active contacts, without a contact-sweep loop.

    ``budget`` may explicitly override the isolated-wall predictor. It must
    be finite and nonnegative. Normal impulses remain uncapped. The returned
    velocity is the primal QP solution, not a second impulse-based update.
    """
    N, T, target, predicted = contact_parameters(world, ball, active, config)
    beta = predicted if budget is None else jnp.where(active, budget, 0.)
    p = contact_qp(ball.velocity, ball.mass, N, T, target, beta, active)
    out = solver.solve(p)
    n = N.shape[0]
    velocity, slip_bound = out.x[:3], out.x[3:]
    # y satisfies Qx+c+A.T@y=0. Normal lower bounds have negative y;
    # tangent upper bounds have positive y. Do not clip solver violations.
    normal = -out.dual[:n]
    tangent = out.dual[3*n:5*n]-out.dual[n:3*n]
    a, b, c = contact_residuals(N, T, velocity, target, normal, tangent, beta, active)
    momentum = jnp.max(jnp.abs(ball.mass*(velocity-ball.velocity) -
                               N.T @ normal-T.T @ tangent))
    qp_res = jnp.max(jnp.stack((out.feasibility, out.stationarity,
                                out.complementarity)))
    residual = jnp.max(jnp.stack((a,b,c,momentum,qp_res)))
    valid_budget = jnp.all(jnp.isfinite(beta)) & jnp.all(beta >= 0.)
    valid_solution = (jnp.all(jnp.isfinite(out.x)) &
                      jnp.all(jnp.isfinite(out.dual)))
    converged = ((out.status == 1) & valid_budget & valid_solution &
                  (residual <= config.complementarity_tolerance))
    energy_change = .5*ball.mass*jnp.sum(velocity**2-ball.velocity**2)
    return ContactResult(velocity, normal, tangent, beta, target, slip_bound,
                          a, b, c, momentum, qp_res, energy_change,
                          jnp.asarray(1), out.iterations, out.status, converged)


def time_to_collision(world, ball, remaining, velocity_tolerance):
    """STUDENT PART: earliest hit on the current straight drift segment.

    Only approaching planes can be hit. A touching, approaching plane has
    collision time zero. Return the earliest time and whether it lies within remaining; +inf
    means no approaching plane.
    """
    # STUDENT TODO: Implement time_to_collision.
    # During the straight drift, gap_i(t) = gap_i(0) + t * normal_speed_i.
    # Only negative normal speeds approach a wall (normals point inward).
    normal_speed = world.normals @ ball.velocity
    approaching = normal_speed < -velocity_tolerance
    # Clamp tiny negative gaps to zero: an already-touching approach hits now.
    clearance = jnp.maximum(gaps(world, ball.position, ball.radius), 0.)
    denominator = jnp.where(approaching, -normal_speed, 1.)
    candidate_times = jnp.where(approaching, clearance / denominator, jnp.inf)
    earliest = jnp.min(candidate_times)
    # Keep the actual earliest time even if it is beyond the remaining interval.
    hit = jnp.isfinite(earliest) & (earliest <= remaining)
    return earliest, hit


def step(world: World, ball: Ball, dt, gravity, solver: Solver,
         config=PhysicsConfig()) -> StepResult:
    """Supplied event driver: gravity kick, swept drift, one QP per event.

    Simultaneous contacts are resolved in ONE solve, but successive impacts
    within dt require successive events. A failed solve, positive impact
    energy jump, penetration, or event overflow is reported, not concealed
    by projecting a penetrated endpoint. Gravity uses a first-order split,
    not exact accelerated trajectories. Near-zero speeds use tolerances.
    """
    kicked = ball._replace(velocity=ball.velocity+dt*gravity)
    initial_valid = (jnp.min(gaps(world, ball.position, ball.radius)) >=
                     -config.contact_tolerance) & (dt > 0)
    # remaining, state, events, ok, residual, energy, QP calls, QP iterations,
    # QP status (1 means that every attempted QP succeeded)
    initial = (jnp.asarray(dt), kicked, jnp.asarray(0), initial_valid,
               jnp.asarray(0.), jnp.asarray(0.), jnp.asarray(0), jnp.asarray(0),
               jnp.asarray(1))
    def cond(s):
        remaining, _, events, ok, _, _, _, _, _ = s
        return (remaining > 1e-12) & (events < config.max_events) & ok
    def body(s):
        remaining, current, events, ok, residual, energy, calls, iterations, status = s
        hit_time, hit = time_to_collision(world, current, remaining,
                                          config.velocity_tolerance)
        def collide(_):
            at_hit = current._replace(position=current.position+hit_time*current.velocity)
            active = gaps(world, at_hit.position, at_hit.radius) <= config.contact_tolerance
            out = resolve_contact(world, at_hit, active, solver, config)
            updated = at_hit._replace(velocity=out.u)
            valid = out.converged & (out.energy_change <= config.energy_tolerance)
            accepted = jax.tree.map(lambda a,b:jnp.where(valid,a,b), updated, at_hit)
            return (jnp.maximum(remaining-hit_time, 0.), accepted, events+1,
                    ok & valid, jnp.maximum(residual, contact_error(out)),
                    jnp.maximum(energy, out.energy_change), calls+out.qp_solves,
                    iterations+out.qp_iterations,
                    jnp.where(out.qp_status == 1, status, out.qp_status))
        def drift(_):
            final = current._replace(position=current.position+remaining*current.velocity)
            return (jnp.asarray(0.), final, events, ok, residual, energy, calls,
                    iterations, status)
        return solver.cond(hit, collide, drift, None)
    remaining, final, events, ok, residual, energy, calls, iterations, status = (
        solver.while_loop(cond, body, initial))
    phi = jnp.min(gaps(world, final.position, final.radius))
    ok = ok & (remaining <= 1e-12) & (phi >= -config.contact_tolerance)
    return StepResult(final, remaining, events, ok, phi, residual, energy, calls,
                      iterations, status)


def make_rollout(solver, steps: int, config=PhysicsConfig()):
    """JIT(scan(time, vmap(world))); equal fixed plane-slot counts required.

    After failure, a world's accepted state is frozen. A vmap of conditionals
    may still execute masked branches on the accelerator: qp_solves counts
    logical accepted/proposed contact events, not hardware kernel launches.
    """
    if not solver.compilable:
        raise ValueError("Compiled rollout requires JAXopt, not SciPy")
    if steps < 1:
        raise ValueError("steps must be positive")
    batched_step = jax.vmap(lambda w,b,dt,g:step(w,b,dt,g,solver,config),
                            in_axes=(0,0,None,None))
    @jax.jit
    def rollout(worlds, balls, dt, gravity):
        alive = jnp.ones(balls.mass.shape, dtype=bool)
        def body(carry, _):
            state, alive = carry
            out = batched_step(worlds, state, dt, gravity)
            def select(a,b):
                mask = alive.reshape(alive.shape+(1,)*(a.ndim-alive.ndim))
                return jnp.where(mask,a,b)
            accepted = jax.tree.map(select,out.ball,state)
            next_alive = alive & out.success
            out = out._replace(ball=accepted, success=next_alive,
                                events=jnp.where(alive,out.events,0),
                                qp_solves=jnp.where(alive,out.qp_solves,0),
                                qp_iterations=jnp.where(alive,out.qp_iterations,0),
                                qp_status=jnp.where(alive,out.qp_status,1))
            return (accepted,next_alive),out
        return jax.lax.scan(body,(balls,alive),None,length=steps)[1]
    return rollout

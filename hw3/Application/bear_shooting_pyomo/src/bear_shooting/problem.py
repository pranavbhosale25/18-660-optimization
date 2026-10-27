"""Student-facing Pyomo task registration; only the actuation U enters IPOPT."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pyomo.environ as pyo
from .layout import RolloutBackend, broadcast, vector
from .pyomo_task import Variables, CompiledTask, expression_vector

@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    expression: object
    weight: float
    scale: float


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    expression: object
    lower: np.ndarray
    upper: np.ndarray
    scale: np.ndarray


class ShootingProblem:
    """min_U objectives(U, rollout(U)) subject to bounds and registered tasks.

    Definitions freeze on first compile/solve/derivative check. Create another
    problem to change definitions. Initial guesses and solver options may change.
    """
    def __init__(self, rollout: RolloutBackend, *, lower=-0.3, upper=0.3, max_jacobian_mb=512):
        if not np.isfinite(max_jacobian_mb) or max_jacobian_mb <= 0:
            raise ValueError("max_jacobian_mb must be positive and finite.")
        self.max_jacobian_mb = float(max_jacobian_mb)
        self.simulator = rollout
        self.layout = rollout.layout
        self.lower = broadcast(lower, self.layout.control_shape, "control lower bounds")
        self.upper = broadcast(upper, self.layout.control_shape, "control upper bounds")
        if np.any(self.lower > self.upper):
            raise ValueError("Control lower bounds exceed upper bounds.")
        self.lower.setflags(write=False)
        self.upper.setflags(write=False)
        self.variables = Variables(self.layout, rollout, self.lower, self.upper)
        self.model = self.variables.model
        self.model.objective_terms = pyo.Expression(pyo.Any)
        self.model.task_constraints = pyo.Block(pyo.Any)
        self.model.objective = pyo.Objective(expr=0.0, sense=pyo.minimize)
        self._objectives: list[ObjectiveSpec] = []
        self._constraints: list[ConstraintSpec] = []
        self._compiled = None
        self._oracle = None
        self._names = set()
        self._model_objective_expr = self.model.objective.expr
        self._model_constraints = []
        self._model_term_exprs = {}

    def _check_name(self, name):
        if self._compiled is not None:
            raise RuntimeError("Task definitions are frozen after compilation. Create a new ShootingProblem.")
        if not isinstance(name, str) or not name or name in self._names:
            raise ValueError("Use a nonempty, unique name for every objective and constraint.")

    def add_objective(self, name, expression, *, weight=1.0, scale=1.0):
        """Add weight * expression / scale to the scalar minimization objective."""
        self._check_name(name)
        expr = expression_vector(expression)
        if expr.size != 1:
            raise ValueError("An objective must be scalar. Use sumsqr or pyomo.quicksum explicitly.")
        if not np.isfinite(weight) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Objective weight must be finite and scale must be finite and positive.")
        self._objectives.append(ObjectiveSpec(name, expr[0], float(weight), float(scale)))
        self.model.objective_terms[name] = expr[0]
        self.model.objective.set_value(pyo.quicksum(
            (o.weight / o.scale) * self.model.objective_terms[o.name] for o in self._objectives))
        self._model_objective_expr = self.model.objective.expr
        self._model_term_exprs[name] = self.model.objective_terms[name].expr
        self._names.add(name)
        return self

    def add_constraint(self, name, expression, *, lower=-np.inf, upper=np.inf, scale=1.0):
        """Enforce lower <= expression <= upper, componentwise in physical units.

        Equal bounds define an equality; infinite bounds define one-sided tasks.
        Matrix expressions and bound/scale arrays are flattened in C order.
        scale changes numerical conditioning, not the physical feasible set.
        """
        self._check_name(name)
        expr = expression_vector(expression)
        n = int(expr.size)
        def bcast(value, label, finite=False):
            return broadcast(np.asarray(value).reshape(-1), (n,), label, finite=finite)
        lb, ub, sc = bcast(lower, "lower"), bcast(upper, "upper"), bcast(scale, "scale", True)
        if np.any(lb > ub) or np.any(np.isposinf(lb)) or np.any(np.isneginf(ub)):
            raise ValueError("Constraint bounds are invalid or reversed.")
        if np.any((lb == -np.inf) & (ub == np.inf)):
            raise ValueError("Each constraint component needs at least one finite bound.")
        if np.any(sc <= 0):
            raise ValueError("Constraint scales must be positive.")
        self._constraints.append(ConstraintSpec(name, expr, lb, ub, sc))
        block = self.model.task_constraints[name]
        block.rows = pyo.ConstraintList()
        for e, l, u, s in zip(expr, lb, ub, sc):
            block.rows.add((None if np.isneginf(l) else float(l / s), e / float(s),
                            None if np.isposinf(u) else float(u / s)))
        self._model_constraints.extend((c, c.expr) for c in block.rows.values())
        self._names.add(name)
        return self

    def add_terminal_position(self, name, part, target, *, tolerance=1e-3, scale=1.0):
        target = vector(target, 3, "target position")
        tol = broadcast(tolerance, (3,), "position tolerance")
        if np.any(tol < 0):
            raise ValueError("Position tolerance cannot be negative.")
        return self.add_constraint(name, self.variables.position(part) - target,
                                   lower=-tol, upper=tol, scale=scale)

    def add_pose_target(self, name, markers, target_points, *, tolerance=1e-3, scale=1.0):
        """Constrain >=3 noncollinear material landmarks to target world points.

        This specifies a landmark pose of a deformable patch; it does not turn
        that patch into a rigid body or constrain all of its internal strains.
        """
        markers = tuple(markers)
        points = np.asarray(target_points, dtype=float)
        if len(markers) < 3 or len(set(markers)) != len(markers) or points.shape != (len(markers), 3):
            raise ValueError("A pose needs at least three distinct markers and matching (M,3) target points.")
        if not np.all(np.isfinite(points)) or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            raise ValueError("Target landmarks must be finite and noncollinear.")
        tol = broadcast(tolerance, points.shape, "pose tolerance").ravel()
        if np.any(tol < 0):
            raise ValueError("Pose tolerance cannot be negative.")
        residual = np.concatenate([self.variables.position(part) - target
                                   for part, target in zip(markers, points)])
        return self.add_constraint(name, residual, lower=-tol, upper=tol, scale=scale)

    def add_slew_limit(self, name, max_rate, *, scale=1.0):
        """Bound inter-interval activation changes per second (not du/dt at jumps)."""
        if self.layout.horizon < 2:
            raise ValueError("A slew limit needs at least two control intervals.")
        rate = broadcast(max_rate, (self.layout.horizon - 1, len(self.layout.actuator_names)), "max_rate")
        if np.any(rate < 0):
            raise ValueError("max_rate must be nonnegative.")
        v = self.variables
        return self.add_constraint(name, (v.u[1:, :] - v.u[:-1, :]) / v.dt,
                                   lower=-rate, upper=rate, scale=scale)

    def _assert_model_consistent(self):
        """Reject direct model edits that would otherwise bypass the registry."""
        message = ("The Pyomo model was modified outside the task registry. Use register methods "
                   "and constructor bounds, and create a new ShootingProblem to change a compiled task.")
        actual_vars = list(self.model.component_data_objects(pyo.Var))
        expected_vars = self.variables._u
        if len(actual_vars) != len(expected_vars) or any(a is not b for a, b in zip(actual_vars, expected_vars)):
            raise RuntimeError(message)
        for var, lo, hi in zip(actual_vars, self.lower.flat, self.upper.flat):
            if var.fixed or not var.is_continuous() or var.lb != lo or var.ub != hi:
                raise RuntimeError(message)
        objectives = list(self.model.component_data_objects(pyo.Objective, active=True))
        if len(objectives) != 1 or objectives[0] is not self.model.objective:
            raise RuntimeError(message)
        if self.model.objective.expr is not self._model_objective_expr or self.model.objective.sense != pyo.minimize:
            raise RuntimeError(message)
        for name, expr in self._model_term_exprs.items():
            if self.model.objective_terms[name].expr is not expr:
                raise RuntimeError(message)
        constraints = list(self.model.component_data_objects(pyo.Constraint, active=True))
        if len(constraints) != len(self._model_constraints):
            raise RuntimeError(message)
        if any(c is not expected or c.expr is not expr
               for c, (expected, expr) in zip(constraints, self._model_constraints)):
            raise RuntimeError(message)

    def compile(self):
        self._assert_model_consistent()
        if self._compiled is None:
            if not self._objectives:
                raise ValueError("Register at least one objective (zero is allowed for a feasibility task).")
            rows = 1 + sum(len(c.lower) for c in self._constraints)
            memory_mb = rows * self.layout.n_controls * 8 / 2**20
            if memory_mb > self.max_jacobian_mb:
                raise MemoryError(
                    f"The dense reduced Jacobian alone needs {memory_mb:.1f} MiB, "
                    f"above max_jacobian_mb={self.max_jacobian_mb}. Use fewer grouped "
                    "actuators/task rows, or explicitly increase the memory budget."
                )
            self._compiled = CompiledTask(self.variables, self._objectives, self._constraints)
        return self._compiled

    @property
    def oracle(self):
        self._assert_model_consistent()
        if self._oracle is None:
            from .oracle import ShootingOracle
            self._oracle = ShootingOracle(self.simulator, self.compile())
        return self._oracle

    def initial_guess(self, controls=None):
        if controls is None:
            return np.clip(np.zeros(self.layout.control_shape), self.lower, self.upper)
        result = broadcast(controls, self.layout.control_shape, "initial controls")
        if np.any(result < self.lower) or np.any(result > self.upper):
            raise ValueError("Initial controls violate actuation bounds; they are not silently clipped.")
        return result

    def rollout(self, controls):
        controls = self.initial_guess(controls)
        if self._oracle is not None:
            self._oracle.invalidate()
        return self.layout.trajectory(self.simulator.rollout(controls))

    def solve(self, initial_controls=None, *, engine="ipopt", ipopt_options=None,
              feasibility_tolerance=1e-4, raise_on_failure=False):
        """Solve the registered shooting problem with IPOPT.

        Parameters
        ----------
        initial_controls : array-like, optional
            Starting control values. A supplied value must broadcast to
            ``layout.control_shape`` and lie within this problem's control
            bounds. When omitted, the bounded zero control is used.
        engine : {"ipopt", "cyipopt"}, optional
            IPOPT interface to use. ``"ipopt"`` selects the supplied native
            C-library interface; ``"cyipopt"`` requires the optional cyipopt
            package. Both choices solve the same registered problem.
        ipopt_options : mapping, optional
            IPOPT option names and values that override the package defaults.
        feasibility_tolerance : float, optional
            Largest accepted scaled task-constraint violation in the returned
            result. Control-bound feasibility is checked separately.
        raise_on_failure : bool, optional
            If true, raise ``RuntimeError`` when IPOPT does not return a result
            that is both solver-successful and independently feasible.

        Returns
        -------
        SolveResult
            Controls, simulated trajectory, objective and constraint reports,
            solver status, work counts, timing, and iteration history. Unless
            ``raise_on_failure`` is true, inspect ``result.success`` or call
            ``result.require_success()`` before accepting the result.
        """
        from .solvers import solve_ipopt
        return solve_ipopt(self, self.initial_guess(initial_controls), engine=engine,
                           ipopt_options=ipopt_options, feasibility_tolerance=feasibility_tolerance,
                           raise_on_failure=raise_on_failure)

    def check_derivatives(self, controls=None, *, directions=3, steps=(1e-2, 3e-3, 1e-3), seed=7):
        """Independent central-difference checks of the FULL shooting derivatives.

        Checks objective AND all constraints, staying within actuation bounds.
        Use several steps: Warp's float32 arithmetic makes tiny steps unreliable.
        The returned errors are diagnostics, not a proof of contact smoothness.
        """
        if directions < 1 or int(directions) != directions:
            raise ValueError("directions must be a positive integer.")
        if not steps or any(not np.isfinite(h) or h <= 0 for h in steps):
            raise ValueError("Finite-difference steps must be positive and finite.")
        x = self.initial_guess((self.lower + self.upper) / 2 if controls is None else controls).ravel()
        oracle = self.oracle
        jac = oracle.total_jacobian(x).copy()
        distance = np.minimum(x - self.lower.ravel(), self.upper.ravel() - x)
        rng = np.random.default_rng(seed)
        results = []
        try:
            for direction_id in range(int(directions)):
                d = rng.normal(size=x.size)
                d[distance <= 1e-7] = 0
                if not np.any(d):
                    raise ValueError("No interior variables available for central differences.")
                d /= np.max(np.abs(d))
                cap = 0.45 * np.min(distance[d != 0] / np.abs(d[d != 0]))
                exact = jac @ d
                for requested in steps:
                    h = min(float(requested), float(cap))
                    plus = oracle.values(x + h * d).copy()
                    minus = oracle.values(x - h * d).copy()
                    finite_difference = (plus - minus) / (2 * h)
                    delta = finite_difference - exact
                    denominator = max(1e-8, np.linalg.norm(exact), np.linalg.norm(finite_difference))
                    row_names = ["objective"] + [f"{c.name}[{i}]"
                        for c in self.compile().constraints for i in range(len(c.lower))]
                    row_errors = [{"name": name, "analytic": float(ad), "finite_difference": float(fd),
                                   "absolute_error": float(abs(fd-ad)),
                                   "relative_error": float(abs(fd-ad) / max(1e-6, abs(ad), abs(fd)))}
                                  for name, ad, fd in zip(row_names, exact, finite_difference)]
                    results.append({"direction": direction_id, "step": h,
                                    "relative_l2_error": float(np.linalg.norm(delta) / denominator),
                                    "max_absolute_error": float(np.max(np.abs(delta))),
                                    "objective_absolute_error": float(abs(delta[0])),
                                    "rows": row_errors})

        finally:
            oracle.values(x)
        return results

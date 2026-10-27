"""Pyomo expressions, external observations, and reduced-space task differentiation.

The public model has only bounded actuation Vars. Observations are lazy Pyomo
ExternalFunction expressions. During differentiation we temporarily treat each
observation as an independent *parameter* in a private expression graph, then
compose its partial derivative with a Warp VJP. No state variables enter IPOPT.
"""
from __future__ import annotations
import numpy as np
import pyomo.environ as pyo
from scipy.sparse import csr_matrix
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_ad
from pyomo.core.expr.visitor import identify_variables, identify_mutable_parameters, replace_expressions
from pyomo.core.expr.expr_common import ExpressionType
from .layout import Layout, vector


def expression_vector(expression):
    """Flatten scalar/list/NumPy arrays in C order without evaluating expressions."""
    values = np.asarray(expression, dtype=object).reshape(-1)
    if not values.size:
        raise ValueError("An empty constraint/objective expression is not meaningful.")
    for expr in values:
        if isinstance(expr, (bool, np.bool_)) or getattr(expr, "is_expression_type", lambda *args: False)(ExpressionType.RELATIONAL):
            raise TypeError("Pass an expression and lower/upper bounds, not a boolean or a comparison.")
        if not (isinstance(expr, (int, float, np.number)) or getattr(expr, "is_numeric_type", lambda: False)()):
            raise TypeError("Use Pyomo expressions built from problem.variables and numerical constants.")
    return values.copy()


def sumsqr(values):
    """Sum of squares for scalar/list/array Pyomo expressions (never calls value)."""
    return pyo.quicksum(v * v for v in expression_vector(values))


def dot(left, right):
    """Scalar dot product of equally sized Pyomo expression arrays."""
    a, b = expression_vector(left), expression_vector(right)
    if a.size != b.size:
        raise ValueError("dot arguments must have the same size.")
    return pyo.quicksum(x * y for x, y in zip(a, b))


class ObservationFunction:
    """Numerical evaluation for the public Pyomo model's external functions.

    The solver's optimized path uses a task-level reverse pass instead of one
    reverse pass for every observation. This path also supports pyo.value and
    Pyomo's own differentiate() on the public model for inspection/testing.
    """
    def __init__(self, simulator):
        self.simulator = simulator
        self.x = self.y = self.revision = None
        self.gradients = {}

    def ensure(self, controls):
        x = vector(controls, self.simulator.layout.n_controls, "external controls")
        revision = getattr(self.simulator, "forward_count", None)
        if self.x is None or not np.array_equal(x, self.x) or revision != self.revision:
            self.x = self.y = self.revision = None
            self.gradients.clear()
            y = self.simulator.rollout(x.reshape(self.simulator.layout.control_shape))
            self.x, self.y = x.copy(), np.asarray(y).copy()
            self.revision = getattr(self.simulator, "forward_count", None)
        return self.y

    def function(self, index, *controls):
        return float(self.ensure(controls)[int(index)])

    def gradient(self, arguments, fixed=None):
        index = int(arguments[0])
        self.ensure(arguments[1:])
        if index not in self.gradients:
            seed = np.zeros(self.simulator.layout.n_observations)
            seed[index] = 1.0
            self.gradients[index] = self.simulator.vjp(seed).ravel().copy()
        # The observation selector is a literal constant, not a decision variable.
        result = [0.0, *self.gradients[index].tolist()]
        if fixed is not None:
            result = [0.0 if is_fixed else v for v, is_fixed in zip(result, fixed)]
        return result


class Variables:
    """u[k,a] is a Pyomo VarData; position/velocity return length-3 expression arrays.

    Control intervals: k=0,...,N-1; observation knots: k=0,...,N (or k=-1
    for terminal). World positions are in metres, velocities in metres/second;
    activation is dimensionless. Arrays use NumPy C order.
    """
    def __init__(self, layout: Layout, simulator, lower, upper):
        self.layout = layout
        self.model = m = pyo.ConcreteModel(name="bear_single_shooting")
        self.N, self.A = layout.control_shape
        m.K = pyo.RangeSet(0, self.N - 1)
        m.A = pyo.RangeSet(0, self.A - 1)
        m.u = pyo.Var(m.K, m.A,
                      bounds=lambda _, k, a: (float(lower[k, a]), float(upper[k, a])),
                      initialize=lambda _, k, a: float(np.clip(0., lower[k, a], upper[k, a])))
        self.u = np.empty(layout.control_shape, dtype=object)
        for k in m.K:
            for a in m.A:
                self.u[k, a] = m.u[k, a]
        self._u = tuple(self.u.flat)
        self._observations = {}
        self._observation_function = ObservationFunction(simulator)
        m.simulation = pyo.ExternalFunction(function=self._observation_function.function,
                                            gradient=self._observation_function.gradient)
        self.dt, self.times = layout.control_dt, layout.times
        self.parts, self.actuators = layout.part_names, layout.actuator_names

    def _observation(self, index):
        if index not in self._observations:
            self._observations[index] = self.model.simulation(index, *self._u)
        return self._observations[index]

    def position(self, part: str, knot: int = -1):
        sl = self.layout.observation_slice(part, knot)
        return np.array([self._observation(i) for i in range(sl.start, sl.stop)], dtype=object)

    def velocity(self, part: str, knot: int = -1):
        sl = self.layout.observation_slice(part, knot, velocity=True)
        return np.array([self._observation(i) for i in range(sl.start, sl.stop)], dtype=object)

    def control(self, actuator: str, interval: int):
        try:
            a = self.actuators.index(actuator)
        except ValueError as exc:
            raise KeyError(f"Unknown actuator {actuator!r}.") from exc
        if isinstance(interval, bool) or not isinstance(interval, (int, np.integer)):
            raise TypeError("The control interval must be an integer.")
        k = int(interval)
        if k < 0:
            k += self.N
        if not 0 <= k < self.N:
            raise IndexError("Control interval is out of range.")
        return self.u[k, a]


class CompiledTask:
    def __init__(self, variables, objectives, constraints):
        self.variables = variables
        self.objectives, self.constraints = tuple(objectives), tuple(constraints)
        self._shadow = pyo.ConcreteModel(name="private_observation_partial_graph")
        self._shadow.y = pyo.Param(range(variables.layout.n_observations), mutable=True, initialize=0.)
        self._y = tuple(self._shadow.y.values())
        self._u = variables._u
        substitute = {id(expr): self._y[i] for i, expr in variables._observations.items()}
        allowed_variables = {id(u) for u in self._u}
        allowed_parameters = {id(y) for y in self._y}

        def compile_expression(expr):
            result = replace_expressions(expr, substitution_map=substitute)
            if any(id(v) not in allowed_variables for v in identify_variables(result, include_fixed=True)):
                raise ValueError("Task contains a foreign/free symbol. Use only this problem's variables.")
            if any(id(p) not in allowed_parameters for p in identify_mutable_parameters(result)):
                raise ValueError("Foreign mutable parameters are not supported: create a new task with numerical constants.")
            return result

        f = pyo.quicksum((o.weight / o.scale) * o.expression for o in objectives)
        self._raw = tuple(compile_expression(o.expression) for o in objectives)
        self._outputs = (compile_expression(f), *(
            compile_expression(expr / scale) for c in constraints for expr, scale in zip(c.expression, c.scale)))
        self.row_count = len(self._outputs)
        self.lower = np.concatenate([c.lower / c.scale for c in constraints]) if constraints else np.zeros(0)
        self.upper = np.concatenate([c.upper / c.scale for c in constraints]) if constraints else np.zeros(0)
        self.physical_scales = np.concatenate([c.scale for c in constraints]) if constraints else np.zeros(0)
        self.physical_lower = np.concatenate([c.lower for c in constraints]) if constraints else np.zeros(0)
        self.physical_upper = np.concatenate([c.upper for c in constraints]) if constraints else np.zeros(0)
        # Sparse identities let us extract only the derivatives visited by Pyomo.
        self._u_index = {id(u): i for i, u in enumerate(self._u)}
        self._y_index = {id(y): i for i, y in enumerate(self._y)}
        self._x_last = self._y_last = None

    def _set(self, controls, observations):
        x = np.asarray(controls).ravel()
        y = np.asarray(observations).ravel()
        # Public model values might have been inspected/changed by the caller.
        for var, value in zip(self._u, x):
            var.set_value(float(value), skip_validation=True)
        if self._y_last is None or not np.array_equal(y, self._y_last):
            for parameter, value in zip(self._y, y):
                parameter.set_value(float(value))
            self._y_last = y.copy()

    def values(self, controls, observations):
        self._set(controls, observations)
        return np.array([pyo.value(expr) for expr in self._outputs], dtype=float)

    def partials(self, controls, observations):
        self._set(controls, observations)
        du = np.zeros((self.row_count, len(self._u)))
        rows, cols, data = [], [], []
        for row, expr in enumerate(self._outputs):
            # reverse_ad computes derivatives of Pyomo algebra only. The
            # observation substitutions prevent simulator callbacks here.
            derivatives = reverse_ad(expr)
            for node, derivative in derivatives.items():
                uid = id(node)
                if uid in self._u_index:
                    du[row, self._u_index[uid]] = float(derivative)
                elif uid in self._y_index and derivative != 0:
                    rows.append(row); cols.append(self._y_index[uid]); data.append(float(derivative))
        return du, csr_matrix((data, (rows, cols)), shape=(self.row_count, len(self._y)))

    def raw_objectives(self, controls, observations):
        self._set(controls, observations)
        return np.array([pyo.value(expr) for expr in self._raw], dtype=float)

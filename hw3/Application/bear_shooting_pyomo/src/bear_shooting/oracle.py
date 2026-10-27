"""Chain rule between algebraic tasks and the Warp simulation's reverse mode."""
from __future__ import annotations
import numpy as np
from .layout import vector


class ShootingOracle:
    """Cached F(U)=[objective; constraints] and its actuation-only Jacobian.

    dF_i/dU = partial(F_i)/partial(U)
              + rollout.vjp(partial(F_i)/partial(observations)).

    One Warp reverse pass per state-dependent scalar row; control-only rows
    (effort/slew terms) require none. No finite-difference production gradients.
    """
    def __init__(self, simulator, task):
        self.simulator = simulator
        self.task = task
        self.layout = simulator.layout
        self._x = None
        self._y = None
        self._values = None
        self._jac = None
        self._sim_revision = None
        self.evaluations = 0
        self.derivative_evaluations = 0

    def invalidate(self):
        self._x = self._y = self._values = self._jac = None
        self._sim_revision = None

    def _ensure(self, x):
        x = vector(x, self.layout.n_controls, "decision vector")
        revision = getattr(self.simulator, "forward_count", None)
        if self._x is not None and np.array_equal(x, self._x) and revision == self._sim_revision:
            return
        # Invalidate FIRST: a failed rollout must not leave a stale tape/cache.
        self.invalidate()
        y = vector(self.simulator.rollout(x.reshape(self.layout.control_shape)),
                   self.layout.n_observations, "rollout observations")
        values = np.asarray(self.task.values(x, y), dtype=float).ravel()
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("A registered objective/constraint returned NaN or infinity.")
        self._x, self._y, self._values = x.copy(), y.copy(), values
        self._sim_revision = getattr(self.simulator, "forward_count", None)
        self.evaluations += 1

    def values(self, x):
        self._ensure(x)
        return self._values.copy()

    def total_jacobian(self, x):
        self._ensure(x)
        if self._jac is None:
            du, dy = self.task.partials(self._x, self._y)
            total = np.asarray(du, dtype=float).copy()
            # Keep algebraic observation partials sparse. Never materialize the
            # enormous Jacobian of every simulated particle wrt every control.
            observation_partials = dy.tocsr()
            for row in range(self.task.row_count):
                seed = observation_partials.getrow(row).toarray().ravel()
                if np.any(seed):
                    total[row] += vector(self.simulator.vjp(seed), self.layout.n_controls, "Warp VJP")
            if not np.all(np.isfinite(total)):
                raise FloatingPointError("Nonfinite total shooting derivative.")
            self._jac = total
            self.derivative_evaluations += 1
        return self._jac.copy()

    # cyipopt's standard callback interface.
    def objective(self, x):
        return float(self.values(x)[0])

    def gradient(self, x):
        return self.total_jacobian(x)[0]

    def constraints(self, x):
        return self.values(x)[1:]

    def jacobian(self, x):
        return self.total_jacobian(x)[1:].ravel(order="C")

    def jacobianstructure(self):
        rows, columns = np.indices((self.task.row_count - 1, self.layout.n_controls))
        return rows.ravel(), columns.ravel()

    def last_observations(self, x):
        self._ensure(x)
        return self._y.copy()

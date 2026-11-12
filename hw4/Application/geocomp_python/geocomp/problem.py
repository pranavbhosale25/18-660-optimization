"""A reusable convex affine-Gaussian, joint-box chance problem."""

from dataclasses import dataclass, field
import numpy as np
import jax.numpy as jnp


@dataclass
class AffineGaussianBox:
    """Minimize .5 x'Qx + c'x subject to P(lower <= b+Hx+noise <= upper)>=alpha.

    Covariance is fixed, positive definite, and independent of x. Deterministic
    box bounds and optional A_ub x <= b_ub are also supported. These assumptions
    give a convex true chance-feasible set. A numerical probability oracle does
    not automatically preserve that exact geometry.

    Let ``m`` be the number of stochastic output features, ``d`` the number of
    decision variables, and ``r`` the number of optional deterministic linear
    inequalities. Required numeric inputs have these layouts:

    - ``bias``: shape ``(m,)``; mean output at a zero decision.
    - ``response``: shape ``(m, d)``; rows are output features and columns are
      decision coordinates.
    - ``covariance``: shape ``(m, m)``; symmetric positive-definite output-error
      covariance.
    - ``lower`` and ``upper``: shape ``(m,)``; feature limits in the same order
      and units as ``bias``.
    - ``Q``: shape ``(d, d)``; symmetric positive-semidefinite quadratic-cost
      matrix.
    - ``bounds``: shape ``(d, 2)``; column zero contains decision lower bounds
      and column one contains decision upper bounds.

    ``alpha`` is a scalar strictly between zero and one. Optional ``c`` has shape
    ``(d,)`` and defaults to zeros. Optional ``A_ub`` and ``b_ub`` have shapes
    ``(r, d)`` and ``(r,)``; omitting both stores empty arrays with shapes
    ``(0, d)`` and ``(0,)``. ``name`` is a string identifier and ``metadata`` is
    an arbitrary dictionary for reporting. Numeric fields are stored as NumPy
    floating-point arrays.
    """

    bias: np.ndarray
    response: np.ndarray
    covariance: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    Q: np.ndarray
    bounds: np.ndarray
    alpha: float = 0.95
    c: np.ndarray | None = None
    A_ub: np.ndarray | None = None
    b_ub: np.ndarray | None = None
    name: str = "affine_gaussian_box"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for key in ("bias", "response", "covariance", "lower", "upper", "Q", "bounds"):
            setattr(self, key, np.asarray(getattr(self, key), dtype=float))
        if self.response.ndim != 2:
            raise ValueError(
                "response: expected a two-dimensional array with shape (m, d); "
                f"got shape {self.response.shape}"
            )
        m, d = self.response.shape
        for key, shape in [
            ("bias", (m,)),
            ("covariance", (m, m)),
            ("lower", (m,)),
            ("upper", (m,)),
            ("Q", (d, d)),
            ("bounds", (d, 2)),
        ]:
            if getattr(self, key).shape != shape:
                raise ValueError(f"{key}: expected shape {shape}")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be strictly between zero and one")
        if not np.all(np.isfinite(self.lower)) or not np.all(np.isfinite(self.upper)):
            raise ValueError("Finite tolerance intervals are required")
        if np.any(self.lower >= self.upper) or np.any(
            self.bounds[:, 0] >= self.bounds[:, 1]
        ):
            raise ValueError("Every interval must have positive width")
        if not np.all(np.isfinite(self.bounds)):
            raise ValueError(
                "Finite decision bounds are required for the LP outer approximation"
            )
        if not np.allclose(self.covariance, self.covariance.T):
            raise ValueError("Covariance must be symmetric")
        np.linalg.cholesky(self.covariance)  # Do not silently jitter a user's model.
        if (
            not np.allclose(self.Q, self.Q.T)
            or np.min(np.linalg.eigvalsh(self.Q)) < -1e-11
        ):
            raise ValueError("Q must be symmetric positive semidefinite")
        self.c = np.zeros(d) if self.c is None else np.asarray(self.c, dtype=float)
        if self.c.shape != (d,):
            raise ValueError("c has the wrong dimension")
        if (self.A_ub is None) != (self.b_ub is None):
            raise ValueError("Supply both A_ub and b_ub, or neither")
        if self.A_ub is None:
            self.A_ub, self.b_ub = np.empty((0, d)), np.empty(0)
        else:
            self.A_ub, self.b_ub = np.asarray(self.A_ub, float), np.asarray(
                self.b_ub, float
            )
            if (
                self.A_ub.ndim != 2
                or self.A_ub.shape[1] != d
                or self.b_ub.shape != (len(self.A_ub),)
            ):
                raise ValueError("Invalid linear inequalities")
        for key in ("bias", "response", "covariance", "Q", "c", "A_ub", "b_ub"):
            if not np.all(np.isfinite(getattr(self, key))):
                raise ValueError(f"{key} contains nonfinite values")

    @property
    def n_decisions(self):
        """Number ``d`` of decision coordinates."""
        return self.response.shape[1]

    @property
    def n_features(self):
        """Number ``m`` of stochastic output features."""
        return self.response.shape[0]

    def mean(self, x):
        """Return the mean output as a JAX array of shape ``(m,)``.

        ``x`` must be an array-like decision vector with shape ``(d,)``.
        """
        return jnp.asarray(self.bias) + jnp.asarray(self.response) @ x

    def objective(self, x):
        """Return the scalar JAX objective for a decision vector of shape ``(d,)``."""
        return 0.5 * x @ jnp.asarray(self.Q) @ x + jnp.asarray(self.c) @ x

    def linear_feasible(self, x, tol=1e-9):
        """Return a Python bool for decision-bound and linear-constraint feasibility."""
        x = np.asarray(x)
        return bool(
            np.all(x >= self.bounds[:, 0] - tol)
            and np.all(x <= self.bounds[:, 1] + tol)
            and np.all(self.A_ub @ x <= self.b_ub + tol)
        )

    def center_guess(self):
        """Least-squares centering is just a guess; the solver checks feasibility."""
        finite_center = np.where(
            np.isfinite(self.lower + self.upper), 0.5 * (self.lower + self.upper), 0.0
        )
        x = np.linalg.lstsq(self.response, finite_center - self.bias, rcond=None)[0]
        return np.clip(x, self.bounds[:, 0] + 1e-7, self.bounds[:, 1] - 1e-7)

"""Illustrative additive and subtractive surrogates, NOT calibrated physics.

Coefficients are normal offsets in millimeters, interpolated with the SAME
partition-of-unity basis used for visualization and stochastic validation.
A process maps command coefficients u to output coefficients b + G u + L z.
The optimization variable x is dimensionless, with u = tolerance * x.
"""

from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from .problem import AffineGaussianBox
from .geometry import MeshBasis, primitive
from .interfaces import GeometryBasis


@dataclass
class ProcessInstance:
    name: str
    process: str
    shape: str
    basis: GeometryBasis
    problem: AffineGaussianBox
    command_scale_mm: float
    description: str

    def command_coefficients(self, x):
        return self.command_scale_mm * jnp.asarray(x)

    def mean_coefficients(self, x):
        return self.problem.mean(jnp.asarray(x))

    def sample_coefficients(self, x, key, n):
        """JAX RNG, explicit PRNG key, correlated Gaussian coefficient draws."""
        z = jax.random.normal(key, (n, self.problem.n_features), dtype=jnp.float64)
        L = jnp.linalg.cholesky(jnp.asarray(self.problem.covariance))
        return self.mean_coefficients(x)[None, :] + z @ L.T

    def generated_vertices(self, x, key, n):
        coeff = self.sample_coefficients(x, key, n)
        return jax.vmap(self.basis.vertices_jax)(coeff)


def formulate_dlp_problem(
    *,
    name: str,
    bias_mm: np.ndarray,
    command_response: np.ndarray,
    covariance_mm2: np.ndarray,
    tolerance_mm: float,
    basis_weights: np.ndarray,
    vertex_area_weights: np.ndarray,
    alpha: float,
) -> AffineGaussianBox:
    """Return the optimization representation for supplied DLP model data.

    Let ``m = bias_mm.shape[0]`` and ``v = basis_weights.shape[0]``. All array
    arguments are NumPy floating-point arrays and use one consistent mode order.

    Args:
        name: Case identifier used in output files and diagnostics.
        bias_mm: Mean uncompensated output coefficients, shape ``(m,)``, in
            millimeters.
        command_response: Response from physical command coefficients to mean
            output coefficients, shape ``(m, m)``. Rows are output modes and
            columns are command modes, both in the same order as ``bias_mm``.
        covariance_mm2: Symmetric positive-definite covariance of the output
            coefficients, shape ``(m, m)``, in square millimeters.
        tolerance_mm: Positive scalar output tolerance and physical scale for
            each dimensionless decision coordinate, in millimeters.
        basis_weights: Coefficient-to-vertex interpolation matrix, shape
            ``(v, m)``. Rows follow mesh-vertex order; columns follow mode order.
        vertex_area_weights: Normalized surface-area weight for each mesh
            vertex, shape ``(v,)``, in the same row order as ``basis_weights``.
        alpha: Required joint success probability as a scalar strictly between
            zero and one.

    Returns:
        A complete :class:`AffineGaussianBox` with ``m`` output features and
        ``m`` dimensionless decisions. Its ``name`` and ``alpha`` fields must
        equal the supplied arguments. That class documents every returned
        field's shape and interpretation.
    """
    # STUDENT TODO: formulate and return the DLP optimization problem.
    return _surface_problem(
        name, bias_mm, command_response, covariance_mm2, tolerance_mm,
        basis_weights, vertex_area_weights, alpha, upper_bound=4.0,
    )


def formulate_cnc_problem(
    *,
    name: str,
    bias_mm: np.ndarray,
    command_response: np.ndarray,
    covariance_mm2: np.ndarray,
    tolerance_mm: float,
    basis_weights: np.ndarray,
    vertex_area_weights: np.ndarray,
    alpha: float,
) -> AffineGaussianBox:
    """Return the optimization representation for supplied CNC model data.

    Let ``m = bias_mm.shape[0]`` and ``v = basis_weights.shape[0]``. All array
    arguments are NumPy floating-point arrays and use one consistent mode order.

    Args:
        name: Case identifier used in output files and diagnostics.
        bias_mm: Mean uncompensated output coefficients, shape ``(m,)``, in
            millimeters.
        command_response: Response from physical command coefficients to mean
            output coefficients, shape ``(m, m)``. Rows are output modes and
            columns are command modes, both in the same order as ``bias_mm``.
        covariance_mm2: Symmetric positive-definite covariance of the output
            coefficients, shape ``(m, m)``, in square millimeters.
        tolerance_mm: Positive scalar output tolerance and physical scale for
            each dimensionless decision coordinate, in millimeters.
        basis_weights: Coefficient-to-vertex interpolation matrix, shape
            ``(v, m)``. Rows follow mesh-vertex order; columns follow mode order.
        vertex_area_weights: Normalized surface-area weight for each mesh
            vertex, shape ``(v,)``, in the same row order as ``basis_weights``.
        alpha: Required joint success probability as a scalar strictly between
            zero and one.

    Returns:
        A complete :class:`AffineGaussianBox` with ``m`` output features and
        ``m`` dimensionless decisions. Its ``name`` and ``alpha`` fields must
        equal the supplied arguments. That class documents every returned
        field's shape and interpretation.
    """
    # STUDENT TODO: formulate and return the CNC optimization problem.
    return _surface_problem(
        name, bias_mm, command_response, covariance_mm2, tolerance_mm,
        basis_weights, vertex_area_weights, alpha, upper_bound=0.0,
    )


def _surface_problem(name, bias, response, covariance, tolerance, weights,
                     area_weights, alpha, *, upper_bound):
    """Shared surface objective and scaling; CNC restricts commands inward."""
    d = len(bias)
    Q = 0.85 * weights.T @ (area_weights[:, None] * weights) + 0.15 * np.eye(d) / d
    return AffineGaussianBox(
        name=name, bias=bias, response=tolerance * response,
        covariance=covariance, lower=np.full(d, -tolerance),
        upper=np.full(d, tolerance), Q=Q, c=np.zeros(d),
        bounds=np.column_stack((np.full(d, -4.0), np.full(d, upper_bound))),
        alpha=alpha,
    )


def build_instance(
    process="dlp", shape="ellipsoid", n_modes=8, alpha=0.95, resolution=2
):
    """Construct one reproducible synthetic primitive instance.

    n_modes changes the approximation of manufacturing error, not just solver
    resolution: refit/recalibrate its covariance when changing it in real use.
    """
    if process not in ("dlp", "cnc"):
        raise ValueError("process must be dlp or cnc")
    mesh = primitive(shape, resolution)
    basis = MeshBasis.from_mesh(mesh, n_modes=n_modes, length_scale=0.34)
    a = basis.anchors
    normals = basis.normals[basis.anchor_indices]
    q = a / np.maximum(np.max(np.abs(basis.vertices), axis=0), 1e-9)
    distances = np.sum((q[:, None, :] - q[None, :, :]) ** 2, axis=2)
    kernel = np.exp(-distances / (2 * 0.9**2))
    mix = kernel / kernel.sum(axis=1, keepdims=True)
    if process == "dlp":
        tolerance = 0.20 if shape == "ellipsoid" else 0.16
        # Projection of anisotropic contraction onto target surface normals.
        # Positive curing offset also shrinks inner holes (inward solid normal).
        shrink = -a * np.array([0.024, 0.018, 0.030])
        contraction = np.einsum("ij,ij->i", shrink, normals)
        cure_bleed = 0.06 + 0.025 * (q[:, 2] + 1.0) / 2.0
        warp = 0.065 * (q[:, 0] ** 2 - q[:, 1] ** 2) * normals[:, 2]
        bias = contraction + cure_bleed + warp
        gains = 0.92 + 0.025 * q[:, 2]
        G = np.diag(gains) @ (0.92 * np.eye(n_modes) + 0.08 * mix)
        sd = tolerance * (0.14 + 0.025 * (q[:, 2] + 1.0) / 2.0)
        corr = 0.72 * np.eye(n_modes) + 0.28 * kernel
        description = (
            "DLP-inspired: anisotropic contraction, positive curing offset, "
            "height-dependent response and smooth warp. All numbers are synthetic; "
            "no exposure, layer, photopolymer or support simulation."
        )
    else:
        tolerance = 0.12 if shape == "cylinder" else 0.14
        side = 1.0 - normals[:, 2] ** 2
        theta = np.arctan2(q[:, 1], q[:, 0])
        bias = (
            0.10
            + 0.12 * side
            + 0.045 * (q[:, 2] + 1.0) / 2.0
            + 0.022 * np.cos(2.0 * theta)
        )
        G = 0.98 * (0.975 * np.eye(n_modes) + 0.025 * mix)
        sd = tolerance * (0.12 + 0.035 * side)
        corr = 0.78 * np.eye(n_modes) + 0.22 * kernel
        description = (
            "CNC-inspired: positive residual stock from compliance/tool offset, "
            "azimuth-dependent bias and correlated residual error. Command is an "
            "inward surface offset, not a cutter-center path or G-code."
        )
    covariance = np.outer(sd, sd) * corr
    # Area-weighted mean-square command plus modest coefficient regularization.
    area = np.zeros(len(mesh.vertices))
    for j in range(3):
        np.add.at(area, np.asarray(mesh.faces)[:, j], np.asarray(mesh.area_faces) / 3.0)
    area /= area.sum()
    formulator = formulate_dlp_problem if process == "dlp" else formulate_cnc_problem
    problem = formulator(
        name=f"{process}_{shape}",
        bias_mm=bias,
        command_response=G,
        covariance_mm2=covariance,
        tolerance_mm=tolerance,
        basis_weights=basis.weights,
        vertex_area_weights=area,
        alpha=alpha,
    )
    return ProcessInstance(
        problem.name, process, shape, basis, problem, tolerance, description
    )


def abstract_instances():
    """Analytically checkable 1-D and low-dimensional abstract problems."""
    yield AffineGaussianBox(
        np.array([0.9]),
        np.eye(1),
        np.array([[0.04]]),
        np.array([-1.0]),
        np.array([1.0]),
        np.eye(1),
        np.array([[-2.0, 2.0]]),
        name="abstract_1d_exact",
    )
    yield AffineGaussianBox(
        np.array([0.9, 0.7]),
        np.eye(2),
        0.04 * np.eye(2),
        -np.ones(2),
        np.ones(2),
        np.eye(2),
        np.array([[-2.0, 2.0], [-2.0, 2.0]]),
        name="abstract_2d_independent",
    )
    yield AffineGaussianBox(
        np.array([0.9, 0.7]),
        np.eye(2),
        np.array([[0.04, 0.024], [0.024, 0.04]]),
        -np.ones(2),
        np.ones(2),
        np.eye(2),
        np.array([[-2.0, 2.0], [-2.0, 2.0]]),
        name="abstract_2d_correlated",
    )


def default_instances():
    return [
        build_instance("dlp", "ellipsoid"),
        build_instance("dlp", "sleeve"),
        build_instance("cnc", "cylinder"),
        build_instance("cnc", "pocket_plate"),
    ]

"""Affine, fixed-reference-frame tube meshes for embedded extrusion.

A coefficient vector contains four channel-major blocks at control stations:
centerline dx, dy, dz and radius correction, all in millimeters. Nonnegative
B-spline weights interpolate each channel. Frames are transported along the
TARGET path once; they are deliberately not recomputed as decisions change.
This small-deformation approximation makes every mesh vertex affine in the
coefficients and gives an explicit, conservative whole-surface error bound.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BSpline
import trimesh


def _unit(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(length < 1e-12):
        raise ValueError("Path contains a zero tangent or an invalid frame")
    return v / length


def transported_frames(path: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation-minimizing discrete frames; reject a locally reversing path."""
    tangent = _unit(np.gradient(path, axis=0))
    normal = np.empty_like(tangent)
    axis = np.eye(3)[np.argmin(np.abs(tangent[0]))]
    normal[0] = _unit(np.cross(tangent[0], axis))
    for i in range(1, len(path)):
        v = np.cross(tangent[i - 1], tangent[i])
        c = float(tangent[i - 1] @ tangent[i])
        if c < -1 + 1e-8:
            raise ValueError("Path reverses its tangent: refine or replace the path")
        # Rodrigues rotation, in a form that also handles parallel tangents.
        n = normal[i - 1]
        n = n + np.cross(v, n) + np.cross(v, np.cross(v, n)) / (1 + c)
        n -= (n @ tangent[i]) * tangent[i]
        normal[i] = _unit(n)
    binormal = _unit(np.cross(tangent, normal))
    return tangent, normal, binormal


def bspline_weights(n_sections: int, n_controls: int) -> tuple[np.ndarray, np.ndarray]:
    """Clamped cubic (or lower-degree) partition-of-unity basis on [0, 1]."""
    if n_controls < 2 or n_sections < n_controls:
        raise ValueError("Require 2 <= n_controls <= n_sections")
    degree = min(3, n_controls - 1)
    interior = np.linspace(0, 1, n_controls - degree + 1)[1:-1]
    knots = np.r_[np.zeros(degree + 1), interior, np.ones(degree + 1)]
    parameters = np.linspace(0, 1, n_sections)
    weights = BSpline.design_matrix(parameters, knots, degree).toarray()
    greville = np.array(
        [np.mean(knots[i + 1 : i + degree + 1]) for i in range(n_controls)]
    )
    return weights, greville


def centerline_primitive(shape: str, n_sections: int = 81) -> np.ndarray:
    """Open, genuinely three-dimensional paths; coordinates are millimeters."""
    if n_sections < 12:
        raise ValueError("Use at least 12 sections to resolve the example path")
    t = np.linspace(0.0, 1.0, n_sections)
    if shape == "helix":
        angle = 2 * np.pi * t - np.pi / 2
        return np.column_stack((6 * np.cos(angle), 6 * np.sin(angle), 12 * (t - 0.5)))
    if shape == "serpentine":
        return np.column_stack(
            (18 * (t - 0.5), 3.5 * np.sin(2 * np.pi * t), 3 * np.sin(np.pi * (t - 0.5)))
        )
    raise ValueError(f"Unknown embedded path {shape!r}; choose helix or serpentine")


@dataclass
class TubeBasis:
    centerline: np.ndarray
    tangent: np.ndarray
    frame_normal: np.ndarray
    frame_binormal: np.ndarray
    weights: np.ndarray  # section x control, NOT vertex x feature
    control_parameters: np.ndarray
    radius_mm: float
    angles: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray  # target surface normals, for interchange
    radial_directions: np.ndarray  # affine radial directions, zero at cap centers
    section_indices: np.ndarray
    anchors: np.ndarray
    anchor_indices: np.ndarray
    feature_limits_mm: np.ndarray
    centerline_budget_mm: float
    radius_budget_mm: float

    @classmethod
    def from_path(
        cls,
        path: np.ndarray,
        radius_mm: float = 0.6,
        n_controls: int = 5,
        n_sides: int = 16,
        centerline_budget_mm: float = 0.18,
        radius_budget_mm: float = 0.06,
    ) -> "TubeBasis":
        path = np.asarray(path, dtype=float)
        if (
            path.ndim != 2
            or path.shape[1] != 3
            or len(path) < 4
            or not np.all(np.isfinite(path))
        ):
            raise ValueError("path must be a finite (n_sections >= 4, 3) array")
        if np.any(np.linalg.norm(np.diff(path, axis=0), axis=1) < 1e-10):
            raise ValueError("Successive path points must be distinct")
        if not (0 < radius_budget_mm < radius_mm and centerline_budget_mm > 0):
            raise ValueError(
                "Require positive budgets and radius_budget < target radius"
            )
        if n_sides < 6:
            raise ValueError("At least six sides are required")
        tangent, fn, fb = transported_frames(path)
        W, controls = bspline_weights(len(path), n_controls)
        angles = np.arange(n_sides) * (2 * np.pi / n_sides)
        dirs = (
            fn[:, None, :] * np.cos(angles)[None, :, None]
            + fb[:, None, :] * np.sin(angles)[None, :, None]
        )
        section_indices = np.r_[
            np.repeat(np.arange(len(path)), n_sides), 0, len(path) - 1
        ]
        radial = np.vstack((dirs.reshape(-1, 3), np.zeros((2, 3))))
        vertices = path[section_indices] + radius_mm * radial
        faces = []
        for i in range(len(path) - 1):
            for j in range(n_sides):
                k = (j + 1) % n_sides
                a, b, c, d = (
                    i * n_sides + j,
                    i * n_sides + k,
                    (i + 1) * n_sides + k,
                    (i + 1) * n_sides + j,
                )
                faces.extend(((a, b, c), (a, c, d)))
        start_cap, end_cap = len(vertices) - 2, len(vertices) - 1
        for j in range(n_sides):
            k = (j + 1) % n_sides
            faces.append((start_cap, k, j))
            faces.append(
                (end_cap, (len(path) - 1) * n_sides + j, (len(path) - 1) * n_sides + k)
            )
        mesh = trimesh.Trimesh(vertices, np.array(faces), process=False)
        mesh.fix_normals()
        anchor_sections = np.rint(controls * (len(path) - 1)).astype(int)
        limits = np.r_[
            np.full(3 * n_controls, centerline_budget_mm / np.sqrt(3)),
            np.full(n_controls, radius_budget_mm),
        ]
        return cls(
            path,
            tangent,
            fn,
            fb,
            W,
            controls,
            radius_mm,
            angles,
            vertices,
            np.asarray(mesh.faces),
            np.asarray(mesh.vertex_normals),
            radial,
            section_indices,
            path[anchor_sections],
            anchor_sections * n_sides,
            limits,
            centerline_budget_mm,
            radius_budget_mm,
        )

    @property
    def n_controls(self) -> int:
        return self.weights.shape[1]

    @property
    def n_features(self) -> int:
        return 4 * self.n_controls

    def fields(self, coefficients):
        """Return (dx, dy, dz, dr) per path section, preserving JAX derivatives."""
        a = jnp.reshape(jnp.asarray(coefficients), (4, self.n_controls))
        return jnp.asarray(self.weights) @ a.T

    def displacement(self, coefficients):
        fields = self.fields(coefficients)[jnp.asarray(self.section_indices)]
        return fields[:, :3] + fields[:, 3, None] * jnp.asarray(self.radial_directions)

    def vertices_jax(self, coefficients):
        return jnp.asarray(self.vertices) + self.displacement(coefficients)

    def centerline_jax(self, coefficients):
        return jnp.asarray(self.centerline) + self.fields(coefficients)[:, :3]

    def radii_jax(self, coefficients):
        return self.radius_mm + self.fields(coefficients)[:, 3]

    def max_displacement_batch(self, coefficients):
        """Exact max over the vertices of each modeled polygonal tube, batched.

        Avoid materializing batch x vertices x 3. Cap-center displacements are
        included explicitly. This tests correspondence distance, not registration
        or the exact closest-point Hausdorff distance.
        """
        a = jnp.reshape(jnp.asarray(coefficients), (-1, 4, self.n_controls))
        fields = jnp.einsum("sk,bck->bsc", jnp.asarray(self.weights), a)
        dc, dr = fields[..., :3], fields[..., 3]
        dn = jnp.einsum("bsi,si->bs", dc, jnp.asarray(self.frame_normal))
        db = jnp.einsum("bsi,si->bs", dc, jnp.asarray(self.frame_binormal))
        projection = dn[..., None] * jnp.cos(jnp.asarray(self.angles)) + db[
            ..., None
        ] * jnp.sin(jnp.asarray(self.angles))
        squared = jnp.sum(dc * dc, axis=-1) + dr * dr
        worst_ring = jnp.max(
            squared + jnp.max(2 * dr[..., None] * projection, axis=-1), axis=-1
        )
        caps = jnp.max(
            jnp.sum(dc[:, jnp.array([0, len(self.centerline) - 1]), :] ** 2, axis=-1),
            axis=-1,
        )
        return jnp.sqrt(jnp.maximum(jnp.maximum(worst_ring, caps), 0.0))

    def mesh(self, coefficients):
        return trimesh.Trimesh(
            np.asarray(self.vertices_jax(coefficients)), self.faces, process=False
        )

    def diagnostics(self, coefficients) -> dict:
        mesh = self.mesh(coefficients)
        f, v = self.faces, np.asarray(mesh.vertices)
        c0 = np.cross(
            self.vertices[f[:, 1]] - self.vertices[f[:, 0]],
            self.vertices[f[:, 2]] - self.vertices[f[:, 0]],
        )
        c = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        return {
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "nonpositive_target_normal_faces": int(np.sum(np.sum(c * c0, axis=1) <= 0)),
            "min_face_area_mm2": float(np.min(mesh.area_faces)),
            "volume_mm3": float(mesh.volume),
            "minimum_equivalent_radius_mm": float(
                jnp.min(self.radii_jax(coefficients))
            ),
            "self_intersection_checked": False,
        }

    def guarantee(self, tolerance: float) -> dict:
        bound = self.centerline_budget_mm + self.radius_budget_mm
        if not np.isclose(tolerance, bound):
            raise ValueError("Tolerance must equal the sum of the allocated budgets")
        return {
            "type": "fixed-frame tube: vector centerline budget + scalar radius budget",
            "nonnegative": bool(np.all(self.weights >= 0)),
            "partition_error": float(np.max(abs(self.weights.sum(axis=1) - 1))),
            "frame_unit_error": float(
                np.max(abs(np.linalg.norm(self.frame_normal, axis=1) - 1))
            ),
            "centerline_budget_mm": self.centerline_budget_mm,
            "radius_budget_mm": self.radius_budget_mm,
            "feature_limits_mm": self.feature_limits_mm.tolist(),
            "bound_mm": bound,
            "scope": "all corresponding points on the triangle surfaces, not CAD or fluid physics",
        }

    def extra_arrays(self) -> dict:
        return {
            "centerline": self.centerline,
            "frame_tangent": self.tangent,
            "frame_normal": self.frame_normal,
            "frame_binormal": self.frame_binormal,
            "section_indices": self.section_indices,
            "radial_directions": self.radial_directions,
            "radius_mm": np.array(self.radius_mm),
            "angles": self.angles,
            "control_parameters": self.control_parameters,
            "feature_limits_mm": self.feature_limits_mm,
        }

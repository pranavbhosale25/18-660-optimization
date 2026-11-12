"""Mesh primitives and a nonnegative partition-of-unity deformation basis."""

from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
import trimesh


def primitive(name, resolution=2):
    """Closed triangular target mesh; all coordinates are millimeters."""
    if name == "ellipsoid":
        mesh = trimesh.creation.icosphere(subdivisions=resolution, radius=1.0)
        mesh.vertices *= np.array([12.0, 9.0, 15.0])
    elif name == "sleeve":
        mesh = trimesh.creation.annulus(r_min=6.0, r_max=10.0, height=16.0, sections=48)
    elif name == "cylinder":
        mesh = trimesh.creation.cylinder(radius=10.0, height=20.0, sections=48)
    elif name == "pocket_plate":
        # Outer 36 x 28 x 10 mm block, 22 x 14 mm blind rectangular pocket,
        # depth 6 mm. Closed solid, not a boolean-engine dependency.
        def loop(a, b, z):
            return [[-a, -b, z], [a, -b, z], [a, b, z], [-a, b, z]]

        vertices = np.array(
            loop(18, 14, -5) + loop(18, 14, 5) + loop(11, 7, 5) + loop(11, 7, -1), float
        )
        faces = []

        def connect(first, second):
            for j in range(4):
                k = (j + 1) % 4
                faces.extend(
                    [
                        [first + j, first + k, second + k],
                        [first + j, second + k, second + j],
                    ]
                )

        connect(0, 4)
        connect(4, 8)
        connect(8, 12)
        faces += [[0, 2, 1], [0, 3, 2], [12, 13, 14], [12, 14, 15]]
        mesh = trimesh.Trimesh(vertices, faces, process=True)
        mesh.fix_normals()
    else:
        raise ValueError(f"Unknown primitive {name!r}")
    # Increase planar-face sampling without changing the target shape.
    if name != "ellipsoid":
        mesh = mesh.subdivide()
        if name == "pocket_plate":
            mesh = mesh.subdivide().subdivide()
    mesh.fix_normals()
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError("Primitive must be a closed consistently wound mesh")
    return mesh


def farthest_points(points, n):
    points = np.asarray(points, float)
    if n < 1 or n > len(points):
        raise ValueError("Invalid number of control centers")
    selected = [int(np.argmax(points[:, 2]))]
    dist = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for _ in range(1, n):
        index = int(np.argmax(dist))
        selected.append(index)
        dist = np.minimum(dist, np.sum((points - points[index]) ** 2, axis=1))
    return np.array(selected)


@dataclass
class MeshBasis:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    anchors: np.ndarray
    anchor_indices: np.ndarray

    @classmethod
    def from_mesh(cls, mesh, n_modes=8, length_scale=0.48):
        v = np.asarray(mesh.vertices).copy()
        normal = np.asarray(mesh.vertex_normals).copy()
        ext = np.maximum(np.ptp(v, axis=0), 1e-9)
        scaled = (v - np.mean(v, axis=0)) / ext
        # Include normals to distinguish nearby inner/outer sleeve surfaces and
        # pocket walls. Euclidean position alone can miss internal features.
        features = np.column_stack([scaled, 0.65 * normal])
        inds = farthest_points(features, n_modes)
        distance = np.sum(
            (features[:, None, :] - features[inds][None, :, :]) ** 2, axis=2
        )
        weights = np.exp(-0.5 * distance / length_scale**2)
        weights /= weights.sum(axis=1, keepdims=True)
        return cls(
            v, np.asarray(mesh.faces).copy(), normal, weights, v[inds].copy(), inds
        )

    def displacement(self, coefficients):
        """Vector displacement, using fixed target normals (not recomputed ones)."""
        scalar = jnp.asarray(self.weights) @ coefficients
        return scalar[:, None] * jnp.asarray(self.normals)

    def max_displacement_batch(self, coefficients):
        """Maximum corresponding-vertex displacement for a batch of coefficient draws."""
        return jnp.max(
            jnp.abs(jnp.asarray(coefficients) @ jnp.asarray(self.weights).T), axis=1
        )

    def vertices_jax(self, coefficients):
        return jnp.asarray(self.vertices) + self.displacement(coefficients)

    def mesh(self, coefficients):
        return trimesh.Trimesh(
            np.asarray(self.vertices_jax(jnp.asarray(coefficients))),
            self.faces,
            process=False,
        )

    def guarantee(self, tolerance):
        """Certificate structure: |coeff_j|<=tau -> every mesh displacement<=tau.

        W>=0 and W1=1 bound the scalar at every vertex. Unit target normals bound
        vector displacement. Barycentric interpolation gives the same bound at
        EVERY point of each corresponding triangle, hence a Hausdorff upper
        bound for the two piecewise-linear surfaces. This does NOT prove absence
        of self-intersection, preserve topology, or bound CAD tessellation error.
        """
        return {
            "nonnegative": bool(np.all(self.weights >= 0)),
            "partition_error": float(np.max(abs(self.weights.sum(axis=1) - 1))),
            "unit_normal_error": float(
                np.max(abs(np.linalg.norm(self.normals, axis=1) - 1))
            ),
            "bound_mm": float(tolerance),
        }

    def diagnostics(self, coefficients):
        mesh = self.mesh(coefficients)
        v = np.asarray(mesh.vertices)
        f = self.faces
        cross0 = np.cross(
            self.vertices[f[:, 1]] - self.vertices[f[:, 0]],
            self.vertices[f[:, 2]] - self.vertices[f[:, 0]],
        )
        cross = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        signed = np.einsum("ij,ij->i", cross, cross0)
        return {
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "nonpositive_target_normal_faces": int(np.sum(signed <= 0)),
            "min_face_area_mm2": float(np.min(mesh.area_faces)),
            "volume_mm3": float(mesh.volume),
            "self_intersection_checked": False,
        }

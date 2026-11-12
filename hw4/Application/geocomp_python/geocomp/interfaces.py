"""Structural interface shared by scalar-surface and vector-tube geometries."""

from typing import Protocol

import jax
import numpy as np
import trimesh


class GeometryBasis(Protocol):
    """A coefficient-to-mesh map plus its conservative displacement bound.

    `weights` and the optional extra serialization arrays are basis-specific;
    callers must use the methods, not assume every model is a scalar normal
    offset. Both existing implementations are affine in their coefficients.
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    anchors: np.ndarray
    anchor_indices: np.ndarray

    def displacement(self, coefficients) -> jax.Array: ...
    def vertices_jax(self, coefficients) -> jax.Array: ...
    def max_displacement_batch(self, coefficients) -> jax.Array: ...
    def mesh(self, coefficients) -> trimesh.Trimesh: ...
    def diagnostics(self, coefficients) -> dict: ...
    def guarantee(self, tolerance: float) -> dict: ...

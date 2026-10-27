"""Fixed material landmarks and non-overlapping actuation channels."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np


def _indices(indices, n: int, name: str) -> np.ndarray:
    raw = np.asarray(indices)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a nonempty, one-dimensional integer array.")
    if np.any(raw < 0) or np.any(raw >= n) or len(np.unique(raw)) != len(raw):
        raise ValueError(f"{name} contains duplicate or out-of-range indices.")
    return raw.astype(np.int32)


@dataclass(frozen=True)
class Part:
    name: str
    indices: np.ndarray
    weights: np.ndarray


class Parts:
    """A named landmark is a fixed weighted average of material vertices.

    Selections happen once on the reference mesh, NEVER on the deformed mesh.
    This avoids hidden, nondifferentiable changes to body-part membership.
    """
    def __init__(self, points, masses=None):
        self.points = np.asarray(points, dtype=float).copy()
        if self.points.ndim != 2 or self.points.shape[1] != 3 or not np.all(np.isfinite(self.points)):
            raise ValueError("points must be a finite (N,3) array.")
        self.masses = np.ones(len(self.points)) if masses is None else np.asarray(masses, dtype=float).copy()
        if self.masses.shape != (len(self.points),) or np.any(self.masses < 0) or not np.all(np.isfinite(self.masses)):
            raise ValueError("masses must be finite, nonnegative, and match the vertices.")
        self._parts: list[Part] = []

    @property
    def names(self):
        return tuple(p.name for p in self._parts)

    def add(self, name: str, indices: Iterable[int], *, weighting: str = "mass", weights=None):
        if not isinstance(name, str) or not name or name in self.names:
            raise ValueError("Each part needs a nonempty, unique name.")
        ids = _indices(list(indices), len(self.points), "part indices")
        if weights is None:
            if weighting not in ("mass", "uniform"):
                raise ValueError("weighting must be 'mass' or 'uniform'.")
            w = self.masses[ids] if weighting == "mass" else np.ones(len(ids))
        else:
            w = np.asarray(weights, dtype=float)
        if w.shape != ids.shape or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("Landmark weights must match the indices, be nonnegative, and have positive sum.")
        self._parts.append(Part(name, ids.copy(), np.asarray(w / w.sum(), dtype=np.float32)))
        return self

    def nearest(self, name: str, point, *, count: int = 8, weighting: str = "mass"):
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("point must be a finite three-vector.")
        if isinstance(count, bool) or int(count) != count or not 1 <= count <= len(self.points):
            raise ValueError("count must be a positive integer no larger than the vertex count.")
        ids = np.argsort(np.sum((self.points - point) ** 2, axis=1), kind="stable")[:int(count)]
        return self.add(name, ids, weighting=weighting)

    def box(self, name: str, lower, upper, *, weighting: str = "mass"):
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
            raise ValueError("box bounds must be ordered three-vectors without NaNs.")
        ids = np.flatnonzero(np.all((self.points >= lower) & (self.points <= upper), axis=1))
        return self.add(name, ids, weighting=weighting)

    def reference(self, name: str):
        for part in self._parts:
            if part.name == name:
                return (self.points[part.indices] * part.weights[:, None]).sum(axis=0)
        raise KeyError(name)

    def packed(self):
        if not self._parts:
            raise ValueError("Register at least one body landmark.")
        offsets = np.r_[0, np.cumsum([len(p.indices) for p in self._parts])].astype(np.int32)
        return offsets, np.concatenate([p.indices for p in self._parts]), np.concatenate([p.weights for p in self._parts])

    def to_dict(self):
        return {p.name: {"indices": p.indices.tolist(), "weights": p.weights.tolist()} for p in self._parts}


@dataclass(frozen=True)
class ActuatorMap:
    """Each tetrahedron takes exactly one channel's activation; -1 means passive.

    Thus U in [lower,upper] directly bounds every active tetrahedron. There is
    no clipping, tanh, neural controller, or hidden sum of overlapping inputs.
    """
    tet_to_channel: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self):
        raw = np.asarray(self.tet_to_channel)
        if raw.ndim != 1 or not raw.size or raw.dtype.kind not in "iu":
            raise ValueError("tet_to_channel must be a nonempty integer vector.")
        if not self.names or len(set(self.names)) != len(self.names) or any(not isinstance(n, str) or not n for n in self.names):
            raise ValueError("Actuator names must be nonempty and unique.")
        if np.any(raw < -1) or np.any(raw >= len(self.names)):
            raise ValueError("Each tetrahedron must map to -1 or a valid channel.")
        if set(raw[raw >= 0].tolist()) != set(range(len(self.names))):
            raise ValueError("Every named channel must actuate at least one tetrahedron.")
        labels = raw.astype(np.int32, copy=True)
        labels.setflags(write=False)
        object.__setattr__(self, "tet_to_channel", labels)
        object.__setattr__(self, "names", tuple(self.names))

    @classmethod
    def per_tet(cls, tet_count: int):
        if isinstance(tet_count, bool) or int(tet_count) != tet_count or tet_count < 1:
            raise ValueError("tet_count must be a positive integer.")
        return cls(np.arange(tet_count, dtype=np.int32), tuple(f"tet_{i}" for i in range(tet_count)))

    @classmethod
    def grid(cls, points, tetrahedra, bins=(2, 2, 2)):
        """Educational spatial grouping, not anatomically labeled muscles."""
        points, tets = np.asarray(points, dtype=float), np.asarray(tetrahedra)
        b = np.asarray(bins)
        if b.shape != (3,) or b.dtype.kind not in "iu" or np.any(b < 1):
            raise ValueError("bins must be three positive integers.")
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("points must be a finite (N,3) array.")
        if tets.ndim != 2 or tets.shape[1] != 4 or not len(tets) or tets.dtype.kind not in "iu" or np.any(tets < 0) or np.any(tets >= len(points)):
            raise ValueError("tetrahedra must contain valid (T,4) vertex indices.")
        centers = points[tets].mean(axis=1)
        span = np.ptp(centers, axis=0)
        normalized = (centers - centers.min(axis=0)) / np.where(span > 0, span, 1)
        ijk = np.minimum((normalized * b).astype(int), b - 1)
        codes = np.ravel_multi_index(ijk.T, tuple(b))
        used, labels = np.unique(codes, return_inverse=True)
        names = tuple("region_" + "_".join(map(str, np.unravel_index(code, tuple(b)))) for code in used)
        return cls(labels.astype(np.int32), names)

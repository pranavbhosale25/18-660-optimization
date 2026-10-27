"""Optional playback export; no rendering or USD package required for optimization."""
from pathlib import Path
import numpy as np


def boundary_triangles(points, tetrahedra):
    points, tets = np.asarray(points), np.asarray(tetrahedra)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points must be a finite (N,3) array.")
    if tets.ndim != 2 or tets.shape[1] != 4 or tets.dtype.kind not in "iu" or not len(tets) or np.any(tets < 0) or np.any(tets >= len(points)):
        raise ValueError("tetrahedra must be a nonempty (T,4) index array.")
    faces = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
                            tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    opposite = np.concatenate([tets[:, 3], tets[:, 2], tets[:, 1], tets[:, 0]])
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    inward = np.einsum("ij,ij->i", np.cross(b-a, c-a), points[opposite]-a) > 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Non-manifold tetrahedral mesh: a face belongs to more than two tetrahedra.")
    return faces[first[counts == 1]].astype(np.int32)


def replay_to_usd(replay_file, output_file):
    try:
        from pxr import Usd, UsdGeom, Vt
    except ImportError as exc:
        raise ImportError("USD export needs usd-core.") from exc
    with np.load(replay_file, allow_pickle=False) as data:
        particles, tets, times = data["particles"], data["tetrahedra"], data["times"]
    if len(particles) != len(times) or len(times) < 2 or not np.all(np.isfinite(particles)):
        raise ValueError("Invalid replay samples.")
    delta = np.diff(times)
    if np.any(delta <= 0) or not np.allclose(delta, delta[0]):
        raise ValueError("USD export expects uniformly spaced, increasing knot times.")
    faces = boundary_triangles(particles[0], tets)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_file))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(float(1 / delta[0]))
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(times) - 1)
    mesh = UsdGeom.Mesh.Define(stage, "/bear")
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr().Set(faces.ravel().tolist())
    for knot, points in enumerate(particles):
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(points, dtype=np.float32)), Usd.TimeCode(knot))
    stage.GetRootLayer().Save()
    return output_file

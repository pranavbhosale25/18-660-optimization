# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Newton's diffsim bear with its neural controller replaced by direct controls.

Reference: newton-physics/newton, examples/diffsim/example_diffsim_bear.py,
commit 516aa47f261e98d8fe0d7a52e609e950b287227b. Material/contact constants are
kept from that example. Newton is an external dependency; the teaching
archive includes the bear asset at ``assets/bear.usd``.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import inspect
import json
import numpy as np
import warp as wp
from .geometry import ActuatorMap, Parts
from .layout import Layout
from .warp_rollout import WarpParticleRollout


def _import_newton():
    try:
        import newton
    except ImportError as exc:
        raise ImportError(
            "The real bear requires Newton 1.5.2 compatible with Warp 1.16.0. "
            "Install this project with the 'bear' extra; see docs/INSTALL.md."
        ) from exc
    for name in ("ModelBuilder", "CollisionPipeline", "solvers"):
        if not hasattr(newton, name):
            raise ImportError(f"Installed 'newton' lacks {name}; install newton-physics/newton.")
    return newton


@dataclass(frozen=True)
class BearPhysics:
    translation: tuple[float, float, float] = (0., 0., 0.5)
    density: float = 1.0
    k_mu: float = 2000.0
    k_lambda: float = 2000.0
    k_damp: float = 2.0
    ground_ke: float = 2000.0
    ground_kf: float = 10.0
    ground_kd: float = 0.1
    ground_mu: float = 0.7

    def __post_init__(self):
        translation = np.asarray(self.translation, dtype=float)
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError("translation must be a finite three-vector.")
        for name in ("density", "k_mu", "k_lambda", "ground_ke"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        for name in ("k_damp", "ground_kf", "ground_kd", "ground_mu"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite.")


class BearScene:
    def __init__(self, model, *, asset: str, device: str, physics: BearPhysics):
        self.model = model
        self.asset = asset
        self.device = device
        self.physics = physics
        self.vertices = model.particle_q.numpy().copy()
        self.tetrahedra = model.tet_indices.numpy().reshape(-1, 4).copy()
        self.masses = model.particle_mass.numpy().copy()
        self.parts = Parts(self.vertices, self.masses)
        self.parts.add("com", np.arange(len(self.vertices)))
        self._rollout_created = False

    @classmethod
    def load(cls, asset: str | Path | None = None, *, device="cpu", physics: BearPhysics | None = None):
        newton = _import_newton()
        physics = BearPhysics() if physics is None else physics
        if asset is None:
            # Do not import newton.examples just to find the asset: that can
            # unnecessarily pull in viewer extras on headless installations.
            asset = Path(newton.__file__).resolve().parent / "examples" / "assets" / "bear.usd"
        asset = Path(asset).expanduser().resolve()
        if not asset.is_file():
            raise FileNotFoundError(f"Bear asset missing: {asset}. Supply asset='/path/to/bear.usd'.")
        if asset.suffix.lower() == ".npz":
            # An explicitly supplied export avoids a USD dependency; never an
            # automatic substitution for the actual bear mesh.
            with np.load(asset, allow_pickle=False) as data:
                points = np.asarray(data["points"], dtype=np.float32)
                indices = np.asarray(data["tetrahedra"], dtype=np.int32).reshape(-1)
        else:
            try:
                from pxr import Usd, UsdGeom
            except ImportError as exc:
                raise ImportError("Loading bear.usd requires the usd-core package (the 'pxr' module).") from exc
            stage = Usd.Stage.Open(str(asset))
            if stage is None:
                raise ValueError(f"Cannot open USD stage {asset}.")
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/root/bear/bear"))
            if not mesh:
                raise ValueError("Expected the Newton bear mesh at /root/bear/bear.")
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
            attr = mesh.GetPrim().GetAttribute("tetraIndices")
            if not attr or attr.Get() is None:
                raise ValueError("The bear mesh has no tetraIndices attribute.")
            indices = np.asarray(attr.Get(), dtype=np.int32).reshape(-1)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("Invalid bear vertex array.")
        if not indices.size or indices.size % 4 or np.any(indices < 0) or np.any(indices >= len(points)):
            raise ValueError("Invalid bear tetrahedron indices.")
        wp.init()
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            builder.add_soft_mesh(
                pos=wp.vec3(*physics.translation), rot=wp.quat_identity(), scale=1.0,
                vel=wp.vec3(0.0), vertices=[wp.vec3(*p) for p in points], indices=indices.tolist(),
                density=physics.density, k_mu=physics.k_mu, k_lambda=physics.k_lambda, k_damp=physics.k_damp,
                tri_ke=0.0, tri_ka=1.0e-8, tri_kd=0.0, tri_drag=0.0, tri_lift=0.0,
            )
            builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(
                ke=physics.ground_ke, kf=physics.ground_kf,
                kd=physics.ground_kd, mu=physics.ground_mu,
            ))
            model = builder.finalize(requires_grad=True)
            model.soft_contact_ke = physics.ground_ke
            model.soft_contact_kf = physics.ground_kf
            model.soft_contact_kd = physics.ground_kd
            model.soft_contact_mu = physics.ground_mu
            # As in the official example: particle-particle contacts do not
            # provide correct gradients in this solver.
            model.particle_grid = None
        return cls(model, asset=str(asset), device=str(device), physics=physics)

    def make_rollout(self, *, horizon=20, control_dt=1.0 / 60.0, substeps=80,
                     actuators: ActuatorMap | None = None, max_state_memory_mb=4096,
                     capture_graph=False):
        if self._rollout_created:
            raise RuntimeError("A BearScene owns one rollout; load another scene for an independent tape.")
        if actuators is None:
            actuators = ActuatorMap.grid(self.vertices, self.tetrahedra)
        if len(actuators.tet_to_channel) != self.model.tet_count:
            raise ValueError("Actuator map must match the FINALIZED model's tetrahedron order/count.")
        rollout = NewtonBearRollout(self, actuators=actuators, horizon=horizon,
                                    control_dt=control_dt, substeps=substeps,
                                    max_state_memory_mb=max_state_memory_mb,
                                    capture_graph=capture_graph)
        self._rollout_created = True
        return rollout

    def export_mesh(self, destination):
        """Export WORLD reference vertices for inspecting selections (not loadable as local mesh)."""
        np.savez_compressed(destination, vertices_world=self.vertices, tetrahedra=self.tetrahedra,
                            masses=self.masses)


class NewtonBearRollout(WarpParticleRollout):
    def __init__(self, scene: BearScene, *, actuators, horizon, control_dt, substeps,
                 max_state_memory_mb, capture_graph=False):
        newton = _import_newton()
        self.scene = scene
        self.model = scene.model
        self.actuator_map = actuators
        self.part_definitions = scene.parts.to_dict()
        layout = Layout(horizon, actuators.names, scene.parts.names, control_dt)
        super().__init__(layout=layout, initial_q=scene.vertices, initial_qd=self.model.particle_qd.numpy(),
                         parts=scene.parts, actuators=actuators, substeps=substeps, device=scene.device,
                         max_state_memory_mb=max_state_memory_mb, capture_graph=capture_graph)
        with wp.ScopedDevice(self.device):
            self.controls = [self.model.control() for _ in range(horizon)]
            for k, control in enumerate(self.controls):
                control.tet_activations = self.activations[k]
            # One-shot contact candidates are valid ONLY for this static infinite
            # ground plane. Include every particle, then let the solver evaluate
            # its current penetration/contact response at each substep.
            gap = max(10.0, float(np.max(np.abs(self.q0[:, 2]))) + 2.0)
            parameters = inspect.signature(newton.CollisionPipeline).parameters
            kwargs = dict(broad_phase="explicit", requires_grad=True)
            if "soft_contact_gap" in parameters:
                kwargs["soft_contact_gap"] = gap
            elif "soft_contact_margin" in parameters:
                kwargs["soft_contact_margin"] = gap
            else:
                raise RuntimeError("Unsupported Newton collision API: expected soft_contact_gap or soft_contact_margin.")
            self.collision_pipeline = newton.CollisionPipeline(self.model, **kwargs)
            self.contacts = self.collision_pipeline.contacts()
            self.collision_pipeline.collide(self.states[0], self.contacts)
            count = int(self.contacts.soft_contact_count.numpy()[0])
            if count != self.model.particle_count:
                raise RuntimeError(
                    f"Expected one static-plane candidate per particle, got {count} for "
                    f"{self.model.particle_count}. Check collision capacity and the Newton API. "
                    "Reusing an incomplete contact set would give incorrect trajectories."
                )
            self.solver = newton.solvers.SolverSemiImplicit(self.model, enable_tri_contact=False)

    def _new_state(self):
        return self.model.state(requires_grad=True)

    def _advance(self, substep_index, control_interval):
        self.states[substep_index].clear_forces()
        self.solver.step(self.states[substep_index], self.states[substep_index + 1],
                         self.controls[control_interval], self.contacts, self.sim_dt)

    def contact_diagnostics(self):
        """Inspect the last trajectory against the static z=0 plane at every substep.

        This is an optional post-processing operation, not part of the NLP or
        its derivative computation. Negative gap means the penalty is active.
        """
        if not self._has_rollout:
            raise RuntimeError("Call rollout before requesting contact diagnostics.")
        radii = self.model.particle_radius.numpy()
        gaps = []
        counts = []
        for state in self.states:
            gap = state.particle_q.numpy()[:, 2] - radii
            gaps.append(float(np.min(gap)))
            counts.append(int(np.count_nonzero(gap < 0.)))
        first = next((i for i, count in enumerate(counts[:-1]) if count), None)
        return {"initial_minimum_gap_m": gaps[0], "final_minimum_gap_m": gaps[-1],
                "minimum_gap_m": min(gaps), "maximum_contacting_particles": max(counts),
                "first_contact_time_s": None if first is None else first * self.sim_dt,
                "substeps_with_contact": sum(count > 0 for count in counts[:-1])}

    def save_replay(self, destination):
        """Save knot-level particles plus material landmark and actuator identities."""
        from dataclasses import asdict
        np.savez_compressed(
            destination, particles=self.particle_positions(), tetrahedra=self.scene.tetrahedra,
            tet_to_channel=self.actuator_map.tet_to_channel, times=self.layout.times,
            metadata=np.array(json.dumps({"asset": self.scene.asset, "physics": asdict(self.scene.physics),
                                          "parts": self.part_definitions,
                                          "actuators": self.layout.actuator_names,
                                          "substeps": self.substeps})),
        )

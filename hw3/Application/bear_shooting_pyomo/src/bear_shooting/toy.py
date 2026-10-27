"""Small nonlinear Warp-only test fixture. This is NOT a bear approximation.

Used to execute the actual IPOPT bridge and tape tests without Newton/USD.
The bear implementation is in newton_bear.py and never falls back to this.
"""
from __future__ import annotations
import numpy as np
import warp as wp
from .geometry import ActuatorMap, Parts
from .layout import Layout
from .warp_rollout import WarpParticleRollout


@wp.kernel
def _toy_step(
    q: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
    rest: wp.array(dtype=wp.vec3), a: wp.array(dtype=wp.float32), dt: float,
    q_next: wp.array(dtype=wp.vec3), v_next: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    u = a[i]
    force = wp.vec3(3.0 * u, 1.5 * u * u, 0.5 * wp.sin(u))
    acceleration = force - 1.2 * (q[i] - rest[i]) - 0.4 * v[i]
    vn = v[i] + dt * acceleration
    v_next[i] = vn
    q_next[i] = q[i] + dt * vn


class _State:
    def __init__(self, q, qd, device):
        self.particle_q = wp.array(q, dtype=wp.vec3, device=device, requires_grad=True)
        self.particle_qd = wp.array(qd, dtype=wp.vec3, device=device, requires_grad=True)


class ToyRollout(WarpParticleRollout):
    def __init__(self, horizon=12, control_dt=0.1, substeps=2, device="cpu"):
        q = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32)
        parts = Parts(q).add("tip", [0]).add("base", [1]).add("side", [2]).add("com", [0, 1, 2])
        actuators = ActuatorMap(np.arange(3, dtype=np.int32), ("tip_drive", "base_drive", "side_drive"))
        layout = Layout(horizon, actuators.names, parts.names, control_dt)
        super().__init__(layout=layout, initial_q=q, initial_qd=np.zeros_like(q), parts=parts,
                         actuators=actuators, substeps=substeps, device=device)
        self.rest_wp = wp.array(q, dtype=wp.vec3, device=device)

    def _new_state(self):
        return _State(self.q0, self.qd0, self.device)

    def _advance(self, substep_index, control_interval):
        left, right = self.states[substep_index:substep_index + 2]
        wp.launch(_toy_step, dim=len(self.q0),
                  inputs=[left.particle_q, left.particle_qd, self.rest_wp,
                          self.activations[control_interval], self.sim_dt],
                  outputs=[right.particle_q, right.particle_qd], device=self.device)

"""Warp rollout/tape management shared by Newton and the small test system.

Important: every substep owns distinct state arrays, every control interval
owns distinct tetrahedron activations, and no primal is modified between a
forward pass and its reverse passes. A two-state ping-pong buffer is NOT safe
for this tape-based full-horizon reverse mode.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import warp as wp
from .geometry import ActuatorMap, Parts
from .layout import Layout, vector


@wp.kernel
def expand_activations(
    controls: wp.array2d(dtype=wp.float32),
    labels: wp.array(dtype=wp.int32),
    interval: int,
    activations: wp.array(dtype=wp.float32),
):
    tet = wp.tid()
    channel = labels[tet]
    value = float(0.0)
    if channel >= 0:
        value = controls[interval, channel]
    activations[tet] = value


@wp.kernel
def observe_parts(
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    offsets: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    weights: wp.array(dtype=wp.float32),
    knot: int,
    part_count: int,
    observations: wp.array(dtype=wp.float32),
):
    part = wp.tid()
    p = wp.vec3(0.0)
    v = wp.vec3(0.0)
    for j in range(offsets[part], offsets[part + 1]):
        p = p + weights[j] * q[indices[j]]
        v = v + weights[j] * qd[indices[j]]
    base = (knot * part_count + part) * 6
    observations[base] = p[0]
    observations[base + 1] = p[1]
    observations[base + 2] = p[2]
    observations[base + 3] = v[0]
    observations[base + 4] = v[1]
    observations[base + 5] = v[2]


class WarpParticleRollout(ABC):
    """Base for differentiable particle simulators; observations use float32.

    Single shooting stores O(horizon * substeps * particle_count) states.
    max_state_memory_mb checks a LOWER BOUND, not a full peak-memory estimate.
    """
    def __init__(
        self, *, layout: Layout, initial_q, initial_qd, parts: Parts,
        actuators: ActuatorMap, substeps: int, device: str = "cpu",
        max_state_memory_mb: float | None = 4096, capture_graph: bool = False,
    ):
        if isinstance(substeps, bool) or int(substeps) != substeps or substeps < 1:
            raise ValueError("substeps must be a positive integer.")
        if layout.actuator_names != actuators.names or layout.part_names != parts.names:
            raise ValueError("Layout names must match the actuator map and parts.")
        if max_state_memory_mb is not None and (not np.isfinite(max_state_memory_mb) or max_state_memory_mb <= 0):
            raise ValueError("max_state_memory_mb must be positive and finite, or None.")
        wp.init()
        self.layout = layout
        self.substeps = int(substeps)
        self.device = wp.get_device(device)
        if capture_graph and not self.device.is_cuda:
            raise ValueError("CUDA graph capture requires a CUDA rollout device.")
        self.capture_graph = bool(capture_graph)
        self.sim_dt = layout.control_dt / self.substeps
        self.q0 = np.asarray(initial_q, dtype=np.float32).copy()
        self.qd0 = np.asarray(initial_qd, dtype=np.float32).copy()
        if self.q0.shape != (len(parts.points), 3) or self.qd0.shape != self.q0.shape:
            raise ValueError("Initial positions/velocities must match the parts' vertex array.")
        if not np.all(np.isfinite(self.q0)) or not np.all(np.isfinite(self.qd0)):
            raise ValueError("Initial state must be finite.")
        state_count = layout.horizon * self.substeps + 1
        # q, qd, f and their adjoints, three float32 components each.
        self.state_memory_lower_bound_mb = state_count * len(self.q0) * 3 * 4 * 6 / 2**20
        if max_state_memory_mb is not None and self.state_memory_lower_bound_mb > max_state_memory_mb:
            raise MemoryError(
                f"State/adjoint storage alone needs at least {self.state_memory_lower_bound_mb:.1f} MiB. "
                "Shorten the horizon or increase the explicitly supplied memory budget. "
                "Do not reduce physics substeps merely to fit memory."
            )
        with wp.ScopedDevice(self.device):
            self.states = [self._new_state() for _ in range(state_count)]
            self.controls_wp = wp.zeros(layout.control_shape, dtype=wp.float32, requires_grad=True)
            self.activations = [wp.zeros(len(actuators.tet_to_channel), dtype=wp.float32, requires_grad=True)
                                for _ in range(layout.horizon)]
            self.labels_wp = wp.array(actuators.tet_to_channel, dtype=wp.int32)
            offsets, indices, weights = parts.packed()
            self.offsets_wp = wp.array(offsets, dtype=wp.int32)
            self.indices_wp = wp.array(indices, dtype=wp.int32)
            self.weights_wp = wp.array(weights, dtype=wp.float32)
            self.observations_wp = wp.zeros(layout.n_observations, dtype=wp.float32, requires_grad=True)
            self.seed_wp = wp.zeros_like(self.observations_wp, requires_grad=False)
        self.tape = wp.Tape()
        self._forward_graph = None
        self._forward_warmed = False
        self._backward_graph = None
        self._backward_warmed = False
        self._has_rollout = False
        self.forward_count = 0
        self.reverse_count = 0

    @abstractmethod
    def _new_state(self):
        """Return a state with distinct particle_q, particle_qd, and forces."""

    @abstractmethod
    def _advance(self, substep_index: int, control_interval: int):
        """Advance states[i] -> states[i+1] with activations[k] and sim_dt."""

    def _observe(self, knot: int):
        state = self.states[knot * self.substeps]
        wp.launch(
            observe_parts, dim=len(self.layout.part_names),
            inputs=[state.particle_q, state.particle_qd, self.offsets_wp,
                    self.indices_wp, self.weights_wp, knot, len(self.layout.part_names)],
            outputs=[self.observations_wp], device=self.device,
        )

    def _record_forward(self):
        """Launch and record one complete differentiable trajectory."""
        with self.tape:
            self._observe(0)
            for k in range(self.layout.horizon):
                wp.launch(
                    expand_activations,
                    dim=len(self.labels_wp),
                    inputs=[self.controls_wp, self.labels_wp, k],
                    outputs=[self.activations[k]],
                    device=self.device,
                )
                for j in range(self.substeps):
                    self._advance(k * self.substeps + j, k)
                self._observe(k + 1)

    def rollout(self, controls: np.ndarray) -> np.ndarray:
        controls = np.asarray(controls, dtype=np.float64)
        if controls.shape != self.layout.control_shape or not np.all(np.isfinite(controls)):
            raise ValueError(f"controls must be finite with shape {self.layout.control_shape}.")
        self._has_rollout = False
        with wp.ScopedDevice(self.device):
            if self._forward_graph is None:
                # Clear old adjoints/operations before changing any primal
                # arrays.  Once captured, the recorded tape must be retained.
                self.tape.reset()
            self.controls_wp.assign(controls.astype(np.float32))
            self.states[0].particle_q.assign(self.q0)
            self.states[0].particle_qd.assign(self.qd0)
            self.observations_wp.zero_()
            if self._forward_graph is not None:
                wp.capture_launch(self._forward_graph)
            else:
                # Reset old adjoints/operations BEFORE overwriting any primal
                # graph.  The first execution loads every kernel.  The second
                # can then be captured without compiling inside CUDA capture.
                if self.capture_graph and self._forward_warmed:
                    with wp.ScopedCapture() as capture:
                        self._record_forward()
                    self._forward_graph = capture.graph
                    self._backward_graph = None
                    self._backward_warmed = False
                    # Capture records launches but does not execute the graph.
                    wp.capture_launch(self._forward_graph)
                else:
                    self._record_forward()
                    self._forward_warmed = True
            values = self.observations_wp.numpy().astype(np.float64)
        self.forward_count += 1
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("Nonfinite simulation output: check dt, material stiffness, and actuation bounds.")
        self._has_rollout = True
        return values

    def vjp(self, observation_seed: np.ndarray) -> np.ndarray:
        if not self._has_rollout:
            raise RuntimeError("vjp requires a successful rollout first.")
        seed = vector(observation_seed, self.layout.n_observations, "observation seed")
        if not np.any(seed):
            return np.zeros(self.layout.control_shape)
        with wp.ScopedDevice(self.device):
            self.seed_wp.assign(seed.astype(np.float32))
            if self._backward_graph is not None:
                wp.capture_launch(self._backward_graph)
            elif self.capture_graph and self._backward_warmed:
                with wp.ScopedCapture() as capture:
                    self._run_backward()
                self._backward_graph = capture.graph
                wp.capture_launch(self._backward_graph)
            else:
                self._run_backward()
                self._backward_warmed = True
            result = self.controls_wp.grad.numpy().astype(np.float64)
        self.reverse_count += 1
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Nonfinite Warp derivative: shorten the horizon and check the physics.")
        return result

    def _run_backward(self):
        """Clear prior adjoints and apply the current observation seed."""
        # zero() retains the recorded forward operations and primal values.
        self.tape.zero()
        self.controls_wp.grad.zero_()
        self.observations_wp.grad.zero_()
        self.tape.backward(grads={self.observations_wp: self.seed_wp})

    def particle_positions(self):
        """Copy the last trajectory at CONTROL KNOTS for export/replay diagnostics."""
        if not self._has_rollout:
            raise RuntimeError("Call rollout first.")
        return np.stack([self.states[k * self.substeps].particle_q.numpy()
                         for k in range(self.layout.horizon + 1)])

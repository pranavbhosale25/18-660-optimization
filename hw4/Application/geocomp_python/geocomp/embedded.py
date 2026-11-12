"""Embedded extrusion in a support bath: an explicit reduced-order surrogate.

This is NOT a Herschel-Bulkley flow solver or a calibrated bioprinting model.
A rheology-inspired mean bias, causal response, and fixed Gaussian residuals
form a convex local model. Shape error includes centerline displacement AND
filament radius, unlike the normal-offset DLP/CNC surface examples.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json

import jax.numpy as jnp
import numpy as np
from scipy.linalg import block_diag
import trimesh

from .problem import AffineGaussianBox
from .processes import ProcessInstance
from .sweep import TubeBasis, centerline_primitive


@dataclass(frozen=True)
class EmbeddedParameters:
    """All defaults are illustrative engineering-scale numbers, not measured fits.

    HB stress is a scalar drag scale only: tau_y + K * (speed/nozzle_radius)**n.
    Compliance maps that stress to a tangent-directed path lag. Transient buoyancy
    drift acts only over an assumed locally yielded/recovery interval, not during
    indefinite storage in an intact yield-stress bath. Radius is an equivalent
    circular radius; noncircular wake shapes and breakup are not represented.
    """

    radius_mm: float = 0.60
    travel_speed_mm_s: float = 5.0
    nozzle_outer_radius_mm: float = 0.45
    bath_yield_stress_pa: float = 8.0
    bath_consistency_pa_s_n: float = 4.0
    bath_flow_index: float = 0.5
    bath_recovery_time_s: float = 0.15
    drag_compliance_mm_pa: float = 0.008
    density_mismatch_kg_m3: float = 80.0
    transient_mobility_mm_s_pa: float = 0.8
    flow_calibration_gain: float = 1.18
    equivalent_radius_offset_mm: float = 0.035
    path_response_gain: float = 0.97
    causal_mixing_fraction: float = 0.10
    centerline_budget_mm: float = 0.18
    radius_budget_mm: float = 0.06
    residual_sd_fraction: float = 0.18
    bath_clearance_mm: float = 3.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        positive = (
            "radius_mm",
            "travel_speed_mm_s",
            "nozzle_outer_radius_mm",
            "bath_consistency_pa_s_n",
            "bath_flow_index",
            "bath_recovery_time_s",
            "flow_calibration_gain",
            "path_response_gain",
            "centerline_budget_mm",
            "radius_budget_mm",
            "residual_sd_fraction",
            "bath_clearance_mm",
        )
        if any(getattr(self, name) <= 0 for name in positive):
            raise ValueError(
                "Radii, speeds, response factors, budgets and noise scales must be positive"
            )
        if (
            self.bath_yield_stress_pa < 0
            or self.drag_compliance_mm_pa < 0
            or self.transient_mobility_mm_s_pa < 0
        ):
            raise ValueError("Yield stress, compliance and mobility cannot be negative")
        if not 0 <= self.causal_mixing_fraction < 1:
            raise ValueError("causal_mixing_fraction must lie in [0, 1)")
        if self.radius_budget_mm >= self.radius_mm:
            raise ValueError("The radius tolerance must be less than the target radius")

    @property
    def tolerance_mm(self) -> float:
        return self.centerline_budget_mm + self.radius_budget_mm

    @property
    def nominal_flow_mm3_s(self) -> float:
        return float(np.pi * self.radius_mm**2 * self.travel_speed_mm_s)


def _causal_kernel(control_s_mm: np.ndarray, memory_mm: float) -> np.ndarray:
    distance = control_s_mm[:, None] - control_s_mm[None, :]
    weights = np.where(
        distance >= -1e-12, np.exp(-np.maximum(distance, 0) / memory_mm), 0.0
    )
    return weights / weights.sum(axis=1, keepdims=True)


@dataclass
class EmbeddedPrintInstance(ProcessInstance):
    basis: TubeBasis
    parameters: EmbeddedParameters
    command_scales_mm: np.ndarray
    bath_bounds_mm: np.ndarray
    model_diagnostics: dict

    def command_coefficients(self, x):
        """Channel-specific scales: unlike DLP/CNC, not one uniform scalar scale."""
        return jnp.asarray(self.command_scales_mm) * jnp.asarray(x)

    def export_extras(self, out: Path, x, samples0, samples1) -> dict:
        """Write a nominal nozzle schedule and bath envelope; return viewer context."""
        out = Path(out)
        basis = self.basis
        coeff_command = self.command_coefficients(x)
        mean0 = self.mean_coefficients(jnp.zeros(self.problem.n_decisions))
        mean1 = self.mean_coefficients(x)
        zero = jnp.zeros(self.problem.n_features)
        path = lambda a: np.asarray(basis.centerline_jax(a))
        command_path = path(coeff_command)
        radius = np.asarray(basis.radii_jax(coeff_command))
        speed = self.parameters.travel_speed_mm_s
        # A circular-area, steady-extrusion command conversion, not calibrated
        # pump dynamics and not exact mass conservation for frozen-frame bends.
        flow = np.pi * radius**2 * speed
        arc = np.r_[
            0.0, np.cumsum(np.linalg.norm(np.diff(command_path, axis=0), axis=1))
        ]
        np.savetxt(
            out / "nozzle_path_optimized.csv",
            np.column_stack(
                (arc / speed, command_path, radius, np.full(len(arc), speed), flow)
            ),
            delimiter=",",
            comments="",
            header="time_s,x_mm,y_mm,z_mm,equivalent_radius_mm,speed_mm_s,nominal_flow_mm3_s",
        )
        np.savetxt(
            out / "nozzle_path_target.csv",
            basis.centerline,
            delimiter=",",
            comments="",
            header="x_mm,y_mm,z_mm",
        )
        bath = trimesh.creation.box(extents=np.diff(self.bath_bounds_mm, axis=0)[0])
        bath.apply_translation(self.bath_bounds_mm.mean(axis=0))
        bath.export(out / "bath_envelope.stl")
        records = {
            "parameters": asdict(self.parameters),
            **self.model_diagnostics,
            "bath_bounds_mm": self.bath_bounds_mm.tolist(),
            "command_channel_order": ["dx", "dy", "dz", "radius"],
            "command_scales_mm": self.command_scales_mm.tolist(),
            "schedule_is_machine_ready": False,
            "schedule_note": "Fixed-speed circular-area conversion; no G-code, pump dynamics or collision planning",
        }
        (out / "embedded_model.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
        return {
            "type": "embedded_extrusion",
            "bath_bounds": self.bath_bounds_mm.tolist(),
            "target_path": path(zero).tolist(),
            "command_path": command_path.tolist(),
            "mean0_path": path(mean0).tolist(),
            "mean1_path": path(mean1).tolist(),
            "samples0_path": [path(a).tolist() for a in samples0],
            "samples1_path": [path(a).tolist() for a in samples1],
            "command_radius_mm": radius.tolist(),
            "command_flow_mm3_s": flow.tolist(),
            "reference_frames_frozen": True,
            "centerline_budget_mm": self.parameters.centerline_budget_mm,
            "radius_budget_mm": self.parameters.radius_budget_mm,
        }


def formulate_embedded_problem(
    *,
    name: str,
    bias_mm: np.ndarray,
    command_response: np.ndarray,
    covariance_mm2: np.ndarray,
    basis: TubeBasis,
    parameters: EmbeddedParameters,
    bath_bounds_mm: np.ndarray,
    alpha: float,
) -> AffineGaussianBox:
    """Return the optimization representation for supplied embedded model data.

    Let ``s, k = basis.weights.shape`` and ``d = 4 * k``. The coefficient and
    decision order is channel-major: all ``dx`` controls, followed by all
    ``dy``, all ``dz``, and all radius controls. Every length is in millimeters.

    Args:
        name: Case identifier used in output files and diagnostics.
        bias_mm: Mean uncompensated output coefficients, shape ``(d,)``, in the
            channel order described above.
        command_response: Response from physical command coefficients to mean
            output coefficients, shape ``(d, d)``. Rows and columns both use the
            channel-major order.
        covariance_mm2: Symmetric positive-definite covariance of the output
            coefficients, shape ``(d, d)``, in square millimeters.
        basis: Tube geometry and interpolation data. Relevant arrays are
            ``weights`` with shape ``(s, k)``, ``feature_limits_mm`` with shape
            ``(d,)``, ``centerline`` with shape ``(s, 3)``, and
            ``control_parameters`` with shape ``(k,)``.
        parameters: Immutable :class:`EmbeddedParameters` containing scalar
            process, geometric-budget, and clearance data. Units are included
            in its field names.
        bath_bounds_mm: Bath-coordinate limits, shape ``(2, 3)``. Row zero is
            the lower XYZ corner and row one is the upper XYZ corner.
        alpha: Required joint success probability as a scalar strictly between
            zero and one.

    Returns:
        A complete :class:`AffineGaussianBox` with ``d`` output features and
        ``d`` dimensionless decisions. Its ``name`` and ``alpha`` fields must
        equal the supplied arguments. Any deterministic linear constraints use
        the standard ``A_ub`` and ``b_ub`` layouts documented by that class.
    """
    # STUDENT TODO: formulate and return the embedded-extrusion optimization problem.
    weights = basis.weights
    sections, controls = weights.shape
    position_scale = parameters.centerline_budget_mm / np.sqrt(3.0)
    radius_scale = parameters.radius_budget_mm
    limits = np.r_[np.full(3 * controls, position_scale),
                   np.full(controls, radius_scale)]

    # Channel-major ordering makes the four quadratic blocks identical.
    difference = np.diff(np.eye(controls), n=2, axis=0)
    block = (0.85 * weights.T @ weights / sections
             + 0.08 * np.eye(controls) / controls
             + 0.07 * difference.T @ difference / controls)
    Q = np.kron(np.eye(4), block) / 4.0
    bound = np.r_[np.full(3 * controls, 6.0),
                  np.full(controls, 0.45 * parameters.radius_mm / radius_scale)]

    # Reserve the largest allowed filament radius at every sampled centerline.
    path_map = np.kron(np.eye(3), position_scale * weights)
    path_map = np.pad(path_map, ((0, 0), (0, controls)))
    target = basis.centerline.T.reshape(-1)
    clearance = 1.45 * parameters.radius_mm
    lower = np.repeat(bath_bounds_mm[0] + clearance, sections)
    upper = np.repeat(bath_bounds_mm[1] - clearance, sections)
    return AffineGaussianBox(
        name=name, bias=bias_mm, response=command_response * limits[None, :],
        covariance=covariance_mm2, lower=-limits, upper=limits,
        Q=Q, c=np.zeros(4 * controls), bounds=np.column_stack((-bound, bound)),
        A_ub=np.vstack((path_map, -path_map)),
        b_ub=np.r_[upper - target, target - lower], alpha=alpha,
    )


def build_embedded_instance(
    shape: str = "helix",
    n_controls: int = 5,
    alpha: float = 0.95,
    n_sections: int = 81,
    n_sides: int = 16,
    parameters: EmbeddedParameters | None = None,
) -> EmbeddedPrintInstance:
    """Build a support-bath example with 4*n_controls continuous decisions.

    Decisions control nozzle centerline dx/dy/dz and equivalent circular radius.
    Volumetric flow rate and speed are not independent optimized variables: the
    nominal flow schedule is computed from the radius command afterward. Bath
    properties stay fixed.
    Residual uncertainty has both along-path and cross-channel correlations.
    """
    par = parameters or EmbeddedParameters()
    basis = TubeBasis.from_path(
        centerline_primitive(shape, n_sections),
        par.radius_mm,
        n_controls,
        n_sides,
        par.centerline_budget_mm,
        par.radius_budget_mm,
    )
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(basis.centerline, axis=0), axis=1))]
    control_s = np.interp(basis.control_parameters, np.linspace(0, 1, n_sections), s)
    shear_rate = par.travel_speed_mm_s / par.nozzle_outer_radius_mm
    drag_stress = (
        par.bath_yield_stress_pa
        + par.bath_consistency_pa_s_n * shear_rate**par.bath_flow_index
    )
    lag_mm = par.drag_compliance_mm_pa * drag_stress
    buoyancy_stress = par.density_mismatch_kg_m3 * 9.81 * par.radius_mm * 1e-3
    drift_z_mm = (
        -par.transient_mobility_mm_s_pa * buoyancy_stress * par.bath_recovery_time_s
    )
    t = np.linspace(0, 1, n_sections)
    # Illustrative height dependence of the local wake; not a constitutive law.
    lag_profile = lag_mm * (0.9 + 0.2 * t)
    raw_drift = -lag_profile[:, None] * basis.tangent
    raw_drift[:, 2] += drift_z_mm
    radius_gain = np.sqrt(par.flow_calibration_gain)
    raw_radius_bias = par.radius_mm * (
        radius_gain - 1
    ) + par.equivalent_radius_offset_mm * (0.85 + 0.30 * t)
    raw_fields = np.column_stack((raw_drift, raw_radius_bias))
    # The fitted field IS the reduced-order mean. The projection residual is
    # reported, not silently included in the Gaussian or in a physical guarantee.
    control_bias = np.linalg.lstsq(basis.weights, raw_fields, rcond=None)[0]
    bias = control_bias.T.reshape(-1)
    projection_error = basis.weights @ control_bias - raw_fields
    memory_mm = max(par.travel_speed_mm_s * par.bath_recovery_time_s, 0.12 * s[-1])
    causal = _causal_kernel(control_s, memory_mm)
    eta = par.causal_mixing_fraction
    smoother = (1 - eta) * np.eye(n_controls) + eta * causal
    G = block_diag(*([par.path_response_gain * smoother] * 3), radius_gain * smoother)
    # First-order coupling: a larger filament experiences more transient gravity
    # drift while the bath is locally yielded. It stays a fixed linear response.
    dz_dr = (
        -par.transient_mobility_mm_s_pa
        * par.density_mismatch_kg_m3
        * 9.81
        * 1e-3
        * par.bath_recovery_time_s
    )
    G[2 * n_controls : 3 * n_controls, 3 * n_controls :] = dz_dr * smoother
    limits = basis.feature_limits_mm.copy()
    u = basis.control_parameters
    along = 0.55 * np.eye(n_controls) + 0.45 * np.exp(
        -0.5 * ((u[:, None] - u[None, :]) / 0.30) ** 2
    )
    channels = np.array(
        [
            [1.0, 0.10, 0.08, 0.06],
            [0.10, 1.0, -0.05, 0.04],
            [0.08, -0.05, 1.0, 0.18],
            [0.06, 0.04, 0.18, 1.0],
        ]
    )
    sd = limits * par.residual_sd_fraction * np.tile(0.94 + 0.12 * u, 4)
    covariance = np.outer(sd, sd) * np.kron(channels, along)
    bath_bounds = np.vstack(
        (
            np.min(basis.vertices, axis=0) - par.bath_clearance_mm,
            np.max(basis.vertices, axis=0) + par.bath_clearance_mm,
        )
    )
    name = f"embedded_{shape}"
    diagnostics = {
        "calibrated": False,
        "drag_stress_scale_pa": float(drag_stress),
        "path_lag_scale_mm": float(lag_mm),
        "transient_vertical_drift_mm": float(drift_z_mm),
        "effective_radius_gain": float(radius_gain),
        "nominal_flow_mm3_s": par.nominal_flow_mm3_s,
        "causal_memory_length_mm": float(memory_mm),
        "mean_field_projection_error_max_mm": float(
            np.max(np.linalg.norm(projection_error[:, :3], axis=1))
        ),
        "projection_residual_included_in_probability_model": False,
        "rheology_model": "HB scalar stress scale + empirical compliance, NOT a fluid solution",
        "frame_model": "frozen target frames, affine small-deformation approximation",
    }
    problem = formulate_embedded_problem(
        name=name,
        bias_mm=bias,
        command_response=G,
        covariance_mm2=covariance,
        basis=basis,
        parameters=par,
        bath_bounds_mm=bath_bounds,
        alpha=alpha,
    )
    description = (
        "Embedded extrusion into a support bath: tangent-directed nozzle/wake lag, "
        "transient density drift, flow-to-radius bias and correlated path/radius errors. "
        "Optimizes 3-D nozzle offsets and filament radius with fixed target frames. "
        "Synthetic reduced-order model; not a fluid, gelation or collision simulation."
    )
    return EmbeddedPrintInstance(
        name,
        "embedded",
        shape,
        basis,
        problem,
        par.tolerance_mm,
        description,
        par,
        limits,
        bath_bounds,
        diagnostics,
    )

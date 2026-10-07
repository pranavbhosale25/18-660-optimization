"""Deterministic physics problems used by the optimizer benchmark.

This module owns problem construction only. Optimizer selection belongs in
``principled_benchmark_runner.py`` and measurement/reporting policy belongs in
``principled_benchmark.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib.resources import files
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
CaseSize: TypeAlias = Literal["small", "full"]
DTypeName: TypeAlias = Literal["float32", "float64"]
CaseKey: TypeAlias = Literal[
    "batched_relaxation",
    "implicit_dynamics",
    "occupancy_inversion",
    "implicit_diffusion",
    "shallow_arch",
]

CASE_KEYS: tuple[CaseKey, ...] = (
    "batched_relaxation",
    "implicit_dynamics",
    "occupancy_inversion",
    "implicit_diffusion",
    "shallow_arch",
)


@dataclass(frozen=True)
class PhysicsCase:
    """One objective and the neutral inputs needed to execute it."""

    key: CaseKey
    title: str
    size: CaseSize
    dtype: DTypeName
    fun: Callable[[Array], Array]
    x0: Array
    minimum_value: float
    robustness_starts: tuple[Array, ...]

    @property
    def dimension(self) -> int:
        return int(self.x0.size)


def make_cases(
    size: CaseSize = "small",
    dtype: DTypeName = "float64",
) -> tuple[PhysicsCase, ...]:
    """Build all five cases at the requested size and precision."""

    _validate_size(size)
    _validate_dtype(dtype)
    return tuple(make_case(key, size=size, dtype=dtype) for key in CASE_KEYS)


def make_case(
    key: CaseKey,
    size: CaseSize = "small",
    dtype: DTypeName = "float64",
) -> PhysicsCase:
    """Build one named case."""

    _validate_size(size)
    _validate_dtype(dtype)
    builders = {
        "batched_relaxation": make_batched_relaxation_case,
        "implicit_dynamics": make_implicit_dynamics_case,
        "occupancy_inversion": make_occupancy_inversion_case,
        "implicit_diffusion": make_implicit_diffusion_case,
        "shallow_arch": make_shallow_arch_case,
    }
    try:
        builder = builders[key]
    except KeyError as error:
        raise ValueError(f"Unknown physics case {key!r}.") from error
    return builder(size=size, dtype=dtype)


def make_batched_relaxation_case(
    size: CaseSize = "small",
    *,
    dtype: DTypeName = "float64",
    condition_number: float = 1.0e4,
) -> PhysicsCase:
    """Build the heterogeneous batched-relaxation problem."""

    _validate_size(size)
    array_dtype = _validate_dtype(dtype)
    if condition_number <= 1.0:
        raise ValueError("condition_number must be greater than one.")

    n = 64 if size == "small" else 1_024
    log_half_range = 0.5 * math.log(condition_number)
    stiffness = jnp.exp(
        jnp.linspace(-log_half_range, log_half_range, n, dtype=array_dtype)
    )
    index = jnp.arange(n, dtype=array_dtype)
    target = jnp.where(jnp.arange(n) % 2 == 0, 1.0, -1.0).astype(array_dtype)
    x0 = jnp.zeros(n, dtype=array_dtype)
    hardening = 0.5

    def objective(state: Array) -> Array:
        mismatch = state - target
        local_energy = stiffness * (0.5 * mismatch**2 + (hardening / 4.0) * mismatch**4)
        return jnp.mean(local_energy)

    starts = (
        0.10 * jnp.sin(0.31 * (index + 1.0)),
        -0.08 * jnp.cos(0.17 * (index + 1.0)),
        0.06 * jnp.sin(0.23 * (index + 1.0)) + 0.04 * jnp.cos(0.41 * (index + 1.0)),
    )
    return PhysicsCase(
        key="batched_relaxation",
        title="Batched heterogeneous internal-variable relaxation",
        size=size,
        dtype=dtype,
        fun=objective,
        x0=x0,
        minimum_value=_fixed_scalar("batched_relaxation_minimum_value"),
        robustness_starts=starts,
    )


def make_implicit_dynamics_case(
    size: CaseSize = "small",
    *,
    dtype: DTypeName = "float64",
) -> PhysicsCase:
    """Build the implicit nonlinear-dynamics substep."""

    _validate_size(size)
    array_dtype = _validate_dtype(dtype)
    n = 8
    dt = 0.25
    grounding_stiffness = 0.5
    coupling_stiffness = 2.0
    onsite_hardening = 5.0
    bond_hardening = 2.0

    index = jnp.arange(n, dtype=array_dtype)
    coordinate = (index + 1.0) / (n + 1.0)
    predictor = 0.45 * jnp.sin(jnp.pi * coordinate) - 0.15 * jnp.sin(
        2.0 * jnp.pi * coordinate
    )
    mass = 1.0 + 0.2 * jnp.cos(jnp.pi * coordinate)
    f = _fixed_array("dynamics_load", array_dtype)

    def objective(state: Array) -> Array:
        fixed_end_differences = jnp.concatenate(
            (state[:1], state[1:] - state[:-1], -state[-1:])
        )
        diff = state - predictor

        return (
            (0.5 / dt**2) * jnp.sum(mass * diff**2)
            + 0.5 * grounding_stiffness * jnp.sum(state**2)
            + 0.5 * coupling_stiffness * jnp.sum(fixed_end_differences**2)
            + (onsite_hardening / 4.0) * jnp.sum(state**4)
            + (bond_hardening / 4.0) * jnp.sum(fixed_end_differences**4)
            - jnp.dot(f, state)
        )

    starts = (
        predictor + 0.03 * jnp.sin(2.0 * jnp.pi * coordinate),
        predictor - 0.03 * jnp.sin(2.0 * jnp.pi * coordinate),
        predictor + 0.02 * jnp.cos(3.0 * jnp.pi * coordinate),
    )
    return PhysicsCase(
        key="implicit_dynamics",
        title="Small implicit nonlinear dynamics step",
        size=size,
        dtype=dtype,
        fun=objective,
        x0=predictor,
        minimum_value=_fixed_scalar("dynamics_minimum_value"),
        robustness_starts=starts,
    )


def make_occupancy_inversion_case(
    size: CaseSize = "small",
    *,
    dtype: DTypeName = "float64",
) -> PhysicsCase:
    """Build the deterministic two-state occupancy inverse problem."""

    _validate_size(size)
    array_dtype = _validate_dtype(dtype)
    n_parameters = 64 if size == "small" else 128
    n_observations = 4_096 if size == "small" else 16_384
    regularization = 0.05

    rng = np.random.default_rng(20_260_821)
    raw = rng.standard_normal((n_observations, n_parameters))
    column_scales = np.exp(np.linspace(-1.5, 1.5, n_parameters))
    sensitivity = jnp.asarray(
        raw * column_scales[None, :] / math.sqrt(n_parameters),
        dtype=array_dtype,
    )
    linear_load = _fixed_array(f"occupancy_linear_load_{size}", array_dtype)
    energy_offset = jnp.asarray(
        _fixed_scalar(f"occupancy_energy_offset_{size}"), dtype=array_dtype
    )
    index = jnp.arange(n_parameters, dtype=array_dtype)
    x0 = jnp.zeros(n_parameters, dtype=array_dtype)

    # Assignment notation: field_coefficients is x; sensitivity is A (row j
    # is a_j^T); regularization is lambda; linear_load is ell; and
    # energy_offset is C.
    def objective(field_coefficients: Array) -> Array:
        logits = sensitivity @ field_coefficients
        normalization_energy = jnp.mean(jnp.logaddexp(0.0, logits))
        fixed_contribution = jnp.dot(linear_load, field_coefficients)
        quadratic_regularization = 0.5 * regularization * jnp.sum(field_coefficients**2)

        return (
            normalization_energy
            - fixed_contribution
            + quadratic_regularization
            + energy_offset
        )

    starts = (
        0.12 * jnp.sin(0.19 * (index + 1.0)),
        -0.10 * jnp.cos(0.13 * (index + 1.0)),
        0.07 * jnp.sin(0.29 * (index + 1.0)) + 0.03 * jnp.cos(0.07 * (index + 1.0)),
    )
    return PhysicsCase(
        key="occupancy_inversion",
        title="Large two-state occupancy inverse calibration",
        size=size,
        dtype=dtype,
        fun=objective,
        x0=x0,
        minimum_value=_fixed_scalar(f"occupancy_minimum_value_{size}"),
        robustness_starts=starts,
    )


def make_implicit_diffusion_case(
    size: CaseSize = "small",
    *,
    dtype: DTypeName = "float64",
) -> PhysicsCase:
    """Build the matrix-free implicit diffusion problem."""

    _validate_size(size)
    array_dtype = _validate_dtype(dtype)
    n = 256 if size == "small" else 512
    diffusion_number = 20.0
    n_modes = 10
    mode_numbers_np = np.unique(
        np.rint(np.geomspace(1.0, n / 2.0, n_modes)).astype(int)
    )
    if mode_numbers_np.size != n_modes:
        raise AssertionError("The selected grid must produce ten distinct modes.")
    mode_numbers = jnp.asarray(mode_numbers_np, dtype=array_dtype)
    grid_index = jnp.arange(1, n + 1, dtype=array_dtype)
    coordinate = grid_index / (n + 1.0)
    eigenvectors = jnp.sin(jnp.pi * coordinate[:, None] * mode_numbers[None, :])
    coefficients = 1.0 / jnp.sqrt(mode_numbers)
    right_hand_side = eigenvectors @ coefficients

    # Assignment notation: temperature is u, diffusion_number is r, and
    # right_hand_side is b.
    def objective(temperature: Array) -> Array:
        fixed_end_differences = jnp.concatenate(
            (
                temperature[:1],
                temperature[1:] - temperature[:-1],
                -temperature[-1:],
            )
        )

        state_energy = 0.5 * jnp.dot(temperature, temperature)
        diffusion_energy = (
            0.5
            * diffusion_number
            * jnp.dot(fixed_end_differences, fixed_end_differences)
        )
        source_work = jnp.dot(right_hand_side, temperature)
        return state_energy + diffusion_energy - source_work

    x0 = jnp.zeros(n, dtype=array_dtype)
    starts = (
        0.08 * jnp.sin(jnp.pi * coordinate),
        -0.06 * jnp.cos(2.0 * jnp.pi * coordinate),
        0.05 * jnp.sin(5.0 * jnp.pi * coordinate)
        + 0.03 * jnp.cos(7.0 * jnp.pi * coordinate),
    )
    return PhysicsCase(
        key="implicit_diffusion",
        title="Matrix-free implicit diffusion step",
        size=size,
        dtype=dtype,
        fun=objective,
        x0=x0,
        minimum_value=_fixed_scalar(f"implicit_diffusion_minimum_value_{size}"),
        robustness_starts=starts,
    )


def make_shallow_arch_case(
    size: CaseSize = "small",
    *,
    dtype: DTypeName = "float64",
) -> PhysicsCase:
    """Build the post-critical shallow-arch problem."""

    _validate_size(size)
    array_dtype = _validate_dtype(dtype)
    postcriticality = 1.0
    imperfection_load = 0.42
    compatibility_stiffness = 30.0
    mode_coupling = 0.8

    def objective(modes: Array) -> Array:
        mode1, mode2 = modes
        flat_shape_unstable = -0.5 * postcriticality * mode1**2
        large_bending_resisted = 0.25 * mode1**4
        imperfection_favors_one_side = imperfection_load * mode1
        secondary_shape_tracks_main_shape = (
            0.5 * compatibility_stiffness * (mode2 - mode_coupling * mode1) ** 2
        )
        return (
            flat_shape_unstable
            + large_bending_resisted
            + imperfection_favors_one_side
            + secondary_shape_tracks_main_shape
        )

    x0 = jnp.zeros(2, dtype=array_dtype)
    starts = tuple(
        jnp.asarray(start, dtype=array_dtype)
        for start in (
            (0.08, 0.08),
            (-0.08, -0.08),
            (0.08, -0.08),
            (-0.08, 0.08),
        )
    )
    return PhysicsCase(
        key="shallow_arch",
        title="Post-critical shallow-arch equilibrium",
        size=size,
        dtype=dtype,
        fun=objective,
        x0=x0,
        minimum_value=_fixed_scalar("shallow_arch_minimum_value"),
        robustness_starts=starts,
    )


def with_start(case: PhysicsCase, x0: Array) -> PhysicsCase:
    """Return a copy of a case with a different initial point."""

    return replace(case, x0=jnp.asarray(x0, dtype=case.x0.dtype))


@lru_cache(maxsize=1)
def _fixed_inputs() -> dict[str, np.ndarray]:
    resource = files(__package__).joinpath("data", "principled_physics_inputs.npz")
    with resource.open("rb") as stream, np.load(stream, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _fixed_array(name: str, dtype: jnp.dtype) -> Array:
    return jnp.asarray(_fixed_inputs()[name], dtype=dtype)


def _fixed_scalar(name: str) -> float:
    return float(_fixed_inputs()[name])


def _validate_size(size: str) -> None:
    if size not in {"small", "full"}:
        raise ValueError("size must be 'small' or 'full'.")


def _validate_dtype(dtype: str) -> jnp.dtype:
    if dtype == "float32":
        return jnp.float32
    if dtype == "float64":
        if not jax.config.x64_enabled:
            raise RuntimeError(
                "float64 was requested, but JAX 64-bit mode is disabled. "
                "Start the benchmark with --dtype float64."
            )
        return jnp.float64
    raise ValueError("dtype must be 'float32' or 'float64'.")

"""Fast checks for the supplied, non-assignment locomotion helpers."""

import numpy as np
import pytest

from bear_shooting import Layout
from examples.bear_run import (
    create_parser,
    resolve_graph_capture,
    resolve_size_defaults,
    stage_ipopt_options,
    traveling_wave_controls,
    validate_continuation,
)
from scripts.verify_bear_run import create_parser as create_verification_parser


def test_traveling_wave_uses_x_bin_phase_and_interval_start_times():
    layout = Layout(
        4,
        ("region_0_0_0", "region_0_1_1", "region_1_0_0"),
        ("com",),
        0.25,
    )
    controls = traveling_wave_controls(layout, amplitude=0.2, frequency=1.0)
    times = np.arange(layout.horizon) * layout.control_dt
    expected_x0 = -0.2 * np.sin(2.0 * np.pi * times)
    expected_x1 = -0.2 * np.sin(2.0 * np.pi * times + 0.5 * np.pi)

    np.testing.assert_allclose(controls[:, 0], expected_x0, atol=1e-15)
    np.testing.assert_allclose(controls[:, 1], expected_x0, atol=1e-15)
    np.testing.assert_allclose(controls[:, 2], expected_x1, atol=1e-15)


def test_continuation_validation_and_distances():
    assert validate_continuation((0.25, 0.5, 1.0), 8.0) == (2.0, 4.0, 8.0)
    for fractions in ((), (0.0, 1.0), (0.7, 0.6, 1.0), (0.5, 0.9)):
        with pytest.raises(ValueError):
            validate_continuation(fractions, 8.0)


@pytest.mark.parametrize(
    "device,requested,expected",
    [
        ("cpu", None, False),
        ("cuda", None, True),
        ("CUDA:1", None, True),
        ("cuda:0", False, False),
        ("cpu", True, True),
    ],
)
def test_graph_capture_cli_resolution(device, requested, expected):
    assert resolve_graph_capture(device, requested) is expected


def test_stage_ipopt_options_use_cpu_appropriate_stopping_tolerance():
    cpu = stage_ipopt_options(17, device="cpu")
    cuda = stage_ipopt_options(17, device="cuda:0")

    assert cpu["max_iter"] == cuda["max_iter"] == 17
    assert cpu["acceptable_tol"] == 5.0e-4
    assert cpu["acceptable_iter"] == 3
    assert "acceptable_tol" not in cuda
    assert "acceptable_iter" not in cuda


def test_size_defaults():
    parser = create_parser()
    default = resolve_size_defaults(parser.parse_args([]))
    quick = resolve_size_defaults(parser.parse_args(["--size", "quick"]))
    small = resolve_size_defaults(parser.parse_args(["--size", "small"]))
    full = resolve_size_defaults(parser.parse_args(["--size", "full"]))

    assert (default.size, default.device, default.output.name) == (
        "quick",
        "cpu",
        "bear_run_quick.npz",
    )
    assert (quick.horizon, quick.distance_body_lengths, quick.continuation) == (
        15,
        0.075,
        (1.0,),
    )
    assert (small.horizon, small.distance_body_lengths, small.continuation) == (
        75,
        0.55,
        (0.95, 1.0),
    )
    assert (full.horizon, full.distance_body_lengths, full.continuation) == (
        150,
        1.0,
        (0.95, 1.0),
    )


def test_verification_defaults_follow_the_default_quick_cpu_run():
    args = create_verification_parser().parse_args([])
    assert args.result.name == "bear_run_quick.npz"
    assert args.device == "cpu"

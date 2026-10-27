"""Report local dependencies and optionally execute a small IPOPT probe."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys


REQUIRED_DISTRIBUTIONS = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pyomo", "pyomo"),
    ("warp", "warp-lang"),
    ("newton", "newton"),
    ("pxr", "usd-core"),
)
OPTIONAL_DISTRIBUTIONS = (("cyipopt", "cyipopt"),)


def _installed_version(module: str, distribution: str) -> str:
    if importlib.util.find_spec(module) is None:
        return "MISSING"
    try:
        return importlib.metadata.version(distribution)
    except Exception as exc:
        return f"ERROR: {exc}"


def collect_report(*, test_ipopt: bool = False) -> dict:
    """Collect environment details without hiding errors from the caller."""
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    errors = []

    for module, distribution in REQUIRED_DISTRIBUTIONS:
        value = _installed_version(module, distribution)
        report[module] = value
        if value == "MISSING" or value.startswith("ERROR:"):
            errors.append(f"required dependency {distribution}: {value}")

    for module, distribution in OPTIONAL_DISTRIBUTIONS:
        report[module] = _installed_version(module, distribution)

    if report["warp"] != "MISSING" and not report["warp"].startswith("ERROR:"):
        try:
            import warp as wp

            wp.init()
            report["warp_devices"] = [str(device) for device in wp.get_devices()]
        except Exception as exc:
            message = str(exc)
            report["warp_initialization"] = {"error": message}
            errors.append(f"Warp initialization: {message}")

    if test_ipopt:
        try:
            from bear_shooting import ShootingProblem, sumsqr
            from bear_shooting.ipopt_c import find_library
            from bear_shooting.toy import ToyRollout

            problem = ShootingProblem(ToyRollout(horizon=1))
            problem.add_objective("probe", sumsqr(problem.variables.u - 0.25))
            result = problem.solve()
            result.require_success()
            report["ipopt_probe"] = {
                "controls": result.controls.tolist(),
                "status": result.status,
                "library": find_library(),
                "modeling_layer": "Pyomo",
            }
        except Exception as exc:
            message = str(exc)
            report["ipopt_probe"] = {"error": message}
            errors.append(f"IPOPT probe: {message}")

    report["ok"] = not errors
    report["errors"] = errors
    return report


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-ipopt",
        action="store_true",
        help="solve a tiny problem through the default native IPOPT interface",
    )
    return parser


def main(argv=None) -> int:
    args = create_parser().parse_args(argv)
    report = collect_report(test_ipopt=args.test_ipopt)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

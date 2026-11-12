"""CLI runner: abstract benchmarks, manufacturing cases, mesh exports, dashboard."""

import argparse
from pathlib import Path
from dataclasses import asdict
import json
import sys
import platform
import numpy as np
import scipy
import jax
from scipy.stats import multivariate_normal
from .probability import GaussianBoxOracle
from .veinott import SolverOptions, solve_veinott
from .processes import build_instance, abstract_instances
from .embedded import build_embedded_instance
from .validation import validate_instance, validate_abstract
from .visualization import export_instance, write_dashboard


_MANUFACTURING_PROBABILITY_BUFFER = 0.004


def _positive_integer(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if count < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return count


def _qmc_point_count(value: str) -> int:
    """Parse a power-of-two QMC point count of at least two."""
    count = _positive_integer(value)
    if count < 2 or count & (count - 1):
        raise argparse.ArgumentTypeError(
            "must be a power of two greater than or equal to 2"
        )
    return count


def _probability_level(value: str) -> float:
    """Parse a finite probability strictly between zero and one."""
    try:
        alpha = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise argparse.ArgumentTypeError("must be strictly between zero and one")
    if alpha + _MANUFACTURING_PROBABILITY_BUFFER >= 1:
        raise argparse.ArgumentTypeError(
            "must be less than 0.996 so the solver's probability buffer remains below one"
        )
    return alpha


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2), encoding="utf-8")


def run_abstract(out, n_points=2048, n_samples=100000):
    out = Path(out) / "abstract"
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for p in abstract_instances():
        oracle = GaussianBoxOracle(p, n_points=n_points)
        result = solve_veinott(
            p,
            oracle,
            SolverOptions(
                absolute_gap=1e-9,
                relative_gap=1e-8,
                probability_buffer=0.0 if oracle.exact else 0.004,
            ),
        )
        validation = validate_abstract(p, result.x, n_samples=n_samples)
        record = {
            "name": p.name,
            "result": asdict(result),
            "validation": validation,
            "baseline_probability": oracle.probability(np.zeros(p.n_decisions)),
        }
        save_json(out / (p.name + ".json"), record)
        summaries.append(record)
        print(
            p.name,
            result.status,
            "iterations",
            result.iterations,
            "p",
            result.probability,
            flush=True,
        )
    save_json(out / "summary.json", summaries)
    write_abstract_figure(out, summaries[1])
    return summaries


def write_abstract_figure(out, record):
    """2-D exact chance region, iterates, and boundary supporting hyperplanes."""
    from scipy.stats import norm
    import plotly.graph_objects as go

    u = np.linspace(-1.75, 0.15, 160)
    v = np.linspace(-1.65, 0.2, 160)
    X, Y = np.meshgrid(u, v)
    P = (norm.cdf((1.0 - 0.9 - X) / 0.2) - norm.cdf((-1.0 - 0.9 - X) / 0.2)) * (
        norm.cdf((1.0 - 0.7 - Y) / 0.2) - norm.cdf((-1.0 - 0.7 - Y) / 0.2)
    )
    fig = go.Figure(
        go.Contour(
            x=u,
            y=v,
            z=P,
            contours={"start": 0.95, "end": 0.95, "size": 0.01, "coloring": "none"},
            showscale=False,
            name="95% joint boundary",
        )
    )
    cuts = record["result"]["chance_cuts"]
    for i, c in enumerate(cuts[:: max(1, len(cuts) // 7)]):
        a = np.asarray(c["normal"])
        rhs = c["rhs"]
        if abs(a[1]) > 1e-8:
            fig.add_trace(
                go.Scatter(
                    x=u,
                    y=(rhs - a[0] * u) / a[1],
                    mode="lines",
                    name="Boundary supporting cut" if i == 0 else None,
                    showlegend=i == 0,
                    line={"dash": "dot", "width": 1},
                )
            )
    points = np.array([h["candidate"] for h in record["result"]["history"]])
    fig.add_trace(
        go.Scatter(x=points[:, 0], y=points[:, 1], mode="markers", name="LP candidates")
    )
    x = record["result"]["x"]
    a = record["result"]["anchor"]
    fig.add_trace(
        go.Scatter(
            x=[x[0]],
            y=[x[1]],
            mode="markers",
            name="Veinott solution",
            marker={"size": 13, "symbol": "star"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[a[0]],
            y=[a[1]],
            mode="markers",
            name="Fixed interior anchor",
            marker={"size": 10, "symbol": "diamond"},
        )
    )
    fig.update_layout(
        title="Abstract exact 2-D problem: joint 95% boundary and Veinott cuts",
        xaxis={"title": "x₁", "range": [-1.75, 0.15]},
        yaxis={"title": "x₂", "range": [-1.65, 0.2]},
        height=680,
        legend={"orientation": "h"},
    )
    fig.write_html(
        str(Path(out) / "supporting_hyperplanes.html"), include_plotlyjs=True
    )


def run_manufacturing(
    name,
    out,
    n_points=2048,
    n_samples=100000,
    verbose=False,
    alpha=0.95,
):
    names = {
        "dlp_ellipsoid": ("dlp", "ellipsoid"),
        "dlp_sleeve": ("dlp", "sleeve"),
        "cnc_cylinder": ("cnc", "cylinder"),
        "cnc_pocket_plate": ("cnc", "pocket_plate"),
    }
    instance = (
        build_embedded_instance(name.removeprefix("embedded_"), alpha=alpha)
        if name.startswith("embedded_")
        else build_instance(*names[name], alpha=alpha)
    )
    p = instance.problem
    oracle = GaussianBoxOracle(p, n_points=n_points)
    baseline = oracle.probability(np.zeros(p.n_decisions))
    print(name, "optimizing; baseline box p =", baseline, flush=True)
    result = solve_veinott(
        p,
        oracle,
        SolverOptions(
            probability_buffer=_MANUFACTURING_PROBABILITY_BUFFER,
            verbose=verbose,
        ),
    )
    if result.status != "converged":
        print(
            "WARNING:", result.status, "feasible incumbent is retained", file=sys.stderr
        )
    print(
        name, result.status, "iterations", result.iterations, "validating", flush=True
    )
    validation = validate_instance(instance, result.x, n_samples=n_samples)
    check = oracle.independent_scrambles(
        result.x, n_points=8192, n_replicates=4, seed=911
    )
    scipy_check = float(
        multivariate_normal.cdf(
            p.upper,
            mean=np.asarray(p.mean(result.x)),
            cov=p.covariance,
            lower_limit=p.lower,
            maxpts=1_000_000,
            abseps=2e-6,
            rng=np.random.default_rng(985),
        )
    )
    folder = Path(out) / name
    folder.mkdir(parents=True, exist_ok=True)
    summary = {
        "name": name,
        "process": instance.process,
        "shape": instance.shape,
        "calibrated": False,
        "target_probability": p.alpha,
        "working_probability": result.working_target,
        "tolerance_mm": instance.command_scale_mm,
        "status": result.status,
        "iterations": result.iterations,
        "objective": result.objective,
        "lp_gap": result.absolute_gap,
        "certificate_label": result.certificate_label,
        "solver_seconds": result.elapsed_seconds,
        "n_qmc_points": n_points,
        "probability_seed": oracle.seed,
        "baseline_box_probability": baseline,
        "optimized_box_probability": result.probability,
        "independent_qmc_checks": check,
        "scipy_probability_check": scipy_check,
        "validation": validation,
        "basis_guarantee": instance.basis.guarantee(instance.command_scale_mm),
        "x": result.x,
        "command_coefficients_mm": np.asarray(instance.command_coefficients(result.x)),
        "mean_output_coefficients_mm": np.asarray(instance.mean_coefficients(result.x)),
        "description": instance.description,
    }
    save_json(folder / "metrics.json", summary)
    save_json(folder / "optimization.json", asdict(result))
    export_instance(instance, result, validation, folder)
    print(
        name,
        "box MC",
        validation["optimized_box"]["estimate"],
        "mesh MC",
        validation["optimized_mesh"]["estimate"],
        flush=True,
    )
    return summary


def main():
    ap = argparse.ArgumentParser(
        prog="python -m geocomp",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--out",
        default="results",
        metavar="DIRECTORY",
        help="Directory in which generated results and the viewer are written.",
    )
    ap.add_argument(
        "--case",
        choices=[
            "all",
            "abstract",
            "embedded",
            "dlp_ellipsoid",
            "dlp_sleeve",
            "cnc_cylinder",
            "cnc_pocket_plate",
            "embedded_helix",
            "embedded_serpentine",
        ],
        default="all",
        help="Individual case or predefined case group to run.",
    )
    ap.add_argument(
        "--alpha",
        type=_probability_level,
        default=0.95,
        metavar="PROBABILITY",
        help="Joint success probability required for each physical manufacturing case.",
    )
    ap.add_argument(
        "--qmc-points",
        type=_qmc_point_count,
        default=2048,
        metavar="COUNT",
        help="Power-of-two quasi-Monte Carlo point count used by the probability oracle.",
    )
    ap.add_argument(
        "--validation-samples",
        type=_positive_integer,
        default=100000,
        metavar="COUNT",
        help="Independent manufacturing draws used for post-solve validation.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print supporting-hyperplane solver progress.",
    )
    ap.add_argument(
        "--viewer-only",
        action="store_true",
        help="Rebuild viewer.html from existing per-case viewer_data.json files without solving.",
    )
    args = ap.parse_args()
    out = Path(args.out)
    if args.viewer_only:
        if not out.is_dir():
            ap.error(f"cannot rebuild viewer: output directory does not exist: {out}")
        viewer = write_dashboard(out)
        if viewer is None:
            ap.error(
                f"cannot rebuild viewer: no per-case viewer_data.json files found under {out}"
            )
        print(viewer)
        return
    out.mkdir(parents=True, exist_ok=True)
    save_json(
        out / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "devices": [str(d) for d in jax.devices()],
        },
    )
    if args.case in ["all", "abstract"]:
        run_abstract(out, args.qmc_points, args.validation_samples)
    cases = (
        [
            "dlp_ellipsoid",
            "dlp_sleeve",
            "cnc_cylinder",
            "cnc_pocket_plate",
            "embedded_helix",
            "embedded_serpentine",
        ]
        if args.case == "all"
        else ([] if args.case == "abstract" else [args.case])
    )
    if args.case == "embedded":
        cases = ["embedded_helix", "embedded_serpentine"]
    for case in cases:
        run_manufacturing(
            case,
            out,
            args.qmc_points,
            args.validation_samples,
            verbose=args.verbose,
            alpha=args.alpha,
        )
    write_dashboard(out)
    summaries = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(out.glob("*/metrics.json"))
    ]
    save_json(out / "summary.json", summaries)
    if summaries:
        import csv

        with (out / "summary.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "case",
                    "tolerance_mm",
                    "status",
                    "iterations",
                    "baseline_mesh_yield",
                    "optimized_mesh_yield",
                    "optimized_coefficient_box_yield",
                    "box_ci_lower",
                    "box_ci_upper",
                ]
            )
            for r in summaries:
                v = r["validation"]
                writer.writerow(
                    [
                        r["name"],
                        r["tolerance_mm"],
                        r["status"],
                        r["iterations"],
                        v["baseline_mesh"]["estimate"],
                        v["optimized_mesh"]["estimate"],
                        v["optimized_box"]["estimate"],
                        v["optimized_box"]["ci_lower"],
                        v["optimized_box"]["ci_upper"],
                    ]
                )
    print("Outputs:", out.resolve(), flush=True)


if __name__ == "__main__":
    main()

"""Collect all 18 manufacturing runs, export figures, and update the writeup."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from geocomp.embedded import build_embedded_instance

ROOT = Path(__file__).parent
OUT = ROOT / "results"
CASES = ("dlp_ellipsoid", "dlp_sleeve", "cnc_cylinder", "cnc_pocket_plate",
         "embedded_helix", "embedded_serpentine")
LEVELS = ("090", "095", "099")


def collect():
    records = []
    for level in LEVELS:
        for name in CASES:
            folder = ROOT / f"results_a{level}" / name
            record = json.loads((folder / "metrics.json").read_text())
            assert record["status"] == "converged", (name, level, record["status"])
            assert record["scipy_probability_check"] >= record["target_probability"]
            assert record["validation"]["optimized_box"]["ci_lower"] > record["target_probability"]
            with np.load(folder / "arrays.npz") as data:
                x = np.asarray(record["x"])
                assert np.all(x >= data["bounds"][:, 0] - 1e-8)
                assert np.all(x <= data["bounds"][:, 1] + 1e-8)
                objective = .5 * x @ data["Q"] @ x
                np.testing.assert_allclose(objective, record["objective"], rtol=1e-10)
            if name.startswith("embedded_"):
                instance = build_embedded_instance(name.removeprefix("embedded_"),
                                                   alpha=record["target_probability"])
                assert instance.problem.linear_feasible(x, tol=1e-7)
            record["folder"] = f"results_a{level}/{name}"
            records.append(record)
    for name in CASES:
        costs = [r["objective"] for r in records if r["name"] == name]
        assert np.all(np.diff(costs) >= -1e-6)
    return records


def summary_tables(records):
    fields = ("case", "alpha", "objective", "iterations", "oracle_probability",
              "scipy_probability", "mc_box", "mc_lower", "mc_upper", "mc_mesh")
    with (OUT / "all_results.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for r in records:
            box = r["validation"]["optimized_box"]
            writer.writerow((r["name"], r["target_probability"], r["objective"], r["iterations"],
                             r["optimized_box_probability"], r["scipy_probability_check"],
                             box["estimate"], box["ci_lower"], box["ci_upper"],
                             r["validation"]["optimized_mesh"]["estimate"]))
    with (OUT / "optimized_variables.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case", "alpha", "coordinate", "x_dimensionless", "command_mm"))
        for r in records:
            for index, (x, u) in enumerate(zip(r["x"], r["command_coefficients_mm"])):
                writer.writerow((r["name"], r["target_probability"], index, x, u))
    (OUT / "all_results.json").write_text(json.dumps(records, indent=2, allow_nan=False))


def plot_yields(records):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for ax, name in zip(axes.ravel(), CASES):
        group = [r for r in records if r["name"] == name]
        alpha = np.array([r["target_probability"] for r in group])
        values = [r["validation"]["optimized_box"] for r in group]
        means = np.array([v["estimate"] for v in values])
        errors = [means - [v["ci_lower"] for v in values],
                  [v["ci_upper"] for v in values] - means]
        ax.errorbar(alpha, means, yerr=errors, fmt="o-", capsize=4, label="Box MC + 95% CI")
        ax.plot(alpha, [r["validation"]["optimized_mesh"]["estimate"] for r in group],
                "s-", label="Whole-mesh MC")
        ax.plot(alpha, alpha, "k--", label="Required probability")
        ax.set(title=name.replace("_", " "), xlabel="Required joint probability",
               ylabel="Estimated success probability", ylim=(.89, 1.006))
        ax.set_xticks([.90, .95, .99])
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Independent validation: 100,000 manufactured parts per case and probability")
    fig.tight_layout()
    fig.savefig(OUT / "validated_yields.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in CASES:
        group = [r for r in records if r["name"] == name]
        ax.plot([r["target_probability"] for r in group], [r["objective"] for r in group],
                "o-", label=name.replace("_", " "))
    ax.set(xlabel="Required joint probability", ylabel="Dimensionless compensation objective",
           title="Higher required yield requires greater compensation cost")
    ax.set_xticks([.90, .95, .99])
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "compensation_cost.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def wireframe(ax, vertices, faces, color, label, *, linewidth=.35, alpha=.5):
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    ax.add_collection3d(Line3DCollection(vertices[edges], colors=color, linewidths=linewidth,
                                        alpha=alpha, label=label))


def plot_geometry(level):
    fig = plt.figure(figsize=(15, 10))
    for index, name in enumerate(CASES):
        data = json.loads((ROOT / f"results_a{level}" / name / "viewer_data.json").read_text())
        target, faces = np.asarray(data["vertices"]), np.asarray(data["faces"])
        factor = 4 if name.startswith("embedded") else 8
        command = target + factor * np.asarray(data["command"])
        ax = fig.add_subplot(2, 3, index + 1, projection="3d")
        wireframe(ax, target, faces, "#525252", "Target", alpha=.45)
        wireframe(ax, command, faces, "#1671ad", f"Command ({factor}× displacement)", alpha=.55)
        points = np.vstack((target, command))
        center = (points.max(axis=0) + points.min(axis=0)) / 2
        half = np.max(np.ptp(points, axis=0)) * .55
        ax.set(xlim=(center[0] - half, center[0] + half),
               ylim=(center[1] - half, center[1] + half),
               zlim=(center[2] - half, center[2] + half),
               xlabel="x / mm", ylabel="y / mm", zlabel="z / mm",
               title=f"{name.replace('_', ' ')}: {factor}×")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=60 if name.endswith("serpentine") else 22,
                     azim=-90 if name.endswith("serpentine") else -58)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Target and optimized command at alpha = {int(level) / 100:.2f}\n"
                 "Display exaggeration only; exported STL coordinates retain their true scale")
    fig.tight_layout()
    fig.savefig(OUT / f"commands_a{level}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def update_writeup(records):
    lines = ["", "## Results from all 18 physical runs", "",
             "- All **18 physical cases converged**. The unrounded optimized decisions and millimeter commands are in [optimized_variables.csv](results/optimized_variables.csv).",
             "- The table gives independent Monte Carlo coefficient-box yield and its individual 95% interval; the whole-mesh yield is listed separately.",
             "- Full precision probability checks, model data, and solver traces are retained in each case's `metrics.json`, `arrays.npz`, and `optimization.json`.", "",
             "| Case | alpha | Cost | Iterations | Box yield [95% CI] | Mesh yield |",
             "| --- | ---: | ---: | ---: | --- | ---: |"]
    for r in records:
        v = r["validation"]["optimized_box"]
        mesh = r["validation"]["optimized_mesh"]["estimate"]
        lines.append(f"| {r['name']} | {r['target_probability']:.2f} | {r['objective']:.6f} | "
                     f"{r['iterations']} | {v['estimate']:.5f} [{v['ci_lower']:.5f}, {v['ci_upper']:.5f}] | {mesh:.5f} |")
    qmc_error = max(abs(r["independent_qmc_checks"]["mean"] - r["optimized_box_probability"]) for r in records)
    scipy_error = max(abs(r["scipy_probability_check"] - r["optimized_box_probability"]) for r in records)
    minimum_headroom = min(r["validation"]["optimized_box"]["ci_lower"] - r["target_probability"] for r in records)
    lines += ["", "## Interpretation", "",
              "- Every optimized coefficient-box Monte Carlo lower confidence limit exceeds its requested target. These are individual intervals, not a simultaneous coverage statement over all 18 experiments.",
              f"- The smallest lower-limit headroom is **{minimum_headroom:.6f}** in probability units.",
              f"- The largest difference between the optimizer probability and the independent QMC mean is **{qmc_error:.3e}**; the corresponding largest SciPy difference is **{scipy_error:.3e}**.",
              "- Compensation cost increases with the requested yield for every geometry: tighter chance constraints shrink the feasible set.",
              "- The uncompensated coefficient-box probabilities are extremely small. Correcting systematic bias produces a large improvement in modeled yield.",
              "- The embedded whole-mesh yield is especially conservative relative to the coefficient box: the sum of separate centerline and radius budgets bounds displacement, but both worst cases need not occur together.",
              "", "![Validated coefficient and mesh yields](results/validated_yields.png)",
              "", "![Compensation cost against required yield](results/compensation_cost.png)",
              "", "## Exaggerated geometry visualizations", "",
              "- DLP/CNC static figures use **8× displacement**; embedded figures use **4× displacement**. Gray is the target and blue is the compensated command.",
              "- The supplied offline viewers include target, command, uncompensated output, compensated output, paired random realizations, bath context, and solver convergence. Select the display factor and realization using their controls.",
              "- Exported STLs retain the actual millimeter coordinates. Exaggeration only changes the display."]
    for level in LEVELS:
        lines += ["", f"- **alpha = {int(level)/100:.2f}:** [Open the supplied interactive viewer](results_a{level}/viewer.html).",
                  "", f"![Exaggerated commands at alpha {int(level)/100:.2f}](results/commands_a{level}.png)"]
    lines += ["", "## Optimized dimensionless variables", "",
              "- Entries below are rounded to six decimals for reading; use the linked metrics or CSV for all available digits.",
              "- Surface vectors follow the supplied mode order. Embedded vectors are shown in four channel blocks, each ordered along the path."]
    for level in LEVELS:
        lines += ["", f"### alpha = {int(level)/100:.2f}", ""]
        for r in records:
            if r["folder"].split("/")[0] != f"results_a{level}":
                continue
            values = np.array(r["x"])
            lines.append(f"- **{r['name']}** ([full precision]({r['folder']}/metrics.json)):")
            if len(values) == 20:
                for channel, block in zip(("dx", "dy", "dz", "radius"), values.reshape(4, -1)):
                    lines.append(f"  - `{channel} = ({', '.join(f'{x:.6f}' for x in block)})`")
            else:
                lines.append(f"  - `x = ({', '.join(f'{x:.6f}' for x in values)})`")
    lines += ["", "ChatGPT was used to understand the problems, assist with equations and "
              "implementation, check the results, and prepare this writeup."]
    path = ROOT / "writeup.md"
    prefix = path.read_text().split("<!-- GENERATED RESULTS -->")[0]
    path.write_text(prefix + "<!-- GENERATED RESULTS -->\n" + "\n".join(lines) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    records = collect()
    summary_tables(records)
    plot_yields(records)
    for level in LEVELS:
        plot_geometry(level)
    update_writeup(records)
    print("Collected", len(records), "converged physical cases in", OUT)


if __name__ == "__main__":
    main()

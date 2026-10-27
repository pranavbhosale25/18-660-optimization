"""Plot the actual nonlinear SQP histories saved by demo.py.

Example: python plot_results.py --input results/cpu64_all
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import SymmetricalLogLocator


def draw_history(ax, history, title):
    iterations = np.arange(len(history))
    for column, label, color in (
        ("merit", "Merit", "#2563eb"),
        ("constraint_violation_l1", "Constraint violation (L1)", "#d97706"),
        ("objective", "Loss", "#15803d"),
    ):
        ax.plot(iterations, history[column], label=label, color=color, linewidth=1.7)
    # Some losses are negative and some are identically zero. A symmetric
    # log scale preserves both without clipping or shifting the recorded data.
    ax.set_yscale("symlog", linthresh=1e-6)
    ticks = SymmetricalLogLocator(linthresh=1e-6, base=10)
    ticks.set_params(numticks=9)
    ax.yaxis.set_major_locator(ticks)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("SQP iteration")
    ax.set_ylabel("Recorded value (symlog)")
    ax.grid(True, alpha=0.22)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/cpu64_all"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.input / "plots"
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.input / "report.json").read_text())
    rows = report["results"]
    if not rows:
        raise ValueError("The solver report contains no problem results.")

    columns = 3
    grid, axes = plt.subplots(
        math.ceil(len(rows) / columns), columns,
        figsize=(16, 3.7 * math.ceil(len(rows) / columns)),
        constrained_layout=True, squeeze=False,
    )
    for ax, row in zip(axes.flat, rows):
        history = np.atleast_1d(np.genfromtxt(
            args.input / "histories" / f"{row['key']}.csv",
            delimiter=",", names=True,
        ))
        needed = {"objective", "constraint_violation_l1", "merit"}
        if not needed.issubset(history.dtype.names):
            raise ValueError("Rerun demo.py to record nonlinear violation and merit.")
        title = f"{row['problem']} | {row['iterations']} iterations"
        draw_history(ax, history, title)
        individual, single = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        draw_history(single, history, title)
        single.legend(fontsize=9)
        individual.savefig(output / f"{row['key']}.png", dpi=170)
        plt.close(individual)
    for ax in axes.flat[len(rows):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if len(rows) < axes.size:
        legend_ax = axes.flat[len(rows)]
        legend_ax.set_visible(True)
        legend_ax.set_axis_off()
        legend_ax.legend(handles, labels, loc="center", fontsize=12, frameon=False)
    else:
        axes.flat[0].legend(handles, labels, fontsize=8)
    grid.suptitle("Exercise 2.1 - SQP convergence", fontsize=17)
    grid.savefig(output / "convergence_all.png", dpi=150)
    plt.close(grid)
    print(f"Wrote {len(rows)} individual plots and convergence_all.png to {output}")


if __name__ == "__main__":
    main()

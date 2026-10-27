#!/usr/bin/env python3
"""Run reproducible CPU/GPU, float32/float64 SQP demonstrations.

Run from this directory after ``pip install -e '.[test]'``. Compilation,
first execution, and blocked steady-state execution are recorded separately.
"""

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--precision", type=int, choices=[32, 64], default=64)
    parser.add_argument(
        "--suite", choices=["small", "large", "all", "precision"], default="small"
    )
    parser.add_argument(
        "--problem",
        action="append",
        help="Select a registry key; repeat for multiple problems.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List problem keys and exit."
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--maxiter", type=int)
    parser.add_argument(
        "--elastic",
        action="store_true",
        help="Use elastic QPs throughout, not just on hard-QP failure.",
    )
    parser.add_argument("--merit-slope", choices=["model", "exact"], default="model")
    parser.add_argument(
        "--no-jit",
        action="store_true",
        help="Debug only: all control loops execute eagerly.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Also solve three HS71 starts using jit(vmap).",
    )
    parser.add_argument("--output", type=Path, default=Path("results/run"))
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Keep failure records but return exit code zero.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    # This must happen BEFORE importing JAX or jax_sqp.
    if args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    from jax_sqp import HISTORY_COLUMNS, Options, Status, configure

    rt = configure(
        platform=args.device,
        precision=args.precision,
        disable_jit=args.no_jit,
        device_index=args.device_index,
    )
    import jax
    import jax.numpy as jnp
    import numpy as np

    from jax_sqp import problems as P

    small = ["hs06", "hs07", "hs26", "hs71", "hs100", "nw18.3", "circle", "conflict"]
    large = [
        "broyden1000",
        "broyden10000",
        "control2000",
        "control10000",
        "rosenbrock1000",
    ]
    registry = {
        "hs06": P.hs06,
        "hs07": P.hs07,
        "hs26": P.hs26,
        "hs71": P.hs71,
        "hs100": P.hs100,
        "nw18.3": P.nw_ex18_3,
        "circle": P.circle,
        "conflict": P.conflicting_linearizations,
        "broyden1000": lambda d: P.broyden_tridiagonal(1000, d),
        "broyden10000": lambda d: P.broyden_tridiagonal(10000, d),
        "control200": lambda d: P.optimal_control(100, d),
        "control2000": lambda d: P.optimal_control(1000, d),
        "control10000": lambda d: P.optimal_control(5000, d),
        "rosenbrock1000": lambda d: P.constrained_rosenbrock(1000, d),
    }
    if args.list:
        print("\n".join(registry))
        return 0
    keys = (
        args.problem
        or {
            "small": small,
            "large": large,
            "all": small + large,
            "precision": ["hs71", "broyden1000", "control200"],
        }[args.suite]
    )
    unknown = set(keys) - registry.keys()
    if unknown:
        parser.error("Unknown problem keys: " + ", ".join(sorted(unknown)))
    opts = Options.for_dtype(
        args.precision,
        qp_mode="elastic" if args.elastic else "auto",
        merit_slope=args.merit_slope,
    )
    if args.maxiter is not None:
        opts = replace(opts, max_iter=args.maxiter)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "histories").mkdir(exist_ok=True)
    (out / "solutions").mkdir(exist_ok=True)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "jax": jax.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": str(rt.device),
        "device_kind": rt.device.device_kind,
        "backend": rt.platform,
        "precision": args.precision,
        "jit_enabled": not args.no_jit,
        "repeats": args.repeats,
        "options": asdict(opts),
        "timing_method": "Separate lower+compile; first blocked execution; median of blocked repeat executions. Input transfer and file I/O excluded.",
        "memory_note": "XLA compiler estimates, not measured peak process or device memory. Null when not supported.",
        "command": " ".join(sys.argv),
    }
    rows = []

    def save():
        (out / "report.json").write_text(
            json.dumps({"metadata": metadata, "results": rows}, indent=2) + "\n"
        )
        if rows:
            with (out / "summary.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    print(
        f"JAX {jax.__version__}; {rt.device}; float{args.precision}; tolerance {opts.tolerance:g}",
        flush=True,
    )
    for key in keys:
        p = registry[key](rt.dtype)
        n, ne, ni = p.dimensions
        x0, params = jax.device_put((p.x0, p.params), rt.device)
        jax.block_until_ready((x0, params))
        solver = p.solver(opts)
        mem = None
        if args.no_jit:
            compiled = solver.run
            compile_s = 0.0
        else:
            t = time.perf_counter()
            compiled = solver.solve.lower(x0, params).compile()
            compile_s = time.perf_counter() - t
            try:
                mem = compiled.memory_analysis()
            except (AttributeError, RuntimeError, NotImplementedError):
                pass
        t = time.perf_counter()
        r = compiled(x0, params)
        jax.block_until_ready(r)
        first_s = time.perf_counter() - t
        samples = []
        for _ in range(args.repeats):
            t = time.perf_counter()
            r = compiled(x0, params)
            jax.block_until_ready(r)
            samples.append(time.perf_counter() - t)
        row = {
            "key": key,
            "problem": p.name,
            "n": n,
            "equalities": ne,
            "inequalities_including_bounds": ni,
            "precision": args.precision,
            "success": bool(r.success),
            "status": Status(int(r.status)).name,
            "iterations": int(r.iterations),
            "objective": float(r.objective),
            "reference_objective": p.reference_objective,
            "primal_inf": float(r.kkt.primal),
            "stationarity_inf": float(r.kkt.stationarity),
            "dual_inf": float(r.kkt.dual),
            "complementarity_inf": float(r.kkt.complementarity),
            "kkt_inf": float(r.kkt.error),
            "kkt_tolerance": opts.tolerance,
            "elastic_iterations": int(r.elastic_steps),
            "penalty": float(r.penalty),
            "qp_iterations": int(r.qp_iterations),
            "cg_iterations": int(r.cg_iterations),
            "merit_evaluations": int(r.line_search_evaluations),
            "compile_seconds": compile_s,
            "first_execution_seconds": first_s,
            "warm_median_seconds": statistics.median(samples),
            "warm_min_seconds": min(samples),
            "xla_temp_bytes": getattr(mem, "temp_size_in_bytes", None),
            "xla_argument_bytes": getattr(mem, "argument_size_in_bytes", None),
            "xla_output_bytes": getattr(mem, "output_size_in_bytes", None),
            "source": p.source,
            "notes": p.notes,
        }
        rows.append(row)
        hist = np.asarray(r.history)[: int(r.iterations) + 1]
        np.savetxt(
            out / "histories" / f"{key}.csv",
            hist,
            delimiter=",",
            header=",".join(HISTORY_COLUMNS),
            comments="",
        )
        np.savez_compressed(
            out / "solutions" / f"{key}.npz",
            x=np.asarray(r.x),
            multipliers=np.asarray(r.multipliers),
        )
        save()  # Preserve completed results even if a later problem fails.
        print(
            f'{p.name:30s} n={n:6d} {row["status"]:23s} it={row["iterations"]:3d} '
            f'f={row["objective"]:.10g} KKT={row["kkt_inf"]:.3e} '
            f'compile={compile_s:.3f}s warm={row["warm_median_seconds"]:.6f}s',
            flush=True,
        )
        del compiled, solver, r
        jax.clear_caches()
    if args.batch:
        p = P.hs71(rt.dtype)
        solver = p.solver(opts)
        xs = jax.device_put(
            jnp.array(
                [[1.0, 5.0, 5.0, 1.0], [1.2, 4.5, 4.0, 1.4], [1.0, 4.7, 3.8, 1.4]],
                dtype=rt.dtype,
            ),
            rt.device,
        )
        t = time.perf_counter()
        batched = (
            jax.vmap(solver.run, in_axes=(0, None))
            if args.no_jit
            else solver.solve_batch.lower(xs, None).compile()
        )
        compile_s = time.perf_counter() - t
        r = batched(xs, None)
        jax.block_until_ready(r)
        times = []
        for _ in range(args.repeats):
            t = time.perf_counter()
            r = batched(xs, None)
            jax.block_until_ready(r)
            times.append(time.perf_counter() - t)
        batch = {
            "problem": "HS71",
            "batch_size": 3,
            "compile_seconds": compile_s,
            "warm_median_seconds": statistics.median(times),
            "success": np.asarray(r.success).tolist(),
            "iterations": np.asarray(r.iterations).tolist(),
            "kkt_inf": np.asarray(r.kkt.error).tolist(),
            "x": np.asarray(r.x).tolist(),
            "note": "Independent starts, same problem; no claim that batching beats sequential execution.",
        }
        metadata["batch_demo"] = batch
        save()
        print("HS71 batch:", json.dumps(batch), flush=True)
    ok = all(r["success"] for r in rows)
    if args.batch:
        ok = ok and all(metadata["batch_demo"]["success"])
    return 0 if ok or args.allow_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

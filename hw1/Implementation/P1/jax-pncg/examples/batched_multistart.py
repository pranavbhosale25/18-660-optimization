from __future__ import annotations

import argparse

import numpy as np

from jax_pncg import PNCGOptions, configure_runtime, make_batched_solver
from jax_pncg.test_problems import rosenbrock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float64"
    )
    args = parser.parse_args()

    runtime = configure_runtime(platform=args.device, precision=args.precision)
    gtol = 5.0e-4 if args.precision == "float32" else 1.0e-8
    starts = np.asarray(
        [[-1.2, 1.0], [-1.0, 2.0], [2.0, 2.0], [0.0, 0.0]], dtype=float
    )

    with runtime.default_device():
        starts_device = runtime.asarray(starts)
        batched_solver = make_batched_solver(
            rosenbrock,
            options=PNCGOptions(maxiter=1000, gtol=gtol),
        )
        result = batched_solver(starts_device)

    result.x.block_until_ready()
    print(f"device: {runtime.device}; dtype: {result.x.dtype}")
    print("solutions:")
    print(result.x)
    print("iterations:", result.nit)
    print("success:   ", result.success)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from jax_pncg import PNCGOptions, configure_runtime, make_solver, status_message
from jax_pncg.test_problems import ROSENBROCK_2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float64"
    )
    args = parser.parse_args()

    runtime = configure_runtime(platform=args.device, precision=args.precision)
    gtol = 1.0e-5 if args.precision == "float32" else 1.0e-8

    with runtime.default_device():
        x0 = runtime.asarray(ROSENBROCK_2.start)
        solver = make_solver(
            ROSENBROCK_2.objective,
            options=PNCGOptions(maxiter=1000, gtol=gtol, beta_method="pr+"),
        )
        result = solver(x0)

    result.x.block_until_ready()
    print(f"device:    {runtime.device}")
    print(f"precision: {result.x.dtype}")
    print(f"status:    {status_message(result.status)}")
    print(f"x*:        {result.x}")
    print(f"f(x*):     {float(result.fun):.8e}")
    print(f"||g||:     {float(result.grad_norm):.8e}")
    print(f"nit/nfev:  {int(result.nit)}/{int(result.nfev)}")


if __name__ == "__main__":
    main()

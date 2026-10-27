"""Optional on-hardware test: python examples/gpu_smoke.py --precision 64.

Requires a configured JAX GPU backend; there is no CPU fallback.
"""

import argparse

p = argparse.ArgumentParser()
p.add_argument("--precision", type=int, choices=[32, 64], default=64)
a = p.parse_args()
from jax_sqp import configure

rt = configure(platform="gpu", precision=a.precision)
import jax

from jax_sqp.problems import hs71, optimal_control

for problem in [hs71(rt.dtype), optimal_control(100, rt.dtype)]:
    x0, params = jax.device_put((problem.x0, problem.params), rt.device)
    result = problem.solver().solve(x0, params)
    jax.block_until_ready(result)
    assert result.x.device.platform == "gpu"
    assert result.x.dtype == rt.dtype
    assert bool(result.success), (problem.name, result.status, result.kkt)
    print(problem.name, result.x.device, result.x.dtype, result.kkt)

import json
import subprocess
import sys

import pytest

from jax_sqp import Options, QPOptions, configure


def test_cpu_and_precision_in_fresh_process():
    code = """
import json
from jax_sqp import configure
rt=configure(platform="cpu",precision=32)
import jax
from jax_sqp.problems import hs71
p=hs71(rt.dtype);r=p.solver().solve(p.x0)
print(json.dumps({"dtype":str(r.x.dtype),"x64":jax.config.x64_enabled,"platform":r.x.device.platform,"success":bool(r.success)}))
"""
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    data = json.loads(out.strip().splitlines()[-1])
    assert data == {
        "dtype": "float32",
        "x64": False,
        "platform": "cpu",
        "success": True,
    }


def test_bad_configuration():
    with pytest.raises(ValueError):
        configure(precision=16)
    with pytest.raises(ValueError):
        configure(platform="unknown")
    with pytest.raises(ValueError):
        Options(history_size=0)
    with pytest.raises(ValueError):
        QPOptions(max_iter=0)


def test_gpu_mode_errors_clearly_when_unavailable():
    import jax

    try:
        jax.devices("gpu")
    except RuntimeError:
        with pytest.raises(RuntimeError, match="unavailable"):
            configure(platform="gpu")
        return
    pytest.skip("A GPU is available; CPU-only missing-backend test does not apply.")

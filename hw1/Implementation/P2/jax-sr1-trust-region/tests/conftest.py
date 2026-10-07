from __future__ import annotations

import os

# Tests use deterministic CPU execution and exercise float64 unless a subprocess
# explicitly selects float32.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

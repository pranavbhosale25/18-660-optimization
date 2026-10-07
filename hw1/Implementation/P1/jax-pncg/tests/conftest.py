from __future__ import annotations

import jax
import pytest


@pytest.fixture(autouse=True)
def default_to_x64_for_each_test():
    # Precision is a process-global JAX option.  Reset it around every test so
    # runtime-switch tests cannot leak float32 mode into numerical tests.
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", True)

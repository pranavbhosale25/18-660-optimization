import os
os.environ.setdefault("JAX_ENABLE_X64","1")
import jax
jax.config.update("jax_enable_x64",True)

import pytest

@pytest.fixture(autouse=True,scope="module")
def _release_jax_compilation_cache():
    yield
    jax.clear_caches()

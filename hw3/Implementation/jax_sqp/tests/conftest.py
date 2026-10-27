import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import jax
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def clean_jax_cache():
    yield
    jax.clear_caches()

import jax
jax.config.update('jax_enable_x64', True)
import pytest
from qp import Solver, SolverConfig

@pytest.fixture(scope='session')
def oracle():
    return Solver(SolverConfig(backend='scipy'))

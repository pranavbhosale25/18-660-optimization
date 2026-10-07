"""Check the compact LP against independently enumerated opponent plans.

Run from the poker folder: python results/validate_formulation.py
"""
import itertools
import json
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linprog

from games import row_lp
from qp import Solver, SolverConfig


def reference_value(kernel, row_types, row_actions, col_types, col_actions):
    # This deliberately different formulation has ONE value variable and a
    # constraint for every complete opponent contingent plan (small tests only).
    plans = []
    for actions in itertools.product(range(col_actions), repeat=col_types):
        policy = np.zeros(col_types * col_actions)
        policy[np.arange(col_types) * col_actions + np.asarray(actions)] = 1
        plans.append(kernel @ policy)
    probabilities = row_types * row_actions
    result = linprog(
        np.r_[np.zeros(probabilities), -1.],
        A_ub=np.c_[-np.asarray(plans), np.ones(len(plans))],
        b_ub=np.zeros(len(plans)),
        A_eq=np.c_[np.kron(np.eye(row_types), np.ones((1, row_actions))),
                    np.zeros(row_types)],
        b_eq=np.ones(row_types),
        bounds=[(0, None)] * probabilities + [(None, None)],
        method='highs',
    )
    assert result.success, result.message
    return -result.fun


def check_case(label, kernel, dimensions, solver):
    rt, ra, ct, ca = dimensions
    problem = row_lp(jnp.asarray(kernel), rt, ra, ct, ca)
    out = solver.solve(problem)
    assert int(out.status) == 1, label
    strategy = np.asarray(out.x[:rt * ra]).reshape(rt, ra)
    np.testing.assert_allclose(strategy.sum(axis=1), 1, atol=1e-10)
    assert strategy.min() >= -1e-10, label
    # Compare in normalized payoff units, so 1e-12 games cannot pass merely
    # because an absolute chip tolerance exceeds their entire payoff range.
    scale = float(np.max(np.abs(kernel))) or 1.
    normalized = np.asarray(kernel, dtype=float) / scale
    value = np.min((normalized.T @ strategy.ravel()).reshape(ct, ca), axis=1).sum()
    expected = reference_value(normalized, *dimensions)
    np.testing.assert_allclose(value, expected, atol=1e-8, rtol=1e-8)
    assert float(out.stationarity) < 1e-8, label
    assert float(out.complementarity) < 1e-8, label
    return dict(case=label,dimensions=list(dimensions),payoff_scale=scale,
                normalized_value=float(value),reference_value=float(expected),
                absolute_error=float(abs(value-expected)))


def main():
    rng = np.random.default_rng(7)
    solver = Solver(SolverConfig(backend='scipy'))
    records = []
    for dims in [(2, 3, 3, 2), (3, 2, 2, 3), (1, 3, 2, 2)]:
        base = rng.normal(size=(dims[0] * dims[1], dims[2] * dims[3]))
        for factor in (1e-12, 1., 1e12):
            records.append(check_case(f'rectangular_{dims}_scale_{factor:g}',
                                      base * factor, dims, solver))
    for label, kernel in [('integer_matching_pennies',np.array([[1,-1],[-1,1]])),
                          ('negative_value',np.full((2,2),-3)),
                          ('zero_kernel',np.zeros((2,2),dtype=int))]:
        records.append(check_case(label,kernel,(1,2,1,2),solver))
    Path(__file__).with_name('formulation_validation.json').write_text(
        json.dumps(records,indent=2)+'\n')
    print(f'{len(records)} independent LP checks passed; maximum normalized error '
          f'{max(r["absolute_error"] for r in records):.3e}.')


if __name__ == '__main__':
    main()

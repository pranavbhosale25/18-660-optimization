"""Exact poker solution + best responses + optional sampling and timing."""
import argparse
import json
from pathlib import Path
import platform
import time
import importlib.metadata
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from qp import Solver,SolverConfig
from games import solve_game,make_batched_game_solver,evaluate
from poker import PokerRules,build_poker,exact_payoff,sample_payoffs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--precision',choices=['float32','float64'],default='float64')
    parser.add_argument('--solver-tol',type=float,default=None)
    parser.add_argument('--maxiter',type=int,default=30000)
    parser.add_argument('--backend',choices=['jaxopt','scipy'],default='jaxopt')
    parser.add_argument('--ranks',type=int,default=3)
    parser.add_argument('--suits',type=int,default=2)
    parser.add_argument('--bids',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--samples',type=int,default=50000)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,default=Path('results/poker.json'))
    args=parser.parse_args()
    if args.repeats<1 or args.samples<2: parser.error('repeats>=1 and samples>=2 required')
    jax.config.update('jax_enable_x64',args.precision=='float64')
    tol=args.solver_tol if args.solver_tol is not None else (1e-8 if args.precision=='float64' else 1e-5)
    rules=PokerRules(args.ranks,args.suits,bids=tuple(args.bids));data=build_poker(rules)
    solver=Solver(SolverConfig(backend=args.backend,tol=tol,maxiter=args.maxiter))
    H=rules.deck_size-1; B=len(rules.bids)
    if solver.compilable:
        run=make_batched_game_solver(solver,H,B,H,B)
    else:
        def run(kernels):
            solved=[solve_game(k,H,B,H,B,solver) for k in kernels]
            return jax.tree.map(lambda *xs:jnp.stack(xs),*solved)
    t0=time.perf_counter();out=run(data.kernels);jax.block_until_ready(out)
    first=time.perf_counter()-t0
    times=[]
    for _ in range(args.repeats):
        t0=time.perf_counter();run_out=run(data.kernels);jax.block_until_ready(run_out)
        times.append(time.perf_counter()-t0)
    policies=out.row
    baselines={}
    for a in range(B):
        baseline=jnp.broadcast_to(jax.nn.one_hot(a,B),(rules.deck_size,H,B))
        baselines[f'always_bid_{rules.bids[a]}']=float(exact_payoff(data,policies,baseline))
    baselines['uniform']=float(exact_payoff(data,policies,jnp.ones_like(policies)/B))
    allmax=jnp.broadcast_to(jax.nn.one_hot(B-1,B),policies.shape)
    _,_,best_values=jax.vmap(evaluate)(data.kernels,policies,allmax)
    samples=sample_payoffs(jax.random.PRNGKey(7),data,out.row,out.col,args.samples,rules)
    arr=np.asarray(samples)
    summary={
        'backend':args.backend,'jax':jax.__version__,'python':platform.python_version(),
        'precision':args.precision,'solver_tolerance':tol,'solver_maxiter':args.maxiter,
        'devices':[str(d) for d in jax.devices()],
        'deck_size':rules.deck_size,'public_boards':rules.deck_size,
        'private_types_per_board':H,'bids':list(rules.bids),'kernel_shape_per_board':[B*H,B*H],
        'first_call_seconds':first,'median_repeat_seconds':float(np.median(times)),
        'timing_note':('First call includes solver compilation; repeats reuse it.' if solver.compilable else
                       'CPU SciPy oracle; NOT JAXopt or GPU performance.'),
        'exact_self_play_payoff':float(exact_payoff(data,out.row,out.col)),
        'maximum_board_nash_gap':float(out.gap.max()),
        'mean_board_nash_gap':float(out.gap.mean()),
        'board_lower_bounds':np.asarray(out.lower).tolist(),
        'board_upper_bounds':np.asarray(out.upper).tolist(),
        'board_nash_gaps':np.asarray(out.gap).tolist(),
        'all_lp_statuses_success':bool(jnp.all(out.status==1)),
        'maximum_lp_feasibility_residual':float(out.lp_residual.max()),
        'maximum_raw_simplex_error':float(out.raw_simplex_error.max()),
        'equilibrium_vs_baselines_exact_chips':baselines,
        'best_response_gain_against_always_max':float(best_values.mean()),
        'always_max_symmetric_nash_gap':float(2*best_values.mean()),
        'sample_mean':float(arr.mean()),'sample_standard_error':float(arr.std(ddof=1)/np.sqrt(len(arr))),
        'samples':args.samples,
        'board_0_hole_cards':np.asarray(data.holes[0]).tolist(),
        'board_0_row_strategy':np.asarray(out.row[0]).tolist(),
        'board_0_col_strategy':np.asarray(out.col[0]).tolist(),
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'),row=np.asarray(out.row),
                        col=np.asarray(out.col),holes=np.asarray(data.holes),
                        lower=np.asarray(out.lower),upper=np.asarray(out.upper),
                        gap=np.asarray(out.gap))
    print(json.dumps(summary,indent=2))
    if not summary['all_lp_statuses_success'] or summary['maximum_board_nash_gap']>max(1e-4,100*tol):
        raise SystemExit('Equilibrium certification failed; inspect solver settings.')

if __name__=='__main__': main()

import argparse
import json
from pathlib import Path
import time
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from qp import Solver,SolverConfig
from physics import step,make_rollout,PhysicsConfig
from worlds import random_worlds
from viewer import write_viewer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--precision',choices=['float32','float64'],default='float64')
    p.add_argument('--solver-tol',type=float,default=None)
    p.add_argument('--maxiter',type=int,default=20000)
    p.add_argument('--friction',type=float,default=.25)
    p.add_argument('--restitution',type=float,default=.6)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--backend',choices=['jaxopt','scipy'],default='jaxopt')
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--steps',type=int,default=180)
    p.add_argument('--planes',type=int,default=8)
    p.add_argument('--dt',type=float,default=1/120)
    p.add_argument('--output',type=Path,default=Path('results/physics.json'))
    a=p.parse_args()
    if a.steps<1 or a.dt<=0: p.error('steps and dt must be positive')
    jax.config.update('jax_enable_x64',a.precision=='float64')
    tol=a.solver_tol if a.solver_tol is not None else (1e-9 if a.precision=='float64' else 1e-5)
    config=PhysicsConfig() if a.precision=='float64' else PhysicsConfig(
        contact_tolerance=1e-5,velocity_tolerance=2e-4,
        complementarity_tolerance=2e-4,energy_tolerance=2e-4)
    solver=Solver(SolverConfig(backend=a.backend,tol=tol,maxiter=a.maxiter))
    worlds,balls=random_worlds(jax.random.PRNGKey(a.seed),batch=a.batch,planes=a.planes,
                              friction=a.friction,restitution=a.restitution)
    gravity=jnp.array([0.,0.,-9.81])
    t0=time.perf_counter()
    if solver.compilable:
        fn=make_rollout(solver,a.steps,config)
        out=fn(worlds,balls,a.dt,gravity);jax.block_until_ready(out)
        first=time.perf_counter()-t0
        t0=time.perf_counter();again=fn(worlds,balls,a.dt,gravity);jax.block_until_ready(again)
        repeat=time.perf_counter()-t0
    else:
        # Deliberate host loop: an oracle, not a substitute for JAX batching.
        outputs=[]
        for k in range(a.batch):
            w=jax.tree.map(lambda v:v[k],worlds);b=jax.tree.map(lambda v:v[k],balls)
            history=[]
            for _ in range(a.steps):
                r=step(w,b,a.dt,gravity,solver,config)
                history.append(r);b=r.ball
                if not bool(r.success):
                    raise RuntimeError(f'Oracle trajectory {k} failed: {r}')
            outputs.append(jax.tree.map(lambda *xs:jnp.stack(xs),*history))
        out=jax.tree.map(lambda *xs:jnp.stack(xs,axis=1),*outputs)
        first=time.perf_counter()-t0;repeat=None
    summary={'backend':a.backend,'jax':jax.__version__,'devices':[str(d) for d in jax.devices()],
             'model':'Version D: fixed-budget velocity QP; point mass, no rotation',
             'precision':a.precision,'solver_tolerance':tol,'solver_maxiter':a.maxiter,
             'seed':a.seed,'friction_coefficient':a.friction,'restitution':a.restitution,
             'contact_tolerance':config.contact_tolerance,
             'complementarity_tolerance':config.complementarity_tolerance,
             'worlds':a.batch,'planes':a.planes,'steps':a.steps,'dt':a.dt,
             'first_call_seconds':first,'repeat_seconds':repeat,
             'all_steps_success':bool(jnp.all(out.success)),
             'minimum_gap':float(out.min_gap.min()),
             'max_contact_residual':float(out.contact_residual.max()),
             'maximum_impact_energy_increase':float(out.max_energy_increase.max()),
             'total_collision_events':int(out.events.sum()),
             'total_qp_solves':int(out.qp_solves.sum()),
             'total_qp_iterations':int(out.qp_iterations.sum()),
             'qp_status_codes':[int(v) for v in np.unique(np.asarray(out.qp_status))],
             'all_qp_status_success':bool(jnp.all(out.qp_status==1)),
             'one_qp_per_event':bool(jnp.all(out.events==out.qp_solves)),
             'timing_note':('First call includes compilation; repeat is a warmed JAXopt rollout.' if solver.compilable
                            else 'CPU SciPy oracle; NOT JAXopt or GPU performance.')}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(summary,indent=2)+'\n')
    positions=np.concatenate((np.asarray(balls.position)[None,...],np.asarray(out.ball.position)))
    np.savez_compressed(a.output.with_suffix('.npz'),positions=positions,
                        velocities=np.asarray(out.ball.velocity),
                        normals=np.asarray(worlds.normals),offsets=np.asarray(worlds.offsets),dt=a.dt,
                        success=np.asarray(out.success),min_gap=np.asarray(out.min_gap),
                        contact_residual=np.asarray(out.contact_residual),
                        energy_increase=np.asarray(out.max_energy_increase),
                        events=np.asarray(out.events),qp_solves=np.asarray(out.qp_solves),
                        qp_iterations=np.asarray(out.qp_iterations),
                        qp_status=np.asarray(out.qp_status),
                        masses=np.asarray(balls.mass),gravity=np.asarray(gravity))
    first_world=jax.tree.map(lambda v:v[0],worlds)
    write_viewer(a.output.with_suffix('.html'),first_world,positions[:,0],balls.radius[0],a.dt,
                  title=f'Version D: one QP per contact event ({a.backend})')
    print(json.dumps(summary,indent=2))
    if not summary['all_steps_success']:
        raise SystemExit('Some trajectories failed. Inspect residuals / decrease dt; do not grade a frozen result as a success.')

if __name__=='__main__':main()

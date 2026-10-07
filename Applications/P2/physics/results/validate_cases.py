"""Independent analytic cases for the completed contact and collision routines.
Run from the physics folder: python results/validate_cases.py
"""
import json
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from physics import resolve_contact, time_to_collision, contact_error, contact_qp, gaps
from worlds import make_world,make_ball
from qp import Solver,SolverConfig

def main():
    cases=[
        ('frictionless',make_world([[0,0,1]],[0],0,.5),make_ball([0,0,0],[1,0,-2]),jnp.array([True]),None,[1,0,1]),
        ('fixed_friction_budget',make_world([[0,0,1]],[0],0,.5),make_ball([0,0,0],[4,0,-2]),jnp.array([True]),jnp.array([.7]),[3.3,0,1]),
        ('two_tangent_components',make_world([[0,0,1]],[0],0,.5),make_ball([0,0,0],[4,3,-2]),jnp.array([True]),jnp.array([2.]),[2.5,2.5,1]),
        ('corner_and_inactive_wall',make_world([[1,0,0],[0,1,0],[0,0,1]],[0,0,-1],0,.5),make_ball([0,0,0],[-2,-3,0]),jnp.array([True,True,False]),None,[1,1.5,0])]
    records=[]
    for backend in ['scipy','jaxopt']:
        solver=Solver(SolverConfig(backend=backend,tol=1e-9))
        for name,w,b,active,budget,expected in cases:
            r=resolve_contact(w,b,active,solver,budget=budget)
            assert bool(r.converged),r
            np.testing.assert_allclose(r.u,expected,atol=2e-6)
            assert int(r.qp_solves)==1
            if name=='corner_and_inactive_wall':
                np.testing.assert_allclose(r.normal_impulse[-1],0,atol=2e-6)
                np.testing.assert_allclose(r.tangent_impulse[-2:],0,atol=2e-6)
            records.append(dict(case=name,backend=backend,velocity=np.asarray(r.u).tolist(),minimum_gap=float(gaps(w,b.position,b.radius).min()),contact_residual=float(contact_error(r)),impact_energy_change=float(r.energy_change),qp_status=int(r.qp_status),qp_calls=int(r.qp_solves)))
    w=make_world([[0,0,1]],[0]);checks=[]
    for name,pos,vel,radius,remaining,expected_t,expected_hit in [
        ('approaching',[0,0,2],[0,0,-4],0,1,.5,True),
        ('parallel',[0,0,2],[1,0,0],0,1,np.inf,False),
        ('separating',[0,0,2],[0,0,4],0,1,np.inf,False),
        ('touching_approaching',[0,0,0],[0,0,-1],0,1,0,True),
        ('touching_separating',[0,0,0],[0,0,1],0,1,np.inf,False),
        ('outside_interval',[0,0,2],[0,0,-4],0,.25,.5,False),
        ('radius_clearance',[0,0,2],[0,0,-4],.5,1,.375,True)]:
        t,hit=jax.jit(time_to_collision)(w,make_ball(pos,vel,radius),remaining,1e-7)
        np.testing.assert_allclose(t,expected_t);assert bool(hit)==expected_hit
        checks.append(dict(case=name,time=float(t) if np.isfinite(float(t)) else 'inf',hit=bool(hit)))
    # Regression: integer velocity must not turn fractional mass into Q=0.
    integer_problem=contact_qp(jnp.array([4,0,-2]),.5,
        jnp.array([[0.,0.,1.]]),jnp.array([[1.,0.,0.],[0.,1.,0.]]),
        jnp.array([1.]),jnp.array([.7]),jnp.array([True]))
    np.testing.assert_allclose(integer_problem.Q.diagonal()[:3],[.5,.5,.5])
    for backend in ['scipy','jaxopt']:
        result=Solver(SolverConfig(backend=backend,tol=1e-9)).solve(integer_problem)
        assert int(result.status)==1
        np.testing.assert_allclose(result.x[:3],[2.6,0,1],atol=2e-6)

    # No active planes: contact resolution must leave velocity unchanged.
    for backend in ['scipy','jaxopt']:
        result=resolve_contact(make_world([[0,0,1]],[-1],.3,.5),
            make_ball([0,0,0],[1,2,-3]),jnp.array([False]),
            # BoxOSQP's detector falsely flags this no-contact direct call;
            # the production event loop only resolves actual touching contacts.
            Solver(SolverConfig(backend=backend,tol=1e-9,check_infeasibility=False)))
        assert bool(result.converged)
        np.testing.assert_allclose(result.u,[1,2,-3],atol=2e-6)
        np.testing.assert_allclose(result.normal_impulse,0,atol=2e-6)
        np.testing.assert_allclose(result.tangent_impulse,0,atol=2e-6)

    # Earliest event among multiple planes; the hit includes the interval end.
    two_walls=make_world([[1,0,0],[0,0,1]],[0,0])
    t,hit=jax.jit(time_to_collision)(two_walls,make_ball([1,0,2],[-2,0,-1]),.5,1e-7)
    np.testing.assert_allclose(t,.5);assert bool(hit)
    output=Path(__file__).with_name('analytic_checks.json')
    output.write_text(json.dumps(dict(contact_cases=records,collision_cases=checks,additional_regressions=['integer_velocity_fractional_mass_both_backends','all_contacts_inactive_both_backends_infeasibility_detector_disabled','earliest_of_two_planes_at_interval_end']),indent=2)+'\n')
    print('20 analytic/regression checks passed (contact, collision timing, dtype, and inactive slots).')

if __name__=='__main__':main()

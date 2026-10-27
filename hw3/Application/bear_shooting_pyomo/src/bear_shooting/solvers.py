"""Solve the reduced Pyomo shooting NLP using the IPOPT C API or cyipopt."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import numpy as np
from .layout import Trajectory



def _violation(values, lower, upper):
    return np.maximum(np.maximum(lower - values, values - upper), 0.0)


@dataclass
class SolveResult:
    controls: np.ndarray
    trajectory: Trajectory
    objective: float
    status: str
    solver_success: bool
    feasible: bool
    max_scaled_constraint_violation: float
    max_control_bound_violation: float
    constraint_reports: dict
    objective_terms: dict
    iterations: int
    elapsed_seconds: float
    engine: str
    evaluations: int
    derivative_evaluations: int
    history: dict = field(default_factory=dict)

    @property
    def success(self):
        return self.solver_success and self.feasible

    def require_success(self):
        if not self.success:
            raise RuntimeError(
                f"IPOPT did not produce an accepted solution: {self.status}; "
                f"scaled task violation={self.max_scaled_constraint_violation:.3g}, "
                f"control-bound violation={self.max_control_bound_violation:.3g}."
            )
        return self

    def summary(self):
        return {"success": self.success, "solver_success": self.solver_success, "feasible": self.feasible,
                "status": self.status, "objective": self.objective, "iterations": self.iterations,
                "max_scaled_constraint_violation": self.max_scaled_constraint_violation,
                "max_control_bound_violation": self.max_control_bound_violation,
                "elapsed_seconds": self.elapsed_seconds, "engine": self.engine,
                "evaluations": self.evaluations, "derivative_evaluations": self.derivative_evaluations,
                "objective_terms": self.objective_terms, "constraints": self.constraint_reports,
                "history": self.history}

    def save(self, destination):
        """Save controls/landmarks and a readable JSON report; includes failed solves."""
        destination = Path(destination)
        if destination.suffix != ".npz":
            destination = destination.with_suffix(".npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, controls=self.controls, observations=self.trajectory.array,
                            times=self.trajectory.layout.times,
                            part_names=np.asarray(self.trajectory.layout.part_names),
                            actuator_names=np.asarray(self.trajectory.layout.actuator_names))
        destination.with_suffix(".json").write_text(json.dumps(self.summary(), indent=2) + "\n")
        return destination


def _cyipopt_solve(problem, x0, options):
    try:
        import cyipopt
    except ImportError as exc:
        raise ImportError("engine='cyipopt' needs cyipopt and a working IPOPT shared library. Use engine='ipopt' with BEAR_IPOPT_LIBRARY to load a shared library directly.") from exc
    oracle = problem.oracle
    nlp = cyipopt.Problem(n=problem.layout.n_controls, m=oracle.task.row_count - 1,
                         problem_obj=oracle, lb=problem.lower.ravel(), ub=problem.upper.ravel(),
                         cl=oracle.task.lower, cu=oracle.task.upper)
    try:
        for key, value in options.items():
            nlp.add_option(key, value)
        solution, info = nlp.solve(x0)
    finally:
        nlp.close()
    status = info.get("status_msg", "unknown")
    if isinstance(status, bytes):
        status = status.decode("utf-8", errors="replace")
    return solution, {"status": str(status), "solver_success": info.get("status") in (0, 1),
                      "iterations": -1, "history": {}}


def solve_ipopt(problem, initial_controls, *, engine="ipopt", ipopt_options=None,
                feasibility_tolerance=1e-4, raise_on_failure=False):
    if engine not in ("ipopt", "cyipopt"):
        raise ValueError("engine must be 'ipopt' or 'cyipopt'; both invoke IPOPT.")
    if not np.isfinite(feasibility_tolerance) or feasibility_tolerance <= 0:
        raise ValueError("feasibility_tolerance must be finite and positive.")
    options = {"hessian_approximation": "limited-memory", "max_iter": 200, "tol": 1e-5,
               "acceptable_tol": 1e-4, "constr_viol_tol": 1e-5,
               "acceptable_constr_viol_tol": 1e-4, "acceptable_iter": 5, "bound_relax_factor": 0.0,
               "honor_original_bounds": "yes", "nlp_scaling_method": "none",
               "mu_strategy": "adaptive", "print_level": 0, "option_file_name": ""}
    if ipopt_options:
        options.update(ipopt_options)
    if options["hessian_approximation"] != "limited-memory":
        raise ValueError("Only limited-memory Hessians are supported; Warp supplies first derivatives only.")
    if float(options["bound_relax_factor"]) != 0.0:
        raise ValueError("bound_relax_factor must stay zero to preserve actuation limits during the solve.")
    if options["option_file_name"] != "":
        raise ValueError("External IPOPT option files are disabled; supply ipopt_options explicitly.")
    oracle = problem.oracle
    start = time.perf_counter()
    from .ipopt_c import solve as _native_solve
    implementation = _native_solve if engine == "ipopt" else _cyipopt_solve
    x, info = implementation(problem, initial_controls.ravel(), options)
    # Re-evaluate the returned decision vector, not whichever trial point the
    # line search happened to simulate last. Never trust solver status alone.
    outputs = oracle.values(x)
    observations = oracle.last_observations(x)
    task = oracle.task
    scaled_violation = _violation(outputs[1:], task.lower, task.upper)
    control_violation = _violation(x, problem.lower.ravel(), problem.upper.ravel())
    max_g = float(np.max(scaled_violation, initial=0.0))
    max_u = float(np.max(control_violation, initial=0.0))
    physical = outputs[1:] * task.physical_scales
    reports, cursor = {}, 0
    for spec in task.constraints:
        count = len(spec.lower)
        value = physical[cursor:cursor + count]
        violation = _violation(value, spec.lower, spec.upper)
        # JSON null encodes unbounded sides; avoid non-standard +/-Infinity.
        as_bounds = lambda a: [float(v) if np.isfinite(v) else None for v in a]
        reports[spec.name] = {"value": value.tolist(), "lower": as_bounds(spec.lower),
                              "upper": as_bounds(spec.upper), "scale": spec.scale.tolist(),
                              "max_physical_violation": float(np.max(violation, initial=0.0))}
        cursor += count
    raw_terms = np.asarray(task.raw_objectives(x, observations)).ravel()
    objective_terms = {spec.name: {"raw_value": float(raw), "weighted_scaled_value": float(raw * spec.weight / spec.scale)}
                       for spec, raw in zip(task.objectives, raw_terms)}
    result = SolveResult(
        controls=x.reshape(problem.layout.control_shape), trajectory=problem.layout.trajectory(observations),
        objective=float(outputs[0]), status=info["status"], solver_success=info["solver_success"],
        feasible=(max_g <= feasibility_tolerance and max_u <= 1e-7),
        max_scaled_constraint_violation=max_g, max_control_bound_violation=max_u,
        constraint_reports=reports, objective_terms=objective_terms,
        iterations=info["iterations"], elapsed_seconds=time.perf_counter() - start,
        engine=engine, evaluations=oracle.evaluations, derivative_evaluations=oracle.derivative_evaluations,
        history=info["history"],
    )
    if raise_on_failure:
        result.require_success()
    return result

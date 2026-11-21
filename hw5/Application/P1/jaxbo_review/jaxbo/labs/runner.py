"""Seed-separated experiment execution, post-run grading, and JSON/CSV export."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
import csv
import json
import time
import jax
import jax.numpy as jnp
from .core import Problem, ExperimentSession, BudgetExhausted, Policy, Observation
from .policies import ConfiguredPolicy, PolicyConfig


@dataclass
class RunResult:
    """Complete in-memory output from one independent experiment episode.

    Attributes:
        task: Problem identifier.
        config: JSON-serializable policy configuration used for the episode.
        seed: Nonnegative algorithm/noise seed supplied to ``run_experiment``.
        score: Post-run assessment of the final recommendation and budget use.
            These values are never returned to the policy during the episode.
        observations: Chronological tuple of charged ``Observation`` objects.
        recommendation_trace: Saved recommendations at requested checkpoints;
            latent diagnostic values are attached only after optimization ends.
        diagnostics: Policy-specific, JSON-serializable diagnostic information.
        elapsed_seconds: Measured wall-clock seconds for episode execution and
            post-run assessment, excluding construction of the policy/problem.
    """

    task: str
    config: dict
    seed: int
    score: dict
    observations: tuple[Observation, ...]
    recommendation_trace: list[dict]
    diagnostics: dict
    elapsed_seconds: float

    def summary(self):
        return dict(task=self.task,seed=self.seed,config=self.config,score=self.score,
                    diagnostics=self.diagnostics,elapsed_seconds=self.elapsed_seconds)

    def save(self, directory: str | Path):
        path = Path(directory)
        path.mkdir(parents=True,exist_ok=True)
        (path/'result.json').write_text(json.dumps(self.summary(),indent=2,allow_nan=False)+'\n')
        records = []
        for obs in self.observations:
            rec = asdict(obs)
            rec['x'] = obs.x.tolist()
            records.append(rec)
        (path/'observations.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
        (path/'recommendations.json').write_text(json.dumps(self.recommendation_trace,indent=2,allow_nan=False)+'\n')
        with (path/'observations.csv').open('w',newline='') as f:
            fields = ['step','cumulative_cost','cumulative_runs','y','noise_variance','constraint','cost','replicates']
            dim = self.observations[0].x.size if self.observations else 0
            fields += [f'x{i}' for i in range(dim)]
            writer = csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            for i,rec in enumerate(records):
                out = {k:v for k,v in rec.items() if k!='x'}
                out.update(step=i+1,**{f'x{j}':v for j,v in enumerate(rec['x'])})
                writer.writerow(out)
        return path


def run_experiment(problem: Problem, config: PolicyConfig | None = None, *, seed: int = 0,
                   policy_factory: Callable | None = None, trace_every: int = 10,
                   validation_runs: int = 256, callback: Callable | None = None) -> RunResult:
    """Run one episode. Costs and initialization are never free.

    Custom factory signature: factory(spec, *, key, initial_points) -> Policy.
    The factory receives no Problem, latent response function, witness, or metadata.
    Callback is observational: callback(step, observation), never oracle scores.
    Post-run scores are never passed back to the policy as episode feedback.
    """
    if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:
        raise ValueError('seed must be a nonnegative integer')
    if isinstance(trace_every,bool) or not isinstance(trace_every,int) or trace_every<0:
        raise ValueError('trace_every must be a nonnegative integer; 0 disables intermediate recommendations')
    if isinstance(validation_runs,bool) or not isinstance(validation_runs,int) or validation_runs<2:
        raise ValueError('validation_runs must be an integer of at least two')
    cfg = PolicyConfig() if config is None else config
    root = jax.random.key(seed)
    # Stable named streams: changing policy call counts never changes simulator
    # noise, the public initial design, or the independent validation batch.
    design_key = jax.random.fold_in(root,101)
    noise_key = jax.random.fold_in(root,202)
    policy_key = jax.random.fold_in(root,303)
    validation_key = jax.random.fold_in(root,404)
    n0 = min(problem.spec.n_initial,problem.spec.capacity)
    initial = problem._initial(design_key,n0)
    if policy_factory is None:
        policy = ConfiguredPolicy(problem.spec,cfg,key=policy_key,initial_points=initial)
    else:
        policy = policy_factory(problem.spec,key=policy_key,initial_points=initial)
    session = ExperimentSession(problem,key=noise_key)
    trace = []
    started = time.perf_counter()
    # No silent catch-all fallback: programming/numerical errors remain visible.
    while session.remaining >= problem.spec.minimum_cost-1e-8:
        try:
            action = policy.ask(session.remaining)
            obs = session.evaluate(action)
        except BudgetExhausted:
            break
        policy.tell(obs)
        step = len(session.history)
        if callback is not None:
            callback(step,obs)
        if trace_every and step % trace_every == 0:
            x = policy.recommend()
            # Record only inputs now. Grading happens AFTER the policy is done.
            trace.append(dict(step=step,spent=session.spent,recommendation=x.tolist()))
    if not session.history:
        raise RuntimeError('No experiment completed; check budget and initial policy requests')
    recommendation = policy.recommend()
    grade = session.finalize(recommendation,validation_key=validation_key,validation_runs=validation_runs)
    final_step = len(session.history)
    if not trace or trace[-1]['step']!=final_step:
        trace.append(dict(step=final_step,spent=session.spent,recommendation=recommendation.tolist()))
    else:
        trace[-1]['recommendation']=recommendation.tolist()
    # Post-run diagnostics are never returned to the policy or callback.
    truth = jax.jit(problem._truth)
    for item in trace:
        t = truth(jnp.asarray(item['recommendation']))
        item.update(true_mean=float(t.mean),feasible=bool(t.constraint<=0))
    grade.update(instance=problem.metadata.get('instance'),split=problem.metadata.get('split'),
                 simulator_version=problem.metadata.get('simulator_version'),
                 budget=problem.spec.budget,budget_unit=problem.spec.budget_unit)
    diagnostic = policy.diagnostics() if hasattr(policy,'diagnostics') else {}
    return RunResult(problem.spec.name,cfg.to_dict() if policy_factory is None else {'custom_policy':True},
                     seed,grade,session.history,trace,diagnostic,time.perf_counter()-started)

# Problem 3.2 — Contact physics

Source: homework PDF, Exercise 3.2, pp. 7–8.

### AI Usage Declaration

I used ChatGPT Sol 6.1 on this problem to understand problem formulation, verify my code and run the required simulations. It was also used to rewrite equations and format tables in markdown. 


## 1. Contact QP

- Introduce a slip bound s_i ≥ ||T_i u||∞ for each active plane:

$$
\min_{u,s}\;\frac m2\|u-v^-\|_2^2+\sum_{i\in\mathcal A}\beta_i s_i,
\quad N_i u\ge d_i,\quad -s_i\mathbf1\le T_i u\le s_i\mathbf1,\quad s_i\ge0.
$$

- For x=[u,s], **Q=diag(m I₃,0)** and **c=[−m v⁻,β]**. Omit the constant m||v⁻||²/2. Return `QP(Q,c,A,lower,upper)`.
- R repeats each identity row twice. Preserve this constraint order for the supplied dual-to-impulse recovery:

| Family | Rows | A block | Active lower / upper |
|---|---:|---|---|
| Normal | n | [N,0] | d / +∞ |
| Positive tangent | 2n | [T,−R] | −∞ / 0 |
| Negative tangent | 2n | [−T,−R] | −∞ / 0 |
| Slip bound | n | [0,I] | 0 / +∞ |

- Inactive normal/tangent rows are unbounded; inactive β=0 and s=0. Keep all slots for batching.
- β is the **prescribed friction-impulse budget**, not μ times the solved normal impulse. Resolve all simultaneous contacts together in **one QP**.

## 2. Collision time

- During straight drift, φ_i(t)=φ_i(0)+t n_iᵀv, where φ_i(0)=n_iᵀx−b_i−radius. Therefore:

$$
t_i=\begin{cases}
\max(\phi_i(0),0)/(-n_i^\top v),&n_i^\top v<-\varepsilon_v,\\
+\infty,&\text{otherwise}.
\end{cases}
$$

- Return the smallest t_i; `hit=True` only when finite and within the remaining interval, including its endpoint.
- Touching + approaching gives t=0. Parallel, separating, or speeds below the approach threshold give no candidate. A later collision retains its finite time with `hit=False`.
- Clamping handles gap roundoff; the event driver separately rejects invalid penetrated states. Experiments use radius=0.

## 3. Verification and measured results

- **4 public tests and 20 analytic/regression checks passed.** Coverage includes both solvers, two tangent components, simultaneous walls, inactive slots, collision edge cases, and integer velocities with fractional mass.
- Fixed the dtype bug: integer velocities no longer truncate fractional mass in Q. Both original TODO markers remain; the supplied event loop and solver adapter are unchanged.
- The supplemental all-inactive direct solve disables BoxOSQP’s infeasibility detector because it raises a false flag for that case. The production event loop calls the QP only at actual contacts.

### Analytic impacts (JAXopt)

| Case | Outgoing velocity | Min gap | Impact ΔKE | Max residual | Status / QP calls |
|---|---|---:|---:|---:|---|
| frictionless | [1.000, 0.000, 1.000] | 0.0 | -1.500 | 3.742e-10 | 1 / 1 |
| fixed_friction_budget | [3.300, 0.000, 1.000] | 0.0 | -4.055 | 1.836e-09 | 1 / 1 |
| two_tangent_components | [2.500, 2.500, 1.000] | 0.0 | -7.750 | 1.500e-09 | 1 / 1 |
| corner_and_inactive_wall | [1.000, 1.500, 0.000] | 0.0 | -4.875 | 3.036e-10 | 1 / 1 |

### Random worlds

- JAXopt, float64, seed 2026; eight worlds, eight planes, μ=0.25, e=0.6; same **1.5-second** duration. Status 1 means solver success.

| Run | dt / QP tolerance | Min gap | Max residual | Max positive impact ΔKE | Status | Events / QP calls | Valid trajectory |
|---|---|---:|---:|---:|---|---|---|
| coarse | 1/60 / 1e-09 | 1.095e-06 | 6.231e-09 | 0.0 | 1 | 33 / 33 | Pass |
| default | 1/120 / 1e-09 | -1.082e-12 | 7.982e-09 | 0.0 | 1 | 78 / 78 | Pass |
| fine | 1/240 / 1e-09 | -1.696e-13 | 7.983e-09 | 0.0 | 1 | 97 / 97 | Pass |
| reference | 1/480 / 1e-09 | -1.541e-13 | 7.981e-09 | 0.0 | 1 | 181 / 181 | Pass |
| tol_1e7 | 1/120 / 1e-07 | -5.191e-11 | 7.740e-07 | 0.0 | 1 | 78 / 78 | Pass |
| tol_1e5 | 1/120 / 1e-05 | -4.441e-16 | 7.847e-05 | 0.0 | 1 | 8 / 8 | Fail |

- Negative gaps are roundoff within the 1e−7 contact tolerance. **One logical QP per event** holds in every run; the two-wall analytic impact also uses one call.
- Zero maximum positive ΔKE means no detected impact-energy increase; it does **not** mean zero dissipation (see negative analytic ΔKE).
- `tol_1e5` is intentionally invalid: rejected worlds freeze, so its event count covers the attempted prefix, not a completed trajectory.

## 4. Time step, QP tolerance, and parallelization

- Compare final positions with dt=1/480 as a **numerical reference**, not an exact solution:

| dt | Steps | Final-position RMS difference |
|---|---:|---:|
| 1/60 | 90 | 0.105528 |
| 1/120 | 180 | 0.024823 |
| 1/240 | 360 | 0.005939 |

- **Integration error:** gravity kick + straight drift approximates acceleration. Smaller dt reduced trajectory differences even with tightly solved QPs.
- **Incomplete optimization:** loose tolerance leaves contact/KKT residuals. At 1e−7, final positions changed by at most 6.358e-08 versus 1e−9, and QP iterations fell from 11913 to 9118. At 1e−5, status 1 was insufficient: residuals exceeded the 2e−6 physical acceptance threshold.
- **Parallel:** independent worlds (`vmap`), constraint assembly, and linear-algebra kernels. **Sequential:** time steps, successive impacts, and solver iterations. Simultaneously touching planes are coupled, not independent QPs.
- Event counts vary with dt, including repeated gravity-driven contacts at rest; fewer events alone does not establish greater accuracy.

## Reproduce

- Run from `hw2/Applications/P2/physics` in a separate environment. Measured with Python 3.12.14 and JAX 0.4.38. The last command is expected to fail trajectory validation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]" "jax==0.4.38" "jaxlib==0.4.38"
python -m pytest -q
python results/validate_cases.py
python -m examples.physics_demo --backend jaxopt --batch 8 --steps 180 --output results/default.json
python -m examples.physics_demo --backend jaxopt --batch 8 --dt 0.016666666666666666 --steps 90 --output results/coarse.json
python -m examples.physics_demo --backend jaxopt --batch 8 --dt 0.004166666666666667 --steps 360 --output results/fine.json
python -m examples.physics_demo --backend jaxopt --batch 8 --dt 0.0020833333333333333 --steps 720 --output results/reference.json
python -m examples.physics_demo --backend jaxopt --batch 8 --steps 180 --solver-tol 1e-7 --output results/tol_1e7.json
python -m examples.physics_demo --backend jaxopt --batch 8 --steps 180 --solver-tol 1e-5 --output results/tol_1e5.json
```

- Each run saves JSON diagnostics, NPZ trajectories, and an HTML viewer. Open `results/default.html`; analytic results are in `results/analytic_checks.json`.

# Problem 3.1 — Poker

Source: homework PDF, Exercise 3.1, pp. 5–6.

### AI Usage Declaration
I used ChatGPT Sol 6.1 on this problem to understand problem formulation, verify my code and run the required simulations. It was also used to rewrite equations and format tables in markdown. 

## Player 1's LP

- Fix community card c. Set p_ia = Pr(bid a | c, own card h_i); each own card has a separate probability distribution.
- K already includes joint deal probabilities. Policies never depend on the opponent's private card.
- The opponent chooses a best response independently for each private card. Introduce one **free** value z_j per opponent card:

$$
\begin{aligned}
\max_{p,z}\quad &\sum_j z_j\\
\text{subject to}\quad
&\sum_a p_{ia}=1 &&\text{for each }i,\\
&p_{ia}\ge0 &&\text{for each }i,a,\\
&z_j\le\sum_{i,a}K_{(i,a),(j,b)}p_{ia} &&\text{for each }j,b.
\end{aligned}
$$

- At the optimum, z_j is the minimum opponent-action contribution for card j. Their sum is player 1's guaranteed payoff. Negative values must remain allowed.

## QP container and code verification

- `row_lp` returns `QP(Q,c,A,lower,upper)` with variables `[p.ravel(), z_scaled]` and **Q=0**; this is an LP.
- One scale s=max|K| (s=1 for a zero kernel) gives K̃=K/s and z_scaled=z/s. The objective is **−sum(z_scaled)/col_types**, a positive multiple of the original minimization objective. Averaging preserves the optimum and avoids tiny objective coefficients.
- A stacks the simplex rows `[I⊗1,0]` (bounds 1,1), probability rows `[I,0]` (bounds 0,+∞), and payoff rows `[K̃ᵀ,−R]` (bounds 0,+∞). R repeats each opponent-type identity row once per action.
- Player 2 uses the same LP on **−Kᵀ**. Evaluate chip payoffs with the original K, not the scaled objective.
- Recheck: **3 public tests + 12 independent LP checks passed**, including integer inputs, negative/zero values, asymmetric type/action counts, and scales 1e−12, 1, 1e12. Maximum normalized value error: **1.11e−16**.
- The TODO marker remains. Integer inputs are now promoted to floating point. `qp.py`, the rules, evaluator, and demo were not changed.

## Bounds and equilibrium quality

$$
L(p)=\sum_j\min_b(K^\top p)_{jb},\qquad
R(q)=\sum_i\max_a(Kq)_{ia},\qquad
\text{gap}=R(q)-L(p).
$$

- L is player 1's guarantee; R is the best-response payoff against player 2. The minimax value lies between them.
- The gap bounds either player's unilateral improvement. Every board's exact value is **0** by symmetry and zero-sum payoffs.
- Full-deck **JAXopt convergence remains unresolved**; its failed policies are not used below. Full-deck tables use the supplied **SciPy/HiGHS oracle**. Six-card tables use JAXopt.

| Deck | Bids | Certified backend | Maximum absolute gap (chips) | All LP statuses successful |
|---|---|---|---:|---|
| 6 | [0, 1] | jaxopt | 5.499e-09 | Yes |
| 6 | [0, 1, 2] | jaxopt | 6.481e-09 | Yes |
| 52 | [0, 1] | scipy | 3.608e-16 | Yes |
| 52 | [0, 1, 2] | scipy | 3.608e-15 | Yes |

## Policies for community card 2C (ID 0)

- C,D label the two suits in the six-card deck; C,D,H,S label the four suits in the full deck. Suits never break ties.
- Each row gives player 1’s conditional probabilities. Bids are the column labels; six-decimal rounding may affect the last digit of a row sum.
- These are **one** certified equilibrium per case. Different optimal mixtures can have the same game value. Full-precision policies for both players are saved in the JSON/NPZ files.

### 6-card deck

| Hole card | {0,1}: P(0) | P(1) | {0,1,2}: P(0) | P(1) | P(2) |
|---|---:|---:|---:|---:|---:|
| 2D | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| 3C | 0.726101 | 0.273899 | 0.848420 | 0.018246 | 0.133333 |
| 3D | 0.726101 | 0.273899 | 0.315087 | 0.284913 | 0.400000 |
| 4C | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| 4D | 0.607232 | 0.392768 | 0.484913 | 0.515087 | 0.000000 |

### 52-card deck

| Hole card | {0,1}: P(0) | P(1) | {0,1,2}: P(0) | P(1) | P(2) |
|---|---:|---:|---:|---:|---:|
| 2D | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| 2H | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| 2S | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| 3C | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| 3D | 1.000000 | 0.000000 | 0.342802 | 0.329521 | 0.327677 |
| 3H | 0.666667 | 0.333333 | 0.342802 | 0.329521 | 0.327677 |
| 3S | 0.000000 | 1.000000 | 0.342802 | 0.329521 | 0.327677 |
| 4C | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| 4D | 1.000000 | 0.000000 | 0.759030 | 0.240970 | 0.000000 |
| 4H | 0.333333 | 0.666667 | 0.759030 | 0.240970 | 0.000000 |
| 4S | 1.000000 | 0.000000 | 0.579729 | 0.420271 | 0.000000 |
| 5C | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| 5D | 0.666667 | 0.333333 | 0.469025 | 0.159623 | 0.371352 |
| 5H | 0.000000 | 1.000000 | 0.469025 | 0.159623 | 0.371352 |
| 5S | 1.000000 | 0.000000 | 0.469025 | 0.159623 | 0.371352 |
| 6C | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| 6D | 1.000000 | 0.000000 | 0.254340 | 0.745660 | 0.000000 |
| 6H | 1.000000 | 0.000000 | 0.674156 | 0.325844 | 0.000000 |
| 6S | 0.333333 | 0.666667 | 0.674156 | 0.325844 | 0.000000 |
| 7C | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| 7D | 0.666667 | 0.333333 | 0.653482 | 0.018841 | 0.327677 |
| 7H | 1.000000 | 0.000000 | 0.653482 | 0.018841 | 0.327677 |
| 7S | 0.000000 | 1.000000 | 0.653482 | 0.018841 | 0.327677 |
| 8C | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| 8D | 1.000000 | 0.000000 | 0.335680 | 0.545288 | 0.119032 |
| 8H | 1.000000 | 0.000000 | 0.335680 | 0.545288 | 0.119032 |
| 8S | 0.333333 | 0.666667 | 0.335680 | 0.545288 | 0.119032 |
| 9C | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| 9D | 0.666667 | 0.333333 | 0.824390 | 0.023831 | 0.151779 |
| 9H | 0.000000 | 1.000000 | 0.824390 | 0.023831 | 0.151779 |
| 9S | 1.000000 | 0.000000 | 0.824390 | 0.023831 | 0.151779 |
| 10C | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| 10D | 1.000000 | 0.000000 | 0.240688 | 0.459295 | 0.300017 |
| 10H | 1.000000 | 0.000000 | 0.240688 | 0.459295 | 0.300017 |
| 10S | 1.000000 | 0.000000 | 0.240688 | 0.459295 | 0.300017 |
| JC | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| JD | 0.000000 | 1.000000 | 0.974169 | 0.000000 | 0.025831 |
| JH | 0.333333 | 0.666667 | 1.000000 | 0.000000 | 0.000000 |
| JS | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| QC | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| QD | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| QH | 0.000000 | 1.000000 | 0.525831 | 0.000000 | 0.474169 |
| QS | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| KC | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| KD | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| KH | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| KS | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| AC | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| AD | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| AH | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| AS | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |

## Does the policy bluff?

- **Yes.** A higher bid can win without showdown against a lower bid. Weak hands therefore sometimes bid positively.
- The examples below have negative expected showdown outcome against both a uniform legal opponent card and the equilibrium opponent conditioned on the same bid. This gives a conservative, explicit meaning of “bluff.”

| Deck / bids | Weak hole card on 2C | Positive-bid probabilities |
|---|---|---|
| 6 / [0, 1] | 3D | P(1)=0.273899 |
| 6 / [0, 1, 2] | 3D | P(1)=0.284913, P(2)=0.400000 |
| 52 / [0, 1] | 3H | P(1)=0.333333 |
| 52 / [0, 1, 2] | 3D | P(1)=0.329521, P(2)=0.327677 |

- On the six-card 2C board, **3D loses every legal showdown**, so its positive bids are clear bluffs. Other mixtures and bluff frequencies may differ across equally optimal policies.

## Best-response bounds on every board

- All numbers below come from the supplied evaluator, in **chips**. Tiny negative gaps around 1e−16 are floating-point roundoff; the mathematical gap is nonnegative.

### 6-card deck

| Community | {0,1}: L | R | Gap | {0,1,2}: L | R | Gap |
|---|---:|---:|---:|---:|---:|---:|
| 2C | -2.750e-09 | 2.750e-09 | 5.499e-09 | -3.240e-09 | 3.240e-09 | 6.481e-09 |
| 2D | -2.750e-09 | 2.750e-09 | 5.499e-09 | -3.240e-09 | 3.240e-09 | 6.480e-09 |
| 3C | -2.750e-09 | 2.750e-09 | 5.499e-09 | -3.240e-09 | 3.240e-09 | 6.480e-09 |
| 3D | -2.750e-09 | 2.750e-09 | 5.499e-09 | -3.240e-09 | 3.240e-09 | 6.480e-09 |
| 4C | -2.750e-09 | 2.750e-09 | 5.499e-09 | -3.240e-09 | 3.240e-09 | 6.480e-09 |
| 4D | -2.750e-09 | 2.750e-09 | 5.499e-09 | -3.240e-09 | 3.240e-09 | 6.480e-09 |

### 52-card deck

| Community | {0,1}: L | R | Gap | {0,1,2}: L | R | Gap |
|---|---:|---:|---:|---:|---:|---:|
| 2C | 1.110e-16 | -1.110e-16 | -2.220e-16 | -1.631e-15 | 1.631e-15 | 3.261e-15 |
| 2D | 1.249e-16 | -1.249e-16 | -2.498e-16 | -4.163e-16 | 4.163e-16 | 8.327e-16 |
| 2H | 1.110e-16 | -1.110e-16 | -2.220e-16 | -3.192e-16 | 3.192e-16 | 6.384e-16 |
| 2S | 1.388e-16 | -1.388e-16 | -2.776e-16 | -4.857e-16 | 4.857e-16 | 9.714e-16 |
| 3C | 1.110e-16 | -1.110e-16 | -2.220e-16 | -1.478e-15 | 1.478e-15 | 2.956e-15 |
| 3D | 1.249e-16 | -1.249e-16 | -2.498e-16 | -1.721e-15 | 1.721e-15 | 3.442e-15 |
| 3H | 1.110e-16 | -1.110e-16 | -2.220e-16 | -3.747e-16 | 3.747e-16 | 7.494e-16 |
| 3S | 1.388e-16 | -1.388e-16 | -2.776e-16 | -3.886e-16 | 3.886e-16 | 7.772e-16 |
| 4C | 1.110e-16 | -1.110e-16 | -2.220e-16 | -1.492e-15 | 1.492e-15 | 2.984e-15 |
| 4D | 9.714e-17 | -9.714e-17 | -1.943e-16 | -3.886e-16 | 3.886e-16 | 7.772e-16 |
| 4H | 1.110e-16 | -1.110e-16 | -2.220e-16 | -3.261e-16 | 3.261e-16 | 6.523e-16 |
| 4S | 1.388e-16 | -1.388e-16 | -2.776e-16 | -4.718e-16 | 4.718e-16 | 9.437e-16 |
| 5C | 1.110e-16 | -1.110e-16 | -2.220e-16 | -1.540e-15 | 1.540e-15 | 3.081e-15 |
| 5D | 1.388e-16 | -1.388e-16 | -2.776e-16 | -4.163e-16 | 4.163e-16 | 8.327e-16 |
| 5H | 1.388e-16 | -1.388e-16 | -2.776e-16 | -3.955e-16 | 3.955e-16 | 7.910e-16 |
| 5S | 1.527e-16 | -1.527e-16 | -3.053e-16 | -1.471e-15 | 1.471e-15 | 2.942e-15 |
| 6C | 9.714e-17 | -9.714e-17 | -1.943e-16 | -2.776e-16 | 2.776e-16 | 5.551e-16 |
| 6D | 9.714e-17 | -9.714e-17 | -1.943e-16 | -1.436e-15 | 1.436e-15 | 2.873e-15 |
| 6H | 1.388e-16 | -1.388e-16 | -2.776e-16 | -3.053e-16 | 3.053e-16 | 6.106e-16 |
| 6S | 1.388e-16 | -1.388e-16 | -2.776e-16 | -2.498e-16 | 2.498e-16 | 4.996e-16 |
| 7C | 1.110e-16 | -1.110e-16 | -2.220e-16 | -1.658e-15 | 1.658e-15 | 3.317e-15 |
| 7D | 1.110e-16 | -1.110e-16 | -2.220e-16 | -1.381e-15 | 1.381e-15 | 2.762e-15 |
| 7H | 1.527e-16 | -1.527e-16 | -3.053e-16 | -2.914e-16 | 2.914e-16 | 5.829e-16 |
| 7S | 1.804e-16 | -1.804e-16 | -3.608e-16 | -4.718e-16 | 4.718e-16 | 9.437e-16 |
| 8C | 8.327e-17 | -8.327e-17 | -1.665e-16 | -1.721e-15 | 1.721e-15 | 3.442e-15 |
| 8D | 8.327e-17 | -8.327e-17 | -1.665e-16 | -1.596e-15 | 1.596e-15 | 3.192e-15 |
| 8H | 8.327e-17 | -8.327e-17 | -1.665e-16 | -1.624e-15 | 1.624e-15 | 3.247e-15 |
| 8S | 1.110e-16 | -1.110e-16 | -2.220e-16 | -4.441e-16 | 4.441e-16 | 8.882e-16 |
| 9C | 8.327e-17 | -8.327e-17 | -1.665e-16 | -3.886e-16 | 3.886e-16 | 7.772e-16 |
| 9D | 8.327e-17 | -8.327e-17 | -1.665e-16 | -2.776e-16 | 2.776e-16 | 5.551e-16 |
| 9H | 8.327e-17 | -8.327e-17 | -1.665e-16 | -3.608e-16 | 3.608e-16 | 7.216e-16 |
| 9S | 8.327e-17 | -8.327e-17 | -1.665e-16 | -1.471e-15 | 1.471e-15 | 2.942e-15 |
| 10C | 8.327e-17 | -8.327e-17 | -1.665e-16 | -1.416e-15 | 1.416e-15 | 2.831e-15 |
| 10D | 8.327e-17 | -8.327e-17 | -1.665e-16 | -2.220e-16 | 2.220e-16 | 4.441e-16 |
| 10H | 8.327e-17 | -8.327e-17 | -1.665e-16 | -1.471e-15 | 1.471e-15 | 2.942e-15 |
| 10S | 8.327e-17 | -8.327e-17 | -1.665e-16 | -3.331e-16 | 3.331e-16 | 6.661e-16 |
| JC | 8.327e-17 | -8.327e-17 | -1.665e-16 | -1.804e-15 | 1.804e-15 | 3.608e-15 |
| JD | 8.327e-17 | -8.327e-17 | -1.665e-16 | -1.416e-15 | 1.416e-15 | 2.831e-15 |
| JH | 1.388e-16 | -1.388e-16 | -2.776e-16 | -1.554e-15 | 1.554e-15 | 3.109e-15 |
| JS | 1.110e-16 | -1.110e-16 | -2.220e-16 | -4.441e-16 | 4.441e-16 | 8.882e-16 |
| QC | 8.327e-17 | -8.327e-17 | -1.665e-16 | -3.053e-16 | 3.053e-16 | 6.106e-16 |
| QD | 8.327e-17 | -8.327e-17 | -1.665e-16 | -2.498e-16 | 2.498e-16 | 4.996e-16 |
| QH | 1.388e-16 | -1.388e-16 | -2.776e-16 | -3.053e-16 | 3.053e-16 | 6.106e-16 |
| QS | 1.665e-16 | -1.665e-16 | -3.331e-16 | -1.499e-15 | 1.499e-15 | 2.998e-15 |
| KC | 8.327e-17 | -8.327e-17 | -1.665e-16 | -4.163e-16 | 4.163e-16 | 8.327e-16 |
| KD | 8.327e-17 | -8.327e-17 | -1.665e-16 | -3.608e-16 | 3.608e-16 | 7.216e-16 |
| KH | 1.388e-16 | -1.388e-16 | -2.776e-16 | -3.053e-16 | 3.053e-16 | 6.106e-16 |
| KS | 1.665e-16 | -1.665e-16 | -3.331e-16 | -3.331e-16 | 3.331e-16 | 6.661e-16 |
| AC | 8.327e-17 | -8.327e-17 | -1.665e-16 | -4.718e-16 | 4.718e-16 | 9.437e-16 |
| AD | 8.327e-17 | -8.327e-17 | -1.665e-16 | -1.471e-15 | 1.471e-15 | 2.942e-15 |
| AH | 8.327e-17 | -8.327e-17 | -1.665e-16 | -1.554e-15 | 1.554e-15 | 3.109e-15 |
| AS | 1.110e-16 | -1.110e-16 | -2.220e-16 | -1.388e-15 | 1.388e-15 | 2.776e-15 |

## Reproduce

- Run from `Applications/P1/poker` in a separate environment from physics. Measured with Python 3.12.14, JAX 0.4.38, float64, tolerance 1e−8, maxiter 30,000, seed 7, and 50,000 sampled deals; certification uses exact evaluator bounds.
- `--repeats 1` changes timing repetition only. JSON contains diagnostics; NPZ contains both players’ full policies and every board’s bounds. Old `*_jaxopt_attempt.*` files are failed historical runs, not certified results.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]" "jax==0.4.38" "jaxlib==0.4.38"
python -m pytest -q
python results/validate_formulation.py
python -m examples.poker_demo --backend jaxopt --bids 0 1 --repeats 1 --output results/six_01.json
python -m examples.poker_demo --backend jaxopt --bids 0 1 2 --repeats 1 --output results/six_012.json
python -m examples.poker_demo --backend scipy --ranks 13 --suits 4 --bids 0 1 --repeats 1 --output results/full_01.json
python -m examples.poker_demo --backend scipy --ranks 13 --suits 4 --bids 0 1 2 --repeats 1 --output results/full_012.json
```

# Graph Report - Student_Distribution-2  (2026-10-04)

## Corpus Check
- Corpus is ~10,985 words - fits in a single context window. You may not need a graph.

## Summary
- 291 nodes · 587 edges · 13 communities (9 shown, 4 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 17 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Linkage Geometry and Simulation
- Contact Physics and Events
- Poker Rules and Games
- Application Instructions and Experiments
- Poker Solver and Tests
- Physics Solver and Tests
- Linkage Visualization and Driver
- Implicit Differentiation and Optimization
- Linkage Instructions and Dependencies
- Poker Example Package
- Physics Example Package
- Physics Package Metadata
- Poker Package Metadata

## God Nodes (most connected - your core abstractions)
1. `FourBarMechanism` - 31 edges
2. `Solver` - 14 edges
3. `Four-bar linkage optimization` - 14 edges
4. `Solver` - 13 edges
5. `step()` - 13 edges
6. `Poker application` - 13 edges
7. `Contact physics application` - 12 edges
8. `main()` - 11 edges
9. `solve_game()` - 11 edges
10. `resolve_contact()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `row_lp()` --uses--> `QP`  [INFERRED]
  Applications/P1/poker/games.py → Applications/P1/poker/qp.py
- `solve_game()` --uses--> `Solver`  [INFERRED]
  Applications/P1/poker/games.py → Applications/P1/poker/qp.py
- `contact_qp()` --uses--> `QP`  [INFERRED]
  Applications/P2/physics/physics.py → Applications/P2/physics/qp.py
- `resolve_contact()` --uses--> `Solver`  [INFERRED]
  Applications/P2/physics/physics.py → Applications/P2/physics/qp.py
- `step()` --uses--> `Solver`  [INFERRED]
  Applications/P2/physics/physics.py → Applications/P2/physics/qp.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Simulation underlies sensitivity and optimization** — implementation_linkage_readme_forward_simulation, implementation_linkage_readme_closure_energy, implementation_linkage_readme_batched_state_sensitivities, implementation_linkage_readme_design_objective_and_gradient, implementation_linkage_readme_gradient_descent_update [EXTRACTED 1.00]

## Communities (13 total, 4 thin omitted)

### Community 0 - "Linkage Geometry and Simulation"
Cohesion: 0.12
Nodes (55): build_link_pose_stack(), checked_simulation_from_lengths(), closed_form_initial_guess(), closure_constraint_norms(), closure_energy(), contact_pairs(), curve_loss_from_states_and_lengths(), curve_loss_with_simulation() (+47 more)

### Community 1 - "Contact Physics and Events"
Cohesion: 0.07
Nodes (46): main(), Ball, contact_jacobians(), contact_parameters(), contact_qp(), contact_residuals(), ContactResult, gaps() (+38 more)

### Community 2 - "Poker Rules and Games"
Cohesion: 0.08
Nodes (37): main(), Exact poker solution + best responses + optional sampling and timing., evaluate(), GameResult, make_batched_game_solver(), normalize_roundoff(), Array, NamedTuple (+29 more)

### Community 3 - "Application Instructions and Experiments"
Cohesion: 0.07
Nodes (29): Poker application, Bids 0 and 1, Bids 0, 1, and 2, 52-card experiments, games.py: LP formulation and game-solving helpers, Authoritative homework PDF, examples/poker_demo.py: experiment driver, poker.py: rules, payoff, deal construction (+21 more)

### Community 4 - "Poker Solver and Tests"
Cohesion: 0.18
Nodes (14): kkt_residuals(), _max(), NamedTuple, QP, QPResult, Shared QP boundary: min .5*x.T Q x + c.T x, lower <= A x <= upper. JAXopt…, One CPU LP/QP solve, with explicit conversion of signed multipliers. Redundant…, Original-unit feasibility, stationarity, and dual/complementarity errors.… (+6 more)

### Community 5 - "Physics Solver and Tests"
Cohesion: 0.18
Nodes (14): kkt_residuals(), _max(), NamedTuple, QP, QPResult, Shared QP boundary: min .5*x.T Q x + c.T x, lower <= A x <= upper. JAXopt…, One CPU LP/QP solve, with explicit conversion of signed multipliers. Redundant…, Original-unit feasibility, stationarity, and dual/complementarity errors.… (+6 more)

### Community 6 - "Linkage Visualization and Driver"
Cohesion: 0.15
Nodes (19): Figure, FuncAnimation, animate_linkage(), _bounds(), _endpoint(), plot_curve_comparison(), plot_linkage_snapshots(), plot_loss_history() (+11 more)

### Community 7 - "Implicit Differentiation and Optimization"
Cohesion: 0.13
Nodes (20): check_design_result(), DesignHistory, DesignResult, Scalar loss, total (4,) gradient, (T,6,4) sensitivities, forward solve., Histories contain num_steps + 1 EVALUATED designs, including the final one.…, Catch unfinished/invalid numerical answers before they drive an update., batched_state_sensitivities(), build_parser() (+12 more)

### Community 8 - "Linkage Instructions and Dependencies"
Cohesion: 0.12
Nodes (20): Four-bar linkage optimization, batched_state_sensitivities student implementation, closure_energy student implementation, design_objective_and_gradient student implementation, Forward linkage simulation, gradient_descent_update student implementation, Authoritative homework PDF, linkage_core.py: model, simulation, objectives, validation (+12 more)

## Knowledge Gaps
- **33 isolated node(s):** `hw2-poker-application`, `hw2-physics-application`, `Authoritative homework PDF`, `Python 3.11 or newer`, `row_lp student implementation` (+28 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 110 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Poker Rules and Games` to `Poker Solver and Tests`, `Linkage Visualization and Driver`?**
  _High betweenness centrality (0.263) - this node is a cross-community bridge._
- **Why does `main()` connect `Contact Physics and Events` to `Physics Solver and Tests`, `Linkage Visualization and Driver`?**
  _High betweenness centrality (0.246) - this node is a cross-community bridge._
- **Why does `build_parser()` connect `Implicit Differentiation and Optimization` to `Linkage Visualization and Driver`?**
  _High betweenness centrality (0.104) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Solver` (e.g. with `resolve_contact()` and `step()`) actually correct?**
  _`Solver` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Solver` (e.g. with `solve_game()` and `oracle()`) actually correct?**
  _`Solver` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `hw2-poker-application`, `hw2-physics-application`, `Authoritative homework PDF` to the rest of the system?**
  _33 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Linkage Geometry and Simulation` be split into smaller, more focused modules?**
  _Cohesion score 0.11948051948051948 - nodes in this community are weakly interconnected._
## Extraction limitations
Semantic token counters are unmeasured placeholders. Raw extraction has 95 dangling-endpoint edges and 61 collapsed same-endpoint edges. Ten physics-to-poker solver references were corrected to the local physics solver after source verification.

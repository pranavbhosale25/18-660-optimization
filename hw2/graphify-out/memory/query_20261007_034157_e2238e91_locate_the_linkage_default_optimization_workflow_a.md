---
type: "query"
date: "2026-10-07T03:41:57.805705+00:00"
question: "Locate the linkage default optimization workflow and writeup artifacts"
contributor: "graphify"
outcome: "useful"
source_nodes: ["optimize.py", "linkage_core.py", "linkage_viz.py"]
---

# Q: Locate the linkage default optimization workflow and writeup artifacts

## Answer

Expanded from graph vocabulary: linkage optimize objective history animation checks. The graph identified linkage_core.py for simulation/objectives, optimize.py for outer design optimization and saved histories, and linkage_viz.py for plots/animations. Direct source inspection and execution confirmed the default artifacts. The distributed checks.py was missing; a separately labeled local harness was added.

## Outcome

- Signal: useful

## Source Nodes

- optimize.py
- linkage_core.py
- linkage_viz.py
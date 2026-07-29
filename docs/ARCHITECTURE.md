# Architecture

The package is divided into core geometry, planner adapters, evaluation,
visualization, utilities, reproducible experiments, and preserved third-party
sources.

Each planner keeps its native representation and native search objective.
Adapters perform only documented map/coordinate conversion. Cross-algorithm
comparison is based on common post-run path and runtime metrics, never on raw
native objective values.

Only success, path geometry, clearance, and planner-reported execution time
are intended for cross-algorithm inference. CPU time and Python-tracked memory
are diagnostic because MATLAB runs in a child process. Iterations, expanded
nodes, generated nodes, and fitness evaluations also retain planner-specific
meanings and are not included in the overall cross-algorithm rank.

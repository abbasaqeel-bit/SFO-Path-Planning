# Experimental Setup

## Benchmark protocol

The experiments evaluate static global path planning in known two-dimensional
workspaces. Every map has continuous Cartesian bounds of 100 by 100 units and
contains only static circular and axis-aligned rectangular obstacles. The
robot is modelled as a disc; collision checking accounts for the radius given
for each map. Start and goal locations are fixed within a map and are shared
by every planner.

Nine planners were evaluated: A*, Dijkstra, RRT, RRT*, Informed RRT*, PRM,
ACO, ACO-GA, and the proposed Three-Phase Incremental SFO. The first six
reference planners use the vendored PythonRobotics implementations. The ACO
adapter invokes the archived Haghrah implementation without modifying its
search mechanism. ACO-GA is executed through the bundled MATLAB bridge for
the paper-faithful independent reproduction. Each planner retains its native
solution representation and internal objective function. Common metrics are
computed only after the planner returns a path.

For each map-planner combination, five independent runs were performed using
a deterministic seed schedule initialized from base seed 2026. Therefore, the
benchmark comprises 9 planners × 3 maps × 5 runs = 135 planned runs. The
benchmark runner stores the complete run record, a static path plot, a GIF
diagnostic, and the planner state for every attempt.

## Map specifications

| Map | Bounds | Start | Goal | Robot radius | Obstacles | Purpose |
|---|---:|---:|---:|---:|---|---|
| Dense-obstacle (`dense_01`) | 100 × 100 | (5, 5) | (95, 95) | 0.5 | 3 circles + 2 rectangles | Tests routing through mixed clutter. |
| Narrow-passage (`narrow_passage_01`) | 100 × 100 | (8, 12) | (92, 88) | 1.0 | 4 circles + 6 rectangles | Tests passage discovery through a central wall opening. |
| Bug-trap (`bug_trap_01`) | 100 × 100 | (48, 50) | (92, 50) | 1.0 | 3 circles + 5 rectangles | Tests escape from a U-shaped trap before goal-directed motion. |

## Planner settings

| Planner | Source / runtime | Parameters used in the benchmark |
|---|---|---|
| A* | PythonRobotics | Grid resolution = 1.0. |
| Dijkstra | PythonRobotics | Grid resolution = 1.0. |
| RRT | PythonRobotics | Rectangle resolution = 1.0; expansion distance = 4.0; path resolution = 0.5; goal-sampling rate = 10%; maximum iterations = 1000. |
| RRT* | PythonRobotics | Rectangle resolution = 1.0; expansion distance = 4.0; path resolution = 0.5; goal-sampling rate = 10%; maximum iterations = 1000; connection-circle distance = 50.0; search until maximum iteration = enabled. |
| Informed RRT* | PythonRobotics | Rectangle resolution = 1.0; expansion distance = 2.0; goal-sampling rate = 10%; maximum iterations = 1000. |
| PRM | PythonRobotics | Rectangle resolution = 1.0; sampled nodes = 500; nearest neighbours = 10; maximum edge length = 30.0. |
| ACO | Archived Haghrah implementation | Ants = 20; steps per ant = 200; colony iterations = 40; α = 1.1; β = 12.0; γ = 0.02; evaporation = 0.5; brushfire iterations = 5; pheromone Q = 10.0. |
| ACO-GA | MATLAB bridge | Runtime = automatic MATLAB/Octave selection; grid resolution = 1.0; population = 80; ACO iterations = 90; GA iterations = 60; ρ = 0.25; α = 2.0; β = 8.0; Q = 100.0; crossover probability = 0.20; mutation probability = 0.05; path-length weight = 0.50; smoothness weight = 0.50; corner cutting = disabled; timeout = 1800 s. |
| Three-Phase Incremental SFO | Project implementation | Population = 30; iterations = 250; candidate attempts = 100; initial step range = [8.0, 16.0]; glide velocity range = [5.0, 18.0]; target velocity range = [0.5, 8.0]; micro-adjustment velocity range = [0.1, 2.5]; goal-connection distance = 18.0; collision resolution = 0.25. |

## Evaluation and reporting

A path is treated as successful only when it reaches the goal, is
collision-free, and passes the common geometric validator. Failed and partial
paths are retained for diagnostic figures but are excluded from path-quality
means. The following metrics are reported:

| Metric | Definition | Aggregation rule |
|---|---|---|
| Success rate | Proportion of valid completed paths. | All five attempts. |
| Shortest path | Minimum valid path length. | Valid completed paths only. |
| Average path | Mean valid path length. | Valid completed paths only. |
| Turning points | Number of heading changes greater than 5 degrees. | Mean over valid completed paths. |
| Smoothness | Total turning angle in radians; lower values indicate smoother paths. | Mean over valid completed paths. |
| Execution time | Wall-clock duration of a planner attempt. | Mean over all attempts, including failures. |

All reported failures remain visible in the success-rate analysis and in
diagnostic path figures. They are not converted into artificial goal-reaching
paths and do not contribute to path-length or smoothness statistics.

## Computing environment

The software requirements are Python 3.11 and MATLAB R2025 or GNU Octave for
ACO-GA. Before submission, add the actual host specification used for the
reported experiment: processor model, installed memory, operating system,
MATLAB release, Python version, and package versions. These hardware values
are not encoded in the source archive and must not be inferred.

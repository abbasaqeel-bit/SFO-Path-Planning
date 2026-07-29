# ACO-GA paper-faithful reproduction

This directory contains an independent MATLAB reproduction of:

Zhang et al., "An Improved Hybrid Ant Colony Optimization and Genetic
Algorithm for Multi-Map Path Planning of Rescuing Robots in Mine Disaster
Scenario", *Machines* 13(6), 474 (2025),
https://doi.org/10.3390/machines13060474.

## Fidelity boundary

The reproduction follows every algorithmic detail explicitly stated in the
paper:

- 8-neighbour grid ACO with a no-revisit tabu set;
- initial pheromone proportional to inverse goal distance;
- transition probability `tau^alpha * (1/d_goal)^beta`;
- safety-aware pheromone deposit `(1-sigma)*Q/L`;
- the high-quality population from the final ACO iteration initializes GA;
- roulette-wheel selection;
- crossover only at common path nodes, preserving connected path sections;
- mutation replaces a subpath between two selected nodes by a newly generated
  feasible subpath;
- the published multi-map objective maximizes weighted reciprocal path length
  plus path smoothness;
- the paper's published parameter values are the defaults.

The paper does not publish author source code and states that data are
available on request. Low-level tie breaking, exact numerical values of W1/W2,
and diagonal corner policy are therefore not reproducible byte-for-byte.
Those choices are exposed as configuration and recorded in result metadata.
The package must be described as a **paper-faithful independent
reproduction**, not as the authors' official implementation.

The bridge does not replace the native objective with the benchmark
objective. Common metrics are calculated only after the native algorithm has
finished.

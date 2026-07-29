# ACO-GA reproduction traceability

Reference: Zhang et al., *Machines* 13(6), 474 (2025),
DOI: https://doi.org/10.3390/machines13060474.

## Explicitly matched

| Published element | Reproduction |
|---|---|
| Grid values 0/1/2/3 | Python-to-MATLAB bridge payload |
| Eight-neighbour ACO and tabu/no revisit | `paper_aco_population.m` |
| Initial pheromone `1 / d(goal)` | `paper_aco_population.m` |
| Transition `tau^alpha * (1/d)^beta` | `paper_aco_population.m` |
| Evaporation and safety deposit `(1-sigma)Q/L` | `paper_aco_population.m` |
| Final high-quality ACO population initializes GA | `paper_aco_population.m` |
| Roulette selection | `paper_ga_optimize.m` |
| Crossover of overlapping path sections | common-node crossover in `paper_ga_optimize.m` |
| Two-node subpath mutation with pheromone guidance | `paper_ga_optimize.m` |
| Multi-map native objective | `paper_multimap_fitness.m` |
| Published parameters | benchmark YAML: 80, 90, 60, 0.25, 2, 8, 100, 0.2, 0.05 |
| MATLAB execution | MATLAB bridge; no Python translation of search loops |

## Unspecified by the publication

The publication does not provide author code and states that data are
available on request. It does not provide:

- numerical values for objective weights `W1` and `W2`;
- tie-breaking order among equal neighbours;
- diagonal corner-cutting policy;
- exact data structures for rectangular-region expansion;
- the authors' random seeds.

The integration therefore exposes these choices in YAML. Defaults are
`W1=W2=0.5` and no diagonal corner cutting. These are declared reproduction
choices and must not be presented as author-specified values.

## Comparison rule

ACO-GA optimizes its own published native objective. The benchmark computes
common path metrics after completion. Native convergence values from
different algorithms are not plotted on a shared numeric fitness axis.

# Haghrah ACO traceability

## Source classification

- Repository: `Haghrah/ACO---Robot-Path-Planning`
- File executed: `PathPlanning.py`, class `ACO_PDG`
- Archived SHA-256:
  `531D2D10633C042670CE765CFE8BA4F3DF87A25CD10881CACFFF9946CBBDEB61`
- Repository license: GPL-3.0
- Related paper: Liu et al., *Soft Computing* 21 (2017), 5829-5839,
  DOI `10.1007/s00500-016-2161-7`

The repository is an independent simulation of the cited paper, not an
official author repository. The benchmark must cite and classify it
accordingly.

## Native implementation preserved

The adapter loads the archived Python file and instantiates `ACO_PDG`
directly. The following source behaviours are retained:

- fixed `20 x 20` grid;
- variable-length eight-neighbour grid chains;
- Euclidean path length as the native objective;
- potential-field heuristic;
- Brushfire obstacle-distance calculation;
- pheromone evaporation and diffusion;
- source geometric path optimizer;
- source dead-end and previous-node handling;
- final colony iteration as the source return value.

The repository demonstration settings are:

| Parameter | Value |
|---|---:|
| Ants | 20 |
| Construction steps | 100 |
| Colony iterations | 20 |
| Alpha | 1.1 |
| Beta | 12 |
| Gamma | 0.02 |
| Evaporation | 0.5 |
| Brushfire iterations | 5 |
| Q | 10 |

ACO does not use the benchmark `PathObjective`,
`FixedWaypointEncoding`, line-of-sight pruning, or post-search repair.

## Necessary interface conversions

1. A benchmark workspace is rasterized to the source's fixed `20 x 20` grid.
2. A cell is occupied when its centre lies within an obstacle after applying
   the benchmark robot radius.
3. Start and goal are converted to their containing grid cells and kept free.
4. Returned grid cells are converted to workspace cell centres.
5. The exact benchmark start and goal replace the first and final cell-centre
   coordinates.
6. Consecutive duplicate points alone are removed.
7. Source plotting is disabled for headless benchmark execution.

These operations are integration conversions. They do not alter the source
search equations or objective.

## Validation and interpretation

The source permits diagonal movement whenever the destination grid cell is
free; it does not prohibit diagonal corner cutting between two occupied
orthogonal neighbours. The common continuous collision checker is deliberately
stricter. A native path rejected by that checker is recorded as a failed run,
not repaired.

The fixed `20 x 20` raster can also remove narrow connectivity present in a
higher-resolution benchmark map. Failure on narrow-passage or bug-trap maps is
therefore a legitimate observed limitation of the preserved source and its
necessary map conversion.

Native convergence values measure source grid-chain length. They may be used
for within-ACO analysis but are not numerically comparable with algorithms
using different native objectives. Cross-algorithm comparison uses common
post-run success, collision, length, turning, clearance, time, and memory
metrics.

# Validation Record

## Scope

The benchmark contains nine planners, three challenge maps, and five
independent seeds per planner/map pair (135 planned records). PSO and
standalone GA are not included.

## Validation policy

1. The experiment directory is cleaned before a complete benchmark run.
2. The batch validates all unique map/algorithm/run combinations.
3. Path-quality statistics use only collision-free paths that reach the goal.
4. Failed and partial paths remain in `tables/failed_and_partial_runs.csv` and
   are excluded from quality means.
5. Every boxplot lists all configured algorithms. Zero-success cases are
   labelled `No valid path (0/5)`.
6. Execution-time summaries include failures; path-quality summaries do not.
7. Each run receives a PNG and GIF diagnostic.
8. RRT-family planners and PRM retain the closest native reachable branch for
   diagnostic plotting when no complete path is found.
9. ACO-GA retains an invalid or partial MATLAB payload for diagnostics when it
   exists, without changing search behavior or success classification.

Diagnostic paths are not replacement solutions and receive no success credit.
The dotted goal-gap line is visual only and is never counted as travelled path.
Native objectives and representations remain unchanged.

## Host verification

Run the following after installing Python 3.11 and MATLAB R2025:

```powershell
.\run_validation.ps1
```

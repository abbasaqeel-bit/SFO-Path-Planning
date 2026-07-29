# Combined Path Figures

The generator creates two figures for each benchmark map without rerunning a
planner:

1. `*_all_planners`: one representative trajectory for every algorithm.
2. `*_failure_diagnostics`: failed, partial, and absent trajectories only.

Each figure is exported as 600-DPI PNG and as vector PDF/SVG. A companion
`*_representatives.csv` records the selected run and the selection rule.

## Representative-run rule

- For an algorithm with successful runs, the selected path is the valid run
  whose length is closest to the median valid length.
- When every run fails but a trajectory exists, the selected attempt has the
  smallest remaining goal distance. Clearance, path length, and run number are
  deterministic tie-breakers.
- When no trajectory exists, the algorithm is listed as `no path` and no
  artificial line is drawn.

## Visual encoding

- Solid line: valid, collision-free complete path.
- Dashed line: complete path rejected for collision.
- Dotted line: partial path that did not reach the goal.
- Red X: collision interval detected by the common validator.
- Open square: last reached point of a partial path.
- Thin grey dotted segment: untravelled gap to the goal.
- Dashed obstacle outline: obstacle inflated by the robot radius.

## Run

```powershell
.\generate_path_comparisons.ps1 `
  "C:\path\to\static_benchmark.zip"
```

The input may also be an extracted benchmark directory.

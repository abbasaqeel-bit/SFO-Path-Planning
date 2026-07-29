<<<<<<< HEAD
# Path Planning Benchmark

Reproducible static global path-planning benchmark for the proposed
Three-Phase Incremental SFO and eight reference planners.

## Included planners

| Family | Planner | Implementation |
|---|---|---|
| Grid search | A*, Dijkstra | PythonRobotics |
| Sampling-based | RRT, RRT*, Informed RRT*, PRM | PythonRobotics |
| Swarm optimisation | ACO | Archived Haghrah implementation |
| Hybrid optimisation | ACO-GA | Paper-faithful independent reproduction |
| Proposed method | Three-Phase Incremental SFO | Project implementation |

Each planner retains its native solution representation, internal objective
function, and source parameters. Cross-planner comparison uses common
post-run physical metrics only.

## Installation

Python 3.11 is required. MATLAB R2025 or GNU Octave is required only for
ACO-GA.

```powershell
py -3.11 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
& ".\.venv\Scripts\python.exe" -m pip install -e ".[dev]"
```

If MATLAB is not on PATH, set the executable location before running:

```powershell
$env:SFO_MATLAB_EXECUTABLE = "C:/Program Files/MATLAB/R2025a/bin/matlab.exe"
```

## Validation

```powershell
.\run_validation.ps1
```

The validation sequence runs the Python test suite, the archived ACO check,
the ACO-GA reproduction self-test, and the ACO-GA map validation.

## Run the benchmark

```powershell
.\run_benchmark.ps1
```

The default configuration is configs/benchmark.yaml. It executes nine
planners on three static maps with five independent runs per planner/map pair
(135 planned runs). Results are written below:

```text
results/experiments/static_benchmark/
```

Every run records the planner result, metrics, a static path figure, and an
animated path diagnostic. Aggregate results are written to tables/, including
summary_table.csv, success rates, descriptive statistics, Friedman tests, and
Holm-adjusted pairwise tests.

## Documentation

- docs/EXPERIMENT_SETUP.md: experimental protocol and parameter tables.
- docs/EXPERIMENT_SETUP.tex: LaTeX-ready experimental setup.
- docs/ARCHITECTURE.md: package architecture.
- docs/ACO_HAGHRAH_TRACEABILITY.md: ACO source traceability.
- docs/ACO_GA_PAPER_TRACEABILITY.md: ACO-GA reproduction traceability.
- docs/THIRD_PARTY_NOTICES.md: third-party notices and licenses.
- docs/DOI_ARCHIVING.md: Zenodo archiving and DOI completion steps.

## Citation

Citation metadata is provided in CITATION.cff. The DOI is intentionally not
included until Zenodo archives a tagged GitHub release and issues a real DOI.

## Reproducibility policy

Only valid, collision-free paths that reach the goal contribute to path-quality
statistics. Failed and partial paths are retained as diagnostic artefacts and
are reported in the success rate. Execution time is reported for every attempt,
including failed attempts.

The root project code is distributed under the MIT license. Third-party
components retain their original licenses and provenance notices.
=======
# SFO-Path-Planning
>>>>>>> e4f32d2f8bb95cb0a822f8a718f56a5fd5b42e41

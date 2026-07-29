"""Publication figures comparing representative paths on a common map.

The module reads completed benchmark artefacts rather than rerunning planners.
For each algorithm it selects one representative run using a documented rule:

* successful algorithms: the collision-free run closest to the median valid
  path length;
* algorithms with no successful run but with trajectories: the failed attempt
  that progressed closest to the goal, with safer/shorter paths breaking ties;
* algorithms that produced no trajectory: a legend-only ``no path`` entry.

Solid lines are valid paths, dashed lines are complete but colliding paths, and
dotted lines are incomplete paths. Collision locations are recomputed with the
same inflated-obstacle geometry used by the common benchmark validator.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
import numpy as np

from sfo_benchmark.core.collision import CollisionChecker
from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.obstacle import CircularObstacle, RectangularObstacle


DISPLAY_NAMES = {
    "astar_pythonrobotics": "A*",
    "dijkstra_pythonrobotics": "Dijkstra",
    "rrt_pythonrobotics": "RRT",
    "rrt_star_pythonrobotics": "RRT*",
    "informed_rrt_star_pythonrobotics": "Informed RRT*",
    "prm_pythonrobotics": "PRM",
    "aco_published_global": "ACO",
    "aco_ga_paper_reproduction": "ACO-GA",
    "three_phase_incremental_sfo": "Proposed SFO",
}

ALGORITHM_COLORS = {
    "astar_pythonrobotics": "#0072B2",
    "dijkstra_pythonrobotics": "#E69F00",
    "rrt_pythonrobotics": "#009E73",
    "rrt_star_pythonrobotics": "#56B4E9",
    "informed_rrt_star_pythonrobotics": "#CC79A7",
    "prm_pythonrobotics": "#D55E00",
    "aco_published_global": "#8C564B",
    "aco_ga_paper_reproduction": "#9467BD",
    "three_phase_incremental_sfo": "#000000",
}

DEFAULT_ORDER = tuple(DISPLAY_NAMES)


@dataclass(frozen=True, slots=True)
class RunArtefact:
    """One completed benchmark run loaded from its run directory."""

    algorithm: str
    run_number: int
    run_directory: Path
    metrics: dict[str, Any]
    points: np.ndarray | None

    @property
    def success(self) -> bool:
        return bool(self.metrics.get("success", False))

    @property
    def path_status(self) -> str:
        return str(self.metrics.get("path_status", "none"))

    @property
    def path_length(self) -> float:
        value = self.metrics.get("path_length")
        return float(value) if value is not None else float("inf")


@dataclass(frozen=True, slots=True)
class RepresentativePath:
    """Representative run plus summary information used in the legend."""

    algorithm: str
    selected: RunArtefact | None
    successful_runs: int
    total_runs: int
    selection_rule: str

    @property
    def display_name(self) -> str:
        return DISPLAY_NAMES.get(self.algorithm, self.algorithm)

    @property
    def state(self) -> str:
        if self.selected is None or self.selected.points is None:
            return "no_path"
        if self.selected.success:
            return "valid"
        if self.selected.path_status == "invalid_complete":
            return "collision"
        return "partial"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _read_trajectory(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return None
    points = np.asarray(
        [[float(row["x"]), float(row["y"])] for row in rows],
        dtype=float,
    )
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Invalid trajectory shape in {path}: {points.shape}")
    return points


def load_map_runs(results_root: Path, map_id: str) -> dict[str, list[RunArtefact]]:
    """Load every algorithm/run artefact recorded for one benchmark map."""
    map_directory = Path(results_root) / map_id
    if not map_directory.is_dir():
        raise FileNotFoundError(f"Map results directory not found: {map_directory}")

    algorithms: dict[str, list[RunArtefact]] = {}
    for algorithm_directory in sorted(
        path for path in map_directory.iterdir() if path.is_dir()
    ):
        runs: list[RunArtefact] = []
        for run_directory in sorted(algorithm_directory.glob("Run_*")):
            metrics_path = run_directory / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = _read_json(metrics_path)
            points = _read_trajectory(run_directory / "trajectory.csv")
            if bool(metrics.get("success", False)) and points is None:
                raise ValueError(
                    "A successful run is missing trajectory.csv: "
                    f"{run_directory}"
                )
            try:
                run_number = int(run_directory.name.split("_")[-1])
            except ValueError as error:
                raise ValueError(
                    f"Invalid run directory name: {run_directory.name}"
                ) from error
            runs.append(
                RunArtefact(
                    algorithm=algorithm_directory.name,
                    run_number=run_number,
                    run_directory=run_directory,
                    metrics=metrics,
                    points=points,
                )
            )
        if runs:
            algorithms[algorithm_directory.name] = runs
    return algorithms


def _remaining_goal_distance(run: RunArtefact, goal: np.ndarray) -> float:
    value = run.metrics.get("goal_distance_remaining")
    if value is not None:
        return float(value)
    if run.points is not None and len(run.points):
        return float(np.linalg.norm(run.points[-1] - goal))
    return float("inf")


def select_representative(
    algorithm: str,
    runs: Iterable[RunArtefact],
    goal: np.ndarray,
) -> RepresentativePath:
    """Select a representative run without best-case cherry-picking."""
    all_runs = list(runs)
    successful = [
        run
        for run in all_runs
        if run.success and run.points is not None and len(run.points) >= 2
    ]
    if successful:
        median_length = median(run.path_length for run in successful)
        selected = min(
            successful,
            key=lambda run: (
                abs(run.path_length - median_length),
                run.run_number,
            ),
        )
        return RepresentativePath(
            algorithm=algorithm,
            selected=selected,
            successful_runs=len(successful),
            total_runs=len(all_runs),
            selection_rule="successful run closest to median valid path length",
        )

    failed_with_paths = [
        run for run in all_runs if run.points is not None and len(run.points)
    ]
    if failed_with_paths:
        selected = min(
            failed_with_paths,
            key=lambda run: (
                _remaining_goal_distance(run, goal),
                -float(
                    run.metrics.get("minimum_clearance")
                    if run.metrics.get("minimum_clearance") is not None
                    else -float("inf")
                ),
                run.path_length,
                run.run_number,
            ),
        )
        return RepresentativePath(
            algorithm=algorithm,
            selected=selected,
            successful_runs=0,
            total_runs=len(all_runs),
            selection_rule=(
                "failed attempt with minimum remaining goal distance; "
                "clearance and length used as tie-breakers"
            ),
        )

    return RepresentativePath(
        algorithm=algorithm,
        selected=None,
        successful_runs=0,
        total_runs=len(all_runs),
        selection_rule="no trajectory generated in any run",
    )


def approximate_collision_locations(
    points: np.ndarray,
    environment: Environment,
    *,
    resolution: float = 0.10,
) -> np.ndarray:
    """Return one marker per contiguous collision interval along a path."""
    if len(points) < 2:
        return np.empty((0, 2), dtype=float)
    checker = CollisionChecker(environment, resolution=resolution)
    samples: list[np.ndarray] = []
    colliding: list[bool] = []
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        distance = float(np.linalg.norm(end - start))
        count = max(2, int(np.ceil(distance / resolution)) + 1)
        alphas = np.linspace(0.0, 1.0, count)
        if index:
            alphas = alphas[1:]
        for alpha in alphas:
            point = start + alpha * (end - start)
            samples.append(point)
            colliding.append(not checker.point_is_free(point))

    clusters: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for point, is_collision in zip(samples, colliding):
        if is_collision:
            current.append(point)
        elif current:
            clusters.append(np.mean(np.asarray(current), axis=0))
            current = []
    if current:
        clusters.append(np.mean(np.asarray(current), axis=0))
    if not clusters:
        return np.empty((0, 2), dtype=float)
    return np.asarray(clusters, dtype=float)


def _algorithm_color(
    algorithm: str,
    index: int,
) -> str | tuple[float, float, float, float]:
    if algorithm in ALGORITHM_COLORS:
        return ALGORITHM_COLORS[algorithm]
    palette = plt.get_cmap("tab10")
    return palette(index % 10)


def _draw_environment(axis: plt.Axes, environment: Environment) -> None:
    obstacle_face = "#C7C9CC"
    obstacle_edge = "#6B6E73"
    inflated_edge = "#7A7D82"
    radius = environment.robot_radius

    for obstacle in environment.obstacles:
        if isinstance(obstacle, CircularObstacle):
            axis.add_patch(
                Circle(
                    tuple(obstacle.center),
                    obstacle.radius,
                    facecolor=obstacle_face,
                    edgecolor=obstacle_edge,
                    linewidth=0.8,
                    alpha=0.72,
                    zorder=1,
                )
            )
            if radius > 0:
                axis.add_patch(
                    Circle(
                        tuple(obstacle.center),
                        obstacle.radius + radius,
                        fill=False,
                        edgecolor=inflated_edge,
                        linestyle="--",
                        linewidth=0.65,
                        alpha=0.8,
                        zorder=2,
                    )
                )
        elif isinstance(obstacle, RectangularObstacle):
            axis.add_patch(
                Rectangle(
                    (obstacle.xmin, obstacle.ymin),
                    obstacle.xmax - obstacle.xmin,
                    obstacle.ymax - obstacle.ymin,
                    facecolor=obstacle_face,
                    edgecolor=obstacle_edge,
                    linewidth=0.8,
                    alpha=0.72,
                    zorder=1,
                )
            )
            if radius > 0:
                axis.add_patch(
                    Rectangle(
                        (obstacle.xmin - radius, obstacle.ymin - radius),
                        obstacle.xmax - obstacle.xmin + 2 * radius,
                        obstacle.ymax - obstacle.ymin + 2 * radius,
                        fill=False,
                        edgecolor=inflated_edge,
                        linestyle="--",
                        linewidth=0.65,
                        alpha=0.8,
                        zorder=2,
                    )
                )


def _legend_label(item: RepresentativePath) -> str:
    rate = f"{item.successful_runs}/{item.total_runs}"
    if item.state == "valid":
        state = "valid"
    elif item.state == "collision":
        state = "collision"
    elif item.state == "partial":
        state = "partial"
    else:
        state = "no path"
    return f"{item.display_name} — {state} ({rate})"


def _write_selection_manifest(
    output_path: Path,
    map_id: str,
    representatives: list[RepresentativePath],
) -> None:
    rows = []
    for item in representatives:
        selected = item.selected
        rows.append(
            {
                "map_id": map_id,
                "algorithm": item.algorithm,
                "display_name": item.display_name,
                "successful_runs": item.successful_runs,
                "total_runs": item.total_runs,
                "representative_state": item.state,
                "selected_run": (
                    selected.run_number if selected is not None else None
                ),
                "selected_path_length": (
                    selected.metrics.get("path_length")
                    if selected is not None
                    else None
                ),
                "remaining_goal_distance": (
                    selected.metrics.get("goal_distance_remaining")
                    if selected is not None
                    else None
                ),
                "selection_rule": item.selection_rule,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_combined_path_figure(
    environment: Environment,
    representatives: list[RepresentativePath],
    output_stem: Path,
    *,
    failures_only: bool = False,
    collision_resolution: float = 0.25,
    dpi: int = 600,
    include_title: bool = False,
) -> list[Path]:
    """Render PNG, PDF and SVG versions of one publication figure."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.2, 7.0))
    _draw_environment(axis, environment)

    handles: list[Any] = []
    labels: list[str] = []
    collision_marker_added = False
    terminal_marker_added = False

    plotted = [
        item
        for item in representatives
        if not failures_only or item.state != "valid"
    ]
    for index, item in enumerate(plotted):
        color = _algorithm_color(item.algorithm, index)
        selected = item.selected
        linewidth = 2.4 if item.algorithm == "three_phase_incremental_sfo" else 1.7

        if item.state == "no_path" or selected is None or selected.points is None:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    marker="x",
                    linestyle="None",
                    markersize=7,
                    markeredgewidth=1.5,
                )
            )
            labels.append(_legend_label(item))
            continue

        points = selected.points
        linestyle = {
            "valid": "-",
            "collision": "--",
            "partial": ":",
        }[item.state]
        axis.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.96,
            zorder=5 + index * 0.01,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
        )
        labels.append(_legend_label(item))

        collisions = approximate_collision_locations(
            points,
            environment,
            resolution=collision_resolution,
        )
        if len(collisions):
            axis.scatter(
                collisions[:, 0],
                collisions[:, 1],
                marker="X",
                s=58,
                facecolor="#D62728",
                edgecolor="white",
                linewidth=0.7,
                zorder=20,
            )
            collision_marker_added = True

        if item.state == "partial":
            axis.scatter(
                points[-1, 0],
                points[-1, 1],
                marker="s",
                s=46,
                facecolor="none",
                edgecolor=color,
                linewidth=1.5,
                zorder=19,
            )
            axis.plot(
                [points[-1, 0], environment.goal[0]],
                [points[-1, 1], environment.goal[1]],
                color="#777777",
                linestyle=(0, (1, 2)),
                linewidth=0.8,
                alpha=0.75,
                zorder=3,
            )
            terminal_marker_added = True

    axis.scatter(
        *environment.start,
        s=86,
        marker="o",
        facecolor="#2E86AB",
        edgecolor="white",
        linewidth=0.8,
        zorder=30,
    )
    axis.scatter(
        *environment.goal,
        s=155,
        marker="*",
        facecolor="#F28E2B",
        edgecolor="#A54F00",
        linewidth=0.8,
        zorder=30,
    )

    status_handles: list[Any] = [
        Patch(
            facecolor="#C7C9CC",
            edgecolor="#6B6E73",
            alpha=0.72,
            label="Obstacle",
        ),
        Line2D(
            [0],
            [0],
            color="#7A7D82",
            linestyle="--",
            linewidth=0.8,
            label="Robot-inflated boundary",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#2E86AB",
            markeredgecolor="white",
            markersize=7,
            label="Start",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#F28E2B",
            markeredgecolor="#A54F00",
            markersize=10,
            label="Goal",
        ),
    ]
    if collision_marker_added:
        status_handles.append(
            Line2D(
                [0],
                [0],
                marker="X",
                color="none",
                markerfacecolor="#D62728",
                markeredgecolor="white",
                markersize=7,
                label="Detected collision interval",
            )
        )
    if terminal_marker_added:
        status_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color="#555555",
                markerfacecolor="none",
                linestyle="None",
                markersize=6,
                label="Last reached point",
            )
        )

    xmin, xmax, ymin, ymax = environment.bounds
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.grid(True, linewidth=0.35, alpha=0.28)
    if include_title:
        suffix = "failure diagnostics" if failures_only else "all planners"
        axis.set_title(f"{environment.map_id.replace('_', ' ')} — {suffix}")

    algorithm_legend = axis.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=False,
        title="Representative trajectories",
        fontsize=8,
        title_fontsize=8,
    )
    axis.add_artist(algorithm_legend)
    axis.legend(
        handles=status_handles,
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=7.5,
    )
    figure.subplots_adjust(right=0.70)

    outputs = []
    for extension in ("png", "pdf", "svg"):
        destination = output_stem.with_suffix(f".{extension}")
        figure.savefig(
            destination,
            dpi=dpi if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        outputs.append(destination)
    plt.close(figure)
    return outputs


def generate_path_comparison_figures(
    *,
    results_root: Path,
    environments: dict[str, Environment],
    output_directory: Path,
    algorithm_order: Iterable[str] = DEFAULT_ORDER,
    collision_resolution: float = 0.25,
    dpi: int = 600,
    include_title: bool = False,
) -> list[Path]:
    """Generate all-planner and failure-only figures for every environment."""
    results_root = Path(results_root)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    manifest_rows: list[Path] = []

    for map_id, environment in environments.items():
        loaded = load_map_runs(results_root, map_id)
        ordered_algorithms = [
            algorithm
            for algorithm in algorithm_order
            if algorithm in loaded
        ]
        ordered_algorithms.extend(
            algorithm
            for algorithm in sorted(loaded)
            if algorithm not in ordered_algorithms
        )
        representatives = [
            select_representative(algorithm, loaded[algorithm], environment.goal)
            for algorithm in ordered_algorithms
        ]

        manifest_path = output_directory / f"{map_id}_representatives.csv"
        _write_selection_manifest(manifest_path, map_id, representatives)
        manifest_rows.append(manifest_path)

        generated.extend(
            save_combined_path_figure(
                environment,
                representatives,
                output_directory / f"{map_id}_all_planners",
                collision_resolution=collision_resolution,
                dpi=dpi,
                include_title=include_title,
            )
        )
        generated.extend(
            save_combined_path_figure(
                environment,
                representatives,
                output_directory / f"{map_id}_failure_diagnostics",
                failures_only=True,
                collision_resolution=collision_resolution,
                dpi=dpi,
                include_title=include_title,
            )
        )

    return [*manifest_rows, *generated]

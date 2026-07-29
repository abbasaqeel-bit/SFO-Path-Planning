"""Rank-based Pareto visualization of path quality, runtime, and reliability."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DISPLAY_NAMES = {
    "astar_pythonrobotics": "A*",
    "dijkstra_pythonrobotics": "Dijkstra",
    "rrt_pythonrobotics": "RRT",
    "rrt_star_pythonrobotics": "RRT*",
    "informed_rrt_star_pythonrobotics": "Informed RRT*",
    "prm_pythonrobotics": "PRM",
    "aco_published_global": "ACO",
    "aco_ga_paper_reproduction": "ACO-GA",
    "three_phase_incremental_sfo": "SFO",
}

ALGORITHM_ORDER = tuple(DISPLAY_NAMES)

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

MAP_TITLES = {
    "dense_01": "Dense",
    "narrow_passage_01": "Narrow passage",
    "bug_trap_01": "Bug trap",
}

LABEL_OFFSETS = {
    "astar_pythonrobotics": (8, 8),
    "dijkstra_pythonrobotics": (8, -10),
    "rrt_pythonrobotics": (8, 8),
    "rrt_star_pythonrobotics": (-8, 8),
    "informed_rrt_star_pythonrobotics": (8, 8),
    "prm_pythonrobotics": (8, -10),
    "aco_published_global": (8, 8),
    "aco_ga_paper_reproduction": (-8, 8),
    "three_phase_incremental_sfo": (8, -11),
}


@dataclass(frozen=True, slots=True)
class QualityPoint:
    map_id: str
    algorithm: str
    total_runs: int
    successful_runs: int
    path_length_median: float | None
    turning_angle_median: float | None
    execution_time_median: float | None
    path_length_rank: float | None = None
    smoothness_rank: float | None = None

    @property
    def display_name(self) -> str:
        return DISPLAY_NAMES.get(self.algorithm, self.algorithm)

    @property
    def success_rate(self) -> float:
        if self.total_runs <= 0:
            return 0.0
        return self.successful_runs / self.total_runs

    @property
    def has_valid_path(self) -> bool:
        return (
            self.successful_runs > 0
            and self.path_length_median is not None
            and self.turning_angle_median is not None
            and self.execution_time_median is not None
        )


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _read_runs(results_root: Path) -> list[dict[str, Any]]:
    candidates = [
        Path(results_root) / "tables" / "all_runs.csv",
        Path(results_root) / "summary.csv",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "Expected tables/all_runs.csv or summary.csv under "
            f"{results_root}."
        )
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _average_ranks(values: list[float]) -> list[float]:
    """Return one-based ascending ranks with average ranks for ties."""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        end = position + 1
        while (
            end < len(indexed)
            and math.isclose(
                indexed[end][1],
                indexed[position][1],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            end += 1
        average = ((position + 1) + end) / 2.0
        for index in range(position, end):
            ranks[indexed[index][0]] = average
        position = end
    return ranks


def summarize_quality_points(results_root: Path) -> list[QualityPoint]:
    """Compute per-map medians from successful collision-free runs only."""
    rows = _read_runs(results_root)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["map_id"]), str(row["algorithm"]))
        groups.setdefault(key, []).append(row)

    map_order = list(dict.fromkeys(str(row["map_id"]) for row in rows))
    algorithm_order = [
        algorithm
        for algorithm in ALGORITHM_ORDER
        if any(key[1] == algorithm for key in groups)
    ]
    algorithm_order.extend(
        algorithm
        for algorithm in sorted({key[1] for key in groups})
        if algorithm not in algorithm_order
    )

    points: list[QualityPoint] = []
    for map_id in map_order:
        map_points: list[QualityPoint] = []
        for algorithm in algorithm_order:
            runs = groups.get((map_id, algorithm), [])
            if not runs:
                continue
            valid = [
                row
                for row in runs
                if _truthy(row.get("success"))
                and _truthy(row.get("collision_free"))
                and str(row.get("path_status")) == "valid_complete"
                and _optional_float(
                    row.get("valid_path_length", row.get("path_length"))
                )
                is not None
            ]
            if valid:
                lengths = [
                    float(
                        _optional_float(
                            row.get(
                                "valid_path_length",
                                row.get("path_length"),
                            )
                        )
                    )
                    for row in valid
                ]
                turning = [
                    float(_optional_float(row["total_turning_angle_rad"]))
                    for row in valid
                    if _optional_float(row.get("total_turning_angle_rad"))
                    is not None
                ]
                runtime = [
                    float(_optional_float(row["execution_time_s"]))
                    for row in valid
                    if _optional_float(row.get("execution_time_s")) is not None
                ]
                if len(turning) != len(valid) or len(runtime) != len(valid):
                    raise ValueError(
                        "A successful run is missing turning angle or runtime: "
                        f"{map_id}/{algorithm}"
                    )
                map_points.append(
                    QualityPoint(
                        map_id=map_id,
                        algorithm=algorithm,
                        total_runs=len(runs),
                        successful_runs=len(valid),
                        path_length_median=float(median(lengths)),
                        turning_angle_median=float(median(turning)),
                        execution_time_median=float(median(runtime)),
                    )
                )
            else:
                map_points.append(
                    QualityPoint(
                        map_id=map_id,
                        algorithm=algorithm,
                        total_runs=len(runs),
                        successful_runs=0,
                        path_length_median=None,
                        turning_angle_median=None,
                        execution_time_median=None,
                    )
                )

        valid_points = [point for point in map_points if point.has_valid_path]
        length_ranks = _average_ranks(
            [float(point.path_length_median) for point in valid_points]
        )
        turning_ranks = _average_ranks(
            [float(point.turning_angle_median) for point in valid_points]
        )
        ranked = {
            point.algorithm: QualityPoint(
                map_id=point.map_id,
                algorithm=point.algorithm,
                total_runs=point.total_runs,
                successful_runs=point.successful_runs,
                path_length_median=point.path_length_median,
                turning_angle_median=point.turning_angle_median,
                execution_time_median=point.execution_time_median,
                path_length_rank=length_rank,
                smoothness_rank=turning_rank,
            )
            for point, length_rank, turning_rank in zip(
                valid_points,
                length_ranks,
                turning_ranks,
                strict=True,
            )
        }
        points.extend(
            ranked.get(point.algorithm, point) for point in map_points
        )
    return points


def _pareto_front(points: Iterable[QualityPoint]) -> list[QualityPoint]:
    valid = [point for point in points if point.has_valid_path]
    front = []
    for candidate in valid:
        dominated = any(
            other.algorithm != candidate.algorithm
            and float(other.path_length_median)
            <= float(candidate.path_length_median)
            and float(other.turning_angle_median)
            <= float(candidate.turning_angle_median)
            and (
                float(other.path_length_median)
                < float(candidate.path_length_median)
                or float(other.turning_angle_median)
                < float(candidate.turning_angle_median)
            )
            for other in valid
        )
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda point: float(point.path_length_rank))


def _bubble_area(runtime_seconds: float) -> float:
    return 55.0 + 90.0 * math.log10(1.0 + max(runtime_seconds, 0.0))


def write_quality_data(points: list[QualityPoint], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "map_id",
        "algorithm",
        "display_name",
        "total_runs",
        "successful_runs",
        "success_rate_percent",
        "path_length_median",
        "turning_angle_median_rad",
        "execution_time_median_s",
        "path_length_rank",
        "smoothness_rank",
        "pareto_nondominated",
    ]
    fronts = {
        map_id: {
            point.algorithm
            for point in _pareto_front(
                candidate for candidate in points if candidate.map_id == map_id
            )
        }
        for map_id in {point.map_id for point in points}
    }
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "map_id": point.map_id,
                    "algorithm": point.algorithm,
                    "display_name": point.display_name,
                    "total_runs": point.total_runs,
                    "successful_runs": point.successful_runs,
                    "success_rate_percent": 100.0 * point.success_rate,
                    "path_length_median": point.path_length_median,
                    "turning_angle_median_rad": point.turning_angle_median,
                    "execution_time_median_s": point.execution_time_median,
                    "path_length_rank": point.path_length_rank,
                    "smoothness_rank": point.smoothness_rank,
                    "pareto_nondominated": (
                        point.algorithm in fronts[point.map_id]
                    ),
                }
            )
    return path


def save_rank_pareto_figure(
    points: list[QualityPoint],
    output_stem: Path,
    *,
    dpi: int = 600,
    include_title: bool = False,
) -> list[Path]:
    """Save PNG/PDF/SVG forms of the rank-based Pareto comparison."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    map_order = list(dict.fromkeys(point.map_id for point in points))
    figure, axes = plt.subplots(
        1,
        len(map_order),
        figsize=(13.6, 4.8),
        squeeze=False,
    )
    axes_row = axes[0]

    for axis, map_id in zip(axes_row, map_order, strict=True):
        map_points = [point for point in points if point.map_id == map_id]
        valid = [point for point in map_points if point.has_valid_path]
        absent = [point.display_name for point in map_points if not point.has_valid_path]
        maximum_rank = max(len(valid), 1)

        front = _pareto_front(valid)
        if len(front) >= 2:
            axis.plot(
                [float(point.path_length_rank) for point in front],
                [float(point.smoothness_rank) for point in front],
                linestyle="--",
                linewidth=1.0,
                color="0.55",
                zorder=1,
            )

        for point in valid:
            color = ALGORITHM_COLORS.get(point.algorithm, "#555555")
            reliability = point.success_rate
            axis.scatter(
                float(point.path_length_rank),
                float(point.smoothness_rank),
                s=_bubble_area(float(point.execution_time_median)),
                color=color,
                alpha=0.38 + 0.58 * reliability,
                edgecolor=color,
                linewidth=2.4 if reliability < 1.0 else 1.2,
                zorder=3,
            )
            offset = LABEL_OFFSETS.get(point.algorithm, (7, 7))
            alignment = "right" if offset[0] < 0 else "left"
            weight = "bold" if point.algorithm == "three_phase_incremental_sfo" else "normal"
            axis.annotate(
                (
                    f"{point.display_name} "
                    f"· {point.successful_runs}/{point.total_runs}"
                ),
                (
                    float(point.path_length_rank),
                    float(point.smoothness_rank),
                ),
                xytext=offset,
                textcoords="offset points",
                ha=alignment,
                va="center",
                fontsize=7.4,
                fontweight=weight,
                zorder=4,
            )

        axis.set_xlim(0.5, maximum_rank + 0.5)
        axis.set_ylim(0.5, maximum_rank + 0.5)
        axis.set_xticks(range(1, maximum_rank + 1))
        axis.set_yticks(range(1, maximum_rank + 1))
        axis.set_xlabel("Path-length rank")
        axis.set_ylabel("Smoothness rank")
        axis.set_title(MAP_TITLES.get(map_id, map_id.replace("_", " ")))
        axis.grid(True, linewidth=0.45, alpha=0.30)
        axis.annotate(
            "Ideal",
            xy=(0.62, 0.62),
            xytext=(1.25, 1.25),
            arrowprops={"arrowstyle": "->", "linewidth": 0.8, "color": "0.45"},
            fontsize=7.5,
            color="0.35",
        )
        if absent:
            axis.text(
                0.0,
                -0.20,
                "No valid path: " + ", ".join(absent),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7.2,
                color="#B22222",
                wrap=True,
            )

    if include_title:
        figure.suptitle(
            "Path-quality, runtime, and reliability comparison",
            fontsize=12,
        )

    size_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="0.45",
            markersize=math.sqrt(_bubble_area(seconds)) / 1.7,
            label=f"{seconds:g} s",
        )
        for seconds in (3.0, 30.0, 300.0)
    ]
    front_handle = Line2D(
        [],
        [],
        linestyle="--",
        linewidth=1.0,
        color="0.55",
        label="Pareto front",
    )
    figure.legend(
        handles=[*size_handles, front_handle],
        loc="lower center",
        ncol=4,
        frameon=False,
        title="Bubble size: median execution time (log-scaled)",
        fontsize=7.5,
        title_fontsize=7.5,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.subplots_adjust(
        left=0.06,
        right=0.99,
        top=0.88 if include_title else 0.94,
        bottom=0.28,
        wspace=0.28,
    )

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


def generate_rank_pareto_outputs(
    *,
    results_root: Path,
    output_directory: Path,
    dpi: int = 600,
    include_title: bool = False,
) -> list[Path]:
    points = summarize_quality_points(results_root)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    data_path = write_quality_data(
        points,
        output_directory / "rank_pareto_plot_data.csv",
    )
    figures = save_rank_pareto_figure(
        points,
        output_directory / "rank_pareto_path_quality",
        dpi=dpi,
        include_title=include_title,
    )
    return [data_path, *figures]

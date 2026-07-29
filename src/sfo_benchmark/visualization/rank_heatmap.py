"""Square-cell performance rank heatmaps.

The figure deliberately ranks algorithms *within each map*.  This avoids
combining metrics with incompatible units (metres, radians, seconds, and
clearance) in one colour scale.  Numerical labels are always the original
median values; colour represents only the ordinal within-map rank.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colors, patches, pyplot as plt
    from matplotlib.cm import ScalarMappable
except ModuleNotFoundError:  # permits data-only inspection without plotting deps
    matplotlib = None
    colors = patches = plt = ScalarMappable = None


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

MAP_TITLES = {
    "dense_01": "Dense-obstacle map",
    "narrow_passage_01": "Narrow-passage map",
    "bug_trap_01": "Bug-trap map",
}


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    key: str
    label: str
    source_column: str | None
    lower_is_better: bool


METRICS = (
    MetricDefinition(
        key="path_length",
        label="Path\nlength",
        source_column="valid_path_length",
        lower_is_better=True,
    ),
    MetricDefinition(
        key="turning",
        label="Turning\nangle",
        source_column="total_turning_angle_rad",
        lower_is_better=True,
    ),
    MetricDefinition(
        key="runtime",
        label="Execution\ntime",
        source_column="execution_time_s",
        lower_is_better=True,
    ),
    MetricDefinition(
        key="clearance",
        label="Minimum\nclearance",
        source_column="minimum_clearance",
        lower_is_better=False,
    ),
    MetricDefinition(
        key="success",
        label="Success\nruns",
        source_column=None,
        lower_is_better=False,
    ),
)


@dataclass(frozen=True, slots=True)
class AlgorithmHeatmapRow:
    """Medians and diagnostic status for one algorithm on one map."""

    map_id: str
    algorithm: str
    total_runs: int
    valid_runs: int
    values: dict[str, float | None]
    failure_code: str | None

    @property
    def display_name(self) -> str:
        return DISPLAY_NAMES.get(self.algorithm, self.algorithm)

    @property
    def has_valid_path(self) -> bool:
        return self.valid_runs > 0 and self.values["path_length"] is not None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _path_length(row: dict[str, Any]) -> float | None:
    """Return the validated length, falling back for older result schemas."""
    validated = _optional_float(row.get("valid_path_length"))
    return validated if validated is not None else _optional_float(row.get("path_length"))


def _read_runs(results_root: Path) -> list[dict[str, Any]]:
    candidates = (
        Path(results_root) / "tables" / "all_runs.csv",
        Path(results_root) / "summary.csv",
    )
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "Expected tables/all_runs.csv or summary.csv below "
            f"{results_root}."
        )
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _is_valid_completed_run(row: dict[str, Any]) -> bool:
    status = str(row.get("path_status", "")).strip().lower()
    length = _path_length(row)
    return (
        _truthy(row.get("success"))
        and _truthy(row.get("collision_free"))
        and status == "valid_complete"
        and length is not None
    )


def _failure_code(rows: Iterable[dict[str, Any]]) -> str:
    """Return a concise, auditable code when no valid completed run exists."""
    attempts = list(rows)
    if not attempts:
        return "NA"

    statuses = [str(row.get("path_status", "")).strip().lower() for row in attempts]
    messages = " ".join(
        str(row.get("message", "")).strip().lower() for row in attempts
    )
    has_partial = any("partial" in status for status in statuses)
    has_collision = (
        any("collision" in status for status in statuses)
        or "collision" in messages
        or any(not _truthy(row.get("collision_free")) for row in attempts)
    )
    no_path = (
        all(status in {"", "none"} for status in statuses)
        or "no path" in messages
        or "did not produce" in messages
        or "no_initial_population" in messages
    )
    if has_partial and has_collision:
        return "MIX"
    if no_path:
        return "NP"
    if has_collision:
        return "COL"
    if has_partial:
        return "MIX"
    return "FAIL"


def _median_column(
    rows: Iterable[dict[str, Any]],
    column: str,
) -> float | None:
    values = [_optional_float(row.get(column)) for row in rows]
    usable = [float(value) for value in values if value is not None]
    return float(median(usable)) if usable else None


def _median_path_length(rows: Iterable[dict[str, Any]]) -> float | None:
    values = [_path_length(row) for row in rows]
    usable = [float(value) for value in values if value is not None]
    return float(median(usable)) if usable else None


def _average_ranks(values: list[float], *, lower_is_better: bool) -> list[float]:
    """Return one-based ranks with average ranks for ties."""
    indexed = sorted(
        enumerate(values),
        key=lambda item: item[1],
        reverse=not lower_is_better,
    )
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


def summarize_heatmap_rows(results_root: Path) -> list[AlgorithmHeatmapRow]:
    """Summarize medians strictly from valid, completed, collision-free paths."""
    rows = _read_runs(results_root)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["map_id"]), str(row["algorithm"]))
        grouped.setdefault(key, []).append(row)

    map_order = list(dict.fromkeys(str(row["map_id"]) for row in rows))
    present_algorithms = {algorithm for _, algorithm in grouped}
    algorithm_order = [
        algorithm for algorithm in ALGORITHM_ORDER if algorithm in present_algorithms
    ]
    algorithm_order.extend(
        algorithm
        for algorithm in sorted(present_algorithms)
        if algorithm not in algorithm_order
    )

    output: list[AlgorithmHeatmapRow] = []
    for map_id in map_order:
        for algorithm in algorithm_order:
            attempts = grouped.get((map_id, algorithm), [])
            valid = [row for row in attempts if _is_valid_completed_run(row)]
            values: dict[str, float | None] = {
                "path_length": _median_path_length(valid),
                "turning": _median_column(valid, "total_turning_angle_rad"),
                "runtime": _median_column(valid, "execution_time_s"),
                "clearance": _median_column(valid, "minimum_clearance"),
                "success": (
                    (len(valid) / len(attempts)) if attempts else None
                ),
            }
            output.append(
                AlgorithmHeatmapRow(
                    map_id=map_id,
                    algorithm=algorithm,
                    total_runs=len(attempts),
                    valid_runs=len(valid),
                    values=values,
                    failure_code=None if valid else _failure_code(attempts),
                )
            )
    return output


def _format_value(metric: MetricDefinition, row: AlgorithmHeatmapRow) -> str:
    value = row.values[metric.key]
    if value is None:
        return row.failure_code or "NA"
    if metric.key == "success":
        return f"{row.valid_runs}/{row.total_runs}"
    if metric.key == "path_length":
        return f"{value:.1f}"
    if metric.key == "turning":
        return f"{value:.2f}"
    if metric.key == "runtime":
        return f"{value:.2f}" if value < 10 else f"{value:.1f}"
    if metric.key == "clearance":
        return f"{value:.3f}" if value < 0.1 else f"{value:.2f}"
    return f"{value:.3g}"


def _rank_lookup(
    rows: list[AlgorithmHeatmapRow],
    metric: MetricDefinition,
) -> tuple[dict[str, float], int]:
    eligible = [
        row
        for row in rows
        if row.has_valid_path and row.values[metric.key] is not None
    ]
    values = [float(row.values[metric.key]) for row in eligible]
    if not values:
        return {}, 0
    ranks = _average_ranks(values, lower_is_better=metric.lower_is_better)
    return {
        row.algorithm: rank for row, rank in zip(eligible, ranks, strict=True)
    }, len(eligible)


def write_heatmap_data(rows: list[AlgorithmHeatmapRow], destination: Path) -> Path:
    """Write the exact labels, ranks, and status codes used in the figure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    by_map: dict[str, list[AlgorithmHeatmapRow]] = {}
    for row in rows:
        by_map.setdefault(row.map_id, []).append(row)

    fieldnames = [
        "map_id",
        "map_title",
        "algorithm",
        "display_name",
        "total_runs",
        "valid_runs",
        "failure_code",
        *[f"{metric.key}_median" for metric in METRICS[:-1]],
        "success_rate",
        *[f"{metric.key}_rank" for metric in METRICS],
    ]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for map_id, map_rows in by_map.items():
            lookups = {
                metric.key: _rank_lookup(map_rows, metric)
                for metric in METRICS
            }
            for row in map_rows:
                record: dict[str, Any] = {
                    "map_id": map_id,
                    "map_title": MAP_TITLES.get(
                        map_id, map_id.replace("_", " ").title()
                    ),
                    "algorithm": row.algorithm,
                    "display_name": row.display_name,
                    "total_runs": row.total_runs,
                    "valid_runs": row.valid_runs,
                    "failure_code": row.failure_code or "",
                    "path_length_median": row.values["path_length"],
                    "turning_median": row.values["turning"],
                    "runtime_median": row.values["runtime"],
                    "clearance_median": row.values["clearance"],
                    "success_rate": row.values["success"],
                }
                for metric in METRICS:
                    record[f"{metric.key}_rank"] = lookups[metric.key][0].get(
                        row.algorithm
                    )
                writer.writerow(record)
    return destination


def _rank_colormap() -> "colors.LinearSegmentedColormap":
    """Pale green -> green -> blue palette matching the approved reference."""
    # Rank 1 uses the left (dark-blue) end; higher ranks approach pale green.
    return colors.LinearSegmentedColormap.from_list(
        "green_blue_rank",
        ["#0B4F8A", "#2E92BB", "#78C6C0", "#BCE2BD", "#E4F2DB"],
        N=256,
    )


def _text_color(rank: float, maximum_rank: int) -> str:
    if maximum_rank <= 1:
        return "white"
    quality = 1.0 - ((rank - 1.0) / (maximum_rank - 1.0))
    return "white" if quality >= 0.57 else "#111111"


def _save_one_map_heatmap(
    *,
    map_id: str,
    rows: list[AlgorithmHeatmapRow],
    output_stem: Path,
    dpi: int,
    include_title: bool,
) -> list[Path]:
    """Save one large half-page figure with geometrically square cells."""
    if plt is None or colors is None or patches is None or ScalarMappable is None:
        raise RuntimeError(
            "This figure requires matplotlib. Install the optional "
            "package requirements before generating outputs."
        )
    # 7.25 x 6.75 in prints comfortably at roughly half an A4/Letter page.
    figure, axis = plt.subplots(figsize=(7.25, 6.75))
    figure.patch.set_facecolor("white")
    rank_map = {
        metric.key: _rank_lookup(rows, metric) for metric in METRICS
    }
    maximum_rank = max((count for _, count in rank_map.values()), default=1)
    normalization = colors.Normalize(vmin=1.0, vmax=max(maximum_rank, 2))
    cmap = _rank_colormap()

    for row_index, row in enumerate(rows):
        for column_index, metric in enumerate(METRICS):
            rank = rank_map[metric.key][0].get(row.algorithm)
            if rank is None:
                facecolor = "#F9D7D3"
                text_color = "#8A1C17"
            else:
                facecolor = cmap(normalization(rank))
                text_color = _text_color(rank, maximum_rank)
            tile = patches.Rectangle(
                (column_index, row_index),
                1.0,
                1.0,
                facecolor=facecolor,
                edgecolor="white",
                linewidth=2.2,
            )
            axis.add_patch(tile)
            axis.text(
                column_index + 0.5,
                row_index + 0.5,
                _format_value(metric, row),
                ha="center",
                va="center",
                fontsize=9.8,
                fontweight="bold",
                color=text_color,
            )

    column_positions = [index + 0.5 for index in range(len(METRICS))]
    row_positions = [index + 0.5 for index in range(len(rows))]
    axis.set_xlim(0, len(METRICS))
    axis.set_ylim(len(rows), 0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks(column_positions, [metric.label for metric in METRICS])
    axis.set_yticks(row_positions, [row.display_name for row in rows])
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", length=0, pad=8, labelsize=10.5)
    axis.tick_params(axis="y", length=0, pad=9, labelsize=11.5)
    for label in axis.get_yticklabels():
        label.set_fontweight(
            "bold" if label.get_text() == "SFO" else "normal"
        )
    for spine in axis.spines.values():
        spine.set_visible(False)

    title = MAP_TITLES.get(map_id, map_id.replace("_", " ").title())
    if include_title:
        title = f"Performance rank heatmap - {title}"
    axis.set_title(title, fontsize=14, fontweight="bold", pad=34)

    colorbar = figure.colorbar(
        ScalarMappable(norm=normalization, cmap=cmap),
        ax=axis,
        fraction=0.050,
        pad=0.035,
    )
    ticks = [1] if maximum_rank <= 1 else [1, maximum_rank]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(
        ["1 (best)"] if len(ticks) == 1 else ["1 (best)", f"{maximum_rank} (worst)"]
    )
    colorbar.ax.tick_params(labelsize=8.5, length=0)
    colorbar.set_label("Within-map rank", fontsize=9.5, labelpad=8)

    figure.text(
        0.16,
        0.052,
        "Cell labels: medians across valid, completed, collision-free runs; "
        "success is valid runs / total runs.",
        ha="left",
        va="center",
        fontsize=8.4,
    )
    figure.text(
        0.16,
        0.025,
        "Failure codes: COL = collision; MIX = partial/collision; "
        "NP = no path; NA = no recorded run.",
        ha="left",
        va="center",
        fontsize=8.4,
        color="#8A1C17",
    )
    figure.subplots_adjust(left=0.24, right=0.91, top=0.84, bottom=0.12)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    destinations: list[Path] = []
    for extension in ("png", "pdf", "svg"):
        destination = output_stem.with_suffix(f".{extension}")
        figure.savefig(
            destination,
            dpi=dpi if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        destinations.append(destination)
    plt.close(figure)
    return destinations


def generate_rank_heatmaps(
    *,
    results_root: Path,
    output_directory: Path,
    dpi: int = 600,
    include_title: bool = False,
) -> list[Path]:
    """Generate one square-cell heatmap per benchmark map."""
    rows = summarize_heatmap_rows(results_root)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    generated = [
        write_heatmap_data(rows, output_directory / "performance_rank_heatmap_data.csv")
    ]
    map_order = list(dict.fromkeys(row.map_id for row in rows))
    for map_id in map_order:
        map_rows = [row for row in rows if row.map_id == map_id]
        generated.extend(
            _save_one_map_heatmap(
                map_id=map_id,
                rows=map_rows,
                output_stem=output_directory / f"{map_id}_performance_rank_heatmap",
                dpi=dpi,
                include_title=include_title,
            )
        )
    return generated

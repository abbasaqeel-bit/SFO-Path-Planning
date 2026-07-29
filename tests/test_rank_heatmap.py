from __future__ import annotations

import csv
from pathlib import Path

import pytest

from sfo_benchmark.visualization.rank_heatmap import (
    generate_rank_heatmaps,
    summarize_heatmap_rows,
)


def _write_rows(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "map_id",
        "algorithm",
        "success",
        "collision_free",
        "path_status",
        "valid_path_length",
        "total_turning_angle_rad",
        "execution_time_s",
        "minimum_clearance",
        "message",
    ]
    values = [
        {
            "map_id": "dense_01",
            "algorithm": "astar_pythonrobotics",
            "success": "True",
            "collision_free": "True",
            "path_status": "valid_complete",
            "valid_path_length": "101.0",
            "total_turning_angle_rad": "2.0",
            "execution_time_s": "4.0",
            "minimum_clearance": "0.5",
            "message": "ok",
        },
        {
            "map_id": "dense_01",
            "algorithm": "aco_published_global",
            "success": "False",
            "collision_free": "False",
            "path_status": "partial_collision",
            "valid_path_length": "",
            "total_turning_angle_rad": "",
            "execution_time_s": "10.0",
            "minimum_clearance": "",
            "message": "collision detected",
        },
    ]
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(values)


def test_heatmap_summary_uses_valid_runs_and_diagnostic_codes(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _write_rows(results / "tables" / "all_runs.csv")

    rows = summarize_heatmap_rows(results)
    aco = next(row for row in rows if row.algorithm == "aco_published_global")
    assert aco.valid_runs == 0
    assert aco.failure_code == "MIX"
    astar = next(row for row in rows if row.algorithm == "astar_pythonrobotics")
    assert astar.values["path_length"] == 101.0
    assert astar.values["success"] == 1.0


def test_square_heatmap_writes_all_formats(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    results = tmp_path / "results"
    _write_rows(results / "tables" / "all_runs.csv")

    output = tmp_path / "output"
    generated = generate_rank_heatmaps(
        results_root=results,
        output_directory=output,
        dpi=100,
    )
    names = {path.name for path in generated}
    assert "performance_rank_heatmap_data.csv" in names
    assert "dense_01_performance_rank_heatmap.png" in names
    assert "dense_01_performance_rank_heatmap.pdf" in names
    assert "dense_01_performance_rank_heatmap.svg" in names

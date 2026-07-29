from __future__ import annotations

import csv
from pathlib import Path

from sfo_benchmark.visualization.pareto_quality import (
    _pareto_front,
    generate_rank_pareto_outputs,
    summarize_quality_points,
)


def _write_results(root: Path) -> None:
    table = root / "tables" / "all_runs.csv"
    table.parent.mkdir(parents=True)
    rows = [
        {
            "map_id": "map_1",
            "algorithm": "algorithm_a",
            "success": True,
            "collision_free": True,
            "path_status": "valid_complete",
            "valid_path_length": 10.0,
            "total_turning_angle_rad": 4.0,
            "execution_time_s": 1.0,
        },
        {
            "map_id": "map_1",
            "algorithm": "algorithm_a",
            "success": True,
            "collision_free": True,
            "path_status": "valid_complete",
            "valid_path_length": 12.0,
            "total_turning_angle_rad": 2.0,
            "execution_time_s": 3.0,
        },
        {
            "map_id": "map_1",
            "algorithm": "algorithm_b",
            "success": True,
            "collision_free": True,
            "path_status": "valid_complete",
            "valid_path_length": 9.0,
            "total_turning_angle_rad": 5.0,
            "execution_time_s": 2.0,
        },
        {
            "map_id": "map_1",
            "algorithm": "algorithm_b",
            "success": False,
            "collision_free": False,
            "path_status": "partial",
            "valid_path_length": "",
            "total_turning_angle_rad": 1.0,
            "execution_time_s": 2.0,
        },
        {
            "map_id": "map_1",
            "algorithm": "algorithm_c",
            "success": False,
            "collision_free": False,
            "path_status": "none",
            "valid_path_length": "",
            "total_turning_angle_rad": "",
            "execution_time_s": 1.0,
        },
    ]
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_quality_summary_ranks_failures_and_outputs(tmp_path: Path) -> None:
    _write_results(tmp_path)
    points = summarize_quality_points(tmp_path)
    by_name = {point.algorithm: point for point in points}

    assert by_name["algorithm_a"].path_length_median == 11.0
    assert by_name["algorithm_a"].turning_angle_median == 3.0
    assert by_name["algorithm_a"].successful_runs == 2
    assert by_name["algorithm_b"].successful_runs == 1
    assert by_name["algorithm_c"].successful_runs == 0
    assert by_name["algorithm_c"].path_length_rank is None
    assert len(_pareto_front(points)) == 2

    generated = generate_rank_pareto_outputs(
        results_root=tmp_path,
        output_directory=tmp_path / "output",
        dpi=72,
    )
    assert len(generated) == 4
    assert all(path.exists() and path.stat().st_size > 0 for path in generated)

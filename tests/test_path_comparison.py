from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.obstacle import RectangularObstacle
from sfo_benchmark.visualization.combined_paths import (
    approximate_collision_locations,
    generate_path_comparison_figures,
    load_map_runs,
    select_representative,
)


def _write_run(
    root: Path,
    *,
    algorithm: str,
    run_number: int,
    success: bool,
    path_status: str,
    points: list[list[float]] | None,
    path_length: float | None,
    goal_distance: float | None,
) -> None:
    directory = root / "test_map" / algorithm / f"Run_{run_number:03d}"
    directory.mkdir(parents=True)
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                "success": success,
                "path_status": path_status,
                "path_length": path_length,
                "goal_distance_remaining": goal_distance,
                "minimum_clearance": 1.0 if success else -0.2,
            }
        ),
        encoding="utf-8",
    )
    if points is not None:
        with (directory / "trajectory.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["waypoint_index", "x", "y"])
            for index, point in enumerate(points):
                writer.writerow([index, *point])


def test_representatives_and_publication_outputs(tmp_path: Path) -> None:
    lengths = [12.0, 10.0, 11.0, 30.0, 9.0]
    for number, length in enumerate(lengths, start=1):
        _write_run(
            tmp_path,
            algorithm="three_phase_incremental_sfo",
            run_number=number,
            success=True,
            path_status="valid_complete",
            points=[[0.0, 0.0], [10.0, 10.0]],
            path_length=length,
            goal_distance=0.0,
        )
    _write_run(
        tmp_path,
        algorithm="prm_pythonrobotics",
        run_number=1,
        success=False,
        path_status="invalid_complete",
        points=[[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]],
        path_length=14.2,
        goal_distance=0.0,
    )
    _write_run(
        tmp_path,
        algorithm="aco_published_global",
        run_number=1,
        success=False,
        path_status="none",
        points=None,
        path_length=None,
        goal_distance=None,
    )

    environment = Environment(
        map_id="test_map",
        bounds=(0.0, 10.0, 0.0, 10.0),
        start=np.array([0.0, 0.0]),
        goal=np.array([10.0, 10.0]),
        robot_radius=0.2,
        obstacles=(
            RectangularObstacle(4.0, 6.0, 4.0, 6.0),
        ),
    )
    loaded = load_map_runs(tmp_path, "test_map")
    representative = select_representative(
        "three_phase_incremental_sfo",
        loaded["three_phase_incremental_sfo"],
        environment.goal,
    )
    assert representative.selected is not None
    assert representative.selected.path_length == 11.0

    output = tmp_path / "figures"
    generated = generate_path_comparison_figures(
        results_root=tmp_path,
        environments={"test_map": environment},
        output_directory=output,
        dpi=72,
    )
    assert len(generated) == 7
    assert all(path.exists() and path.stat().st_size > 0 for path in generated)

    collisions = approximate_collision_locations(
        np.array([[0.0, 0.0], [10.0, 10.0]]),
        environment,
    )
    assert collisions.shape == (1, 2)
    assert np.allclose(collisions[0], [5.0, 5.0], atol=0.25)

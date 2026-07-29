from pathlib import Path

import numpy as np

from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.path import Path as PlannedPath
from sfo_benchmark.visualization.path_animation import save_path_gif


def test_gif_is_written_for_partial_and_absent_paths(tmp_path: Path) -> None:
    environment = Environment(
        map_id="animation",
        bounds=(0.0, 10.0, 0.0, 10.0),
        start=np.array([0.0, 0.0]),
        goal=np.array([10.0, 10.0]),
    )
    partial = PlannedPath(np.array([[0.0, 0.0], [3.0, 4.0]]))
    partial_output = save_path_gif(
        environment,
        partial,
        "planner",
        tmp_path / "partial.gif",
        success=False,
    )
    absent_output = save_path_gif(
        environment,
        None,
        "planner",
        tmp_path / "absent.gif",
        success=False,
    )
    assert partial_output.stat().st_size > 0
    assert absent_output.stat().st_size > 0

"""Tests for segment-level path clearance."""

import numpy as np

from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.obstacle import CircularObstacle
from sfo_benchmark.core.path import Path
from sfo_benchmark.evaluation.metrics import minimum_clearance


def test_circle_segment_clearance() -> None:
    environment = Environment(
        map_id="clearance",
        bounds=(0.0, 10.0, 0.0, 10.0),
        start=np.array([0.0, 0.0]),
        goal=np.array([10.0, 0.0]),
        obstacles=(CircularObstacle(np.array([5.0, 5.0]), 1.0),),
    )
    path = Path(np.array([[0.0, 0.0], [10.0, 0.0]]))
    assert minimum_clearance(path, environment) == 4.0

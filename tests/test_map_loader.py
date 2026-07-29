"""Tests for YAML-compatible map construction."""

import numpy as np

from sfo_benchmark.core.map_loader import environment_from_mapping
from sfo_benchmark.core.obstacle import (
    CircularObstacle,
    RectangularObstacle,
)


def test_environment_from_mapping() -> None:
    environment = environment_from_mapping(
        "map",
        {
            "bounds": [0, 20, 0, 20],
            "start": [1, 1],
            "goal": [19, 19],
            "robot_radius": 0.5,
            "obstacles": [
                {"type": "circle", "center": [5, 5], "radius": 2},
                {"type": "rectangle", "bounds": [10, 12, 4, 8]},
            ],
        },
    )
    assert np.allclose(environment.start, [1, 1])
    assert isinstance(environment.obstacles[0], CircularObstacle)
    assert isinstance(environment.obstacles[1], RectangularObstacle)

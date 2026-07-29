"""Tests for rectangular obstacle geometry."""

import numpy as np

from sfo_benchmark.core.obstacle import RectangularObstacle


def test_rectangle_signed_clearance() -> None:
    rectangle = RectangularObstacle(2.0, 4.0, 2.0, 4.0)
    assert rectangle.clearance(np.array([1.0, 3.0])) == 1.0
    assert rectangle.clearance(np.array([3.0, 3.0])) == -1.0

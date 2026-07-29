import numpy as np

from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.obstacle import CircularObstacle
from sfo_benchmark.planners.pythonrobotics.grid_conversion import (
    environment_to_grid_problem,
)


def test_grid_conversion_contains_bounds_and_obstacle() -> None:
    environment = Environment(
        map_id="grid",
        bounds=(0.0, 10.0, 0.0, 10.0),
        start=np.array([1.0, 1.0]),
        goal=np.array([9.0, 9.0]),
        obstacles=(CircularObstacle(np.array([5.0, 5.0]), 1.5),),
    )

    problem = environment_to_grid_problem(environment, resolution=1.0)
    points = set(zip(problem.obstacle_x, problem.obstacle_y, strict=True))

    assert (0.0, 0.0) in points
    assert (10.0, 10.0) in points
    assert (5.0, 5.0) in points

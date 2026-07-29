import numpy as np

from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.obstacle import (
    CircularObstacle,
    RectangularObstacle,
)
from sfo_benchmark.planners.pythonrobotics.sampling_conversion import (
    environment_to_sampling_problem,
)


def test_sampling_conversion_supports_rectangles() -> None:
    environment = Environment(
        map_id="sampling",
        bounds=(0.0, 10.0, 0.0, 10.0),
        start=np.array([1.0, 1.0]),
        goal=np.array([9.0, 9.0]),
        obstacles=(
            CircularObstacle(np.array([3.0, 3.0]), 1.0),
            RectangularObstacle(5.0, 7.0, 4.0, 6.0),
        ),
    )
    problem = environment_to_sampling_problem(
        environment,
        rectangle_resolution=1.0,
    )

    assert len(problem.obstacle_list) > 1
    assert problem.play_area == [0.0, 10.0, 0.0, 10.0]
    assert len(problem.obstacle_x) > len(problem.obstacle_list)

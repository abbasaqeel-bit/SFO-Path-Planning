import numpy as np
import pytest

from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.obstacle import CircularObstacle
from sfo_benchmark.evaluation.benchmark import BenchmarkRunner
from sfo_benchmark.planners.base import PlannerConfig
from sfo_benchmark.planners.pythonrobotics import AStarAdapter, DijkstraAdapter


@pytest.fixture
def environment() -> Environment:
    return Environment(
        map_id="adapter_test",
        bounds=(0.0, 20.0, 0.0, 20.0),
        start=np.array([2.0, 2.0]),
        goal=np.array([18.0, 18.0]),
        robot_radius=0.25,
        obstacles=(CircularObstacle(np.array([10.0, 10.0]), 3.0),),
    )


@pytest.mark.parametrize("adapter_class", [AStarAdapter, DijkstraAdapter])
def test_adapter_returns_valid_path(adapter_class, environment) -> None:
    planner = adapter_class(
        PlannerConfig(
            name=adapter_class.__name__,
            parameters={"resolution": 1.0},
        )
    )

    result = BenchmarkRunner(environment).run(planner, seed=7, run_id=1)

    assert result.success, result.message
    assert result.path is not None
    assert result.path_length is not None
    assert result.path_length > np.linalg.norm(environment.goal - environment.start)
    assert result.metadata["source_modified"] is False

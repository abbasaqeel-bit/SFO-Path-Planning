"""Integration test for benchmark result normalization."""

import numpy as np

from sfo_benchmark.core.environment import Environment
from sfo_benchmark.evaluation.benchmark import BenchmarkRunner
from sfo_benchmark.planners.base import PlannerConfig
from sfo_benchmark.planners.null_planner import NullPlanner


def test_runner_normalizes_metrics() -> None:
    environment = Environment(
        map_id="empty",
        bounds=(0.0, 10.0, 0.0, 10.0),
        start=np.array([0.0, 0.0]),
        goal=np.array([3.0, 4.0]),
    )
    planner = NullPlanner(PlannerConfig(name="null"))

    result = BenchmarkRunner(environment).run(
        planner,
        seed=1,
        run_id=1,
    )

    assert result.success
    assert result.path is not None
    assert result.path_length == result.path.length()
    assert result.path_length >= 5.0
    assert result.waypoint_count == 3
    assert result.segment_count == 2
    assert result.minimum_clearance == float("inf")

from __future__ import annotations

import numpy as np

from sfo_benchmark.core.environment import Environment
from sfo_benchmark.planners.aco import PublishedACOAdapter
from sfo_benchmark.planners.base import PlannerConfig


def open_environment() -> Environment:
    return Environment(
        map_id="open",
        bounds=(0.0, 10.0, 0.0, 10.0),
        start=np.array([1.0, 1.0]),
        goal=np.array([9.0, 9.0]),
    )


def test_aco_uses_variable_grid_chain() -> None:
    planner = PublishedACOAdapter(
        PlannerConfig(
            name="aco",
            parameters={
                "ant_count": 8,
                "colony_iterations": 8,
                "step_count": 100,
            },
        )
    )
    result = planner.plan(open_environment(), seed=11, run_id=1)
    assert result.metadata["fixed_waypoint_encoding"] is False
    assert (
        result.metadata["representation"]
        == "native_haghrah_20x20_grid_chain"
    )
    assert result.metadata["source_invoked_directly"] is True
    assert result.metadata["objective_usage"] == "native_internal_search"
    assert result.metadata["grid_size"] == 20
    assert len(result.metadata["fitness_history"]) == 8
    assert result.path is not None

import numpy as np

from sfo_benchmark.core.environment import Environment
from sfo_benchmark.planners.base import PlannerConfig
from sfo_benchmark.planners.sfo import ThreePhaseSFOAdapter


def test_sfo_adapter_completes_easy_open_map() -> None:
    environment = Environment(
        map_id="easy",
        bounds=(0.0, 30.0, 0.0, 30.0),
        start=np.array([5.0, 5.0]),
        goal=np.array([25.0, 25.0]),
        obstacles=(),
    )
    planner = ThreePhaseSFOAdapter(
        PlannerConfig(
            name="sfo",
            parameters={
                "population_size": 8,
                "max_iterations": 30,
                "max_glide_iterations": 15,
                "min_completed_for_target": 1,
                "goal_connection_distance": 30.0,
                "target_improvement_window": 3,
                "force_micro_last_iterations": 5,
                "micro_stagnation_limit": 3,
                "micro_attempts_per_waypoint": 2,
            },
        )
    )
    result = planner.plan(environment, seed=42, run_id=1)
    assert result.success
    assert result.path is not None
    assert np.allclose(result.path.points[0], environment.start)
    assert np.allclose(result.path.points[-1], environment.goal)
    assert result.metadata["source_equations_modified"] is False

import numpy as np

from sfo_benchmark.core.environment import Environment
from sfo_benchmark.planners.sfo import (
    ThreePhaseSFOConfig,
    ThreePhaseSFOEngine,
)


def make_engine() -> ThreePhaseSFOEngine:
    environment = Environment(
        map_id="open",
        bounds=(0.0, 100.0, 0.0, 100.0),
        start=np.array([5.0, 5.0]),
        goal=np.array([95.0, 95.0]),
        obstacles=(),
    )
    return ThreePhaseSFOEngine(
        environment,
        ThreePhaseSFOConfig(
            population_size=2,
            max_iterations=5,
        ),
        seed=42,
    )


def test_relative_arc_reference_supports_different_waypoint_counts() -> None:
    engine = make_engine()
    path = [
        np.array([0.0, 0.0]),
        np.array([5.0, 0.0]),
        np.array([10.0, 0.0]),
    ]
    point = engine.point_at_relative_arc(path, 0.25)
    assert np.allclose(point, [2.5, 0.0])


def test_line_of_sight_pruning_removes_redundant_points() -> None:
    engine = make_engine()
    path = [
        np.array([5.0, 5.0]),
        np.array([20.0, 20.0]),
        np.array([40.0, 40.0]),
    ]
    pruned, removed = engine.prune_redundant_waypoints(path)
    assert removed == 1
    assert len(pruned) == 2

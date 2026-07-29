import numpy as np

from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.path import Path
from sfo_benchmark.evaluation.research_metrics import compute_research_metrics


def test_research_metrics_for_straight_path() -> None:
    environment = Environment(
        map_id="metrics",
        bounds=(0.0, 10.0, 0.0, 10.0),
        start=np.array([0.0, 0.0]),
        goal=np.array([3.0, 4.0]),
    )
    path = Path(np.array([[0.0, 0.0], [3.0, 4.0]]))
    metrics = compute_research_metrics(
        path=path,
        environment=environment,
        success=True,
        execution_time_s=0.1,
        cpu_time_s=0.08,
        peak_memory_mb=1.2,
    )
    assert metrics.path_length == 5.0
    assert metrics.path_efficiency == 1.0
    assert metrics.collision_free
    assert metrics.path_status == "valid_complete"
    assert metrics.valid_path_length == 5.0
    assert metrics.goal_distance_remaining == 0.0


def test_partial_path_is_not_counted_as_valid_length() -> None:
    environment = Environment(
        map_id="partial",
        bounds=(0.0, 10.0, 0.0, 10.0),
        start=np.array([0.0, 0.0]),
        goal=np.array([10.0, 10.0]),
    )
    path = Path(np.array([[0.0, 0.0], [4.0, 4.0]]))
    metrics = compute_research_metrics(
        path=path,
        environment=environment,
        success=False,
        execution_time_s=0.1,
        cpu_time_s=0.08,
        peak_memory_mb=1.2,
    )
    assert metrics.path_status == "partial"
    assert metrics.path_length is not None
    assert metrics.valid_path_length is None
    assert not metrics.success

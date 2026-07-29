import numpy as np

from sfo_benchmark.planners.base import PlannerConfig
from sfo_benchmark.planners.pythonrobotics.rrt_adapters import (
    _best_partial_tree_path,
)


def test_partial_tree_path_supports_integer_parent_indices() -> None:
    class Node:
        def __init__(self, x: float, y: float, parent: int) -> None:
            self.x = x
            self.y = y
            self.parent = parent

    nodes = [
        Node(0.0, 0.0, -1),
        Node(1.0, 0.0, 0),
        Node(2.0, 0.0, 1),
    ]
    path = _best_partial_tree_path(nodes, np.array([3.0, 0.0]))
    assert path == [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]


def test_partial_tree_path_supports_object_parents() -> None:
    class Node:
        def __init__(self, x: float, y: float, parent=None) -> None:
            self.x = x
            self.y = y
            self.parent = parent

    root = Node(0.0, 0.0)
    middle = Node(1.0, 0.0, root)
    leaf = Node(2.0, 0.0, middle)
    path = _best_partial_tree_path(
        [root, middle, leaf],
        np.array([3.0, 0.0]),
    )
    assert path == [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
from sfo_benchmark.planners.pythonrobotics import (
    InformedRRTStarAdapter,
    PRMAdapter,
    RRTAdapter,
    RRTStarAdapter,
    ThetaStarAdapter,
)


def test_extended_adapter_identifiers() -> None:
    cases = [
        (ThetaStarAdapter, "ThetaStarPlanner"),
        (RRTAdapter, "RRT"),
        (RRTStarAdapter, "RRTStar"),
        (InformedRRTStarAdapter, "InformedRRTStar"),
        (PRMAdapter, "ProbabilisticRoadMap"),
    ]

    for adapter_class, expected in cases:
        adapter = adapter_class(
            PlannerConfig(
                name=expected,
                parameters={},
            )
        )
        assert adapter.upstream_algorithm == expected

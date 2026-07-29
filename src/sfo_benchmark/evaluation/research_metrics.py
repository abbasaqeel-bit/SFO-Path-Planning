"""Research-oriented path and runtime metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from sfo_benchmark.core.collision import CollisionChecker
from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.path import Path
from sfo_benchmark.evaluation.metrics import minimum_clearance, total_turning_angle


@dataclass(frozen=True, slots=True)
class ResearchMetrics:
    success: bool
    collision_free: bool
    path_status: str
    reaches_goal: bool
    goal_distance_remaining: float | None
    valid_path_length: float | None
    path_length: float | None
    waypoint_count: int | None
    segment_count: int | None
    total_turning_angle_rad: float | None
    average_turning_angle_rad: float | None
    minimum_clearance: float | None
    straight_line_distance: float
    path_efficiency: float | None
    execution_time_s: float
    cpu_time_s: float
    peak_memory_mb: float
    expanded_nodes: int | None
    generated_nodes: int | None
    maximum_open_set_size: int | None
    search_iterations: int | None
    fitness_evaluations: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def compute_research_metrics(
    *,
    path: Path | None,
    environment: Environment,
    success: bool,
    execution_time_s: float,
    cpu_time_s: float,
    peak_memory_mb: float,
    metadata: dict[str, Any] | None = None,
    fitness_evaluations: int | None = None,
) -> ResearchMetrics:
    metadata = metadata or {}
    straight = float(np.linalg.norm(environment.goal - environment.start))

    common = dict(
        success=bool(success),
        execution_time_s=float(execution_time_s),
        cpu_time_s=float(cpu_time_s),
        peak_memory_mb=float(peak_memory_mb),
        straight_line_distance=straight,
        expanded_nodes=_optional_int(metadata.get("expanded_nodes")),
        generated_nodes=_optional_int(metadata.get("generated_nodes")),
        maximum_open_set_size=_optional_int(metadata.get("maximum_open_set_size")),
        search_iterations=_optional_int(metadata.get("search_iterations")),
        fitness_evaluations=fitness_evaluations,
    )

    if path is None:
        return ResearchMetrics(
            collision_free=False,
            path_status="none",
            reaches_goal=False,
            goal_distance_remaining=None,
            valid_path_length=None,
            path_length=None,
            waypoint_count=None,
            segment_count=None,
            total_turning_angle_rad=None,
            average_turning_angle_rad=None,
            minimum_clearance=None,
            path_efficiency=None,
            **common,
        )

    length = float(path.length())
    turning = float(total_turning_angle(path))
    angle_count = max(path.waypoint_count - 2, 0)
    collision_free = CollisionChecker(environment).path_is_free(path)
    goal_distance = float(
        np.linalg.norm(path.points[-1] - environment.goal)
    )
    reaches_goal = bool(goal_distance <= 1e-6)
    valid = bool(success and collision_free and reaches_goal)
    common["success"] = valid
    if valid:
        path_status = "valid_complete"
    elif reaches_goal:
        path_status = "invalid_complete"
    else:
        path_status = "partial"

    return ResearchMetrics(
        collision_free=collision_free,
        path_status=path_status,
        reaches_goal=reaches_goal,
        goal_distance_remaining=goal_distance,
        valid_path_length=length if valid else None,
        path_length=length,
        waypoint_count=path.waypoint_count,
        segment_count=max(path.waypoint_count - 1, 0),
        total_turning_angle_rad=turning,
        average_turning_angle_rad=turning / angle_count if angle_count else 0.0,
        minimum_clearance=float(minimum_clearance(path, environment)),
        path_efficiency=straight / length if length > 0.0 else math.nan,
        **common,
    )

"""Planner-independent path metrics."""

from __future__ import annotations

import math

import numpy as np

from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.path import Path


def total_turning_angle(path: Path) -> float:
    """Return the sum of absolute internal turning angles in radians."""
    if path.waypoint_count < 3:
        return 0.0

    vectors = np.diff(path.points, axis=0)
    total = 0.0
    for first, second in zip(vectors[:-1], vectors[1:], strict=True):
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator == 0.0:
            continue
        cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
        total += abs(math.acos(cosine))
    return float(total)


def minimum_clearance(
    path: Path,
    environment: Environment,
    *,
    rectangle_samples: int = 201,
) -> float:
    """Return obstacle clearance along the complete path polyline.

    Circle-segment clearance is analytic. Axis-aligned rectangle clearance is
    sampled deterministically at a fixed resolution to keep the implementation
    transparent and reproducible.
    """
    if not environment.obstacles:
        return math.inf

    clearance = math.inf
    for start, end in zip(path.points[:-1], path.points[1:], strict=True):
        for obstacle in environment.obstacles:
            value = obstacle.segment_clearance(
                start,
                end,
                samples=rectangle_samples,
            )
            clearance = min(clearance, value - environment.robot_radius)
    return float(clearance)

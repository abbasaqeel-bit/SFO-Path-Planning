"""Conversion from the common environment to PythonRobotics obstacle points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sfo_benchmark.core.environment import Environment


@dataclass(frozen=True, slots=True)
class GridProblem:
    obstacle_x: list[float]
    obstacle_y: list[float]
    resolution: float
    robot_radius: float


def _axis_values(lower: float, upper: float, resolution: float) -> np.ndarray:
    count = int(round((upper - lower) / resolution))
    values = lower + np.arange(count + 1, dtype=float) * resolution
    if values[-1] < upper:
        values = np.append(values, upper)
    else:
        values[-1] = upper
    return values


def environment_to_grid_problem(
    environment: Environment,
    *,
    resolution: float,
) -> GridProblem:
    if resolution <= 0.0:
        raise ValueError("resolution must be positive.")

    xmin, xmax, ymin, ymax = environment.bounds
    xs = _axis_values(xmin, xmax, resolution)
    ys = _axis_values(ymin, ymax, resolution)

    obstacle_points: set[tuple[float, float]] = set()

    for x in xs:
        obstacle_points.add((float(x), float(ymin)))
        obstacle_points.add((float(x), float(ymax)))
    for y in ys:
        obstacle_points.add((float(xmin), float(y)))
        obstacle_points.add((float(xmax), float(y)))

    for x in xs:
        for y in ys:
            point = np.array([x, y], dtype=float)
            if any(
                obstacle.contains(point, environment.robot_radius)
                for obstacle in environment.obstacles
            ):
                obstacle_points.add((float(x), float(y)))

    ordered = sorted(obstacle_points)
    return GridProblem(
        obstacle_x=[point[0] for point in ordered],
        obstacle_y=[point[1] for point in ordered],
        resolution=float(resolution),
        robot_radius=max(float(resolution) * 1.0e-6, 1.0e-9),
    )

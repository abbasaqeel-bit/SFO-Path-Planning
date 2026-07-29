"""Environment conversion for PythonRobotics sampling planners."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.obstacle import (
    CircularObstacle,
    RectangularObstacle,
)


@dataclass(frozen=True, slots=True)
class SamplingProblem:
    obstacle_list: list[tuple[float, float, float]]
    obstacle_x: list[float]
    obstacle_y: list[float]
    random_area: list[float]
    play_area: list[float]


def environment_to_sampling_problem(
    environment: Environment,
    *,
    rectangle_resolution: float,
) -> SamplingProblem:
    """Convert all obstacles to conservative circular samples."""
    if rectangle_resolution <= 0.0:
        raise ValueError("rectangle_resolution must be positive.")

    circles: list[tuple[float, float, float]] = []

    for obstacle in environment.obstacles:
        if isinstance(obstacle, CircularObstacle):
            circles.append(
                (
                    float(obstacle.center[0]),
                    float(obstacle.center[1]),
                    float(obstacle.radius),
                )
            )
        elif isinstance(obstacle, RectangularObstacle):
            circles.extend(
                _rectangle_to_circles(
                    obstacle,
                    rectangle_resolution,
                )
            )
        else:
            raise TypeError(
                f"Unsupported obstacle type: {type(obstacle)!r}"
            )

    xmin, xmax, ymin, ymax = environment.bounds

    # Add boundary samples so PRM's sampling domain matches the map.
    boundary_radius = max(rectangle_resolution * 0.5, 0.25)
    boundary_points = _boundary_circles(
        xmin,
        xmax,
        ymin,
        ymax,
        rectangle_resolution,
        boundary_radius,
    )
    circles_with_boundary = [*circles, *boundary_points]

    obstacle_x = [circle[0] for circle in circles_with_boundary]
    obstacle_y = [circle[1] for circle in circles_with_boundary]

    return SamplingProblem(
        obstacle_list=circles,
        obstacle_x=obstacle_x,
        obstacle_y=obstacle_y,
        random_area=[float(min(xmin, ymin)), float(max(xmax, ymax))],
        play_area=[float(xmin), float(xmax), float(ymin), float(ymax)],
    )


def _rectangle_to_circles(
    obstacle: RectangularObstacle,
    resolution: float,
) -> list[tuple[float, float, float]]:
    radius = resolution * math.sqrt(2.0) / 2.0
    xs = np.arange(
        obstacle.xmin,
        obstacle.xmax + resolution * 0.5,
        resolution,
    )
    ys = np.arange(
        obstacle.ymin,
        obstacle.ymax + resolution * 0.5,
        resolution,
    )
    return [
        (float(x), float(y), float(radius))
        for x in xs
        for y in ys
    ]


def _boundary_circles(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    resolution: float,
    radius: float,
) -> list[tuple[float, float, float]]:
    xs = np.arange(xmin, xmax + resolution * 0.5, resolution)
    ys = np.arange(ymin, ymax + resolution * 0.5, resolution)

    points = [
        *((float(x), float(ymin), radius) for x in xs),
        *((float(x), float(ymax), radius) for x in xs),
        *((float(xmin), float(y), radius) for y in ys),
        *((float(xmax), float(y), radius) for y in ys),
    ]
    return list(dict.fromkeys(points))

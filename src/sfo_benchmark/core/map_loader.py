"""Construction of benchmark environments from validated YAML mappings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sfo_benchmark.utils.config import load_yaml

from .environment import Environment
from .obstacle import CircularObstacle, Obstacle, RectangularObstacle


def _build_obstacle(payload: dict[str, Any]) -> Obstacle:
    obstacle_type = str(payload.get("type", "")).lower()

    if obstacle_type == "circle":
        return CircularObstacle(
            center=np.asarray(payload["center"], dtype=float),
            radius=float(payload["radius"]),
        )

    if obstacle_type in {"rectangle", "rect"}:
        if "bounds" in payload:
            xmin, xmax, ymin, ymax = map(float, payload["bounds"])
        else:
            xmin = float(payload["xmin"])
            xmax = float(payload["xmax"])
            ymin = float(payload["ymin"])
            ymax = float(payload["ymax"])
        return RectangularObstacle(xmin, xmax, ymin, ymax)

    raise ValueError(f"Unsupported obstacle type: {obstacle_type!r}")


def environment_from_mapping(map_id: str, payload: dict[str, Any]) -> Environment:
    """Build one environment from a map configuration mapping."""
    bounds = tuple(map(float, payload["bounds"]))
    if len(bounds) != 4:
        raise ValueError("Map bounds must contain four numbers.")

    obstacles_payload = payload.get("obstacles", [])
    if not isinstance(obstacles_payload, list):
        raise ValueError("Map obstacles must be a list.")

    obstacles = tuple(_build_obstacle(item) for item in obstacles_payload)

    return Environment(
        map_id=map_id,
        bounds=bounds,
        start=np.asarray(payload["start"], dtype=float),
        goal=np.asarray(payload["goal"], dtype=float),
        robot_radius=float(payload.get("robot_radius", 0.0)),
        obstacles=obstacles,
    )


def load_environment(path: Path, map_id: str) -> Environment:
    """Load one named environment from a YAML map collection."""
    document = load_yaml(path)
    maps = document.get("maps")
    if not isinstance(maps, dict):
        raise ValueError("The YAML file must contain a top-level 'maps' mapping.")
    if map_id not in maps:
        available = ", ".join(sorted(map(str, maps)))
        raise KeyError(f"Unknown map_id {map_id!r}. Available maps: {available}")
    payload = maps[map_id]
    if not isinstance(payload, dict):
        raise ValueError(f"Map {map_id!r} must be a mapping.")
    return environment_from_mapping(map_id, payload)

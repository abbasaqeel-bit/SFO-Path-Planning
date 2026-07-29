"""Adapter for the official PythonRobotics A* implementation."""

from __future__ import annotations

from typing import Any

from .common import PythonRoboticsGridAdapter
from .vendor_loader import load_vendor_module


class AStarAdapter(PythonRoboticsGridAdapter):
    upstream_algorithm = "AStarPlanner"
    upstream_file = "PathPlanning/AStar/a_star.py"

    def _planner_class(self) -> type[Any]:
        module = load_vendor_module(
            self.upstream_file,
            "sfo_vendor_pythonrobotics_astar",
        )
        return module.AStarPlanner

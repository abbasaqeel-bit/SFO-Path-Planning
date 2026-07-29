"""Adapter for the official PythonRobotics Dijkstra implementation."""

from __future__ import annotations

from typing import Any

from .common import PythonRoboticsGridAdapter
from .vendor_loader import load_vendor_module


class DijkstraAdapter(PythonRoboticsGridAdapter):
    upstream_algorithm = "DijkstraPlanner"
    upstream_file = "PathPlanning/Dijkstra/dijkstra.py"

    def _planner_class(self) -> type[Any]:
        module = load_vendor_module(
            self.upstream_file,
            "sfo_vendor_pythonrobotics_dijkstra",
        )
        return module.DijkstraPlanner

"""Adapter for PythonRobotics Theta*."""

from __future__ import annotations

from typing import Any

from .common import PythonRoboticsGridAdapter
from .vendor_loader import load_vendor_module


class ThetaStarAdapter(PythonRoboticsGridAdapter):
    upstream_algorithm = "ThetaStarPlanner"
    upstream_file = "PathPlanning/ThetaStar/theta_star.py"

    def _planner_class(self) -> type[Any]:
        module = load_vendor_module(
            self.upstream_file,
            "sfo_vendor_pythonrobotics_theta_star",
        )
        module.use_theta_star = True
        return module.ThetaStarPlanner

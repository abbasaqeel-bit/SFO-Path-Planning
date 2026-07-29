"""Adapters for RRT, RRT*, and Informed RRT*."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from sfo_benchmark.core.environment import Environment

from .sampling_common import PythonRoboticsSamplingAdapter
from .vendor_loader import load_vendor_module


def _vendor_pathplanning_root() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "third_party"
        / "pythonrobotics"
        / "PathPlanning"
    )


def _vendor_root() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "third_party"
        / "pythonrobotics"
    )


def _best_partial_tree_path(
    nodes: list[Any],
    goal: np.ndarray,
) -> list[list[float]] | None:
    """Extract the native tree branch closest to the goal for diagnostics.

    PythonRobotics RRT/RRT* nodes store ``parent`` as another node object,
    whereas Informed RRT* stores ``parent`` as an integer index into
    ``node_list``.  Supporting both native representations is an adapter-only
    diagnostic fix; it does not modify either planner's search.
    """
    if not nodes:
        return None
    best = min(
        nodes,
        key=lambda node: float(
            np.hypot(float(node.x) - goal[0], float(node.y) - goal[1])
        ),
    )
    chain: list[list[float]] = []
    seen: set[int] = set()
    node: Any | None = best
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        chain.append([float(node.x), float(node.y)])
        parent = getattr(node, "parent", None)
        if isinstance(parent, (int, np.integer)):
            parent_index = int(parent)
            if parent_index < 0 or parent_index >= len(nodes):
                node = None
            else:
                node = nodes[parent_index]
        else:
            node = parent
    chain.reverse()
    return chain if len(chain) >= 2 else None


class RRTAdapter(PythonRoboticsSamplingAdapter):
    upstream_algorithm = "RRT"
    upstream_file = "PathPlanning/RRT/rrt.py"

    def _run_upstream(
        self,
        *,
        environment: Environment,
        seed: int,
        sampling_problem: Any,
    ) -> tuple[list[list[float]] | None, dict[str, Any]]:
        module = load_vendor_module(
            self.upstream_file,
            "sfo_vendor_pythonrobotics_rrt",
        )
        parameters = self.config.parameters
        planner = module.RRT(
            start=environment.start.tolist(),
            goal=environment.goal.tolist(),
            obstacle_list=sampling_problem.obstacle_list,
            rand_area=sampling_problem.random_area,
            expand_dis=float(parameters.get("expand_dis", 4.0)),
            path_resolution=float(
                parameters.get("path_resolution", 0.5)
            ),
            goal_sample_rate=int(
                parameters.get("goal_sample_rate", 10)
            ),
            max_iter=int(parameters.get("max_iter", 1000)),
            play_area=sampling_problem.play_area,
            robot_radius=environment.robot_radius,
        )
        path = planner.planning(animation=False)
        partial_path = _best_partial_tree_path(
            planner.node_list,
            environment.goal,
        )
        return path, {
            "iterations": int(parameters.get("max_iter", 1000)),
            "generated_nodes": len(planner.node_list),
            "tree_nodes": [
                [float(node.x), float(node.y)]
                for node in planner.node_list
            ],
            "partial_path": partial_path,
            "algorithm_parameters": {
                "expand_dis": float(
                    parameters.get("expand_dis", 4.0)
                ),
                "path_resolution": float(
                    parameters.get("path_resolution", 0.5)
                ),
                "goal_sample_rate": int(
                    parameters.get("goal_sample_rate", 10)
                ),
                "max_iter": int(
                    parameters.get("max_iter", 1000)
                ),
            },
        }


class RRTStarAdapter(PythonRoboticsSamplingAdapter):
    upstream_algorithm = "RRTStar"
    upstream_file = "PathPlanning/RRTStar/rrt_star.py"

    def _run_upstream(
        self,
        *,
        environment: Environment,
        seed: int,
        sampling_problem: Any,
    ) -> tuple[list[list[float]] | None, dict[str, Any]]:
        path_root = str(_vendor_pathplanning_root())
        if path_root not in sys.path:
            sys.path.insert(0, path_root)

        module = load_vendor_module(
            self.upstream_file,
            "sfo_vendor_pythonrobotics_rrt_star",
        )
        parameters = self.config.parameters
        planner = module.RRTStar(
            start=environment.start.tolist(),
            goal=environment.goal.tolist(),
            obstacle_list=sampling_problem.obstacle_list,
            rand_area=sampling_problem.random_area,
            expand_dis=float(parameters.get("expand_dis", 4.0)),
            path_resolution=float(
                parameters.get("path_resolution", 0.5)
            ),
            goal_sample_rate=int(
                parameters.get("goal_sample_rate", 10)
            ),
            max_iter=int(parameters.get("max_iter", 1000)),
            connect_circle_dist=float(
                parameters.get("connect_circle_dist", 50.0)
            ),
            search_until_max_iter=bool(
                parameters.get("search_until_max_iter", True)
            ),
            robot_radius=environment.robot_radius,
        )
        path = planner.planning(animation=False)
        partial_path = _best_partial_tree_path(
            planner.node_list,
            environment.goal,
        )
        return path, {
            "iterations": int(parameters.get("max_iter", 1000)),
            "generated_nodes": len(planner.node_list),
            "tree_nodes": [
                [float(node.x), float(node.y)]
                for node in planner.node_list
            ],
            "partial_path": partial_path,
            "algorithm_parameters": {
                "expand_dis": float(
                    parameters.get("expand_dis", 4.0)
                ),
                "path_resolution": float(
                    parameters.get("path_resolution", 0.5)
                ),
                "goal_sample_rate": int(
                    parameters.get("goal_sample_rate", 10)
                ),
                "max_iter": int(
                    parameters.get("max_iter", 1000)
                ),
                "connect_circle_dist": float(
                    parameters.get(
                        "connect_circle_dist",
                        50.0,
                    )
                ),
                "search_until_max_iter": bool(
                    parameters.get(
                        "search_until_max_iter",
                        True,
                    )
                ),
            },
        }


class InformedRRTStarAdapter(PythonRoboticsSamplingAdapter):
    upstream_algorithm = "InformedRRTStar"
    upstream_file = (
        "PathPlanning/InformedRRTStar/informed_rrt_star.py"
    )

    def _run_upstream(
        self,
        *,
        environment: Environment,
        seed: int,
        sampling_problem: Any,
    ) -> tuple[list[list[float]] | None, dict[str, Any]]:
        vendor_root = str(_vendor_root())
        if vendor_root not in sys.path:
            sys.path.insert(0, vendor_root)

        module = load_vendor_module(
            self.upstream_file,
            "sfo_vendor_pythonrobotics_informed_rrt_star",
        )
        parameters = self.config.parameters
        planner = module.InformedRRTStar(
            start=environment.start.tolist(),
            goal=environment.goal.tolist(),
            obstacle_list=sampling_problem.obstacle_list,
            rand_area=sampling_problem.random_area,
            expand_dis=float(parameters.get("expand_dis", 2.0)),
            goal_sample_rate=int(
                parameters.get("goal_sample_rate", 10)
            ),
            max_iter=int(parameters.get("max_iter", 1000)),
        )
        path = planner.informed_rrt_star_search(animation=False)
        nodes = planner.node_list or []
        partial_path = _best_partial_tree_path(nodes, environment.goal)
        return path, {
            "iterations": int(parameters.get("max_iter", 1000)),
            "generated_nodes": len(nodes),
            "tree_nodes": [
                [float(node.x), float(node.y)]
                for node in nodes
            ],
            "partial_path": partial_path,
            "algorithm_parameters": {
                "expand_dis": float(
                    parameters.get("expand_dis", 2.0)
                ),
                "goal_sample_rate": int(
                    parameters.get("goal_sample_rate", 10)
                ),
                "max_iter": int(
                    parameters.get("max_iter", 1000)
                ),
            },
        }

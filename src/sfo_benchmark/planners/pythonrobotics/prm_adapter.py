"""Adapter for the PythonRobotics PRM implementation."""

from __future__ import annotations

import heapq
from typing import Any

import numpy as np

from sfo_benchmark.core.environment import Environment

from .sampling_common import PythonRoboticsSamplingAdapter
from .vendor_loader import load_vendor_module


class PRMAdapter(PythonRoboticsSamplingAdapter):
    upstream_algorithm = "ProbabilisticRoadMap"
    upstream_file = (
        "PathPlanning/ProbabilisticRoadMap/"
        "probabilistic_road_map.py"
    )

    def _run_upstream(
        self,
        *,
        environment: Environment,
        seed: int,
        sampling_problem: Any,
    ) -> tuple[list[list[float]] | None, dict[str, Any]]:
        module = load_vendor_module(
            self.upstream_file,
            "sfo_vendor_pythonrobotics_prm",
        )
        parameters = self.config.parameters

        module.N_SAMPLE = int(
            parameters.get("n_sample", 500)
        )
        module.N_KNN = int(parameters.get("n_knn", 10))
        module.MAX_EDGE_LEN = float(
            parameters.get("max_edge_len", 30.0)
        )

        start_x, start_y = map(float, environment.start)
        goal_x, goal_y = map(float, environment.goal)
        robot_radius = max(environment.robot_radius, 0.5)
        obstacle_tree = module.KDTree(
            np.vstack(
                (
                    sampling_problem.obstacle_x,
                    sampling_problem.obstacle_y,
                )
            ).T
        )
        sample_x, sample_y = module.sample_points(
            start_x,
            start_y,
            goal_x,
            goal_y,
            robot_radius,
            sampling_problem.obstacle_x,
            sampling_problem.obstacle_y,
            obstacle_tree,
            np.random.default_rng(seed),
        )
        road_map = module.generate_road_map(
            sample_x,
            sample_y,
            robot_radius,
            obstacle_tree,
        )
        rx, ry = module.dijkstra_planning(
            start_x,
            start_y,
            goal_x,
            goal_y,
            road_map,
            sample_x,
            sample_y,
        )

        path = (
            [[float(x), float(y)] for x, y in zip(rx, ry)]
            if rx and ry
            else None
        )

        partial_path = (
            None
            if path is not None
            else self._best_partial_roadmap_path(
                road_map,
                sample_x,
                sample_y,
                environment.goal,
            )
        )
        return path, {
            "iterations": None,
            "generated_nodes": (
                int(module.N_SAMPLE) + 2
            ),
            "partial_path": partial_path,
            "algorithm_parameters": {
                "n_sample": int(module.N_SAMPLE),
                "n_knn": int(module.N_KNN),
                "max_edge_len": float(module.MAX_EDGE_LEN),
            },
        }

    @staticmethod
    def _best_partial_roadmap_path(
        road_map: list[list[int]],
        sample_x: list[float],
        sample_y: list[float],
        goal: np.ndarray,
    ) -> list[list[float]] | None:
        """Reconstruct the reachable roadmap branch closest to the goal."""
        start_index = len(road_map) - 2
        distances = {start_index: 0.0}
        parents = {start_index: -1}
        queue = [(0.0, start_index)]
        while queue:
            cost, current = heapq.heappop(queue)
            if cost != distances.get(current):
                continue
            for neighbor in road_map[current]:
                edge = float(
                    np.hypot(
                        sample_x[neighbor] - sample_x[current],
                        sample_y[neighbor] - sample_y[current],
                    )
                )
                candidate = cost + edge
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    parents[neighbor] = current
                    heapq.heappush(queue, (candidate, neighbor))
        reachable = [index for index in distances if index != start_index]
        if not reachable:
            return None
        best = min(
            reachable,
            key=lambda index: float(
                np.hypot(
                    sample_x[index] - goal[0],
                    sample_y[index] - goal[1],
                )
            ),
        )
        indices = []
        current = best
        while current != -1:
            indices.append(current)
            current = parents[current]
        indices.reverse()
        return [
            [float(sample_x[index]), float(sample_y[index])]
            for index in indices
        ]

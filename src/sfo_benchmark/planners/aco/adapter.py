"""Adapter for the archived Haghrah ACO robot path-planning implementation."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from sfo_benchmark.core.collision import CollisionChecker
from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.path import Path
from sfo_benchmark.core.result import PlanningResult
from sfo_benchmark.planners.base import BasePlanner

from .vendor_loader import (
    ARCHIVE_RELATIVE_PATH,
    load_original_aco_module,
)


GridNode = tuple[int, int]


class PublishedACOAdapter(BasePlanner):
    """Run the repository's unmodified ``ACO_PDG`` search.

    Haghrah's repository is an independent simulation of Liu et al. (2017),
    not the authors' official implementation.  The archived code is executed
    directly and retains its fixed 20x20 grid, potential-field heuristic,
    Brushfire calculation, pheromone diffusion, geometric optimizer, and
    path-length objective.
    """

    upstream_repository = "Haghrah/ACO---Robot-Path-Planning"
    upstream_archive = ARCHIVE_RELATIVE_PATH.as_posix()
    upstream_file = "PathPlanning.py"
    upstream_url = (
        "https://github.com/Haghrah/ACO---Robot-Path-Planning"
    )
    archive_sha256 = (
        "531D2D10633C042670CE765CFE8BA4F3DF87A25CD10881CACFFF9946CBBDEB61"
    )
    reference_paper = (
        "J. Liu, J. Yang, H. Liu, and X. Tian, An improved ant colony "
        "algorithm for robot path planning, Soft Computing 21 (2017), "
        "5829-5839, DOI 10.1007/s00500-016-2161-7"
    )
    native_objective_id = "haghrah_aco_path_length_v1"
    grid_size = 20

    @staticmethod
    def _world_to_grid(
        point: np.ndarray,
        environment: Environment,
    ) -> GridNode:
        """Map a workspace point to its containing source-grid cell."""
        xmin, xmax, ymin, ymax = environment.bounds
        x, y = np.asarray(point, dtype=float)
        ix = int(np.floor((x - xmin) / (xmax - xmin) * 20.0))
        iy = int(np.floor((y - ymin) / (ymax - ymin) * 20.0))
        return (
            int(np.clip(ix, 0, 19)),
            int(np.clip(iy, 0, 19)),
        )

    @staticmethod
    def _grid_to_world(
        node: GridNode,
        environment: Environment,
    ) -> np.ndarray:
        """Map a source-grid cell to its workspace centre."""
        xmin, xmax, ymin, ymax = environment.bounds
        dx = (xmax - xmin) / 20.0
        dy = (ymax - ymin) / 20.0
        return np.array(
            [
                xmin + (float(node[0]) + 0.5) * dx,
                ymin + (float(node[1]) + 0.5) * dy,
            ],
            dtype=float,
        )

    @classmethod
    def _grid_obstacles(
        cls,
        environment: Environment,
        *,
        start: GridNode,
        goal: GridNode,
    ) -> list[GridNode]:
        """Create source obstacle cells by centre-point occupancy.

        The source represents every obstacle as an occupied grid coordinate.
        The benchmark robot radius is supplied as the obstacle margin. Start
        and goal cells remain free, matching the source input contract.
        """
        occupied: list[GridNode] = []
        for ix in range(cls.grid_size):
            for iy in range(cls.grid_size):
                node = (ix, iy)
                if node == start or node == goal:
                    continue
                point = cls._grid_to_world(node, environment)
                if any(
                    obstacle.contains(point, environment.robot_radius)
                    for obstacle in environment.obstacles
                ):
                    occupied.append(node)
        return occupied

    @classmethod
    def _native_path_to_common(
        cls,
        nodes: list[GridNode],
        environment: Environment,
    ) -> Path | None:
        if not nodes:
            return None
        points = np.asarray(
            [cls._grid_to_world(node, environment) for node in nodes],
            dtype=float,
        )
        points[0] = environment.start.copy()
        points[-1] = environment.goal.copy()

        if len(points) < 2:
            return None
        keep = np.ones(len(points), dtype=bool)
        keep[1:] = np.any(np.diff(points, axis=0) != 0.0, axis=1)
        points = points[keep]
        return Path(points) if len(points) >= 2 else None

    def plan(
        self,
        environment: Environment,
        *,
        seed: int,
        run_id: int,
    ) -> PlanningResult:
        parameters = dict(self.config.parameters)

        # Defaults reproduce the repository's __main__ invocation.
        ant_count = int(parameters.get("ant_count", 20))
        step_count = int(parameters.get("step_count", 100))
        alpha = float(parameters.get("alpha", 1.1))
        beta = float(parameters.get("beta", 12.0))
        gamma = float(parameters.get("gamma", 0.02))
        evaporation_rate = float(
            parameters.get("evaporation_rate", 0.5)
        )
        brushfire_iter = int(parameters.get("brushfire_iter", 5))
        colony_iterations = int(parameters.get("colony_iterations", 20))
        pheromone_q = float(parameters.get("pheromone_q", 10.0))

        if ant_count < 1 or step_count < 1 or colony_iterations < 1:
            raise ValueError(
                "ant_count, step_count, and colony_iterations must be positive."
            )
        if brushfire_iter < 1:
            raise ValueError("brushfire_iter must be positive.")

        start_node = self._world_to_grid(environment.start, environment)
        goal_node = self._world_to_grid(environment.goal, environment)
        obstacles = self._grid_obstacles(
            environment,
            start=start_node,
            goal=goal_node,
        )

        module = load_original_aco_module()
        previous_random_state = np.random.get_state()
        native_history: list[float] = []
        final_complete_paths: list[tuple[float, list[GridNode]]] = []
        native_error: str | None = None
        started = time.perf_counter()
        try:
            np.random.seed(seed)
            planner = module.ACO_PDG(
                self.grid_size,
                self.grid_size,
                obstacles,
                ant_count,
                step_count,
                alpha,
                beta,
                gamma,
                evaporation_rate,
                np.asarray(start_node, dtype=int),
                np.asarray(goal_node, dtype=int),
                brushfire_iter,
                1,
            )

            # One native colony iteration per call preserves the pheromone grid
            # and RNG stream while exposing the otherwise internal iteration
            # history. The last call is exactly the source's returned population.
            for _ in range(colony_iterations):
                final_complete_paths = planner.run_ant_colony(pheromone_q)
                native_history.append(
                    float(final_complete_paths[0][0])
                    if final_complete_paths
                    else float("inf")
                )
        except Exception as error:  # Preserve an upstream failure as a run result.
            native_error = f"{type(error).__name__}: {error}"
        finally:
            np.random.set_state(previous_random_state)
        execution_time = time.perf_counter() - started

        native_nodes: list[GridNode] = []
        native_fitness = float("inf")
        if final_complete_paths:
            native_fitness = float(final_complete_paths[0][0])
            native_nodes = [
                (int(node[0]), int(node[1]))
                for node in final_complete_paths[0][1]
            ]

        path = self._native_path_to_common(native_nodes, environment)
        collision_free = bool(
            path is not None
            and CollisionChecker(environment).path_is_free(path)
        )
        success = bool(path is not None and collision_free)

        metadata: dict[str, Any] = {
            "family": "population_based_optimization",
            "upstream_repository": self.upstream_repository,
            "upstream_archive": self.upstream_archive,
            "upstream_file": self.upstream_file,
            "upstream_url": self.upstream_url,
            "archive_sha256": self.archive_sha256,
            "reference_paper": self.reference_paper,
            "repository_relationship": (
                "independent simulation of the paper; not official author code"
            ),
            "source_modified": False,
            "source_invoked_directly": True,
            "visualization_suppressed_only": True,
            "representation": "native_haghrah_20x20_grid_chain",
            "fixed_waypoint_encoding": False,
            "native_objective_id": self.native_objective_id,
            "native_objective": "Euclidean grid-chain path length",
            "objective_usage": "native_internal_search",
            "fitness_history": native_history,
            "fitness_history_objective_id": self.native_objective_id,
            "fitness_history_cross_algorithm_comparable": False,
            "native_best_fitness": native_fitness,
            "grid_size": self.grid_size,
            "map_conversion": "workspace_to_native_20x20_cell_centres",
            "occupied_grid_cells": len(obstacles),
            "robot_radius_applied_during_grid_occupancy": True,
            "ant_count": ant_count,
            "step_count": step_count,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "evaporation_rate": evaporation_rate,
            "brushfire_iter": brushfire_iter,
            "colony_iterations": colony_iterations,
            "pheromone_q": pheromone_q,
            "native_complete_paths_in_final_iteration": len(
                final_complete_paths
            ),
            "native_error": native_error,
            "path_collision_free_under_common_validator": collision_free,
            "search_iterations": colony_iterations,
            "generated_nodes": ant_count * step_count * colony_iterations,
            "adapter_geometry_cleanup": (
                "endpoint coordinate restoration and consecutive duplicate "
                "removal only"
            ),
        }

        if native_error is not None:
            message = (
                "Archived ACO_PDG raised an upstream runtime error: "
                f"{native_error}"
            )
        elif success:
            message = (
                "Unmodified archived ACO_PDG produced a final-iteration path "
                "that passed common validation."
            )
        elif path is None:
            message = (
                "Archived ACO_PDG returned no complete path in its final "
                "colony iteration."
            )
        else:
            message = (
                "Archived ACO_PDG returned a path, but the mapped path failed "
                "common collision validation."
            )

        return PlanningResult(
            algorithm=self.name,
            map_id=environment.map_id,
            run_id=run_id,
            seed=seed,
            success=success,
            execution_time=execution_time,
            path=path,
            iterations=colony_iterations,
            fitness_evaluations=ant_count * colony_iterations,
            message=message,
            metadata=metadata,
        )

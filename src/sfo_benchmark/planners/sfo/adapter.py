"""Benchmark adapter for the frozen Three-Phase Incremental SFO."""

from __future__ import annotations

from dataclasses import asdict
import math
import time
from typing import Any

import numpy as np

from sfo_benchmark.core.collision import CollisionChecker
from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.path import Path
from sfo_benchmark.core.result import PlanningResult
from sfo_benchmark.planners.base import BasePlanner

from .three_phase_engine import (
    ThreePhaseSFOEngine,
    config_from_parameters,
)


class ThreePhaseSFOAdapter(BasePlanner):
    """Expose the submitted final SFO through the common planner API."""

    upstream_algorithm = "Three-Phase Incremental SFO"
    reference_file = (
        "third_party/sfo_reference/"
        "three_phase_sfo_reference.py"
    )

    def plan(
        self,
        environment: Environment,
        *,
        seed: int,
        run_id: int,
    ) -> PlanningResult:
        config = config_from_parameters(self.config.parameters)
        started = time.perf_counter()

        try:
            engine = ThreePhaseSFOEngine(
                environment,
                config,
                seed=seed,
            )
            output = engine.run()
            execution_time = time.perf_counter() - started

            points = np.asarray(output["final_path"], dtype=float)
            path = Path(points)
            collision_free = CollisionChecker(
                environment,
                resolution=config.collision_resolution,
            ).path_is_free(path)
            reaches_goal = bool(
                np.allclose(points[0], environment.start)
                and np.allclose(points[-1], environment.goal)
            )
            success = bool(
                output["completed"]
                and collision_free
                and reaches_goal
            )

            statistics = asdict(output["statistics"])
            finite_history = [
                float(value)
                for value in output["best_length_history"]
                if math.isfinite(float(value))
            ]
            completed_count = sum(
                individual.completed
                for individual in output["swarm"]
            )

            metadata: dict[str, Any] = {
                "family": "proposed_incremental_metaheuristic",
                "upstream_algorithm": self.upstream_algorithm,
                "reference_file": self.reference_file,
                "source_equations_modified": False,
                "structural_refactoring_only": True,
                "variable_waypoint_count": True,
                "relative_arc_length_matching": True,
                "fitness_history": finite_history,
                "mode_history": output["mode_history"],
                "completed_count_history": (
                    output["completed_count_history"]
                ),
                "completed_paths": completed_count,
                "population_size": config.population_size,
                "max_iterations": config.max_iterations,
                "final_length": float(output["final_length"]),
                "final_waypoint_count": len(points),
                "final_segment_count": len(points) - 1,
                "phase_iterations": statistics["phase_iterations"],
                "phase_transitions": statistics[
                    "phase_transitions"
                ],
                "candidate_attempts": statistics[
                    "candidate_attempts"
                ],
                "generated_segments": statistics[
                    "generated_segments"
                ],
                "accepted_glide_segments": statistics[
                    "accepted_glide_segments"
                ],
                "accepted_target_moves": statistics[
                    "accepted_target_moves"
                ],
                "accepted_micro_moves": statistics[
                    "accepted_micro_moves"
                ],
                "rejected_outside_map": statistics[
                    "rejected_outside_map"
                ],
                "rejected_collision": statistics[
                    "rejected_collision"
                ],
                "rejected_non_improving": statistics[
                    "rejected_non_improving"
                ],
                "failed_glide_extensions": statistics[
                    "failed_glide_extensions"
                ],
                "cloned_failed_individuals": statistics[
                    "cloned_failed_individuals"
                ],
                "direct_goal_connections": statistics[
                    "direct_goal_connections"
                ],
                "pruned_waypoints": statistics[
                    "pruned_waypoints"
                ],
                "collision_checks": statistics[
                    "collision_checks"
                ],
                "parameters": asdict(config),
            }

            return PlanningResult(
                algorithm=self.name,
                map_id=environment.map_id,
                run_id=run_id,
                seed=seed,
                success=success,
                execution_time=execution_time,
                path=path,
                iterations=statistics["iterations_executed"],
                fitness_evaluations=statistics[
                    "candidate_attempts"
                ],
                message=(
                    "Three-Phase Incremental SFO generated a "
                    "collision-free completed path."
                    if success
                    else (
                        "SFO stopped without a validated completed "
                        "path to the goal."
                    )
                ),
                metadata=metadata,
            )

        except (RuntimeError, ValueError, IndexError) as error:
            return PlanningResult(
                algorithm=self.name,
                map_id=environment.map_id,
                run_id=run_id,
                seed=seed,
                success=False,
                execution_time=time.perf_counter() - started,
                message=f"Three-Phase SFO failed: {error}",
                metadata={
                    "family": (
                        "proposed_incremental_metaheuristic"
                    ),
                    "upstream_algorithm": self.upstream_algorithm,
                    "reference_file": self.reference_file,
                    "error_type": type(error).__name__,
                },
            )

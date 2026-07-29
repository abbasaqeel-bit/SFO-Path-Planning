"""Frozen Three-Phase Incremental SFO path-planning engine.

This module is a structural refactoring of the submitted final SFO script.
The mathematical update rules, acceptance rules, and phase-switching logic
are preserved. Environment, random seed, collision checking, and logging are
injected by the benchmark framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from sfo_benchmark.core.collision import CollisionChecker
from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.path import Path


@dataclass(frozen=True, slots=True)
class ThreePhaseSFOConfig:
    population_size: int = 30
    max_iterations: int = 250
    max_candidate_attempts: int = 100

    initial_step_min: float = 8.0
    initial_step_max: float = 16.0

    glide_v_min: float = 5.0
    glide_v_max: float = 18.0

    target_v_min: float = 0.5
    target_v_max: float = 8.0

    micro_v_min: float = 0.1
    micro_v_max: float = 2.5

    goal_connection_distance: float = 18.0

    w_glide: float = 0.35
    alpha_glide: float = 6.0
    beta_glide: float = 3.0

    w_target: float = 0.20
    c1_target: float = 0.80
    c2_target: float = 1.60
    noise_target: float = 2.00

    w_micro: float = 0.10
    c2_micro: float = 1.20
    epsilon_micro: float = 0.50

    levy_beta: float = 1.5

    min_completed_for_target: int = 5
    max_glide_iterations: int = 120
    target_improvement_window: int = 30
    target_min_relative_improvement: float = 0.05
    force_micro_last_iterations: int = 30
    micro_stagnation_limit: int = 15
    micro_attempts_per_waypoint: int = 10
    micro_prune_before_move: bool = True
    micro_prune_after_move: bool = True

    collision_resolution: float = 0.25

    def __post_init__(self) -> None:
        if self.population_size <= 0:
            raise ValueError("population_size must be positive.")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if self.max_candidate_attempts <= 0:
            raise ValueError("max_candidate_attempts must be positive.")
        if self.target_improvement_window <= 0:
            raise ValueError(
                "target_improvement_window must be positive."
            )
        if self.micro_stagnation_limit <= 0:
            raise ValueError(
                "micro_stagnation_limit must be positive."
            )


@dataclass(slots=True)
class Individual:
    path: list[np.ndarray] = field(default_factory=list)
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=float)
    )
    completed_pbest_path: list[np.ndarray] | None = None
    completed_pbest_length: float = math.inf
    travelled_length: float = 0.0
    remaining_distance: float = math.inf
    partial_fitness: float = math.inf
    completed: bool = False
    frozen: bool = False
    failed: bool = False


@dataclass(slots=True)
class SFOStatistics:
    iterations_executed: int = 0
    phase_iterations: dict[str, int] = field(
        default_factory=lambda: {
            "glide": 0,
            "target": 0,
            "micro": 0,
        }
    )
    candidate_attempts: int = 0
    generated_segments: int = 0
    accepted_glide_segments: int = 0
    accepted_target_moves: int = 0
    accepted_micro_moves: int = 0
    rejected_outside_map: int = 0
    rejected_collision: int = 0
    rejected_non_improving: int = 0
    failed_glide_extensions: int = 0
    cloned_failed_individuals: int = 0
    direct_goal_connections: int = 0
    pruned_waypoints: int = 0
    collision_checks: int = 0
    phase_transitions: list[dict[str, Any]] = field(
        default_factory=list
    )


class ThreePhaseSFOEngine:
    """Execute the frozen three-phase SFO on one benchmark environment."""

    def __init__(
        self,
        environment: Environment,
        config: ThreePhaseSFOConfig,
        *,
        seed: int,
    ) -> None:
        self.environment = environment
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.collision_checker = CollisionChecker(
            environment,
            resolution=config.collision_resolution,
        )
        self.stats = SFOStatistics()

    # ------------------------------------------------------------
    # Geometry and path utilities
    # ------------------------------------------------------------
    @staticmethod
    def copy_path(path: list[np.ndarray]) -> list[np.ndarray]:
        return [point.copy() for point in path]

    @staticmethod
    def path_length(path: list[np.ndarray]) -> float:
        if len(path) < 2:
            return 0.0
        points = np.asarray(path, dtype=float)
        return float(
            np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        )

    def point_inside_map(self, point: np.ndarray) -> bool:
        return self.environment.is_inside_bounds(point)

    def segment_is_collision_free(
        self,
        start: np.ndarray,
        end: np.ndarray,
    ) -> bool:
        self.stats.collision_checks += 1
        return self.collision_checker.segment_is_free(start, end)

    def evaluate_partial_path(
        self,
        path: list[np.ndarray],
    ) -> tuple[float, float, float]:
        travelled = self.path_length(path)
        remaining = float(
            np.linalg.norm(self.environment.goal - path[-1])
        )
        if np.allclose(path[-1], self.environment.goal):
            remaining = 0.0
        return travelled + remaining, travelled, remaining

    @staticmethod
    def cumulative_lengths(path: list[np.ndarray]) -> np.ndarray:
        points = np.asarray(path, dtype=float)
        if len(points) == 1:
            return np.array([0.0])
        segment_lengths = np.linalg.norm(
            np.diff(points, axis=0),
            axis=1,
        )
        return np.concatenate(([0.0], np.cumsum(segment_lengths)))

    def relative_arc_position(
        self,
        path: list[np.ndarray],
        point_index: int,
    ) -> float:
        cumulative = self.cumulative_lengths(path)
        total = cumulative[-1]
        if total <= 1e-12:
            return 0.0
        return float(cumulative[point_index] / total)

    def point_at_relative_arc(
        self,
        path: list[np.ndarray],
        ratio: float,
    ) -> np.ndarray:
        ratio = float(np.clip(ratio, 0.0, 1.0))
        points = np.asarray(path, dtype=float)
        cumulative = self.cumulative_lengths(path)
        total = cumulative[-1]
        if total <= 1e-12:
            return points[0].copy()

        target_distance = ratio * total
        segment_index = (
            np.searchsorted(
                cumulative,
                target_distance,
                side="right",
            )
            - 1
        )
        segment_index = int(
            np.clip(segment_index, 0, len(points) - 2)
        )
        start_distance = cumulative[segment_index]
        end_distance = cumulative[segment_index + 1]
        denominator = end_distance - start_distance
        if denominator <= 1e-12:
            return points[segment_index].copy()

        local_ratio = (
            target_distance - start_distance
        ) / denominator
        return (
            points[segment_index]
            + local_ratio
            * (points[segment_index + 1] - points[segment_index])
        )

    def prune_redundant_waypoints(
        self,
        path: list[np.ndarray],
    ) -> tuple[list[np.ndarray], int]:
        if len(path) <= 2:
            return self.copy_path(path), 0

        pruned = self.copy_path(path)
        removed = 0
        index = 1

        while index < len(pruned) - 1:
            if self.segment_is_collision_free(
                pruned[index - 1],
                pruned[index + 1],
            ):
                del pruned[index]
                removed += 1
                if index > 1:
                    index -= 1
            else:
                index += 1

        return pruned, removed

    def apply_pruning_to_individual(
        self,
        individual: Individual,
    ) -> bool:
        if not individual.completed:
            return False

        pruned_path, removed = self.prune_redundant_waypoints(
            individual.path
        )
        if removed == 0:
            return False

        pruned_length = self.path_length(pruned_path)
        if pruned_length >= individual.travelled_length - 1e-9:
            return False

        individual.path = pruned_path
        individual.travelled_length = pruned_length
        individual.partial_fitness = pruned_length
        individual.remaining_distance = 0.0
        self.stats.pruned_waypoints += removed

        if pruned_length < individual.completed_pbest_length - 1e-9:
            individual.completed_pbest_path = self.copy_path(
                pruned_path
            )
            individual.completed_pbest_length = pruned_length

        return True

    # ------------------------------------------------------------
    # Random and SFO utilities
    # ------------------------------------------------------------
    def random_unit_vector(self) -> np.ndarray:
        vector = self.rng.normal(size=2)
        magnitude = float(np.linalg.norm(vector))
        if magnitude < 1e-12:
            return np.array([1.0, 0.0], dtype=float)
        return vector / magnitude

    def vector_with_random_magnitude(
        self,
        minimum: float,
        maximum: float,
    ) -> np.ndarray:
        return (
            self.random_unit_vector()
            * self.rng.uniform(minimum, maximum)
        )

    def limit_vector_magnitude(
        self,
        vector: np.ndarray,
        minimum: float,
        maximum: float,
    ) -> np.ndarray:
        magnitude = float(np.linalg.norm(vector))
        if magnitude < 1e-12:
            return self.vector_with_random_magnitude(
                minimum,
                maximum,
            )
        if magnitude > maximum:
            return vector * (maximum / magnitude)
        if magnitude < minimum:
            return vector * (minimum / magnitude)
        return vector

    def levy_flight_2d(self) -> np.ndarray:
        beta = self.config.levy_beta
        numerator = (
            math.gamma(1.0 + beta)
            * math.sin(math.pi * beta / 2.0)
        )
        denominator = (
            math.gamma((1.0 + beta) / 2.0)
            * beta
            * 2.0 ** ((beta - 1.0) / 2.0)
        )
        sigma_u = (numerator / denominator) ** (1.0 / beta)
        u = self.rng.normal(0.0, sigma_u, size=2)
        v = self.rng.normal(0.0, 1.0, size=2)
        step = u / (np.abs(v) ** (1.0 / beta) + 1e-12)
        return np.clip(step, -3.0, 3.0)

    def generate_glide_velocity(
        self,
        current_velocity: np.ndarray,
    ) -> np.ndarray:
        gaussian_component = self.rng.normal(0.0, 1.0, size=2)
        levy_component = self.levy_flight_2d()
        velocity = (
            self.config.w_glide * current_velocity
            + self.config.alpha_glide * gaussian_component
            + self.config.beta_glide * levy_component
        )
        return self.limit_vector_magnitude(
            velocity,
            self.config.glide_v_min,
            self.config.glide_v_max,
        )

    def generate_target_displacement(
        self,
        current_point: np.ndarray,
        current_velocity: np.ndarray,
        pbest_reference: np.ndarray,
        gbest_reference: np.ndarray,
    ) -> np.ndarray:
        r1 = self.rng.random(2)
        r2 = self.rng.random(2)
        noise = self.rng.normal(
            0.0,
            self.config.noise_target,
            size=2,
        )
        displacement = (
            self.config.w_target * current_velocity
            + self.config.c1_target
            * r1
            * (pbest_reference - current_point)
            + self.config.c2_target
            * r2
            * (gbest_reference - current_point)
            + noise
        )
        return self.limit_vector_magnitude(
            displacement,
            self.config.target_v_min,
            self.config.target_v_max,
        )

    def generate_micro_displacement(
        self,
        current_point: np.ndarray,
        current_velocity: np.ndarray,
        gbest_reference: np.ndarray,
    ) -> np.ndarray:
        r2 = self.rng.random(2)
        epsilon = self.rng.normal(
            0.0,
            self.config.epsilon_micro,
            size=2,
        )
        displacement = (
            self.config.w_micro * current_velocity
            + self.config.c2_micro
            * r2
            * (gbest_reference - current_point)
            + epsilon
        )
        return self.limit_vector_magnitude(
            displacement,
            self.config.micro_v_min,
            self.config.micro_v_max,
        )

    # ------------------------------------------------------------
    # Initialization and Glide
    # ------------------------------------------------------------
    def generate_initial_valid_segment(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(self.config.max_candidate_attempts):
            self.stats.candidate_attempts += 1
            velocity = self.vector_with_random_magnitude(
                self.config.initial_step_min,
                self.config.initial_step_max,
            )
            candidate = self.environment.start + velocity
            if not self.point_inside_map(candidate):
                self.stats.rejected_outside_map += 1
                continue
            if not self.segment_is_collision_free(
                self.environment.start,
                candidate,
            ):
                self.stats.rejected_collision += 1
                continue
            self.stats.generated_segments += 1
            return candidate, velocity
        raise RuntimeError(
            "Could not generate a valid initial segment."
        )

    def initialize_swarm(self) -> list[Individual]:
        swarm: list[Individual] = []
        for _ in range(self.config.population_size):
            first_point, initial_velocity = (
                self.generate_initial_valid_segment()
            )
            path = [
                self.environment.start.copy(),
                first_point.copy(),
            ]
            partial, travelled, remaining = (
                self.evaluate_partial_path(path)
            )
            swarm.append(
                Individual(
                    path=path,
                    velocity=initial_velocity.copy(),
                    travelled_length=travelled,
                    remaining_distance=remaining,
                    partial_fitness=partial,
                )
            )
        return swarm

    @staticmethod
    def current_partial_gbest_index(
        swarm: list[Individual],
    ) -> int | None:
        active_indices = [
            index
            for index, individual in enumerate(swarm)
            if not individual.frozen
        ]
        if not active_indices:
            return None
        return min(
            active_indices,
            key=lambda index: (
                swarm[index].partial_fitness,
                swarm[index].remaining_distance,
                swarm[index].travelled_length,
            ),
        )

    def try_direct_goal_connection(
        self,
        individual: Individual,
    ) -> bool:
        current = individual.path[-1]
        distance_to_goal = float(
            np.linalg.norm(self.environment.goal - current)
        )
        if distance_to_goal > self.config.goal_connection_distance:
            return False
        if not self.segment_is_collision_free(
            current,
            self.environment.goal,
        ):
            self.stats.rejected_collision += 1
            return False

        individual.path.append(self.environment.goal.copy())
        individual.velocity = self.environment.goal - current
        individual.completed = True
        individual.failed = False
        (
            individual.partial_fitness,
            individual.travelled_length,
            individual.remaining_distance,
        ) = self.evaluate_partial_path(individual.path)
        individual.completed_pbest_path = self.copy_path(
            individual.path
        )
        individual.completed_pbest_length = (
            individual.travelled_length
        )
        self.stats.generated_segments += 1
        self.stats.direct_goal_connections += 1
        return True

    def extend_glide_individual(
        self,
        individual: Individual,
    ) -> bool:
        if individual.completed or individual.frozen:
            return True
        if self.try_direct_goal_connection(individual):
            return True

        current_point = individual.path[-1].copy()
        base_velocity = individual.velocity.copy()

        for _ in range(self.config.max_candidate_attempts):
            self.stats.candidate_attempts += 1
            candidate_velocity = self.generate_glide_velocity(
                base_velocity
            )
            candidate_point = current_point + candidate_velocity

            if not self.point_inside_map(candidate_point):
                self.stats.rejected_outside_map += 1
                continue
            if not self.segment_is_collision_free(
                current_point,
                candidate_point,
            ):
                self.stats.rejected_collision += 1
                continue

            individual.path.append(candidate_point.copy())
            individual.velocity = candidate_velocity.copy()
            individual.failed = False
            (
                individual.partial_fitness,
                individual.travelled_length,
                individual.remaining_distance,
            ) = self.evaluate_partial_path(individual.path)
            self.stats.generated_segments += 1
            self.stats.accepted_glide_segments += 1
            self.try_direct_goal_connection(individual)
            return True

        individual.failed = True
        self.stats.failed_glide_extensions += 1
        return False

    @staticmethod
    def nearest_successful_index(
        failed_individual: Individual,
        swarm: list[Individual],
        successful_indices: list[int],
    ) -> int:
        failed_endpoint = failed_individual.path[-1]
        return min(
            successful_indices,
            key=lambda index: np.linalg.norm(
                swarm[index].path[-1] - failed_endpoint
            ),
        )

    def replace_failed_glide_individuals(
        self,
        swarm: list[Individual],
        successful_indices: list[int],
    ) -> None:
        if not successful_indices:
            return

        for individual in swarm:
            if not individual.failed:
                continue
            source_index = self.nearest_successful_index(
                individual,
                swarm,
                successful_indices,
            )
            source = swarm[source_index]
            individual.path = self.copy_path(source.path)
            individual.velocity = self.vector_with_random_magnitude(
                self.config.glide_v_min,
                self.config.glide_v_max,
            )
            individual.completed = source.completed
            individual.failed = False
            (
                individual.partial_fitness,
                individual.travelled_length,
                individual.remaining_distance,
            ) = self.evaluate_partial_path(individual.path)

            if individual.completed:
                individual.completed_pbest_path = self.copy_path(
                    individual.path
                )
                individual.completed_pbest_length = (
                    individual.travelled_length
                )
            self.stats.cloned_failed_individuals += 1

    # ------------------------------------------------------------
    # Completed-path optimization
    # ------------------------------------------------------------
    @staticmethod
    def completed_indices(
        swarm: list[Individual],
    ) -> list[int]:
        return [
            index
            for index, individual in enumerate(swarm)
            if individual.completed
        ]

    def global_best_completed_path(
        self,
        swarm: list[Individual],
    ) -> tuple[list[np.ndarray] | None, float, int | None]:
        indices = self.completed_indices(swarm)
        if not indices:
            return None, math.inf, None
        best_index = min(
            indices,
            key=lambda index: swarm[index].travelled_length,
        )
        return (
            self.copy_path(swarm[best_index].path),
            swarm[best_index].travelled_length,
            best_index,
        )

    def completed_pbest_reference(
        self,
        individual: Individual,
        current_path: list[np.ndarray],
        waypoint_index: int,
    ) -> np.ndarray:
        ratio = self.relative_arc_position(
            current_path,
            waypoint_index,
        )
        if individual.completed_pbest_path is None:
            return current_path[waypoint_index].copy()
        return self.point_at_relative_arc(
            individual.completed_pbest_path,
            ratio,
        )

    def optimize_completed_path(
        self,
        individual: Individual,
        gbest_path: list[np.ndarray],
        mode: str,
    ) -> bool:
        if not individual.completed or len(individual.path) <= 2:
            return False

        improved = False
        waypoint_indices = np.arange(
            1,
            len(individual.path) - 1,
        )
        self.rng.shuffle(waypoint_indices)

        for waypoint_index in waypoint_indices:
            if waypoint_index >= len(individual.path) - 1:
                continue

            current_path = self.copy_path(individual.path)
            previous_point = current_path[waypoint_index - 1]
            current_point = current_path[waypoint_index]
            next_point = current_path[waypoint_index + 1]

            ratio = self.relative_arc_position(
                current_path,
                waypoint_index,
            )
            gbest_reference = self.point_at_relative_arc(
                gbest_path,
                ratio,
            )
            pbest_reference = self.completed_pbest_reference(
                individual,
                current_path,
                waypoint_index,
            )

            attempts = (
                self.config.micro_attempts_per_waypoint
                if mode == "micro"
                else 1
            )
            best_candidate_path = None
            best_candidate_length = individual.travelled_length
            best_displacement = None

            for _ in range(attempts):
                self.stats.candidate_attempts += 1

                if mode == "target":
                    displacement = self.generate_target_displacement(
                        current_point,
                        individual.velocity,
                        pbest_reference,
                        gbest_reference,
                    )
                elif mode == "micro":
                    displacement = self.generate_micro_displacement(
                        current_point,
                        individual.velocity,
                        gbest_reference,
                    )
                else:
                    raise ValueError(
                        "Completed-path optimization supports "
                        "Target and Micro only."
                    )

                candidate_point = current_point + displacement
                if not self.point_inside_map(candidate_point):
                    self.stats.rejected_outside_map += 1
                    continue
                if not self.segment_is_collision_free(
                    previous_point,
                    candidate_point,
                ):
                    self.stats.rejected_collision += 1
                    continue
                if not self.segment_is_collision_free(
                    candidate_point,
                    next_point,
                ):
                    self.stats.rejected_collision += 1
                    continue

                candidate_path = self.copy_path(current_path)
                candidate_path[waypoint_index] = (
                    candidate_point.copy()
                )
                candidate_length = self.path_length(candidate_path)

                if (
                    candidate_length
                    < best_candidate_length - 1e-9
                ):
                    best_candidate_path = candidate_path
                    best_candidate_length = candidate_length
                    best_displacement = displacement.copy()
                else:
                    self.stats.rejected_non_improving += 1

            if best_candidate_path is not None:
                individual.path = best_candidate_path
                individual.velocity = best_displacement
                individual.travelled_length = best_candidate_length
                individual.partial_fitness = best_candidate_length
                individual.remaining_distance = 0.0
                improved = True
                if mode == "target":
                    self.stats.accepted_target_moves += 1
                else:
                    self.stats.accepted_micro_moves += 1

        if (
            individual.travelled_length
            < individual.completed_pbest_length - 1e-9
        ):
            individual.completed_pbest_path = self.copy_path(
                individual.path
            )
            individual.completed_pbest_length = (
                individual.travelled_length
            )
        return improved

    # ------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------
    def _record_transition(
        self,
        *,
        iteration: int,
        source: str,
        target: str,
        reason: str,
    ) -> None:
        self.stats.phase_transitions.append(
            {
                "iteration": iteration,
                "from": source,
                "to": target,
                "reason": reason,
            }
        )

    def run(self) -> dict[str, Any]:
        swarm = self.initialize_swarm()
        mode = "glide"

        target_stagnation = 0
        target_best_history: list[float] = []
        micro_stagnation = 0

        best_completed_path = None
        best_completed_length = math.inf

        mode_history: list[str] = []
        best_length_history: list[float] = []
        completed_count_history: list[int] = []

        partial_best_index = self.current_partial_gbest_index(swarm)
        if partial_best_index is None:
            raise RuntimeError("No active individual after initialization.")
        current_gbest_path = self.copy_path(
            swarm[partial_best_index].path
        )

        for iteration in range(self.config.max_iterations):
            self.stats.iterations_executed = iteration + 1
            self.stats.phase_iterations[mode] += 1

            if mode == "glide":
                successful_indices: list[int] = []
                for index, individual in enumerate(swarm):
                    if self.extend_glide_individual(individual):
                        successful_indices.append(index)

                self.replace_failed_glide_individuals(
                    swarm,
                    successful_indices,
                )

                completed_count = sum(
                    individual.completed for individual in swarm
                )
                completed_path, completed_length, _ = (
                    self.global_best_completed_path(swarm)
                )

                if (
                    completed_path is not None
                    and completed_length < best_completed_length
                ):
                    best_completed_path = self.copy_path(
                        completed_path
                    )
                    best_completed_length = completed_length

                partial_best_index = self.current_partial_gbest_index(
                    swarm
                )
                if partial_best_index is not None:
                    current_gbest_path = self.copy_path(
                        swarm[partial_best_index].path
                    )

                enough_completed_paths = (
                    completed_count
                    >= self.config.min_completed_for_target
                )
                glide_time_finished = (
                    iteration + 1
                    >= self.config.max_glide_iterations
                )
                can_start_target = (
                    enough_completed_paths
                    or (
                        glide_time_finished
                        and completed_count >= 1
                    )
                )

                if can_start_target:
                    reason = (
                        f"{completed_count} completed paths available"
                        if enough_completed_paths
                        else (
                            "maximum Glide duration "
                            f"({self.config.max_glide_iterations} "
                            "iterations) reached"
                        )
                    )
                    self._record_transition(
                        iteration=iteration + 1,
                        source="glide",
                        target="target",
                        reason=reason,
                    )
                    mode = "target"
                    for individual in swarm:
                        if not individual.completed:
                            individual.frozen = True

                    (
                        current_gbest_path,
                        best_completed_length,
                        _,
                    ) = self.global_best_completed_path(swarm)
                    if current_gbest_path is None:
                        raise RuntimeError(
                            "Target started without a completed path."
                        )
                    best_completed_path = self.copy_path(
                        current_gbest_path
                    )
                    target_stagnation = 0
                    target_best_history = [best_completed_length]

            elif mode == "target":
                (
                    current_gbest_path,
                    _,
                    _,
                ) = self.global_best_completed_path(swarm)
                if current_gbest_path is None:
                    raise RuntimeError(
                        "No completed path available in Target."
                    )

                for individual in swarm:
                    if individual.completed:
                        self.optimize_completed_path(
                            individual,
                            current_gbest_path,
                            mode="target",
                        )

                (
                    current_gbest_path,
                    current_best_length,
                    _,
                ) = self.global_best_completed_path(swarm)

                if current_best_length < best_completed_length - 1e-9:
                    best_completed_length = current_best_length
                    best_completed_path = self.copy_path(
                        current_gbest_path
                    )
                    target_stagnation = 0
                else:
                    target_stagnation += 1

                target_best_history.append(best_completed_length)

                remaining_iterations = (
                    self.config.max_iterations - iteration - 1
                )
                force_micro_by_time = (
                    remaining_iterations
                    <= self.config.force_micro_last_iterations
                )
                insufficient_target_improvement = False
                target_relative_improvement = math.inf

                if (
                    len(target_best_history)
                    >= self.config.target_improvement_window + 1
                ):
                    old_window_best = target_best_history[
                        -self.config.target_improvement_window - 1
                    ]
                    current_window_best = target_best_history[-1]
                    target_relative_improvement = (
                        old_window_best - current_window_best
                    ) / max(abs(old_window_best), 1e-12)
                    insufficient_target_improvement = (
                        target_relative_improvement
                        < self.config.target_min_relative_improvement
                    )

                if (
                    insufficient_target_improvement
                    or force_micro_by_time
                ):
                    reason = (
                        "Target improvement below "
                        f"{100 * self.config.target_min_relative_improvement:.1f}% "
                        f"during {self.config.target_improvement_window} cycles"
                        if insufficient_target_improvement
                        else (
                            "last "
                            f"{self.config.force_micro_last_iterations} "
                            "iterations reached"
                        )
                    )
                    self._record_transition(
                        iteration=iteration + 1,
                        source="target",
                        target="micro",
                        reason=reason,
                    )
                    mode = "micro"
                    micro_stagnation = 0

            elif mode == "micro":
                (
                    current_gbest_path,
                    _,
                    _,
                ) = self.global_best_completed_path(swarm)
                if current_gbest_path is None:
                    raise RuntimeError(
                        "No completed path available in Micro."
                    )

                for individual in swarm:
                    if not individual.completed:
                        continue
                    if self.config.micro_prune_before_move:
                        self.apply_pruning_to_individual(individual)
                    self.optimize_completed_path(
                        individual,
                        current_gbest_path,
                        mode="micro",
                    )
                    if self.config.micro_prune_after_move:
                        self.apply_pruning_to_individual(individual)

                (
                    current_gbest_path,
                    current_best_length,
                    _,
                ) = self.global_best_completed_path(swarm)

                if current_best_length < best_completed_length - 1e-9:
                    best_completed_length = current_best_length
                    best_completed_path = self.copy_path(
                        current_gbest_path
                    )
                    micro_stagnation = 0
                else:
                    micro_stagnation += 1

                if (
                    micro_stagnation
                    >= self.config.micro_stagnation_limit
                ):
                    mode_history.append(mode)
                    best_length_history.append(
                        best_completed_length
                    )
                    completed_count_history.append(
                        sum(ind.completed for ind in swarm)
                    )
                    break
            else:
                raise ValueError(f"Unknown mode: {mode}")

            mode_history.append(mode)
            best_length_history.append(
                best_completed_length
                if math.isfinite(best_completed_length)
                else math.nan
            )
            completed_count_history.append(
                sum(ind.completed for ind in swarm)
            )

        if best_completed_path is None:
            final_path = current_gbest_path
            final_completed = False
            final_length = self.path_length(current_gbest_path)
        else:
            final_path = best_completed_path
            final_completed = True
            final_length = best_completed_length

        return {
            "swarm": swarm,
            "final_path": final_path,
            "final_length": final_length,
            "completed": final_completed,
            "mode_history": mode_history,
            "best_length_history": best_length_history,
            "completed_count_history": completed_count_history,
            "statistics": self.stats,
        }


def config_from_parameters(
    parameters: dict[str, Any],
) -> ThreePhaseSFOConfig:
    """Build a typed config from benchmark planner parameters."""
    defaults = ThreePhaseSFOConfig()
    values = {
        field_name: parameters.get(
            field_name,
            getattr(defaults, field_name),
        )
        for field_name in defaults.__dataclass_fields__
    }
    return ThreePhaseSFOConfig(**values)

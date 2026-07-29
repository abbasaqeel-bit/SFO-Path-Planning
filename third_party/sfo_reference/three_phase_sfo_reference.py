
"""
Final Experiment: Correct Three-Phase Incremental SFO Path Planning
===========================================================

Phase 1 - Glide:
    - All individuals start from START with one valid random segment.
    - Unfinished individuals grow their paths incrementally, one segment per
      iteration.
    - A new segment is accepted only if it stays inside the map and does not
      intersect any obstacle.
    - The algorithm switches to Target when either enough completed paths are
      available or the maximum Glide duration is reached with at least one
      completed path. Unfinished paths are then frozen.

Phase 2 - Target:
    - Only completed paths are improved.
    - Intermediate waypoints are moved using the Target equation.
    - A waypoint move is accepted only if:
          1) both adjacent segments remain collision-free, and
          2) the complete path becomes shorter.
    - If the global best completed path does not improve for 10 iterations,
      the algorithm switches to Micro.

Phase 3 - Micro:
    - Only completed paths are refined with smaller movements.
    - The same collision-free and shortening acceptance rule is used.
    - The algorithm stops after Micro stagnation or MAX_ITERATIONS.

Important:
    - The number of waypoints in completed paths may differ.
    - Therefore, the Gbest reference for a waypoint is obtained from the same
      relative arc-length location on the global-best path.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

MAP_WIDTH = 100.0
MAP_HEIGHT = 100.0

START = np.array([5.0, 10.0], dtype=float)
GOAL = np.array([95.0, 90.0], dtype=float)

POPULATION_SIZE = 30
MAX_ITERATIONS = 250
MAX_CANDIDATE_ATTEMPTS = 100

# Initial valid random segment
INITIAL_STEP_MIN = 8.0
INITIAL_STEP_MAX = 16.0

# Glide step limits
GLIDE_V_MIN = 5.0
GLIDE_V_MAX = 18.0

# Target movement limits
TARGET_V_MIN = 0.5
TARGET_V_MAX = 8.0

# Micro movement limits
MICRO_V_MIN = 0.1
MICRO_V_MAX = 2.5

GOAL_CONNECTION_DISTANCE = 18.0

# SFO Glide parameters
W_GLIDE = 0.35

ALPHA_GLIDE = 6.0   # Gaussian exploration scaling
BETA_GLIDE = 3.0    # Lévy-flight exploration scaling

# SFO Target parameters
W_TARGET = 0.20
C1_TARGET = 0.80
C2_TARGET = 1.60
NOISE_TARGET = 2.00

# SFO Micro parameters
W_MICRO = 0.10
C2_MICRO = 1.20
EPSILON_MICRO = 0.50

LEVY_BETA = 1.5

# Mode switching
MIN_COMPLETED_FOR_TARGET = 5
MAX_GLIDE_ITERATIONS = 120
TARGET_IMPROVEMENT_WINDOW = 30
TARGET_MIN_RELATIVE_IMPROVEMENT = 0.05
FORCE_MICRO_LAST_ITERATIONS = 30
MICRO_STAGNATION_LIMIT = 15
MICRO_ATTEMPTS_PER_WAYPOINT = 10
MICRO_PRUNE_BEFORE_MOVE = True
MICRO_PRUNE_AFTER_MOVE = True

# Obstacles: (xmin, ymin, xmax, ymax)
OBSTACLES = [
    (25.0, 10.0, 40.0, 65.0),
    (52.0, 38.0, 68.0, 92.0),
    (75.0, 5.0, 87.0, 55.0),
]

# Visualization
SAVE_EVERY = 1
ANIMATION_FPS = 10
CREATE_GIF = True


# ============================================================
# 2. Data structure
# ============================================================

@dataclass
class Individual:
    path: list = field(default_factory=list)
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=float)
    )

    # Historical personal best completed path
    completed_pbest_path: list | None = None
    completed_pbest_length: float = math.inf

    travelled_length: float = 0.0
    remaining_distance: float = math.inf
    partial_fitness: float = math.inf

    completed: bool = False
    frozen: bool = False
    failed: bool = False


# ============================================================
# 3. Geometry utilities
# ============================================================

def point_inside_rectangle(point, rectangle):
    x, y = point
    xmin, ymin, xmax, ymax = rectangle
    return xmin <= x <= xmax and ymin <= y <= ymax


def point_inside_map(point):
    return (
        0.0 <= point[0] <= MAP_WIDTH
        and 0.0 <= point[1] <= MAP_HEIGHT
    )


def orientation(a, b, c):
    value = (
        (b[1] - a[1]) * (c[0] - b[0])
        - (b[0] - a[0]) * (c[1] - b[1])
    )

    if np.isclose(value, 0.0):
        return 0

    return 1 if value > 0.0 else 2


def on_segment(a, b, c):
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def segments_intersect(p1, p2, q1, q2):
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0 and on_segment(p1, q2, p2):
        return True
    if o3 == 0 and on_segment(q1, p1, q2):
        return True
    if o4 == 0 and on_segment(q1, p2, q2):
        return True

    return False


def segment_intersects_rectangle(p1, p2, rectangle):
    xmin, ymin, xmax, ymax = rectangle

    if point_inside_rectangle(p1, rectangle):
        return True

    if point_inside_rectangle(p2, rectangle):
        return True

    corners = [
        np.array([xmin, ymin], dtype=float),
        np.array([xmax, ymin], dtype=float),
        np.array([xmax, ymax], dtype=float),
        np.array([xmin, ymax], dtype=float),
    ]

    edges = [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    ]

    return any(
        segments_intersect(p1, p2, edge_start, edge_end)
        for edge_start, edge_end in edges
    )


def segment_is_collision_free(p1, p2):
    return not any(
        segment_intersects_rectangle(p1, p2, obstacle)
        for obstacle in OBSTACLES
    )


def complete_path_is_collision_free(path):
    return all(
        segment_is_collision_free(path[index], path[index + 1])
        for index in range(len(path) - 1)
    )


# ============================================================
# 4. Path utilities
# ============================================================

def copy_path(path):
    return [point.copy() for point in path]


def path_length(path):
    if len(path) < 2:
        return 0.0

    points = np.asarray(path, dtype=float)

    return float(
        np.sum(
            np.linalg.norm(
                np.diff(points, axis=0),
                axis=1,
            )
        )
    )


def evaluate_partial_path(path):
    travelled = path_length(path)
    remaining = float(np.linalg.norm(GOAL - path[-1]))

    if np.allclose(path[-1], GOAL):
        remaining = 0.0

    return travelled + remaining, travelled, remaining


def cumulative_lengths(path):
    points = np.asarray(path, dtype=float)

    if len(points) == 1:
        return np.array([0.0])

    segment_lengths = np.linalg.norm(
        np.diff(points, axis=0),
        axis=1,
    )

    return np.concatenate(
        ([0.0], np.cumsum(segment_lengths))
    )


def relative_arc_position(path, point_index):
    cumulative = cumulative_lengths(path)
    total = cumulative[-1]

    if total <= 1e-12:
        return 0.0

    return float(cumulative[point_index] / total)


def point_at_relative_arc(path, ratio):
    ratio = float(np.clip(ratio, 0.0, 1.0))

    points = np.asarray(path, dtype=float)
    cumulative = cumulative_lengths(path)
    total = cumulative[-1]

    if total <= 1e-12:
        return points[0].copy()

    target_distance = ratio * total

    segment_index = np.searchsorted(
        cumulative,
        target_distance,
        side="right",
    ) - 1

    segment_index = int(
        np.clip(
            segment_index,
            0,
            len(points) - 2,
        )
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
        * (
            points[segment_index + 1]
            - points[segment_index]
        )
    )


def prune_redundant_waypoints(path):
    """
    Remove intermediate waypoints using line-of-sight pruning.

    For three consecutive points:
        P_(j-1) -> P_j -> P_(j+1)

    P_j is deleted when P_(j-1) can connect directly to P_(j+1)
    without intersecting an obstacle.

    The operation is repeated because deleting one point can make another
    neighboring point redundant.
    """
    if len(path) <= 2:
        return copy_path(path), False

    pruned_path = copy_path(path)
    changed = False
    index = 1

    while index < len(pruned_path) - 1:
        previous_point = pruned_path[index - 1]
        next_point = pruned_path[index + 1]

        if segment_is_collision_free(
            previous_point,
            next_point,
        ):
            del pruned_path[index]
            changed = True

            if index > 1:
                index -= 1
        else:
            index += 1

    return pruned_path, changed


def apply_pruning_to_individual(individual):
    """
    Apply line-of-sight pruning to one completed individual and update all
    path measurements and personal-best memory when an improvement occurs.
    """
    if not individual.completed:
        return False

    pruned_path, changed = prune_redundant_waypoints(
        individual.path
    )

    if not changed:
        return False

    pruned_length = path_length(pruned_path)

    if pruned_length >= individual.travelled_length - 1e-9:
        return False

    individual.path = pruned_path
    individual.travelled_length = pruned_length
    individual.partial_fitness = pruned_length
    individual.remaining_distance = 0.0

    if (
        pruned_length
        < individual.completed_pbest_length
        - 1e-9
    ):
        individual.completed_pbest_path = copy_path(
            pruned_path
        )
        individual.completed_pbest_length = pruned_length

    return True


# ============================================================
# 5. Random and SFO utilities
# ============================================================

def random_unit_vector():
    vector = rng.normal(size=2)
    magnitude = float(np.linalg.norm(vector))

    if magnitude < 1e-12:
        return np.array([1.0, 0.0], dtype=float)

    return vector / magnitude


def vector_with_random_magnitude(minimum, maximum):
    return (
        random_unit_vector()
        * rng.uniform(minimum, maximum)
    )


def limit_vector_magnitude(vector, minimum, maximum):
    magnitude = float(np.linalg.norm(vector))

    if magnitude < 1e-12:
        return vector_with_random_magnitude(
            minimum,
            maximum,
        )

    if magnitude > maximum:
        return vector * (maximum / magnitude)

    if magnitude < minimum:
        return vector * (minimum / magnitude)

    return vector


def levy_flight_2d(beta=LEVY_BETA):
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

    u = rng.normal(0.0, sigma_u, size=2)
    v = rng.normal(0.0, 1.0, size=2)

    step = u / (
        np.abs(v) ** (1.0 / beta)
        + 1e-12
    )

    return np.clip(step, -3.0, 3.0)


def generate_glide_velocity(current_velocity):
    gaussian_component = rng.normal(
        0.0,
        1.0,
        size=2,
    )

    levy_component = levy_flight_2d()

    velocity = (
        W_GLIDE * current_velocity
        + ALPHA_GLIDE * gaussian_component
        + BETA_GLIDE * levy_component
    )

    return limit_vector_magnitude(
        velocity,
        GLIDE_V_MIN,
        GLIDE_V_MAX,
    )


def generate_target_displacement(
    current_point,
    current_velocity,
    pbest_reference,
    gbest_reference,
):
    r1 = rng.random(2)
    r2 = rng.random(2)

    noise = rng.normal(
        0.0,
        NOISE_TARGET,
        size=2,
    )

    displacement = (
        W_TARGET * current_velocity
        + C1_TARGET
        * r1
        * (pbest_reference - current_point)
        + C2_TARGET
        * r2
        * (gbest_reference - current_point)
        + noise
    )

    return limit_vector_magnitude(
        displacement,
        TARGET_V_MIN,
        TARGET_V_MAX,
    )


def generate_micro_displacement(
    current_point,
    current_velocity,
    gbest_reference,
):
    """
    Original SFO Micro equation:

        v(t+1) = w*v(t) + psi2*r2*(gbest - x(t)) + epsilon

    No Pbest term is used in Micro mode.
    """
    r2 = rng.random(2)

    epsilon = rng.normal(
        0.0,
        EPSILON_MICRO,
        size=2,
    )

    displacement = (
        W_MICRO * current_velocity
        + C2_MICRO
        * r2
        * (gbest_reference - current_point)
        + epsilon
    )

    return limit_vector_magnitude(
        displacement,
        MICRO_V_MIN,
        MICRO_V_MAX,
    )


# ============================================================
# 6. Initialization
# ============================================================

def generate_initial_valid_segment():
    for _ in range(MAX_CANDIDATE_ATTEMPTS):
        velocity = vector_with_random_magnitude(
            INITIAL_STEP_MIN,
            INITIAL_STEP_MAX,
        )

        candidate = START + velocity

        if not point_inside_map(candidate):
            continue

        if not segment_is_collision_free(
            START,
            candidate,
        ):
            continue

        return candidate, velocity

    raise RuntimeError(
        "Could not generate a valid initial segment."
    )


def initialize_swarm():
    swarm = []

    for _ in range(POPULATION_SIZE):
        first_point, initial_velocity = (
            generate_initial_valid_segment()
        )

        path = [
            START.copy(),
            first_point.copy(),
        ]

        (
            partial_fitness,
            travelled,
            remaining,
        ) = evaluate_partial_path(path)

        individual = Individual(
            path=path,
            velocity=initial_velocity.copy(),
            travelled_length=travelled,
            remaining_distance=remaining,
            partial_fitness=partial_fitness,
        )

        swarm.append(individual)

    return swarm


# ============================================================
# 7. Glide phase
# ============================================================

def current_partial_gbest_index(swarm):
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


def try_direct_goal_connection(individual):
    current = individual.path[-1]

    distance_to_goal = float(
        np.linalg.norm(GOAL - current)
    )

    if distance_to_goal > GOAL_CONNECTION_DISTANCE:
        return False

    if not segment_is_collision_free(
        current,
        GOAL,
    ):
        return False

    individual.path.append(GOAL.copy())
    individual.velocity = GOAL - current
    individual.completed = True
    individual.failed = False

    (
        individual.partial_fitness,
        individual.travelled_length,
        individual.remaining_distance,
    ) = evaluate_partial_path(individual.path)

    individual.completed_pbest_path = copy_path(
        individual.path
    )
    individual.completed_pbest_length = (
        individual.travelled_length
    )

    return True


def extend_glide_individual(
    individual,
):
    if individual.completed or individual.frozen:
        return True

    if try_direct_goal_connection(individual):
        return True

    current_point = individual.path[-1].copy()
    base_velocity = individual.velocity.copy()

    for _ in range(MAX_CANDIDATE_ATTEMPTS):
        candidate_velocity = generate_glide_velocity(
            current_velocity=base_velocity
        )

        candidate_point = (
            current_point
            + candidate_velocity
        )

        if not point_inside_map(candidate_point):
            continue

        if not segment_is_collision_free(
            current_point,
            candidate_point,
        ):
            continue

        individual.path.append(
            candidate_point.copy()
        )
        individual.velocity = (
            candidate_velocity.copy()
        )
        individual.failed = False

        (
            individual.partial_fitness,
            individual.travelled_length,
            individual.remaining_distance,
        ) = evaluate_partial_path(individual.path)

        try_direct_goal_connection(individual)

        return True

    individual.failed = True
    return False


def nearest_successful_index(
    failed_individual,
    swarm,
    successful_indices,
):
    failed_endpoint = failed_individual.path[-1]

    return min(
        successful_indices,
        key=lambda index: np.linalg.norm(
            swarm[index].path[-1]
            - failed_endpoint
        ),
    )


def replace_failed_glide_individuals(
    swarm,
    successful_indices,
):
    if not successful_indices:
        return

    for individual in swarm:
        if not individual.failed:
            continue

        source_index = nearest_successful_index(
            individual,
            swarm,
            successful_indices,
        )

        source = swarm[source_index]

        individual.path = copy_path(source.path)
        individual.velocity = (
            vector_with_random_magnitude(
                GLIDE_V_MIN,
                GLIDE_V_MAX,
            )
        )
        individual.completed = source.completed
        individual.failed = False

        (
            individual.partial_fitness,
            individual.travelled_length,
            individual.remaining_distance,
        ) = evaluate_partial_path(individual.path)

        if individual.completed:
            individual.completed_pbest_path = copy_path(
                individual.path
            )
            individual.completed_pbest_length = (
                individual.travelled_length
            )


# ============================================================
# 8. Completed-path optimization
# ============================================================

def completed_indices(swarm):
    return [
        index
        for index, individual in enumerate(swarm)
        if individual.completed
    ]


def global_best_completed_path(swarm):
    indices = completed_indices(swarm)

    if not indices:
        return None, math.inf, None

    best_index = min(
        indices,
        key=lambda index: swarm[index].travelled_length,
    )

    return (
        copy_path(swarm[best_index].path),
        swarm[best_index].travelled_length,
        best_index,
    )


def completed_pbest_reference(
    individual,
    current_path,
    waypoint_index,
):
    ratio = relative_arc_position(
        current_path,
        waypoint_index,
    )

    if individual.completed_pbest_path is None:
        return current_path[waypoint_index].copy()

    return point_at_relative_arc(
        individual.completed_pbest_path,
        ratio,
    )


def optimize_completed_path(
    individual,
    gbest_path,
    mode,
):
    """
    Improve one completed path by moving its intermediate waypoints.

    Target:
        one candidate displacement is tested per waypoint.

    Micro:
        several small candidate displacements are tested per waypoint, and
        only the best collision-free shortening move is accepted.

    A waypoint move is accepted only when:
    - the candidate lies inside the map;
    - both adjacent segments are collision-free;
    - the complete path becomes shorter.
    """
    if not individual.completed:
        return False

    if len(individual.path) <= 2:
        return False

    improved = False

    waypoint_indices = np.arange(
        1,
        len(individual.path) - 1,
    )

    rng.shuffle(waypoint_indices)

    for waypoint_index in waypoint_indices:
        # The path may change after every accepted waypoint move.
        if waypoint_index >= len(individual.path) - 1:
            continue

        current_path = copy_path(individual.path)

        previous_point = current_path[
            waypoint_index - 1
        ]
        current_point = current_path[
            waypoint_index
        ]
        next_point = current_path[
            waypoint_index + 1
        ]

        ratio = relative_arc_position(
            current_path,
            waypoint_index,
        )

        gbest_reference = point_at_relative_arc(
            gbest_path,
            ratio,
        )

        pbest_reference = completed_pbest_reference(
            individual,
            current_path,
            waypoint_index,
        )

        attempts = (
            MICRO_ATTEMPTS_PER_WAYPOINT
            if mode == "micro"
            else 1
        )

        best_candidate_path = None
        best_candidate_length = (
            individual.travelled_length
        )
        best_displacement = None

        for _ in range(attempts):
            if mode == "target":
                displacement = generate_target_displacement(
                    current_point=current_point,
                    current_velocity=individual.velocity,
                    pbest_reference=pbest_reference,
                    gbest_reference=gbest_reference,
                )
            elif mode == "micro":
                displacement = generate_micro_displacement(
                    current_point=current_point,
                    current_velocity=individual.velocity,
                    gbest_reference=gbest_reference,
                )
            else:
                raise ValueError(
                    "Completed-path optimization supports "
                    "Target and Micro only."
                )

            candidate_point = (
                current_point
                + displacement
            )

            if not point_inside_map(candidate_point):
                continue

            if not segment_is_collision_free(
                previous_point,
                candidate_point,
            ):
                continue

            if not segment_is_collision_free(
                candidate_point,
                next_point,
            ):
                continue

            candidate_path = copy_path(current_path)
            candidate_path[
                waypoint_index
            ] = candidate_point.copy()

            candidate_length = path_length(
                candidate_path
            )

            if (
                candidate_length
                < best_candidate_length
                - 1e-9
            ):
                best_candidate_path = candidate_path
                best_candidate_length = candidate_length
                best_displacement = displacement.copy()

        if best_candidate_path is not None:
            individual.path = best_candidate_path
            individual.velocity = best_displacement
            individual.travelled_length = (
                best_candidate_length
            )
            individual.partial_fitness = (
                best_candidate_length
            )
            individual.remaining_distance = 0.0
            improved = True

    if (
        individual.travelled_length
        < individual.completed_pbest_length
        - 1e-9
    ):
        individual.completed_pbest_path = copy_path(
            individual.path
        )
        individual.completed_pbest_length = (
            individual.travelled_length
        )

    return improved


# ============================================================
# 9. Main algorithm
# ============================================================

def run_final_sfo():
    swarm = initialize_swarm()

    mode = "glide"

    target_stagnation = 0
    target_best_history = []
    micro_stagnation = 0

    best_completed_path = None
    best_completed_length = math.inf

    history = []
    gbest_history = []
    mode_history = []
    best_length_history = []

    # Initial current Gbest from partial paths
    partial_best_index = current_partial_gbest_index(
        swarm
    )

    current_gbest_path = copy_path(
        swarm[partial_best_index].path
    )

    for iteration in range(MAX_ITERATIONS):
        if mode == "glide":
            successful_indices = []

            for index, individual in enumerate(swarm):
                success = extend_glide_individual(
                    individual,
                )

                if success:
                    successful_indices.append(index)

            replace_failed_glide_individuals(
                swarm,
                successful_indices,
            )

            completed_count = sum(
                individual.completed
                for individual in swarm
            )

            completed_path, completed_length, _ = (
                global_best_completed_path(swarm)
            )

            if (
                completed_path is not None
                and completed_length
                < best_completed_length
            ):
                best_completed_path = copy_path(
                    completed_path
                )
                best_completed_length = (
                    completed_length
                )

            # Current partial Gbest is retained only for observation and
            # reporting. It does not enter the Glide equation.
            partial_best_index = (
                current_partial_gbest_index(swarm)
            )

            if partial_best_index is not None:
                current_gbest_path = copy_path(
                    swarm[partial_best_index].path
                )

            enough_completed_paths = (
                completed_count
                >= MIN_COMPLETED_FOR_TARGET
            )

            glide_time_finished = (
                iteration + 1
                >= MAX_GLIDE_ITERATIONS
            )

            can_start_target = (
                enough_completed_paths
                or (
                    glide_time_finished
                    and completed_count >= 1
                )
            )

            if can_start_target:
                transition_reason = (
                    f"{completed_count} completed paths available"
                    if enough_completed_paths
                    else
                    f"maximum Glide duration "
                    f"({MAX_GLIDE_ITERATIONS} iterations) reached"
                )

                print(
                    f"Switching Glide -> Target at "
                    f"iteration {iteration + 1}: "
                    f"{transition_reason}."
                )

                mode = "target"

                # Freeze unfinished paths. Only completed paths participate
                # in Target and Micro optimization.
                for individual in swarm:
                    if not individual.completed:
                        individual.frozen = True

                (
                    current_gbest_path,
                    best_completed_length,
                    _,
                ) = global_best_completed_path(swarm)

                best_completed_path = copy_path(
                    current_gbest_path
                )

                target_stagnation = 0
                target_best_history = [best_completed_length]

        elif mode == "target":
            (
                current_gbest_path,
                current_best_length,
                _,
            ) = global_best_completed_path(swarm)

            improved_any = False

            for individual in swarm:
                if not individual.completed:
                    continue

                if optimize_completed_path(
                    individual,
                    current_gbest_path,
                    mode="target",
                ):
                    improved_any = True

            (
                current_gbest_path,
                current_best_length,
                _,
            ) = global_best_completed_path(swarm)

            if (
                current_best_length
                < best_completed_length
                - 1e-9
            ):
                best_completed_length = (
                    current_best_length
                )
                best_completed_path = copy_path(
                    current_gbest_path
                )
                target_stagnation = 0
            else:
                target_stagnation += 1

            # Store the best completed length achieved at every Target cycle.
            target_best_history.append(
                best_completed_length
            )

            remaining_iterations = (
                MAX_ITERATIONS - iteration - 1
            )

            force_micro_by_time = (
                remaining_iterations
                <= FORCE_MICRO_LAST_ITERATIONS
            )

            insufficient_target_improvement = False
            target_relative_improvement = math.inf

            if (
                len(target_best_history)
                >= TARGET_IMPROVEMENT_WINDOW + 1
            ):
                old_window_best = (
                    target_best_history[
                        -TARGET_IMPROVEMENT_WINDOW - 1
                    ]
                )
                current_window_best = (
                    target_best_history[-1]
                )

                target_relative_improvement = (
                    old_window_best
                    - current_window_best
                ) / max(
                    abs(old_window_best),
                    1e-12,
                )

                insufficient_target_improvement = (
                    target_relative_improvement
                    < TARGET_MIN_RELATIVE_IMPROVEMENT
                )

            if (
                insufficient_target_improvement
                or force_micro_by_time
            ):
                transition_reason = (
                    "Target improvement below "
                    f"{100 * TARGET_MIN_RELATIVE_IMPROVEMENT:.1f}% "
                    f"during {TARGET_IMPROVEMENT_WINDOW} cycles"
                    if insufficient_target_improvement
                    else
                    f"last {FORCE_MICRO_LAST_ITERATIONS} "
                    "iterations reached"
                )

                print(
                    f"Switching Target -> Micro at "
                    f"iteration {iteration + 1}: "
                    f"{transition_reason}."
                )

                if math.isfinite(
                    target_relative_improvement
                ):
                    print(
                        "Target relative improvement over "
                        f"the last {TARGET_IMPROVEMENT_WINDOW} "
                        f"cycles = "
                        f"{100 * target_relative_improvement:.3f}%"
                    )

                mode = "micro"
                micro_stagnation = 0

        elif mode == "micro":
            (
                current_gbest_path,
                current_best_length,
                _,
            ) = global_best_completed_path(swarm)

            for individual in swarm:
                if not individual.completed:
                    continue

                if MICRO_PRUNE_BEFORE_MOVE:
                    apply_pruning_to_individual(
                        individual
                    )

                optimize_completed_path(
                    individual,
                    current_gbest_path,
                    mode="micro",
                )

                if MICRO_PRUNE_AFTER_MOVE:
                    apply_pruning_to_individual(
                        individual
                    )

            (
                current_gbest_path,
                current_best_length,
                _,
            ) = global_best_completed_path(swarm)

            if (
                current_best_length
                < best_completed_length
                - 1e-9
            ):
                best_completed_length = (
                    current_best_length
                )
                best_completed_path = copy_path(
                    current_gbest_path
                )
                micro_stagnation = 0
            else:
                micro_stagnation += 1

            if (
                micro_stagnation
                >= MICRO_STAGNATION_LIMIT
            ):
                break

        else:
            raise ValueError(
                f"Unknown mode: {mode}"
            )

        if iteration % SAVE_EVERY == 0:
            history.append([
                copy_path(individual.path)
                for individual in swarm
            ])
            gbest_history.append(
                copy_path(current_gbest_path)
            )
            mode_history.append(mode)
            best_length_history.append(
                best_completed_length
                if math.isfinite(best_completed_length)
                else math.nan
            )

        if (iteration + 1) % 10 == 0:
            completed_count = sum(
                individual.completed
                for individual in swarm
            )

            printable_best = (
                best_completed_length
                if math.isfinite(best_completed_length)
                else -1.0
            )

            print(
                f"Iteration {iteration + 1:3d} | "
                f"Mode={mode:6s} | "
                f"Completed={completed_count:2d}/{POPULATION_SIZE} | "
                f"Best completed length={printable_best:8.3f}"
                + (
                    f" | Glide limit="
                    f"{iteration + 1}/{MAX_GLIDE_ITERATIONS}"
                    if mode == "glide"
                    else ""
                )
                + (
                    f" | Target window="
                    f"{len(target_best_history)}/"
                    f"{TARGET_IMPROVEMENT_WINDOW + 1}"
                    if mode == "target"
                    else ""
                )
            )

    if best_completed_path is None:
        final_path = current_gbest_path
        final_completed = False
        final_length = path_length(
            current_gbest_path
        )
    else:
        final_path = best_completed_path
        final_completed = True
        final_length = best_completed_length

    return {
        "swarm": swarm,
        "final_path": final_path,
        "final_length": final_length,
        "completed": final_completed,
        "history": history,
        "gbest_history": gbest_history,
        "mode_history": mode_history,
        "best_length_history": best_length_history,
    }


# ============================================================
# 10. Visualization
# ============================================================

def draw_obstacles(axis):
    for xmin, ymin, xmax, ymax in OBSTACLES:
        rectangle = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            alpha=0.35,
        )
        axis.add_patch(rectangle)


def plot_final_result(result):
    figure, axis = plt.subplots(
        figsize=(8, 8)
    )

    draw_obstacles(axis)

    for individual in result["swarm"]:
        points = np.asarray(
            individual.path,
            dtype=float,
        )

        if len(points) >= 2:
            axis.plot(
                points[:, 0],
                points[:, 1],
                linewidth=0.8,
                alpha=0.12,
            )

    best_points = np.asarray(
        result["final_path"],
        dtype=float,
    )

    axis.plot(
        best_points[:, 0],
        best_points[:, 1],
        marker="o",
        linewidth=3.0,
        label="Best path",
    )

    axis.scatter(
        START[0],
        START[1],
        s=100,
        label="Start",
    )

    axis.scatter(
        GOAL[0],
        GOAL[1],
        s=100,
        label="Goal",
    )

    axis.set_title(
        "Three-Phase SFO with Micro Pruning\n"
        f"Completed={result['completed']}, "
        f"Length={result['final_length']:.3f}, "
        f"Segments={len(result['final_path']) - 1}"
    )

    axis.set_xlim(0, MAP_WIDTH)
    axis.set_ylim(0, MAP_HEIGHT)
    axis.set_aspect(
        "equal",
        adjustable="box",
    )
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        "final_sfo_three_phase_path.png",
        dpi=200,
    )

    plt.show()


def plot_best_length_history(result):
    values = np.asarray(
        result["best_length_history"],
        dtype=float,
    )

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.plot(values, linewidth=2.0)
    axis.set_title(
        "Best Completed Path Length"
    )
    axis.set_xlabel("Saved iteration")
    axis.set_ylabel("Path length")
    axis.grid(True)

    figure.tight_layout()

    figure.savefig(
        "final_sfo_best_length_history.png",
        dpi=200,
    )

    plt.show()


def animate_search(result):
    history = result["history"]
    gbest_history = result["gbest_history"]
    mode_history = result["mode_history"]

    if not history:
        return

    figure, axis = plt.subplots(
        figsize=(8, 8)
    )

    def update(frame_index):
        axis.clear()
        draw_obstacles(axis)

        for path in history[frame_index]:
            points = np.asarray(
                path,
                dtype=float,
            )

            if len(points) >= 2:
                axis.plot(
                    points[:, 0],
                    points[:, 1],
                    linewidth=0.9,
                    alpha=0.16,
                )

        best_points = np.asarray(
            gbest_history[frame_index],
            dtype=float,
        )

        axis.plot(
            best_points[:, 0],
            best_points[:, 1],
            marker="o",
            linewidth=3.0,
            label="Current Gbest",
        )

        axis.scatter(
            START[0],
            START[1],
            s=100,
            label="Start",
        )

        axis.scatter(
            GOAL[0],
            GOAL[1],
            s=100,
            label="Goal",
        )

        axis.set_title(
            f"Iteration={frame_index * SAVE_EVERY} | "
            f"Mode={mode_history[frame_index]}"
        )

        axis.set_xlim(0, MAP_WIDTH)
        axis.set_ylim(0, MAP_HEIGHT)
        axis.set_aspect(
            "equal",
            adjustable="box",
        )
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.grid(True)
        axis.legend(loc="upper right")

    animation = FuncAnimation(
        figure,
        update,
        frames=len(history),
        interval=1000 / ANIMATION_FPS,
        repeat=True,
    )

    animation.save(
        "final_sfo_three_phase_behavior.gif",
        writer=PillowWriter(
            fps=ANIMATION_FPS
        ),
        dpi=120,
    )

    plt.close(figure)


# ============================================================
# 11. Main
# ============================================================

if __name__ == "__main__":
    result = run_final_sfo()

    print("\nFinal result")
    print("-" * 70)
    print(
        f"Completed path      : {result['completed']}"
    )
    print(
        f"Final path length   : {result['final_length']:.6f}"
    )
    print(
        f"Number of segments  : {len(result['final_path']) - 1}"
    )
    print("Final path points:")

    for index, point in enumerate(
        result["final_path"]
    ):
        print(
            f"{index:3d}: "
            f"({point[0]:.6f}, {point[1]:.6f})"
        )

    if CREATE_GIF:
        print("\nCreating GIF animation...")
        animate_search(result)
        print(
            "Saved: final_sfo_three_phase_behavior.gif"
        )

    plot_final_result(result)
    plot_best_length_history(result)

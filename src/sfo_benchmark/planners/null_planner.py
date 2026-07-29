"""Architecture-test planner.

This planner is not a research baseline. It returns a deterministic detour
around the central test obstacle to verify the complete v0.2 pipeline.
"""

from __future__ import annotations

import time

import numpy as np

from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.path import Path
from sfo_benchmark.core.result import PlanningResult

from .base import BasePlanner


class NullPlanner(BasePlanner):
    """Return a fixed start-to-goal polyline for infrastructure testing."""

    def plan(
        self,
        environment: Environment,
        *,
        seed: int,
        run_id: int,
    ) -> PlanningResult:
        started_at = time.perf_counter()

        midpoint = np.array(
            [
                (environment.start[0] + environment.goal[0]) / 2.0,
                environment.bounds[3] - 10.0,
            ],
            dtype=float,
        )
        path = Path(np.vstack((environment.start, midpoint, environment.goal)))
        elapsed = time.perf_counter() - started_at

        return PlanningResult(
            algorithm=self.name,
            map_id=environment.map_id,
            run_id=run_id,
            seed=seed,
            success=True,
            execution_time=elapsed,
            path=path,
            message="Infrastructure-only deterministic planner.",
        )

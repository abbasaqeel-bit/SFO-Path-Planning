from __future__ import annotations
from dataclasses import dataclass
from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.result import PlanningResult
from sfo_benchmark.planners.base import BasePlanner
from .metrics import minimum_clearance,total_turning_angle
from .validator import PathValidator

@dataclass(slots=True)
class BenchmarkRunner:
    environment: Environment
    def run(self, planner: BasePlanner, *, seed: int, run_id: int)->PlanningResult:
        r=planner.plan(self.environment,seed=seed,run_id=run_id)
        if r.path is None: r.success=False; return r
        valid,msg=PathValidator(self.environment).validate(r.path); r.success=bool(r.success and valid)
        if not valid: r.message=msg
        r.path_length=r.path.length(); r.waypoint_count=r.path.waypoint_count; r.segment_count=r.path.segment_count
        r.total_turning_angle=total_turning_angle(r.path); r.minimum_clearance=minimum_clearance(r.path,self.environment)
        return r

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .path import Path

@dataclass(slots=True)
class PlanningResult:
    algorithm: str
    map_id: str
    run_id: int
    seed: int
    success: bool
    execution_time: float
    path: Path|None=None
    path_length: float|None=None
    waypoint_count: int|None=None
    segment_count: int|None=None
    total_turning_angle: float|None=None
    minimum_clearance: float|None=None
    iterations: int|None=None
    fitness_evaluations: int|None=None
    message: str=""
    metadata: dict[str,Any]=field(default_factory=dict)
    def to_record(self)->dict[str,Any]:
        return {"algorithm":self.algorithm,"map_id":self.map_id,"run_id":self.run_id,"seed":self.seed,"success":self.success,"execution_time":self.execution_time,"path":None if self.path is None else self.path.points.tolist(),"path_length":self.path_length,"waypoint_count":self.waypoint_count,"segment_count":self.segment_count,"total_turning_angle":self.total_turning_angle,"minimum_clearance":self.minimum_clearance,"iterations":self.iterations,"fitness_evaluations":self.fitness_evaluations,"message":self.message,"metadata":self.metadata}

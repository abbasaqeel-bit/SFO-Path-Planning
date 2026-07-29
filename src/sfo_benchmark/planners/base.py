from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.result import PlanningResult

@dataclass(frozen=True, slots=True)
class PlannerConfig:
    name: str
    parameters: dict[str,Any]=field(default_factory=dict)

class BasePlanner(ABC):
    def __init__(self, config: PlannerConfig)->None: self.config=config
    @property
    def name(self)->str: return self.config.name
    @abstractmethod
    def plan(self, environment: Environment, *, seed: int, run_id: int)->PlanningResult: ...

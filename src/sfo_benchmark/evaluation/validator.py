from __future__ import annotations
import numpy as np
from sfo_benchmark.core.collision import CollisionChecker
from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.path import Path

class PathValidator:
    def __init__(self, environment: Environment, tolerance: float=1e-6)->None:
        self.environment=environment; self.tolerance=tolerance; self.checker=CollisionChecker(environment)
    def validate(self, path: Path)->tuple[bool,str]:
        if not np.allclose(path.points[0],self.environment.start,atol=self.tolerance): return False,"invalid start"
        if not np.allclose(path.points[-1],self.environment.goal,atol=self.tolerance): return False,"invalid goal"
        if not self.checker.path_is_free(path): return False,"collision detected"
        return True,"valid path"

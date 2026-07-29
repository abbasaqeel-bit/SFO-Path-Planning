from __future__ import annotations
import numpy as np
from .environment import Environment
from .path import Path

class CollisionChecker:
    def __init__(self, environment: Environment, resolution: float=0.25)->None:
        if resolution<=0: raise ValueError("resolution must be positive")
        self.environment=environment; self.resolution=float(resolution)
    def point_is_free(self, point: np.ndarray)->bool:
        p=np.asarray(point,float)
        return self.environment.is_inside_bounds(p) and not any(o.contains(p,self.environment.robot_radius) for o in self.environment.obstacles)
    def segment_is_free(self, start: np.ndarray, end: np.ndarray)->bool:
        a=np.asarray(start,float); b=np.asarray(end,float); d=float(np.linalg.norm(b-a))
        n=max(2,int(np.ceil(d/self.resolution))+1)
        return all(self.point_is_free(a+t*(b-a)) for t in np.linspace(0,1,n))
    def path_is_free(self, path: Path)->bool:
        return all(self.segment_is_free(a,b) for a,b in zip(path.points[:-1],path.points[1:]))

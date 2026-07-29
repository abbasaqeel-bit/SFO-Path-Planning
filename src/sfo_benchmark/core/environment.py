from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .obstacle import Obstacle

@dataclass(frozen=True, slots=True)
class Environment:
    map_id: str
    bounds: tuple[float,float,float,float]
    start: np.ndarray
    goal: np.ndarray
    obstacles: tuple[Obstacle,...]=field(default_factory=tuple)
    robot_radius: float=0.0
    def __post_init__(self)->None:
        xmin,xmax,ymin,ymax=self.bounds
        if not(xmin<xmax and ymin<ymax): raise ValueError("invalid bounds")
        s=np.asarray(self.start,float); g=np.asarray(self.goal,float)
        if s.shape!=(2,) or g.shape!=(2,): raise ValueError("start and goal must have shape (2,)")
        object.__setattr__(self,"start",s); object.__setattr__(self,"goal",g)
        if not self.is_inside_bounds(s) or not self.is_inside_bounds(g): raise ValueError("endpoint outside bounds")
    def is_inside_bounds(self, point: np.ndarray)->bool:
        xmin,xmax,ymin,ymax=self.bounds; x,y=np.asarray(point,float)
        return bool(xmin<=x<=xmax and ymin<=y<=ymax)

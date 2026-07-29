from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True, slots=True)
class Path:
    points: np.ndarray
    def __post_init__(self)->None:
        p=np.asarray(self.points,float)
        if p.ndim!=2 or p.shape[1]!=2 or len(p)<2: raise ValueError("points must have shape (N,2), N>=2")
        object.__setattr__(self,"points",p)
    @property
    def waypoint_count(self)->int: return int(len(self.points))
    @property
    def segment_count(self)->int: return self.waypoint_count-1
    def length(self)->float: return float(np.linalg.norm(np.diff(self.points,axis=0),axis=1).sum())

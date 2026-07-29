"""Obstacle primitives used by the common benchmark environment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .types import Point


class Obstacle(ABC):
    """Abstract two-dimensional obstacle."""

    @abstractmethod
    def contains(self, point: Point, margin: float = 0.0) -> bool:
        """Return whether a point lies inside the inflated obstacle."""

    @abstractmethod
    def clearance(self, point: Point) -> float:
        """Return signed Euclidean clearance from a point to the obstacle."""

    @abstractmethod
    def segment_clearance(
        self,
        start: Point,
        end: Point,
        *,
        samples: int = 101,
    ) -> float:
        """Return approximate minimum clearance along a segment."""


@dataclass(frozen=True, slots=True)
class CircularObstacle(Obstacle):
    """Circular obstacle defined by center and radius."""

    center: Point
    radius: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        if center.shape != (2,):
            raise ValueError("CircularObstacle.center must have shape (2,).")
        if self.radius <= 0.0:
            raise ValueError("CircularObstacle.radius must be positive.")
        object.__setattr__(self, "center", center)

    def contains(self, point: Point, margin: float = 0.0) -> bool:
        point = np.asarray(point, dtype=float)
        return bool(np.linalg.norm(point - self.center) <= self.radius + margin)

    def clearance(self, point: Point) -> float:
        point = np.asarray(point, dtype=float)
        return float(np.linalg.norm(point - self.center) - self.radius)

    def segment_clearance(
        self,
        start: Point,
        end: Point,
        *,
        samples: int = 101,
    ) -> float:
        del samples
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        segment = end - start
        denominator = float(np.dot(segment, segment))
        if denominator == 0.0:
            return self.clearance(start)
        ratio = float(np.dot(self.center - start, segment) / denominator)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        closest = start + ratio * segment
        return self.clearance(closest)


@dataclass(frozen=True, slots=True)
class RectangularObstacle(Obstacle):
    """Axis-aligned rectangular obstacle."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self) -> None:
        if not (self.xmin < self.xmax and self.ymin < self.ymax):
            raise ValueError("RectangularObstacle bounds are invalid.")

    def contains(self, point: Point, margin: float = 0.0) -> bool:
        x, y = np.asarray(point, dtype=float)
        return bool(
            self.xmin - margin <= x <= self.xmax + margin
            and self.ymin - margin <= y <= self.ymax + margin
        )

    def clearance(self, point: Point) -> float:
        x, y = np.asarray(point, dtype=float)

        dx = max(self.xmin - x, 0.0, x - self.xmax)
        dy = max(self.ymin - y, 0.0, y - self.ymax)
        outside_distance = float(np.hypot(dx, dy))

        if not self.contains(point):
            return outside_distance

        inside_distance = min(
            x - self.xmin,
            self.xmax - x,
            y - self.ymin,
            self.ymax - y,
        )
        return -float(inside_distance)

    def segment_clearance(
        self,
        start: Point,
        end: Point,
        *,
        samples: int = 101,
    ) -> float:
        if samples < 2:
            raise ValueError("samples must be at least 2.")
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        return float(
            min(
                self.clearance(start + alpha * (end - start))
                for alpha in np.linspace(0.0, 1.0, samples)
            )
        )

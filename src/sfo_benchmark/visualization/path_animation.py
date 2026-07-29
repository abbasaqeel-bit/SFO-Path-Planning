"""Animated per-run path visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
import numpy as np

from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.obstacle import CircularObstacle, RectangularObstacle
from sfo_benchmark.core.path import Path as PlannedPath


def _animation_points(
    path: PlannedPath | None,
    maximum_frames: int = 100,
) -> np.ndarray:
    if path is None:
        return np.empty((0, 2), dtype=float)
    points = np.asarray(path.points, dtype=float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total <= 0.0:
        return points[:1]
    count = min(maximum_frames, max(2, int(np.ceil(total)) + 1))
    samples = np.linspace(0.0, total, count)
    return np.column_stack(
        [
            np.interp(samples, cumulative, points[:, dimension])
            for dimension in range(2)
        ]
    )


def save_path_gif(
    environment: Environment,
    path: PlannedPath | None,
    algorithm: str,
    output_path: Path,
    *,
    success: bool,
    message: str = "",
    fps: int = 12,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 7))

    for obstacle in environment.obstacles:
        if isinstance(obstacle, CircularObstacle):
            axis.add_patch(Circle(tuple(obstacle.center), obstacle.radius, alpha=0.35))
        elif isinstance(obstacle, RectangularObstacle):
            axis.add_patch(
                Rectangle(
                    (obstacle.xmin, obstacle.ymin),
                    obstacle.xmax - obstacle.xmin,
                    obstacle.ymax - obstacle.ymin,
                    alpha=0.35,
                )
            )

    axis.scatter(*environment.start, s=80, label="Start")
    axis.scatter(*environment.goal, s=130, marker="*", label="Goal")
    xmin, xmax, ymin, ymax = environment.bounds
    axis.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="X", ylabel="Y")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linewidth=0.4, alpha=0.5)
    status = "SUCCESS" if success else "FAILED / DIAGNOSTIC"
    axis.set_title(f"{algorithm} — {environment.map_id} — {status}")

    points = _animation_points(path)
    color = "#1f77b4" if success else "#d62728"
    style = "-" if success else "--"
    line, = axis.plot([], [], linestyle=style, color=color, linewidth=2.0)
    head = axis.scatter([], [], s=35, color=color)
    if path is None:
        axis.text(
            0.5,
            0.04,
            "No path was produced by the planner",
            transform=axis.transAxes,
            ha="center",
            color=color,
            weight="bold",
        )
    elif not success:
        axis.plot(
            [path.points[-1, 0], environment.goal[0]],
            [path.points[-1, 1], environment.goal[1]],
            linestyle=":",
            color="0.45",
            linewidth=1.0,
        )
    if message:
        axis.text(
            0.5,
            -0.08,
            message[:140],
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            wrap=True,
        )
    axis.legend(fontsize=8)

    def update(frame: int):
        visible = points[: frame + 1]
        if len(visible):
            line.set_data(visible[:, 0], visible[:, 1])
            head.set_offsets(visible[-1:])
        else:
            line.set_data([], [])
            head.set_offsets(np.empty((0, 2)))
        return line, head

    animation = FuncAnimation(
        figure,
        update,
        frames=max(1, len(points)),
        interval=1000 / max(fps, 1),
        blit=False,
        repeat=False,
    )
    animation.save(
        output_path,
        writer=PillowWriter(fps=max(fps, 1)),
        dpi=110,
    )
    plt.close(figure)
    return output_path

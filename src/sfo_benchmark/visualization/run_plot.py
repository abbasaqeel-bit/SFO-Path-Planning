"""Static visualization of complete and diagnostic planner output."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.obstacle import CircularObstacle, RectangularObstacle
from sfo_benchmark.core.path import Path as PlannedPath


def save_run_plot(
    environment: Environment,
    path: PlannedPath | None,
    algorithm: str,
    output_path: Path,
    *,
    success: bool = True,
    message: str = "",
    dpi: int = 300,
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

    if path is not None:
        color = "#1f77b4" if success else "#d62728"
        style = "-" if success else "--"
        label = "Valid complete path" if success else "Partial / invalid path"
        axis.plot(
            path.points[:, 0],
            path.points[:, 1],
            linestyle=style,
            color=color,
            linewidth=1.8,
            label=label,
        )
        axis.scatter(path.points[:, 0], path.points[:, 1], s=7, color=color)
        if not success:
            axis.scatter(
                path.points[-1, 0],
                path.points[-1, 1],
                marker="x",
                s=65,
                color=color,
                label="Last reached point",
            )
            axis.plot(
                [path.points[-1, 0], environment.goal[0]],
                [path.points[-1, 1], environment.goal[1]],
                linestyle=":",
                color="0.45",
                linewidth=1.0,
                label="Untravelled goal gap",
            )
    else:
        axis.text(
            0.5,
            0.03,
            "No path was produced by the planner",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            color="#d62728",
            weight="bold",
        )

    xmin, xmax, ymin, ymax = environment.bounds
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    status = "SUCCESS" if success else "FAILED / DIAGNOSTIC"
    axis.set_title(f"{algorithm} — {environment.map_id} — {status}")
    if message:
        axis.text(
            0.5,
            -0.10,
            message[:160],
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            wrap=True,
        )
    axis.grid(True, linewidth=0.4, alpha=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path

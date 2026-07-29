"""Convergence-curve rendering for optimization planners."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_convergence_plot(
    history: Sequence[float],
    output_path: Path,
    *,
    algorithm: str,
    dpi: int = 300,
) -> Path | None:
    if not history:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(range(1, len(history) + 1), history, linewidth=1.5)
    axis.set_xlabel("Recorded fitness evaluation")
    axis.set_ylabel("Best objective value")
    axis.set_title(f"{algorithm} convergence")
    axis.grid(True, linewidth=0.4, alpha=0.5)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path

"""Per-run artifact storage."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from sfo_benchmark.core.path import Path as PlannedPath
from sfo_benchmark.utils.serialization import write_json


class RunLogger:
    def __init__(self, run_directory: Path) -> None:
        self.run_directory = Path(run_directory)
        self.run_directory.mkdir(parents=True, exist_ok=True)

    def json(self, filename: str, payload: dict[str, Any]) -> Path:
        output = self.run_directory / filename
        write_json(output, payload)
        return output

    def text(self, filename: str, content: str) -> Path:
        output = self.run_directory / filename
        output.write_text(content, encoding="utf-8")
        return output

    def trajectory(self, path: PlannedPath | None) -> Path | None:
        if path is None:
            return None
        output = self.run_directory / "trajectory.csv"
        with output.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["waypoint_index", "x", "y"])
            for index, point in enumerate(path.points):
                writer.writerow([index, float(point[0]), float(point[1])])
        return output

import numpy as np

from sfo_benchmark.core.path import Path
from sfo_benchmark.evaluation.run_logger import RunLogger


def test_run_logger_creates_artifacts(tmp_path) -> None:
    logger = RunLogger(tmp_path / "Run_001")
    logger.json("config.json", {"seed": 7})
    logger.json("metrics.json", {"success": True})
    logger.text("console.log", "ok")
    logger.trajectory(Path(np.array([[0.0, 0.0], [1.0, 1.0]])))

    assert (tmp_path / "Run_001" / "config.json").exists()
    assert (tmp_path / "Run_001" / "metrics.json").exists()
    assert (tmp_path / "Run_001" / "trajectory.csv").exists()

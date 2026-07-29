"""Repeatable multi-run experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.result import PlanningResult
from sfo_benchmark.planners.base import BasePlanner
from sfo_benchmark.utils.provenance import collect_provenance
from sfo_benchmark.utils.random_seed import set_global_seed
from sfo_benchmark.utils.serialization import append_csv, write_json

from .benchmark import BenchmarkRunner


PlannerFactory = Callable[[], BasePlanner]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Settings controlling repeated algorithm trials."""

    name: str
    runs: int
    base_seed: int
    output_directory: Path

    def __post_init__(self) -> None:
        if self.runs <= 0:
            raise ValueError("runs must be positive.")


class ExperimentRunner:
    """Execute independent trials and save raw, auditable records."""

    def __init__(
        self,
        environment: Environment,
        config: ExperimentConfig,
        *,
        project_root: Path,
    ) -> None:
        self.environment = environment
        self.config = config
        self.project_root = project_root

    def run(self, planner_factory: PlannerFactory) -> list[PlanningResult]:
        """Run all configured trials with deterministic independent seeds."""
        output = self.config.output_directory
        output.mkdir(parents=True, exist_ok=True)

        provenance = collect_provenance(self.project_root)
        write_json(output / f"{self.config.name}_provenance.json", provenance)

        results: list[PlanningResult] = []
        csv_path = output / f"{self.config.name}_results.csv"

        for run_index in range(self.config.runs):
            run_id = run_index + 1
            seed = self.config.base_seed + run_index
            set_global_seed(seed)

            planner = planner_factory()
            result = BenchmarkRunner(self.environment).run(
                planner,
                seed=seed,
                run_id=run_id,
            )
            result.seed = seed
            result.metadata["experiment"] = self.config.name
            result.metadata["provenance_file"] = (
                f"{self.config.name}_provenance.json"
            )

            record = result.to_record()
            write_json(
                output / f"{self.config.name}_run_{run_id:03d}.json",
                record,
            )
            append_csv(csv_path, record)
            results.append(result)

        return results

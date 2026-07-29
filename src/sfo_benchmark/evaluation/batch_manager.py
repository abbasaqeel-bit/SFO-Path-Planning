"""Multi-planner, multi-map, repeated benchmark execution."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
import shutil
import time
import tracemalloc
from typing import Any, Callable

from sfo_benchmark.core.environment import Environment
from sfo_benchmark.evaluation.benchmark import BenchmarkRunner
from sfo_benchmark.evaluation.research_metrics import compute_research_metrics
from sfo_benchmark.evaluation.run_logger import RunLogger
from sfo_benchmark.planners.base import BasePlanner
from sfo_benchmark.utils.provenance import collect_provenance
from sfo_benchmark.utils.random_seed import set_global_seed
from sfo_benchmark.utils.serialization import append_csv, write_json
from sfo_benchmark.visualization.run_plot import save_run_plot
from sfo_benchmark.visualization.path_animation import save_path_gif
from sfo_benchmark.visualization.convergence_plot import save_convergence_plot
from sfo_benchmark.evaluation.reporting import EvaluationReporter


PlannerFactory = Callable[[], BasePlanner]


@dataclass(frozen=True, slots=True)
class PlannerEntry:
    name: str
    family: str
    factory: PlannerFactory
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BatchConfig:
    experiment_name: str
    runs: int
    base_seed: int
    output_directory: Path
    save_plots: bool = True
    save_gifs: bool = True
    clean_output: bool = True

    def __post_init__(self) -> None:
        if self.runs <= 0:
            raise ValueError("runs must be positive.")


class BatchBenchmarkManager:
    def __init__(
        self,
        *,
        environments: list[Environment],
        planners: list[PlannerEntry],
        config: BatchConfig,
        project_root: Path,
    ) -> None:
        if not environments or not planners:
            raise ValueError("Environments and planners cannot be empty.")
        self.environments = environments
        self.planners = planners
        self.config = config
        self.project_root = Path(project_root)

    def run(self) -> list[dict[str, Any]]:
        experiment_root = (
            Path(self.config.output_directory) / self.config.experiment_name
        )
        if self.config.clean_output and experiment_root.exists():
            output_root = Path(self.config.output_directory).resolve()
            resolved = experiment_root.resolve()
            if resolved.parent != output_root or not resolved.name:
                raise ValueError(
                    f"Refusing to clean unsafe experiment path: {resolved}"
                )
            shutil.rmtree(resolved)
        experiment_root.mkdir(parents=True, exist_ok=True)
        write_json(
            experiment_root / "provenance.json",
            collect_provenance(self.project_root),
        )

        summary_csv = experiment_root / "summary.csv"
        records: list[dict[str, Any]] = []

        for environment in self.environments:
            for planner_entry in self.planners:
                for run_index in range(self.config.runs):
                    record = self._run_once(
                        environment,
                        planner_entry,
                        run_index + 1,
                        self.config.base_seed + run_index,
                        experiment_root,
                    )
                    append_csv(summary_csv, record)
                    records.append(record)

        write_json(experiment_root / "summary.json", {"records": records})
        self._validate_complete_batch(records)
        EvaluationReporter(experiment_root, records).generate_all()
        return records

    def _validate_complete_batch(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        expected = (
            len(self.environments) * len(self.planners) * self.config.runs
        )
        if len(records) != expected:
            raise RuntimeError(
                f"Incomplete benchmark: expected {expected} records, "
                f"received {len(records)}."
            )
        keys = {
            (
                record["map_id"],
                record["algorithm"],
                int(record["run_number"]),
            )
            for record in records
        }
        if len(keys) != expected:
            raise RuntimeError(
                "Duplicate or missing map/algorithm/run combinations detected."
            )

    def _run_once(
        self,
        environment: Environment,
        planner_entry: PlannerEntry,
        run_number: int,
        seed: int,
        experiment_root: Path,
    ) -> dict[str, Any]:
        set_global_seed(seed)
        gc.collect()

        run_directory = (
            experiment_root
            / environment.map_id
            / planner_entry.name
            / f"Run_{run_number:03d}"
        )
        logger = RunLogger(run_directory)
        logger.json(
            "config.json",
            {
                "experiment": self.config.experiment_name,
                "map_id": environment.map_id,
                "algorithm": planner_entry.name,
                "family": planner_entry.family,
                "run_number": run_number,
                "seed": seed,
                "parameters": planner_entry.parameters,
            },
        )

        planner = planner_entry.factory()
        tracemalloc.start()
        cpu_start = time.process_time()
        result = BenchmarkRunner(environment).run(
            planner,
            seed=seed,
            run_id=run_number,
        )
        cpu_time = time.process_time() - cpu_start
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        metrics = compute_research_metrics(
            path=result.path,
            environment=environment,
            success=result.success,
            execution_time_s=result.execution_time,
            cpu_time_s=cpu_time,
            peak_memory_mb=peak_bytes / 1048576.0,
            metadata=result.metadata,
            fitness_evaluations=result.fitness_evaluations,
        )

        logger.json("metrics.json", metrics.to_dict())
        logger.json("planner_state.json", result.metadata)
        logger.trajectory(result.path)
        logger.text(
            "console.log",
            (
                f"algorithm={planner_entry.name}\n"
                f"map={environment.map_id}\n"
                f"run={run_number}\n"
                f"seed={seed}\n"
                f"success={result.success}\n"
                f"message={result.message}\n"
            ),
        )

        plot_name = None
        if self.config.save_plots:
            plot_name = "path.png"
            save_run_plot(
                environment,
                result.path,
                planner_entry.name,
                run_directory / plot_name,
                success=result.success,
                message=result.message,
            )

        animation_name = None
        if self.config.save_gifs:
            animation_name = "path.gif"
            save_path_gif(
                environment,
                result.path,
                planner_entry.name,
                run_directory / animation_name,
                success=result.success,
                message=result.message,
            )

        convergence_name = None
        convergence_history = result.metadata.get("fitness_history", [])
        if self.config.save_plots and convergence_history:
            convergence_name = "convergence.png"
            save_convergence_plot(
                convergence_history,
                run_directory / convergence_name,
                algorithm=planner_entry.name,
            )

        logger.json(
            "manifest.json",
            {
                "config": "config.json",
                "metrics": "metrics.json",
                "trajectory": "trajectory.csv" if result.path is not None else None,
                "planner_state": "planner_state.json",
                "console_log": "console.log",
                "plot": plot_name,
                "animation": animation_name,
                "convergence_plot": convergence_name,
            },
        )

        return {
            "experiment_name": self.config.experiment_name,
            "map_id": environment.map_id,
            "algorithm": planner_entry.name,
            "family": planner_entry.family,
            "run_number": run_number,
            "seed": seed,
            **metrics.to_dict(),
            "message": result.message,
            "run_directory": str(run_directory),
        }

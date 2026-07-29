"""Run the complete static path-planning benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from sfo_benchmark.core.map_loader import load_environment
from sfo_benchmark.evaluation.batch_manager import (
    BatchBenchmarkManager,
    BatchConfig,
    PlannerEntry,
)
from sfo_benchmark.planners.base import PlannerConfig
from sfo_benchmark.planners.aco import PublishedACOAdapter
from sfo_benchmark.planners.pythonrobotics import (
    AStarAdapter,
    DijkstraAdapter,
    InformedRRTStarAdapter,
    PRMAdapter,
    RRTAdapter,
    RRTStarAdapter,
    ThetaStarAdapter,
)
from sfo_benchmark.planners.sfo import ThreePhaseSFOAdapter
from sfo_benchmark.planners.matlab_aco_ga import MatlabACOGAAdapter
from sfo_benchmark.utils.config import load_yaml


ADAPTERS = {
    "astar": AStarAdapter,
    "dijkstra": DijkstraAdapter,
    "theta_star": ThetaStarAdapter,
    "rrt": RRTAdapter,
    "rrt_star": RRTStarAdapter,
    "informed_rrt_star": InformedRRTStarAdapter,
    "prm": PRMAdapter,
    "aco": PublishedACOAdapter,
    "aco_ga_matlab": MatlabACOGAAdapter,
    "sfo": ThreePhaseSFOAdapter,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/benchmark.yaml",
    )
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    payload = load_yaml(project_root / arguments.config)

    environment_payload = payload["environments"]
    environments = [
        load_environment(
            project_root / environment_payload["maps_file"],
            map_id,
        )
        for map_id in environment_payload["map_ids"]
    ]

    planners = []
    for item in payload["planners"]:
        adapter_class = ADAPTERS[str(item["adapter"])]
        name = str(item["name"])
        family = str(item["family"])
        parameters = dict(item.get("parameters", {}))
        planners.append(
            PlannerEntry(
                name=name,
                family=family,
                parameters=parameters,
                factory=lambda adapter_class=adapter_class,
                name=name,
                parameters=parameters: adapter_class(
                    PlannerConfig(
                        name=name,
                        parameters=parameters,
                    )
                ),
            )
        )

    experiment = payload["experiment"]
    records = BatchBenchmarkManager(
        environments=environments,
        planners=planners,
        config=BatchConfig(
            experiment_name=str(experiment["name"]),
            runs=int(experiment["runs"]),
            base_seed=int(experiment["base_seed"]),
            output_directory=(
                project_root / experiment["output_directory"]
            ),
            save_plots=bool(experiment.get("save_plots", True)),
            save_gifs=bool(experiment.get("save_gifs", True)),
            clean_output=bool(experiment.get("clean_output", True)),
        ),
        project_root=project_root,
    ).run()

    print(f"total_runs={len(records)}")
    print(
        "successful_runs="
        f"{sum(bool(record['success']) for record in records)}"
    )


if __name__ == "__main__":
    main()

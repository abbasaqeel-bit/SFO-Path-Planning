from pathlib import Path
import json

import pandas as pd

from sfo_benchmark.evaluation.reporting import EvaluationReporter


def test_evaluation_reporter_creates_tables_and_native_convergence(tmp_path: Path):
    root = tmp_path / "exp"
    run_dir = root / "map_a" / "algo_a" / "Run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "planner_state.json").write_text(
        json.dumps({"fitness_history": [10.0, 8.0, 7.0]}), encoding="utf-8"
    )
    records = [{
        "map_id": "map_a", "algorithm": "algo_a", "success": True,
        "path_length": 12.0, "execution_time_s": 0.1, "cpu_time_s": 0.08,
        "peak_memory_mb": 1.0, "total_turning_angle_rad": 0.5,
        "minimum_clearance": 2.0, "path_efficiency": 0.9,
        "expanded_nodes": 10, "generated_nodes": 20,
        "search_iterations": 3, "fitness_evaluations": 30,
    }]
    EvaluationReporter(root, records).generate_all()
    assert (root / "tables" / "benchmark_results.xlsx").exists()
    assert (root / "tables" / "summary_table.csv").exists()
    assert (
        root
        / "figures"
        / "convergence"
        / "map_a_algo_a_native_convergence.png"
    ).exists()
    assert (root / "tables" / "friedman_tests.csv").exists()
    assert (root / "tables" / "pairwise_wilcoxon_holm.csv").exists()
    assert not (
        root
        / "figures"
        / "convergence"
        / "map_a_convergence_comparison.png"
    ).exists()


def test_statistics_are_empty_when_pairing_key_is_unavailable():
    successful = pd.DataFrame(
        [
            {
                "map_id": "map_a",
                "algorithm": "algo_a",
                "success": True,
                "path_length": 12.0,
            }
        ]
    )

    friedman, pairwise = EvaluationReporter._statistical_tests(successful)

    assert friedman.empty
    assert pairwise.empty


def test_failed_lengths_are_excluded_and_boxplots_keep_all_algorithms(
    tmp_path: Path,
) -> None:
    root = tmp_path / "complete"
    records = []
    for run_number in range(1, 6):
        records.extend(
            [
                {
                    "map_id": "bug",
                    "algorithm": "successful",
                    "run_number": run_number,
                    "success": True,
                    "path_length": 10.0 + run_number,
                    "execution_time_s": 0.1,
                },
                {
                    "map_id": "bug",
                    "algorithm": "failed",
                    "run_number": run_number,
                    "success": False,
                    "path_length": 999.0,
                    "execution_time_s": 0.2,
                },
            ]
        )
    EvaluationReporter(root, records).generate_all()
    summary = pd.read_csv(root / "tables" / "summary_table.csv")
    successful = summary[summary["algorithm"] == "successful"].iloc[0]
    failed = summary[summary["algorithm"] == "failed"].iloc[0]
    assert successful["path_length_mean"] == 13.0
    assert pd.isna(failed["path_length_mean"])
    assert failed["successful_runs"] == 0
    assert (root / "figures" / "boxplots" / "bug_path_length.png").exists()
    assert (root / "tables" / "failed_and_partial_runs.csv").exists()

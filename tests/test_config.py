"""Regression test for YAML configuration loading."""

from pathlib import Path

from sfo_benchmark.utils.config import load_yaml


def test_load_yaml_mapping(tmp_path: Path) -> None:
    config_file = tmp_path / "sample.yaml"
    config_file.write_text("value: 42\n", encoding="utf-8")
    assert load_yaml(config_file) == {"value": 42}


def test_benchmark_config_contains_only_accepted_planners() -> None:
    project_root = Path(__file__).resolve().parents[1]
    payload = load_yaml(
        project_root / "configs" / "benchmark.yaml"
    )
    planner_names = {
        str(item["name"]) for item in payload["planners"]
    }

    assert len(planner_names) == 9
    assert "ga_aco_ga_repository_baseline" not in planner_names
    assert not any(name.startswith("ga_") for name in planner_names)
    assert "aco_ga_paper_reproduction" in planner_names
    assert "pso_pythonrobotics" not in planner_names
    assert payload["experiment"]["runs"] == 5
    assert len(payload["environments"]["map_ids"]) == 3
    assert payload["experiment"]["save_gifs"] is True
    assert payload["experiment"]["clean_output"] is True

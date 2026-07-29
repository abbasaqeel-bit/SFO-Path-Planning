from pathlib import Path
import hashlib
import zipfile

from sfo_benchmark.planners.aco.vendor_loader import (
    ARCHIVE_RELATIVE_PATH,
    load_original_aco_module,
)


EXPECTED_ARCHIVE_SHA256 = (
    "531d2d10633c042670ce765cfe8ba4f3df87a25cd10881cacfff9946cbbdeb61"
)


def test_original_repository_archive_is_preserved() -> None:
    archive = Path(ARCHIVE_RELATIVE_PATH)
    assert archive.is_file()
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == (
        EXPECTED_ARCHIVE_SHA256
    )
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    prefix = "ACO---Robot-Path-Planning-master/"
    assert prefix + "PathPlanning.py" in names
    assert prefix + "README.md" in names
    assert prefix + "LICENSE" in names


def test_adapter_invokes_original_class_without_common_objective() -> None:
    text = Path(
        "src/sfo_benchmark/planners/aco/adapter.py"
    ).read_text(encoding="utf-8")
    assert "load_original_aco_module()" in text
    assert "module.ACO_PDG(" in text
    assert "PathObjective" not in text
    assert "FixedWaypointEncoding(" not in text
    assert "_prune_line_of_sight" not in text


def test_native_source_features_and_example_parameters() -> None:
    archive = Path(ARCHIVE_RELATIVE_PATH)
    with zipfile.ZipFile(archive) as bundle:
        source = bundle.read(
            "ACO---Robot-Path-Planning-master/PathPlanning.py"
        ).decode("utf-8")
    assert "class ACO_PDG" in source
    assert "self.Potential_Field_Heuristic(brushfire_iter)" in source
    assert "def pheromone_diffusion" in source
    assert "def geometric_path_optimizer" in source
    assert "d_pheromone[grid[0], grid[1]] += Q /" in source
    assert "20, 100, 1.1, 12, 0.02, 0.5" in source
    assert "5, 20)" in source
    assert "run_ant_colony(10)" in source


def test_native_loader_exposes_unmodified_aco_class() -> None:
    module = load_original_aco_module()
    assert module.ACO_PDG.__name__ == "ACO_PDG"

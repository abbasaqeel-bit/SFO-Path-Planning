"""Generate path overlays from an existing result directory or ZIP."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Iterator
import zipfile

from sfo_benchmark.core.map_loader import load_environment
from sfo_benchmark.visualization.combined_paths import (
    generate_path_comparison_figures,
)


def _find_results_root(root: Path) -> Path:
    if (root / "summary.csv").exists():
        return root
    candidates = [
        path.parent
        for path in root.rglob("summary.csv")
        if path.parent.is_dir()
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Could not identify one benchmark root containing summary.csv "
            f"under {root}; found {len(candidates)} candidates."
        )
    return candidates[0]


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract a ZIP after rejecting members that escape the destination."""
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if destination != target and destination not in target.parents:
            raise ValueError(f"Unsafe ZIP member path: {member.filename}")
    archive.extractall(destination)


@contextmanager
def resolved_results_root(path: Path) -> Iterator[Path]:
    """Yield an extracted benchmark root for a directory or ZIP input."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        yield _find_results_root(path)
        return
    if not path.is_file() or path.suffix.lower() != ".zip":
        raise FileNotFoundError(
            f"Results input must be a benchmark directory or ZIP: {path}"
        )
    with tempfile.TemporaryDirectory(prefix="sfo_path_comparison_") as temporary:
        extraction = Path(temporary)
        with zipfile.ZipFile(path) as archive:
            _safe_extract(archive, extraction)
        yield _find_results_root(extraction)


def _map_ids_from_summary(results_root: Path) -> list[str]:
    import csv

    with (results_root / "summary.csv").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        rows = csv.DictReader(stream)
        return list(dict.fromkeys(str(row["map_id"]) for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create combined path figures from existing benchmark "
            "artefacts without rerunning any planner."
        )
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Benchmark directory or ZIP containing summary.csv.",
    )
    parser.add_argument(
        "--maps",
        default="configs/maps.yaml",
        help="Map YAML used for the benchmark.",
    )
    parser.add_argument(
        "--output",
        default="results/path_comparisons",
        help="Directory receiving PNG/PDF/SVG figures and selection CSVs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution (default: 600).",
    )
    parser.add_argument(
        "--collision-resolution",
        type=float,
        default=0.25,
        help=(
            "Sampling resolution used to mark collision intervals "
            "(default: 0.25, matching the common benchmark validator)."
        ),
    )
    parser.add_argument(
        "--include-title",
        action="store_true",
        help="Include a title inside each figure.",
    )
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    maps_path = Path(arguments.maps)
    if not maps_path.is_absolute():
        maps_path = project_root / maps_path
    output_path = Path(arguments.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    with resolved_results_root(Path(arguments.results)) as results_root:
        map_ids = _map_ids_from_summary(results_root)
        environments = {
            map_id: load_environment(maps_path, map_id)
            for map_id in map_ids
        }
        generated = generate_path_comparison_figures(
            results_root=results_root,
            environments=environments,
            output_directory=output_path,
            collision_resolution=float(arguments.collision_resolution),
            dpi=int(arguments.dpi),
            include_title=bool(arguments.include_title),
        )

    print(f"generated_files={len(generated)}")
    print(f"output_directory={output_path.resolve()}")


if __name__ == "__main__":
    main()

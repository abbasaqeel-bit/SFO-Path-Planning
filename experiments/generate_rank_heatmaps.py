"""Create large square-cell heatmaps from existing benchmark data."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Iterator
import zipfile

from sfo_benchmark.visualization.rank_heatmap import (
    generate_rank_heatmaps,
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
            "Expected exactly one benchmark directory containing summary.csv "
            f"below {root}; found {len(candidates)}."
        )
    return candidates[0]


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if destination != target and destination not in target.parents:
            raise ValueError(f"Unsafe ZIP member path: {member.filename}")
    archive.extractall(destination)


@contextmanager
def resolved_results_root(path: Path) -> Iterator[Path]:
    """Yield a results root whether input is a directory or an archive."""
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        yield _find_results_root(source)
        return
    if not source.is_file() or source.suffix.lower() != ".zip":
        raise FileNotFoundError(
            f"Results input must be a benchmark directory or ZIP: {source}"
        )
    with tempfile.TemporaryDirectory(prefix="sfo_rank_heatmap_") as temporary:
        extraction = Path(temporary)
        with zipfile.ZipFile(source) as archive:
            _safe_extract(archive, extraction)
        yield _find_results_root(extraction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create one half-page, square-cell performance rank heatmap for "
            "each map in an existing benchmark result directory or ZIP."
        )
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Benchmark directory or ZIP containing summary.csv.",
    )
    parser.add_argument(
        "--output",
        default="results/rank_heatmaps",
        help="Destination for PNG, PDF, SVG, and CSV plot data.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution in dots per inch (default: 600).",
    )
    parser.add_argument(
        "--include-title",
        action="store_true",
        help="Include a longer figure title inside each plot.",
    )
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output = Path(arguments.output)
    if not output.is_absolute():
        output = project_root / output

    with resolved_results_root(Path(arguments.results)) as results_root:
        generated = generate_rank_heatmaps(
            results_root=results_root,
            output_directory=output,
            dpi=int(arguments.dpi),
            include_title=bool(arguments.include_title),
        )

    print(f"generated_files={len(generated)}")
    print(f"output_directory={output.resolve()}")


if __name__ == "__main__":
    main()

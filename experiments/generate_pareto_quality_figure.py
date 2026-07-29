"""Generate the rank-Pareto figure from result artefacts."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Iterator
import zipfile

from sfo_benchmark.visualization.pareto_quality import (
    generate_rank_pareto_outputs,
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
            "Expected exactly one benchmark root containing summary.csv under "
            f"{root}; found {len(candidates)}."
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
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        yield _find_results_root(path)
        return
    if not path.is_file() or path.suffix.lower() != ".zip":
        raise FileNotFoundError(
            f"Results input must be a benchmark directory or ZIP: {path}"
        )
    with tempfile.TemporaryDirectory(prefix="sfo_rank_pareto_") as temporary:
        extraction = Path(temporary)
        with zipfile.ZipFile(path) as archive:
            _safe_extract(archive, extraction)
        yield _find_results_root(extraction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a rank-based Pareto figure from completed benchmark "
            "results without rerunning planners."
        )
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Benchmark directory or ZIP containing summary.csv.",
    )
    parser.add_argument(
        "--output",
        default="results/quality_figures",
        help="Output directory for PNG/PDF/SVG and plot-data CSV.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--include-title", action="store_true")
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output = Path(arguments.output)
    if not output.is_absolute():
        output = project_root / output

    with resolved_results_root(Path(arguments.results)) as results_root:
        generated = generate_rank_pareto_outputs(
            results_root=results_root,
            output_directory=output,
            dpi=int(arguments.dpi),
            include_title=bool(arguments.include_title),
        )

    print(f"generated_files={len(generated)}")
    print(f"output_directory={output.resolve()}")


if __name__ == "__main__":
    main()

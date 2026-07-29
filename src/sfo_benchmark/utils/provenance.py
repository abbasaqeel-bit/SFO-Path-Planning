"""Experiment provenance and dependency-version collection."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_provenance(project_root: Path) -> dict[str, Any]:
    """Collect software, platform, and repository metadata."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {
            "numpy": _package_version("numpy"),
            "PyYAML": _package_version("PyYAML"),
            "scipy": _package_version("scipy"),
            "matplotlib": _package_version("matplotlib"),
            "pandas": _package_version("pandas"),
            "openpyxl": _package_version("openpyxl"),
        },
        "git_commit": _git_commit(project_root),
    }

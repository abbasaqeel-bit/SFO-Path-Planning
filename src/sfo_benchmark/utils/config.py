"""YAML configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a UTF-8 YAML file whose root must be a mapping."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a mapping at the root of configuration file: {path}"
        )

    return payload

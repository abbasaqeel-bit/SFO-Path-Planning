"""Result serialization helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write formatted UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


def _csv_safe(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, allow_nan=True)
    return value


def append_csv(path: Path, record: dict[str, Any]) -> None:
    """Append one normalized result record to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {key: _csv_safe(value) for key, value in record.items()}
    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(normalized.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(normalized)

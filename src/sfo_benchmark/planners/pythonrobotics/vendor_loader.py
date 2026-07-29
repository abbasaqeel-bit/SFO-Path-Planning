"""Safe loading of the unmodified vendored PythonRobotics source files."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


@lru_cache(maxsize=None)
def load_vendor_module(relative_path: str, module_name: str) -> ModuleType:
    source_path = _project_root() / "third_party" / "pythonrobotics" / relative_path
    if not source_path.is_file():
        raise FileNotFoundError(f"Vendored PythonRobotics file not found: {source_path}")

    specification = importlib.util.spec_from_file_location(module_name, source_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load PythonRobotics module: {source_path}")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.show_animation = False
    return module

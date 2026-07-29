"""Load the archived Haghrah ACO repository without extracting or patching it."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import ModuleType
import zipimport


ARCHIVE_RELATIVE_PATH = Path(
    "third_party/aco_haghrah_original/"
    "ACO---Robot-Path-Planning-master.zip"
)
ARCHIVE_PREFIX = "ACO---Robot-Path-Planning-master"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


class _NoOpPlot:
    """Disable source visualization without changing its search behaviour."""

    @staticmethod
    def figure(*args: object, **kwargs: object) -> None:
        del args, kwargs

    @staticmethod
    def imshow(*args: object, **kwargs: object) -> None:
        del args, kwargs

    @staticmethod
    def show(*args: object, **kwargs: object) -> None:
        del args, kwargs


@lru_cache(maxsize=1)
def load_original_aco_module() -> ModuleType:
    archive = _project_root() / ARCHIVE_RELATIVE_PATH
    if not archive.is_file():
        raise FileNotFoundError(f"Original ACO archive not found: {archive}")

    importer = zipimport.zipimporter(
        f"{archive.as_posix()}/{ARCHIVE_PREFIX}"
    )
    code = importer.get_code("PathPlanning")
    if code is None:
        raise ImportError("PathPlanning.py is missing from the ACO archive.")

    module = ModuleType("sfo_vendor_haghrah_aco_path_planning")
    module.__file__ = (
        f"{archive.as_posix()}/{ARCHIVE_PREFIX}/PathPlanning.py"
    )
    exec(code, module.__dict__)

    # run_ant_colony() opens three interactive plots per colony iteration.
    # Replacing only the module's plotting handle makes headless benchmark
    # execution possible and does not alter any ACO state or equation.
    module.plt = _NoOpPlot()
    return module

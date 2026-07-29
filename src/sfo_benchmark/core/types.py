"""Shared numeric type aliases."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Point: TypeAlias = npt.NDArray[np.float64]
PathArray: TypeAlias = npt.NDArray[np.float64]

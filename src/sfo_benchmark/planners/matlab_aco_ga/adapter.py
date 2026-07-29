from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat

from sfo_benchmark.core.collision import CollisionChecker
from sfo_benchmark.core.environment import Environment
from sfo_benchmark.core.path import Path as BenchmarkPath
from sfo_benchmark.core.result import PlanningResult
from sfo_benchmark.planners.base import BasePlanner


class MatlabRuntimeUnavailable(RuntimeError):
    pass


class _MatlabAcoGaBase(BasePlanner):
    bridge_name = "run_aco_ga_paper_bridge"
    method_label = "ACO-GA"
    matlab_source_directory = Path(
        "third_party/aco_ga_paper_reproduction/matlab"
    )
    implementation_status = "paper_faithful_independent_reproduction"

    @staticmethod
    def _runtime(parameters: dict[str, Any]) -> tuple[str, str]:
        requested = str(parameters.get("runtime", "auto")).lower()
        octave = shutil.which(str(parameters.get("octave_executable", "octave")))
        configured_matlab = str(
            parameters.get(
                "matlab_executable",
                os.environ.get("SFO_MATLAB_EXECUTABLE", "matlab"),
            )
        )
        matlab = shutil.which(configured_matlab)
        if matlab is None:
            candidates = (
                Path(r"E:\Program Files\MATLAB\R2025b\bin\matlab.exe"),
                Path(r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"),
                Path(r"E:\Program Files\MATLAB\R2025a\bin\matlab.exe"),
                Path(r"C:\Program Files\MATLAB\R2025a\bin\matlab.exe"),
            )
            matlab = next(
                (str(candidate) for candidate in candidates if candidate.is_file()),
                None,
            )
        if requested in {"auto", "octave"} and octave:
            return "octave", octave
        if requested in {"auto", "matlab"} and matlab:
            return "matlab", matlab
        raise MatlabRuntimeUnavailable(
            "Neither GNU Octave nor MATLAB was found. Install Octave, or set "
            "runtime/matlab_executable in the planner parameters."
        )

    @staticmethod
    def _grid(environment: Environment, resolution: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int,int], tuple[int,int]]:
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        xmin, xmax, ymin, ymax = environment.bounds
        xs = np.arange(xmin, xmax + resolution * 0.5, resolution, dtype=float)
        ys = np.arange(ymin, ymax + resolution * 0.5, resolution, dtype=float)
        checker = CollisionChecker(environment, resolution=max(resolution/3.0, 0.05))
        # MATLAB rows correspond to y, columns to x.
        grid = np.ones((len(ys), len(xs)), dtype=np.int32)
        for r, y in enumerate(ys):
            for c, x in enumerate(xs):
                if not checker.point_is_free(np.array([x, y], dtype=float)):
                    grid[r, c] = 0
        start = (int(np.argmin(abs(ys-environment.start[1]))), int(np.argmin(abs(xs-environment.start[0]))))
        goal = (int(np.argmin(abs(ys-environment.goal[1]))), int(np.argmin(abs(xs-environment.goal[0]))))
        grid[start] = 2; grid[goal] = 3
        return grid, xs, ys, start, goal

    @staticmethod
    def _to_path(
        raw: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        environment: Environment,
        *,
        require_goal: bool = True,
    ) -> BenchmarkPath:
        rc = np.asarray(raw, dtype=float)
        if rc.ndim != 2 or rc.shape[1] != 2 or len(rc) < 2:
            raise ValueError("MATLAB returned an invalid path")
        if not np.all(np.isfinite(rc)):
            raise ValueError("MATLAB returned non-finite path coordinates")
        rounded = np.rint(rc).astype(int)
        if not np.allclose(rc, rounded, atol=1e-9):
            raise ValueError("MATLAB returned non-grid path coordinates")
        if (
            np.any(rounded[:, 0] < 1)
            or np.any(rounded[:, 0] > len(ys))
            or np.any(rounded[:, 1] < 1)
            or np.any(rounded[:, 1] > len(xs))
        ):
            raise ValueError("MATLAB returned out-of-bounds path coordinates")
        rows = rounded[:, 0] - 1
        cols = rounded[:, 1] - 1
        pts = np.column_stack((xs[cols], ys[rows]))
        start_index = (
            int(np.argmin(abs(ys - environment.start[1]))),
            int(np.argmin(abs(xs - environment.start[0]))),
        )
        goal_index = (
            int(np.argmin(abs(ys - environment.goal[1]))),
            int(np.argmin(abs(xs - environment.goal[0]))),
        )
        if tuple((rounded[0] - 1).tolist()) != start_index:
            raise ValueError("MATLAB path does not begin at the benchmark start")
        if (
            require_goal
            and tuple((rounded[-1] - 1).tolist()) != goal_index
        ):
            raise ValueError("MATLAB path does not end at the benchmark goal")
        # The benchmark maps can use non-grid-aligned endpoints. The MATLAB
        # search starts at the nearest grid cells; restore the exact physical
        # endpoints only after verifying that the returned cell indices match.
        pts[0] = environment.start
        if require_goal:
            pts[-1] = environment.goal
        keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-12]
        return BenchmarkPath(pts[keep])

    @staticmethod
    def _matlab_text(value: Any, default: str = "") -> str:
        if value is None:
            return default
        array = np.asarray(value)
        if array.size == 0:
            return default
        if array.dtype.kind in {"U", "S"}:
            return "".join(array.reshape(-1).astype(str).tolist()).strip()
        item = array.squeeze()
        return str(item).strip()

    def plan(self, environment: Environment, *, seed: int, run_id: int) -> PlanningResult:
        p = dict(self.config.parameters)
        started = time.perf_counter()
        try:
            runtime_name, executable = self._runtime(p)
        except MatlabRuntimeUnavailable as exc:
            return PlanningResult(self.name, environment.map_id, run_id, seed, False,
                                  time.perf_counter()-started, message=str(exc),
                                  metadata={"runtime_required": "GNU Octave or MATLAB", "bridge": self.bridge_name})
        resolution = float(p.get("resolution", 1.0))
        grid, xs, ys, start, goal = self._grid(environment, resolution)
        maps_stack = grid[:, :, None]
        project_root = Path(__file__).resolve().parents[4]
        matlab_dir = project_root / self.matlab_source_directory
        map_weights = np.asarray(
            p.get("map_weights", [1.0] * maps_stack.shape[2]),
            dtype=float,
        ).reshape(-1)
        if (
            len(map_weights) != maps_stack.shape[2]
            or np.any(map_weights < 0)
            or float(np.sum(map_weights)) <= 0
        ):
            raise ValueError(
                "map_weights must contain one non-negative value per map"
            )
        map_weights = map_weights / np.sum(map_weights)
        payload = {
            "maps_stack": maps_stack,
            "start_rc": np.array([start[0]+1, start[1]+1], dtype=np.int32),
            "goal_rc": np.array([goal[0]+1, goal[1]+1], dtype=np.int32),
            "seed": np.array(seed, dtype=np.int64),
            "population_size": np.array(
                int(p.get("population_size", 80))
            ),
            "aco_iterations": np.array(int(p.get("aco_iterations", 90))),
            "ga_iterations": np.array(int(p.get("ga_iterations", 60))),
            "rho": np.array(float(p.get("rho", 0.25))),
            "alpha": np.array(float(p.get("alpha", 2.0))),
            "beta": np.array(float(p.get("beta", 8.0))),
            "Q": np.array(float(p.get("Q", 100.0))),
            "Pc": np.array(float(p.get("Pc", 0.2))),
            "Pm": np.array(float(p.get("Pm", 0.05))),
            "gamma_weight": np.array(float(p.get("gamma", 0.5))),
            "delta_weight": np.array(float(p.get("delta", 0.5))),
            "length_weight": np.array(
                float(p.get("length_weight", 0.5))
            ),
            "smoothness_weight": np.array(
                float(p.get("smoothness_weight", 0.5))
            ),
            "map_weights": map_weights,
            "allow_corner_cutting": np.array(
                bool(p.get("allow_corner_cutting", False))
            ),
            "max_walk_attempts": np.array(int(p.get("max_walk_attempts", 500))),
        }
        with tempfile.TemporaryDirectory(prefix="sfo_matlab_") as td:
            input_file = Path(td)/"input.mat"; output_file = Path(td)/"output.mat"
            savemat(input_file, payload, do_compression=True)
            escaped_dir = str(matlab_dir).replace("'", "''")
            escaped_in = str(input_file).replace("'", "''")
            escaped_out = str(output_file).replace("'", "''")
            expression = f"addpath('{escaped_dir}'); {self.bridge_name}('{escaped_in}','{escaped_out}');"
            if runtime_name == "octave":
                command = [executable, "--quiet", "--no-gui", "--eval", expression]
            else:
                command = [executable, "-batch", expression]
            try:
                completed = subprocess.run(command, capture_output=True, text=True,
                    timeout=float(p.get("timeout_seconds", 1800)), check=False)
            except subprocess.TimeoutExpired:
                return PlanningResult(self.name, environment.map_id, run_id, seed, False,
                    time.perf_counter()-started, message="MATLAB/Octave execution timed out",
                    metadata={"runtime": runtime_name, "bridge": self.bridge_name})
            if completed.returncode != 0 or not output_file.exists():
                message = (completed.stderr or completed.stdout or "runtime failed")[-2000:]
                return PlanningResult(self.name, environment.map_id, run_id, seed, False,
                    time.perf_counter()-started, message=message,
                    metadata={
                        "runtime": runtime_name,
                        "bridge": self.bridge_name,
                        "returncode": completed.returncode,
                        "runtime_stdout_tail": completed.stdout[-4000:],
                        "runtime_stderr_tail": completed.stderr[-4000:],
                    })
            result = loadmat(output_file, squeeze_me=True)
        runtime_metadata = {
            "runtime": runtime_name,
            "bridge": self.bridge_name,
            "runtime_stdout_tail": completed.stdout[-4000:],
            "runtime_stderr_tail": completed.stderr[-4000:],
            "failure_stage": self._matlab_text(
                result.get("failure_stage"), "unknown"
            ),
            "validation_reason": self._matlab_text(
                result.get("validation_reason"), "unknown"
            ),
            "initial_population_count": int(
                np.asarray(
                    result.get("initial_population_count", 0)
                ).item()
            ),
            "max_grid_jump": float(
                np.asarray(result.get("max_grid_jump", np.nan)).item()
            ),
        }
        success = bool(np.asarray(result.get("success", 0)).item())
        if not success:
            diagnostic_path = None
            if "best_path" in result:
                try:
                    diagnostic_path = self._to_path(
                        result["best_path"],
                        xs,
                        ys,
                        environment,
                        require_goal=False,
                    )
                except (TypeError, ValueError):
                    diagnostic_path = None
            return PlanningResult(self.name, environment.map_id, run_id, seed, False,
                float(np.asarray(result.get("execution_time", time.perf_counter()-started)).item()),
                path=diagnostic_path,
                message=(
                    "MATLAB/Octave planner did not produce a valid path "
                    f"(stage={runtime_metadata['failure_stage']}, "
                    f"reason={runtime_metadata['validation_reason']})"
                ),
                metadata=runtime_metadata)
        try:
            path = self._to_path(result["best_path"], xs, ys, environment)
        except (KeyError, TypeError, ValueError) as exc:
            return PlanningResult(
                self.name,
                environment.map_id,
                run_id,
                seed,
                False,
                float(
                    np.asarray(
                        result.get(
                            "execution_time",
                            time.perf_counter() - started,
                        )
                    ).item()
                ),
                message=f"Invalid MATLAB path payload: {exc}",
                metadata=runtime_metadata,
            )
        checker = CollisionChecker(environment, resolution=float(p.get("validation_resolution", 0.25)))
        valid = checker.path_is_free(path)
        history = np.atleast_1d(
            np.asarray(result.get("fitness_history", []), dtype=float)
        )
        history = history[np.isfinite(history)].tolist()
        total_iterations = int(p.get("ga_iterations", 60))
        if self.bridge_name == "run_aco_ga_paper_bridge":
            total_iterations += int(p.get("aco_iterations", 90))
        metadata = {
            **runtime_metadata,
            "representation": "native_variable_length_grid_path",
            "implementation_status": self.implementation_status,
            "native_objective": self._matlab_text(
                result.get("objective_scope"),
                "repository_native_objective",
            ),
            "best_native_fitness": float(
                np.asarray(
                    result.get("best_fitness", np.nan)
                ).item()
            ),
            "fitness_history": history,
            "official_author_source_available": False,
            "integration_feasibility_repair": False,
            "single_map_benchmark_mode": True,
            "search_iterations": total_iterations,
        }
        return PlanningResult(
            algorithm=self.name, map_id=environment.map_id, run_id=run_id, seed=seed,
            success=valid, execution_time=float(np.asarray(result.get("execution_time", time.perf_counter()-started)).item()),
            path=path,
            path_length=path.length() if valid else None,
            waypoint_count=path.waypoint_count if valid else None,
            segment_count=path.segment_count if valid else None,
            iterations=total_iterations,
            message="" if valid else "Runtime path failed common collision validation",
            metadata=metadata,
        )


class MatlabACOGAAdapter(_MatlabAcoGaBase):
    """Paper-faithful independent ACO-GA reproduction."""

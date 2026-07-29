from pathlib import Path
import math
import os
import shutil

import numpy as np
import pytest

from sfo_benchmark.core.collision import CollisionChecker
from sfo_benchmark.core.environment import Environment
from sfo_benchmark.planners.base import PlannerConfig
from sfo_benchmark.planners.matlab_aco_ga import MatlabACOGAAdapter
from sfo_benchmark.visualization.run_plot import save_run_plot


def env():
    return Environment('open',(0,4,0,4),np.array([0.,0.]),np.array([4.,4.]))


def test_grid_coordinate_round_trip():
    planner=MatlabACOGAAdapter(PlannerConfig('aco_ga',{'resolution':1.0}))
    grid,xs,ys,start,goal=planner._grid(env(),1.0)
    assert grid[start]==2 and grid[goal]==3
    path=planner._to_path(np.array([[1,1],[3,3],[5,5]]),xs,ys,env())
    assert np.allclose(path.points[0],env().start)
    assert np.allclose(path.points[-1],env().goal)


def test_grid_conversion_rejects_out_of_bounds_indices():
    planner=MatlabACOGAAdapter(PlannerConfig('aco_ga',{'resolution':1.0}))
    _,xs,ys,_,_=planner._grid(env(),1.0)
    with pytest.raises(ValueError, match='out-of-bounds'):
        planner._to_path(np.array([[1,1],[99,99]]),xs,ys,env())


def test_grid_conversion_rejects_wrong_endpoints():
    planner=MatlabACOGAAdapter(PlannerConfig('aco_ga',{'resolution':1.0}))
    _,xs,ys,_,_=planner._grid(env(),1.0)
    with pytest.raises(ValueError, match='begin'):
        planner._to_path(np.array([[2,1],[5,5]]),xs,ys,env())


def test_missing_runtime_is_graceful(monkeypatch):
    import sfo_benchmark.planners.matlab_aco_ga.adapter as mod
    monkeypatch.setattr(mod.shutil,'which',lambda _: None)
    monkeypatch.setattr(mod.Path, 'is_file', lambda _: False)
    result=MatlabACOGAAdapter(PlannerConfig('hybrid',{})).plan(env(),seed=1,run_id=0)
    assert not result.success
    assert 'Octave' in result.message


def test_matlab_runtime_can_be_selected_from_environment(
    monkeypatch,
    tmp_path,
):
    import sfo_benchmark.planners.matlab_aco_ga.adapter as mod

    executable = tmp_path / "matlab.exe"
    executable.touch()
    monkeypatch.setenv("SFO_MATLAB_EXECUTABLE", str(executable))
    monkeypatch.setattr(
        mod.shutil,
        "which",
        lambda value: str(executable) if value == str(executable) else None,
    )

    assert MatlabACOGAAdapter._runtime({"runtime": "matlab"}) == (
        "matlab",
        str(executable),
    )


def test_paper_reproduction_sources_and_bridge_exist():
    root=Path(__file__).resolve().parents[1]
    reproduction = (
        root / 'third_party' / 'aco_ga_paper_reproduction' / 'matlab'
    )
    paper_files = {
        'run_aco_ga_paper_bridge.m',
        'paper_aco_population.m',
        'paper_ga_optimize.m',
        'paper_multimap_fitness.m',
        'paper_avoid_obstacles.m',
        'paper_validate_grid_path.m',
    }
    assert paper_files.issubset({path.name for path in reproduction.iterdir()})
    adapter = MatlabACOGAAdapter(PlannerConfig('paper', {}))
    assert adapter.bridge_name == 'run_aco_ga_paper_bridge'
    assert adapter.implementation_status == (
        'paper_faithful_independent_reproduction'
    )


def test_live_matlab_bridge_constructs_valid_plot(
    tmp_path,
):
    executable = os.environ.get('SFO_TEST_MATLAB_EXECUTABLE')
    if not executable:
        pytest.skip('Set SFO_TEST_MATLAB_EXECUTABLE for live MATLAB test')
    if not Path(executable).is_file() and shutil.which(executable) is None:
        pytest.skip('Configured MATLAB executable was not found')

    parameters = {
        'runtime': 'matlab',
        'matlab_executable': executable,
        'resolution': 1.0,
        'population_size': 12,
        'aco_iterations': 5,
        'ga_iterations': 4,
        'max_walk_attempts': 100,
        'timeout_seconds': 120,
    }
    bridge_name = 'run_aco_ga_paper_bridge'
    planner = MatlabACOGAAdapter(PlannerConfig(bridge_name, parameters))
    result = planner.plan(env(), seed=7, run_id=1)

    assert result.success, result.message
    assert result.path is not None
    assert CollisionChecker(env()).path_is_free(result.path)
    assert np.allclose(result.path.points[0], env().start)
    assert np.allclose(result.path.points[-1], env().goal)
    assert result.metadata['failure_stage'] == 'none'
    assert result.metadata['validation_reason'] == 'ok'
    assert result.metadata['max_grid_jump'] <= math.sqrt(2) + 1e-12
    expected_history = 9
    assert result.metadata['implementation_status'] == (
        'paper_faithful_independent_reproduction'
    )
    assert result.metadata['native_objective'] == (
        'native_published_multimap_objective'
    )
    assert len(result.metadata['fitness_history']) == expected_history

    image = save_run_plot(
        env(),
        result.path,
        bridge_name,
        tmp_path / f'{bridge_name}.png',
    )
    assert image.is_file()
    assert image.stat().st_size > 0

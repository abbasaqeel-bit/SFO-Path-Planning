from sfo_benchmark.planners.sfo import (
    ThreePhaseSFOConfig,
    config_from_parameters,
)


def test_reference_parameters_are_frozen_as_defaults() -> None:
    config = ThreePhaseSFOConfig()
    assert config.population_size == 30
    assert config.max_iterations == 250
    assert config.w_glide == 0.35
    assert config.alpha_glide == 6.0
    assert config.beta_glide == 3.0
    assert config.w_target == 0.20
    assert config.c1_target == 0.80
    assert config.c2_target == 1.60
    assert config.w_micro == 0.10
    assert config.c2_micro == 1.20
    assert config.target_improvement_window == 30
    assert config.target_min_relative_improvement == 0.05


def test_config_override_is_explicit() -> None:
    config = config_from_parameters(
        {"population_size": 10, "max_iterations": 20}
    )
    assert config.population_size == 10
    assert config.max_iterations == 20
    assert config.w_glide == 0.35

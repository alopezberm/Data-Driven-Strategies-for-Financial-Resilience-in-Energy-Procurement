"""
settings.py

Project settings and lightweight runtime configuration.
This module centralizes operational defaults used across training, policy,
backtesting, and reporting workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config.constants import (
    ALLOW_SHIFT_ON_HOLIDAYS,
    ALLOW_SHIFT_ON_WEEKENDS,
    COURSE_NAME,
    DEFAULT_DAILY_VOLUME,
    DEFAULT_EXTREME_COST_QUANTILE,
    DEFAULT_FIGURE_HEIGHT,
    DEFAULT_FIGURE_WIDTH,
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_HEDGE_RATIO_ON_BUY_FUTURE,
    DEFAULT_QUANTILES,
    DEFAULT_REFERENCE_STRATEGY,
    DEFAULT_ROLLING_WINDOWS,
    DEFAULT_SHIFT_FRACTION,
    DEFAULT_SHIFT_PENALTY_PER_MWH,
    DEFAULT_SMALL_FIGURE_HEIGHT,
    DEFAULT_SMALL_FIGURE_WIDTH,
    DEFAULT_STATIC_HEDGE_RATIO,
    MIN_ABS_RISK_PREMIUM_TO_HEDGE,
    MIN_ABS_RISK_PREMIUM_TO_SHIFT,
    MIN_REL_RISK_PREMIUM_TO_HEDGE,
    MIN_REL_RISK_PREMIUM_TO_SHIFT,
    PROJECT_NAME,
    PROJECT_SHORT_NAME,
    TARGET_COLUMN,
)


@dataclass
class TrainingSettings:
    """Settings used during model training and validation."""

    target_column: str = TARGET_COLUMN
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON
    quantiles: list[float] = field(default_factory=lambda: list(DEFAULT_QUANTILES))
    rolling_windows: list[int] = field(default_factory=lambda: list(DEFAULT_ROLLING_WINDOWS))
    static_hedge_ratio: float = DEFAULT_STATIC_HEDGE_RATIO


@dataclass
class PolicySettings:
    """Settings governing the heuristic decision policy."""

    min_abs_risk_premium_to_hedge: float = MIN_ABS_RISK_PREMIUM_TO_HEDGE
    min_rel_risk_premium_to_hedge: float = MIN_REL_RISK_PREMIUM_TO_HEDGE
    min_abs_risk_premium_to_shift: float = MIN_ABS_RISK_PREMIUM_TO_SHIFT
    min_rel_risk_premium_to_shift: float = MIN_REL_RISK_PREMIUM_TO_SHIFT
    allow_shift_on_weekends: bool = ALLOW_SHIFT_ON_WEEKENDS
    allow_shift_on_holidays: bool = ALLOW_SHIFT_ON_HOLIDAYS


@dataclass
class SimulationSettings:
    """Settings used for policy and baseline simulation."""

    default_daily_volume: float = DEFAULT_DAILY_VOLUME
    hedge_ratio_on_buy_future: float = DEFAULT_HEDGE_RATIO_ON_BUY_FUTURE
    shift_fraction: float = DEFAULT_SHIFT_FRACTION
    shift_penalty_per_mwh: float = DEFAULT_SHIFT_PENALTY_PER_MWH
    reference_strategy_name: str = DEFAULT_REFERENCE_STRATEGY
    extreme_cost_quantile: float = DEFAULT_EXTREME_COST_QUANTILE


@dataclass
class VisualizationSettings:
    """Default figure settings for project plots."""

    figure_width: int = DEFAULT_FIGURE_WIDTH
    figure_height: int = DEFAULT_FIGURE_HEIGHT
    small_figure_width: int = DEFAULT_SMALL_FIGURE_WIDTH
    small_figure_height: int = DEFAULT_SMALL_FIGURE_HEIGHT


@dataclass
class ProjectMetadata:
    """High-level metadata for the project."""

    project_name: str = PROJECT_NAME
    project_short_name: str = PROJECT_SHORT_NAME
    course_name: str = COURSE_NAME


@dataclass
class ProjectSettings:
    """Top-level settings container for the whole project."""

    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    policy: PolicySettings = field(default_factory=PolicySettings)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)


def get_default_settings() -> ProjectSettings:
    """Return the default project settings object."""
    return ProjectSettings()


if __name__ == "__main__":
    settings = get_default_settings()
    print("Project settings loaded successfully.")
    print(settings)

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
    DEFAULT_LAG_STEPS,
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
    Q90_COLUMN,
    Q95_COLUMN,
    TARGET_COLUMN,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class TrainingSettings:
    """Settings used during model training and validation."""

    target_column: str = TARGET_COLUMN
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON
    quantiles: list[float] = field(default_factory=lambda: list(DEFAULT_QUANTILES))
    rolling_windows: list[int] = field(default_factory=lambda: list(DEFAULT_ROLLING_WINDOWS))
    lag_steps: list[int] = field(default_factory=lambda: list(DEFAULT_LAG_STEPS))
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
class TailRiskSettings:
    """Settings used for tail-risk analysis and extreme-event flagging."""

    high_quantile_column: str = Q90_COLUMN
    extreme_quantile_column: str = Q95_COLUMN
    price_spike_threshold: float | None = None


@dataclass
class FeatureSelectionSettings:
    """Settings used for feature-selection utilities."""

    max_missing_share: float = 0.40
    min_non_null_rows: int = 30
    top_k_importance: int | None = None
    random_state: int = 42
    n_estimators: int = 200


@dataclass
class ExplainabilitySettings:
    """Settings used for feature importance, scenario explanations, and SHAP."""

    permutation_repeats: int = 10
    shap_max_background_samples: int = 200
    shap_max_explanation_rows: int = 200
    top_k_features: int | None = 10
    random_state: int = 42


@dataclass
class RLSettings:
    """Settings used for lightweight RL environment and agent scaffolding."""

    risk_aversion: float = 1.0
    hedge_cost_penalty: float = 0.1
    epsilon: float = 0.1
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    state_rounding_digits: int = 1
    heuristic_hedge_threshold: float = 5.0
    heuristic_shift_threshold: float = 2.0


@dataclass
class PipelineSettings:
    """Settings for pipeline execution behavior."""

    save_outputs: bool = True
    generate_figures: bool = True
    verbose: bool = True


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
    tail_risk: TailRiskSettings = field(default_factory=TailRiskSettings)
    feature_selection: FeatureSelectionSettings = field(default_factory=FeatureSelectionSettings)
    explainability: ExplainabilitySettings = field(default_factory=ExplainabilitySettings)
    rl: RLSettings = field(default_factory=RLSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)


def get_default_settings() -> ProjectSettings:
    """Return the default project settings object."""
    return ProjectSettings()


if __name__ == "__main__":
    settings = get_default_settings()
    logger.info("Project settings loaded successfully.")
    logger.info(f"\n{settings}")

"""
tail_risk_models.py

Utilities for modelling and evaluating tail risk using quantile outputs.
This module builds on quantile models to explicitly analyse extreme scenarios
(high-price risk) relevant for hedging decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.constants import (
    DEFAULT_FORECAST_HORIZON,
    Q50_COLUMN,
    Q90_COLUMN,
    Q95_COLUMN,
    TARGET_COLUMN,
)
from src.config.settings import TailRiskSettings, get_default_settings


class TailRiskModelError(Exception):
    """Raised when tail risk computations fail."""


# =========================
# Configuration
# =========================


@dataclass
class TailRiskConfig:
    """Configuration for tail risk computations."""

    target_column: str = TARGET_COLUMN
    horizon: int = DEFAULT_FORECAST_HORIZON
    high_quantile: float = 0.9
    extreme_quantile: float = 0.95
    price_spike_threshold: float | None = None  # optional absolute threshold

    @classmethod
    def from_settings(
        cls,
        settings: TailRiskSettings,
        target_column: str = TARGET_COLUMN,
        horizon: int = DEFAULT_FORECAST_HORIZON,
    ) -> "TailRiskConfig":
        """Build a tail-risk config from centralized project settings."""
        high_quantile = float(str(settings.high_quantile_column).replace("q_", ""))
        extreme_quantile = float(str(settings.extreme_quantile_column).replace("q_", ""))
        return cls(
            target_column=target_column,
            horizon=horizon,
            high_quantile=high_quantile,
            extreme_quantile=extreme_quantile,
            price_spike_threshold=settings.price_spike_threshold,
        )


def get_default_tail_risk_settings() -> TailRiskSettings:
    """Return default tail-risk settings from the project configuration."""
    return get_default_settings().tail_risk



def _get_config(config: TailRiskConfig | None) -> TailRiskConfig:
    """Resolve an explicit config or build one from project settings."""
    if config is not None:
        return config
    return TailRiskConfig.from_settings(get_default_tail_risk_settings())



def _format_quantile_column(quantile: float) -> str:
    """Format a quantile value into the project's quantile-column naming convention."""
    return f"q_{quantile:g}"


# =========================
# Core utilities
# =========================


def validate_quantile_inputs(df: pd.DataFrame, quantile_columns: list[str]) -> None:
    """Ensure required quantile columns exist."""
    missing = [col for col in quantile_columns if col not in df.columns]
    if missing:
        raise TailRiskModelError(f"Missing quantile columns: {missing}")



def compute_tail_risk_features(
    df: pd.DataFrame,
    config: TailRiskConfig | None = None,
) -> pd.DataFrame:
    """
    Compute tail-risk features based on quantile predictions.

    Expected columns (example):
    - configured high-quantile column
    - configured extreme-quantile column

    Returns dataframe with:
    - tail_spread
    - extreme_spread
    - tail_ratio
    """
    config = _get_config(config)

    q_high_col = _format_quantile_column(config.high_quantile)
    q_extreme_col = _format_quantile_column(config.extreme_quantile)

    validate_quantile_inputs(df, [q_high_col, q_extreme_col])

    result_df = df.copy()

    # Spread between extreme and high quantile (uncertainty in tail)
    result_df["tail_spread"] = result_df[q_extreme_col] - result_df[q_high_col]

    # Spread between high quantile and median-like proxy (if exists)
    median_col = Q50_COLUMN if Q50_COLUMN in result_df.columns else None
    if median_col:
        result_df["extreme_spread"] = result_df[q_extreme_col] - result_df[median_col]
        result_df["tail_ratio"] = result_df[q_extreme_col] / result_df[median_col]
    else:
        result_df["extreme_spread"] = result_df["tail_spread"]
        result_df["tail_ratio"] = result_df[q_extreme_col] / result_df[q_high_col]

    return result_df


# =========================
# Tail risk metrics
# =========================


def compute_tail_risk_metrics(
    df: pd.DataFrame,
    config: TailRiskConfig | None = None,
) -> dict[str, float]:
    """
    Compute summary metrics for tail risk.

    Returns:
    - mean_tail_spread
    - max_tail_spread
    - pct_high_risk_days
    - pct_extreme_risk_days
    """
    config = _get_config(config)

    q_high_col = _format_quantile_column(config.high_quantile)
    q_extreme_col = _format_quantile_column(config.extreme_quantile)

    validate_quantile_inputs(df, [q_high_col, q_extreme_col])

    tail_spread = df[q_extreme_col] - df[q_high_col]

    # Define thresholds
    high_threshold = df[q_high_col].quantile(0.75)
    extreme_threshold = df[q_extreme_col].quantile(0.90)

    pct_high_risk = float((df[q_high_col] > high_threshold).mean())
    pct_extreme_risk = float((df[q_extreme_col] > extreme_threshold).mean())

    return {
        "mean_tail_spread": float(tail_spread.mean()),
        "max_tail_spread": float(tail_spread.max()),
        "pct_high_risk_days": pct_high_risk,
        "pct_extreme_risk_days": pct_extreme_risk,
    }


# =========================
# Extreme event detection
# =========================


def flag_extreme_events(
    df: pd.DataFrame,
    config: TailRiskConfig | None = None,
) -> pd.DataFrame:
    """
    Flag extreme price scenarios based on quantiles or absolute thresholds.

    Adds columns:
    - is_high_risk
    - is_extreme_risk
    """
    config = _get_config(config)

    q_high_col = _format_quantile_column(config.high_quantile)
    q_extreme_col = _format_quantile_column(config.extreme_quantile)

    validate_quantile_inputs(df, [q_high_col, q_extreme_col])

    result_df = df.copy()

    if config.price_spike_threshold is not None:
        threshold = config.price_spike_threshold
        result_df["is_extreme_risk"] = result_df[q_extreme_col] > threshold
    else:
        threshold = result_df[q_extreme_col].quantile(0.90)
        result_df["is_extreme_risk"] = result_df[q_extreme_col] > threshold

    high_threshold = result_df[q_high_col].quantile(0.75)
    result_df["is_high_risk"] = result_df[q_high_col] > high_threshold

    return result_df


# =========================
# Combined pipeline
# =========================


def build_tail_risk_dataset(
    df: pd.DataFrame,
    config: TailRiskConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Full pipeline:
    1. Compute tail risk features
    2. Flag extreme events
    3. Compute summary metrics

    Returns:
    - enriched dataframe
    - metrics dictionary
    """
    config = _get_config(config)

    df_features = compute_tail_risk_features(df, config=config)
    df_flagged = flag_extreme_events(df_features, config=config)
    metrics = compute_tail_risk_metrics(df_flagged, config=config)

    return df_flagged, metrics


# =========================
# Quick test
# =========================


if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            Q50_COLUMN: [50, 55, 60, 58, 62],
            Q90_COLUMN: [60, 65, 70, 68, 75],
            Q95_COLUMN: [65, 70, 78, 72, 82],
        }
    )

    df_out, metrics = build_tail_risk_dataset(example_df)

    print("Tail risk features:")
    print(df_out)

    print("\nMetrics:")
    print(metrics)
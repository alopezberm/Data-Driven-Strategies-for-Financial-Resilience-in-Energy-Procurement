"""
build_lag_features.py

Feature engineering utilities for lag-based variables.
These features use only past information via pandas `.shift()` and therefore
preserve temporal causality when applied to time-ordered data.
"""

from __future__ import annotations

import pandas as pd

from src.config.constants import DATE_COLUMN, DEFAULT_LAG_STEPS, PRIMARY_FUTURE_COLUMN, TARGET_COLUMN
from src.config.settings import TrainingSettings, get_default_settings


def get_default_lag_config(training_settings: TrainingSettings | None = None) -> dict[str, list[int]]:
    """Build the default lag configuration from project settings."""
    if training_settings is None:
        training_settings = get_default_settings().training

    lag_steps = list(training_settings.lag_steps) if training_settings.lag_steps else list(DEFAULT_LAG_STEPS)
    market_lag_steps = sorted({lag for lag in lag_steps if lag in {1, 7, 14, 28}})

    return {
        TARGET_COLUMN: lag_steps,
        PRIMARY_FUTURE_COLUMN: market_lag_steps,
        "Future_M1_OpenInterest": market_lag_steps,
        "Future_M2_Price": market_lag_steps,
        "Future_M2_OpenInterest": market_lag_steps,
    }


class LagFeaturesError(Exception):
    """Raised when lag features cannot be created safely."""


# =========================
# Validation helpers
# =========================

def _validate_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that the input dataframe is suitable for lag feature generation.

    Returns
    -------
    pd.DataFrame
        Sorted copy of the dataframe.
    """
    if df.empty:
        raise LagFeaturesError("Input dataframe is empty.")

    if DATE_COLUMN not in df.columns:
        raise LagFeaturesError(
            f"Input dataframe must contain a '{DATE_COLUMN}' column."
        )

    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    if df[DATE_COLUMN].isna().any():
        invalid_count = int(df[DATE_COLUMN].isna().sum())
        raise LagFeaturesError(
            f"Found {invalid_count} invalid date values while building lag features."
        )

    if df[DATE_COLUMN].duplicated().any():
        raise LagFeaturesError(
            "Input dataframe contains duplicated dates. Lag features require unique chronological rows."
        )

    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    return df



def _validate_lag_config(lag_config: dict[str, list[int]]) -> None:
    """Validate lag configuration structure and values."""
    if not lag_config:
        raise LagFeaturesError("lag_config cannot be empty.")

    for column, lags in lag_config.items():
        if not isinstance(column, str):
            raise LagFeaturesError("All lag_config keys must be column names (strings).")

        if not isinstance(lags, list) or not lags:
            raise LagFeaturesError(f"Lag list for column '{column}' must be a non-empty list.")

        if len(set(lags)) != len(lags):
            raise LagFeaturesError(
                f"Lag list for column '{column}' contains duplicated lag values."
            )

        if any((not isinstance(lag, int) or lag <= 0) for lag in lags):
            raise LagFeaturesError(
                f"All lag values for column '{column}' must be positive integers."
            )


# =========================
# Core feature builder
# =========================

def _create_lag_features(df: pd.DataFrame, lag_config: dict[str, list[int]]) -> pd.DataFrame:
    """
    Create lagged versions of selected columns.

    Missing source columns are skipped intentionally to keep the function robust
    to evolving datasets.
    """
    df = df.copy()

    for column, lags in lag_config.items():
        if column not in df.columns:
            continue

        for lag in sorted(set(lags)):
            lag_column_name = f"{column}_lag_{lag}"
            df[lag_column_name] = df[column].shift(lag)

    return df


# =========================
# Public API
# =========================

def build_lag_features(
    df: pd.DataFrame,
    lag_config: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    """
    Add lag-based features to a time-ordered dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the configured date column and time-dependent variables.
    lag_config : dict[str, list[int]] | None, optional
        Dictionary mapping column names to lists of lag horizons.
        If None, a default configuration tailored to the energy procurement
        project is used.

    Returns
    -------
    pd.DataFrame
        Dataframe enriched with lag features.
    """
    df = _validate_input_dataframe(df)
    lag_config = get_default_lag_config() if lag_config is None else lag_config
    _validate_lag_config(lag_config)

    df = _create_lag_features(df, lag_config)
    return df


if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2024-01-01", periods=10, freq="D"),
            TARGET_COLUMN: range(10),
            PRIMARY_FUTURE_COLUMN: range(100, 110),
            "Future_M1_OpenInterest": range(1000, 1010),
        }
    )

    transformed_df = build_lag_features(example_df)
    print(transformed_df.head(10))

"""
build_lag_features.py

Feature engineering utilities for lag-based variables.
These features use only past information via pandas `.shift()` and therefore
preserve temporal causality when applied to time-ordered data.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_LAG_CONFIG = {
    "Spot_Price_SPEL": [1, 2, 3, 7, 14, 28],
    "Future_M1_Price": [1, 7, 14, 28],
    "Future_M1_OpenInterest": [1, 7, 14, 28],
    "Future_M2_Price": [1, 7, 14, 28],
    "Future_M2_OpenInterest": [1, 7, 14, 28],
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

    if "date" not in df.columns:
        raise LagFeaturesError("Input dataframe must contain a 'date' column.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        invalid_count = int(df["date"].isna().sum())
        raise LagFeaturesError(
            f"Found {invalid_count} invalid date values while building lag features."
        )

    if df["date"].duplicated().any():
        raise LagFeaturesError(
            "Input dataframe contains duplicated dates. Lag features require unique chronological rows."
        )

    df = df.sort_values("date").reset_index(drop=True)
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
        Input dataframe containing a `date` column and time-dependent variables.
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
    lag_config = DEFAULT_LAG_CONFIG if lag_config is None else lag_config
    _validate_lag_config(lag_config)

    df = _create_lag_features(df, lag_config)
    return df


if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "Spot_Price_SPEL": range(10),
            "Future_M1_Price": range(100, 110),
            "Future_M1_OpenInterest": range(1000, 1010),
        }
    )

    transformed_df = build_lag_features(example_df)
    print(transformed_df.head(10))

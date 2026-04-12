"""
build_time_features.py

Feature engineering utilities for time-based variables.
These features are deterministic calendar transformations and do not
introduce leakage when built from the date column alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.constants import DATE_COLUMN


class TimeFeaturesError(Exception):
    """Raised when time features cannot be created safely."""


DAYS_IN_WEEK = 7
MONTHS_IN_YEAR = 12
DAYS_IN_YEAR = 365.25

MONTH_TO_SEASON = {
    12: 1,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 2,
    6: 3,
    7: 3,
    8: 3,
    9: 4,
    10: 4,
    11: 4,
}


# =========================
# Validation helpers
# =========================

def _validate_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that the dataframe contains a usable date column.

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with the date column coerced to datetime.
    """
    if DATE_COLUMN not in df.columns:
        raise TimeFeaturesError(
            f"Input dataframe must contain a '{DATE_COLUMN}' column."
        )

    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    if df[DATE_COLUMN].isna().any():
        invalid_count = int(df[DATE_COLUMN].isna().sum())
        raise TimeFeaturesError(
            f"Found {invalid_count} invalid date values while building time features."
        )

    return df


# =========================
# Feature builders
# =========================

def _add_basic_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add standard calendar-based features derived from the date column."""
    df = df.copy()

    df["day_of_week"] = df[DATE_COLUMN].dt.dayofweek.astype("Int64")
    df["day_of_month"] = df[DATE_COLUMN].dt.day.astype("Int64")
    df["day_of_year"] = df[DATE_COLUMN].dt.dayofyear.astype("Int64")
    df["week_of_year"] = df[DATE_COLUMN].dt.isocalendar().week.astype("Int64")
    df["month"] = df[DATE_COLUMN].dt.month.astype("Int64")
    df["quarter"] = df[DATE_COLUMN].dt.quarter.astype("Int64")
    df["year"] = df[DATE_COLUMN].dt.year.astype("Int64")

    return df



def _add_weekend_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary weekend indicator based on day_of_week."""
    df = df.copy()
    df["is_weekend"] = (df[DATE_COLUMN].dt.dayofweek >= 5).astype("Int64")
    return df



def _add_month_boundary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators for month start and month end."""
    df = df.copy()
    df["is_month_start"] = df[DATE_COLUMN].dt.is_month_start.astype("Int64")
    df["is_month_end"] = df[DATE_COLUMN].dt.is_month_end.astype("Int64")
    return df



def _add_season_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a season code based on month.

    Coding convention:
    - 1: Winter (Dec-Feb)
    - 2: Spring (Mar-May)
    - 3: Summer (Jun-Aug)
    - 4: Autumn (Sep-Nov)
    """
    df = df.copy()

    df["season"] = df[DATE_COLUMN].dt.month.map(MONTH_TO_SEASON).astype("Int64")
    return df



def _add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical encodings for features with circular structure.

    These are often useful for ML models because they preserve periodicity,
    e.g. month 12 is close to month 1.
    """
    df = df.copy()

    day_of_week = df[DATE_COLUMN].dt.dayofweek
    month = df[DATE_COLUMN].dt.month
    day_of_year = df[DATE_COLUMN].dt.dayofyear

    df["day_of_week_sin"] = pd.Series(
        np.sin(2 * np.pi * day_of_week / DAYS_IN_WEEK), index=df.index
    )
    df["day_of_week_cos"] = pd.Series(
        np.cos(2 * np.pi * day_of_week / DAYS_IN_WEEK), index=df.index
    )

    df["month_sin"] = pd.Series(
        np.sin(2 * np.pi * month / MONTHS_IN_YEAR), index=df.index
    )
    df["month_cos"] = pd.Series(
        np.cos(2 * np.pi * month / MONTHS_IN_YEAR), index=df.index
    )

    df["day_of_year_sin"] = pd.Series(
        np.sin(2 * np.pi * day_of_year / DAYS_IN_YEAR), index=df.index
    )
    df["day_of_year_cos"] = pd.Series(
        np.cos(2 * np.pi * day_of_year / DAYS_IN_YEAR), index=df.index
    )

    return df


# =========================
# Public API
# =========================

def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add deterministic time-based features to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the configured date column.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional time-based features.
    """
    df = _validate_date_column(df)
    df = _add_basic_calendar_features(df)
    df = _add_weekend_feature(df)
    df = _add_month_boundary_features(df)
    df = _add_season_feature(df)
    df = _add_cyclical_time_features(df)

    return df


if __name__ == "__main__":
    example_df = pd.DataFrame({DATE_COLUMN: pd.date_range("2024-01-01", periods=5, freq="D")})
    transformed_df = build_time_features(example_df)
    print(transformed_df.head())

"""
build_time_features.py

Feature engineering utilities for time-based variables.
These features are deterministic calendar transformations and do not
introduce leakage when built from the date column alone.
"""

from __future__ import annotations

import numpy as np

import pandas as pd


class TimeFeaturesError(Exception):
    """Raised when time features cannot be created safely."""


# =========================
# Validation helpers
# =========================

def _validate_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that the dataframe contains a usable `date` column.

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with `date` coerced to datetime.
    """
    if "date" not in df.columns:
        raise TimeFeaturesError("Input dataframe must contain a 'date' column.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        invalid_count = int(df["date"].isna().sum())
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

    df["day_of_week"] = df["date"].dt.dayofweek.astype("Int64")
    df["day_of_month"] = df["date"].dt.day.astype("Int64")
    df["day_of_year"] = df["date"].dt.dayofyear.astype("Int64")
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype("Int64")
    df["month"] = df["date"].dt.month.astype("Int64")
    df["quarter"] = df["date"].dt.quarter.astype("Int64")
    df["year"] = df["date"].dt.year.astype("Int64")

    return df



def _add_weekend_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary weekend indicator based on day_of_week."""
    df = df.copy()
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype("Int64")
    return df



def _add_month_boundary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators for month start and month end."""
    df = df.copy()
    df["is_month_start"] = df["date"].dt.is_month_start.astype("Int64")
    df["is_month_end"] = df["date"].dt.is_month_end.astype("Int64")
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

    month_to_season = {
        12: 1, 1: 1, 2: 1,
        3: 2, 4: 2, 5: 2,
        6: 3, 7: 3, 8: 3,
        9: 4, 10: 4, 11: 4,
    }

    df["season"] = df["date"].dt.month.map(month_to_season).astype("Int64")
    return df



def _add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical encodings for features with circular structure.

    These are often useful for ML models because they preserve periodicity,
    e.g. month 12 is close to month 1.
    """
    df = df.copy()

    day_of_week = df["date"].dt.dayofweek
    month = df["date"].dt.month
    day_of_year = df["date"].dt.dayofyear

    df["day_of_week_sin"] = pd.Series(
        np.sin(2 * np.pi * day_of_week / 7), index=df.index
    )
    df["day_of_week_cos"] = pd.Series(
        np.cos(2 * np.pi * day_of_week / 7), index=df.index
    )

    df["month_sin"] = pd.Series(
        np.sin(2 * np.pi * month / 12), index=df.index
    )
    df["month_cos"] = pd.Series(
        np.cos(2 * np.pi * month / 12), index=df.index
    )

    df["day_of_year_sin"] = pd.Series(
        np.sin(2 * np.pi * day_of_year / 365.25), index=df.index
    )
    df["day_of_year_cos"] = pd.Series(
        np.cos(2 * np.pi * day_of_year / 365.25), index=df.index
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
        Input dataframe containing a `date` column.

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
    example_df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5, freq="D")})
    transformed_df = build_time_features(example_df)
    print(transformed_df.head())

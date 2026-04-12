
"""
clean_weather.py

Cleaning utilities for the weather / calendar dataset.
This module standardizes types, validates the date index,
handles duplicates, performs light normalization, and saves
an interim cleaned dataset.
"""

from __future__ import annotations

import pandas as pd

from src.config.paths import WEATHER_CLEAN_FILE
from src.data.load_raw_data import load_weather_data


NUMERIC_WEATHER_COLUMNS = [
    "weather_code",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_mean",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "shortwave_radiation_sum",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "surface_pressure_mean",
    "et0_fao_evapotranspiration_sum",
    "peninsular_max_temperature",
    "peninsular_min_temperature",
    "max_windspeed",
    "daily_temperature_range",
    "rolling_avg_temp",
    "delta_temp_with_previous",
    "std_avg_temperature",
    "Day_of_the_week",
    "Is_weekend",
    "Month",
    "Year",
    "Season",
    "Is_national_holiday",
]

FLAG_COLUMNS = [
    "Is_weekend",
    "Is_national_holiday",
]

INTEGER_LIKE_COLUMNS = [
    "Day_of_the_week",
    "Month",
    "Year",
    "Season",
]

TIMESTAMP_COLUMNS = [
    "sunrise",
    "sunset",
]


class WeatherCleaningError(Exception):
    """Raised when the weather dataset cannot be cleaned correctly."""


# =========================
# Validation helpers
# =========================

def _validate_required_columns(df: pd.DataFrame) -> None:
    """Ensure the minimum required columns are present after loading."""
    required_columns = ["date"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise WeatherCleaningError(
            f"Missing required columns after weather load: {missing_columns}"
        )



def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert known weather and calendar columns to numeric when present.

    Missing columns are ignored to keep the cleaner robust to evolving
    versions of the raw weather dataset.
    """
    df = df.copy()

    for column in NUMERIC_WEATHER_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df



def _convert_timestamp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert UNIX timestamp columns such as sunrise and sunset to pandas datetime.

    This is a light normalization step that preserves the original information while
    making later feature engineering much easier.
    """
    df = df.copy()

    for column in TIMESTAMP_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], unit="s", errors="coerce")

    return df



def _cast_integer_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast calendar and flag columns to pandas nullable integer type where possible.

    Using Int64 allows us to preserve missing values if they appear later.
    """
    df = df.copy()

    for column in FLAG_COLUMNS + INTEGER_LIKE_COLUMNS:
        if column in df.columns:
            df[column] = df[column].round().astype("Int64")

    return df



def _drop_duplicate_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate dates while keeping the last occurrence."""
    df = df.copy()
    duplicated_dates = df.duplicated(subset=["date"], keep="last")
    df = df.loc[~duplicated_dates].copy()
    return df



def _validate_clean_dataframe(df: pd.DataFrame) -> None:
    """Run final checks on the cleaned weather dataframe."""
    if df.empty:
        raise WeatherCleaningError("Weather cleaned dataframe is empty.")

    if df["date"].isna().any():
        raise WeatherCleaningError("Weather cleaned dataframe contains null dates.")

    if df["date"].duplicated().any():
        raise WeatherCleaningError("Weather cleaned dataframe still has duplicated dates.")


# =========================
# Public API
# =========================

def clean_weather_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean an already-loaded weather dataframe.

    This dataframe-level API is useful for tests and for callers that already
    have the raw weather data in memory.
    """
    if df.empty:
        raise WeatherCleaningError("Weather input dataframe is empty.")

    cleaned_df = df.copy()
    _validate_required_columns(cleaned_df)
    cleaned_df["date"] = pd.to_datetime(cleaned_df["date"], errors="coerce")
    cleaned_df = _coerce_numeric_columns(cleaned_df)
    cleaned_df = _convert_timestamp_columns(cleaned_df)
    cleaned_df = _cast_integer_like_columns(cleaned_df)
    cleaned_df = cleaned_df.sort_values("date").reset_index(drop=True)
    cleaned_df = _drop_duplicate_dates(cleaned_df)
    _validate_clean_dataframe(cleaned_df)
    return cleaned_df


def clean_weather_data(save: bool = True) -> pd.DataFrame:
    """
    Load, clean, validate, and optionally save the weather dataset.

    Cleaning steps:
    1. Load raw weather data
    2. Validate required columns
    3. Convert known weather columns to numeric
    4. Convert sunrise/sunset UNIX timestamps to pandas datetime
    5. Cast calendar and flag columns to integer-like types
    6. Sort by date
    7. Remove duplicated dates
    8. Run final validation
    9. Save interim dataset if requested

    Parameters
    ----------
    save : bool, optional
        Whether to save the cleaned dataframe to `data/interim/weather_clean.csv`.

    Returns
    -------
    pd.DataFrame
        Cleaned weather dataframe.
    """
    df = load_weather_data()
    df = clean_weather_dataframe(df)

    if save:
        df.to_csv(WEATHER_CLEAN_FILE, index=False)

    return df


if __name__ == "__main__":
    cleaned_df = clean_weather_data(save=True)
    print(f"Weather cleaned successfully. Final shape: {cleaned_df.shape}")

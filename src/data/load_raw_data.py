"""
load_raw_data.py

Utility functions to load raw datasets for the project.
Each loader returns a pandas DataFrame with basic validation and
standardized date parsing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.config.paths import HOLIDAYS_RAW_FILE, OMIP_RAW_FILE, WEATHER_RAW_FILE


REQUIRED_OMIP_COLUMNS = [
    "Date",
    "Spot_Price_SPEL",
]

REQUIRED_WEATHER_COLUMNS = [
    "date",
]

REQUIRED_HOLIDAYS_COLUMNS = [
    "date",
]


class RawDataLoadError(Exception):
    """Raised when a raw dataset cannot be loaded or validated."""


# =========================
# Generic helpers
# =========================

def _check_file_exists(file_path: Path) -> None:
    """Raise a clear error if the file does not exist."""
    if not file_path.exists():
        raise RawDataLoadError(f"File not found: {file_path}")



def _validate_required_columns(df: pd.DataFrame, required_columns: list[str], dataset_name: str) -> None:
    """Validate that the DataFrame contains the required columns."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise RawDataLoadError(
            f"Missing required columns in {dataset_name}: {missing_columns}"
        )



def _standardize_date_column(
    df: pd.DataFrame,
    source_column: str,
    dataset_name: str,
    output_column: str = "date",
) -> pd.DataFrame:
    """
    Parse and standardize a date column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    source_column : str
        Name of the source date column.
    dataset_name : str
        Name used in error messages.
    output_column : str, optional
        Name of the standardized output date column, by default "date".

    Returns
    -------
    pd.DataFrame
        DataFrame with a standardized pandas datetime column.
    """
    if source_column not in df.columns:
        raise RawDataLoadError(
            f"Date column '{source_column}' not found in {dataset_name}."
        )

    df = df.copy()
    df[output_column] = pd.to_datetime(df[source_column], errors="coerce")

    if df[output_column].isna().any():
        invalid_count = int(df[output_column].isna().sum())
        raise RawDataLoadError(
            f"Found {invalid_count} invalid date values in {dataset_name} ({source_column})."
        )

    return df


# =========================
# Dataset-specific loaders
# =========================

def load_omip_data(file_path: Path = OMIP_RAW_FILE) -> pd.DataFrame:
    """Load and minimally validate the OMIP raw dataset."""
    _check_file_exists(file_path)

    df = pd.read_csv(file_path)
    _validate_required_columns(df, REQUIRED_OMIP_COLUMNS, "OMIP raw data")

    df = _standardize_date_column(df, source_column="Date", dataset_name="OMIP raw data")
    df = df.sort_values("date").reset_index(drop=True)

    return df



def load_weather_data(file_path: Path = WEATHER_RAW_FILE) -> pd.DataFrame:
    """Load and minimally validate the weather raw dataset."""
    _check_file_exists(file_path)

    df = pd.read_csv(file_path)
    _validate_required_columns(df, REQUIRED_WEATHER_COLUMNS, "weather raw data")

    df = _standardize_date_column(df, source_column="date", dataset_name="weather raw data")
    df = df.sort_values("date").reset_index(drop=True)

    return df



def load_holidays_data(file_path: Path = HOLIDAYS_RAW_FILE) -> pd.DataFrame:
    """
    Load and minimally validate the holidays raw dataset.

    This loader assumes a holidays file exists and contains at least a `date` column.
    If your team has not created this file yet, you can add it later without changing
    the interface of the rest of the pipeline.
    """
    _check_file_exists(file_path)

    df = pd.read_csv(file_path)
    _validate_required_columns(df, REQUIRED_HOLIDAYS_COLUMNS, "holidays raw data")

    df = _standardize_date_column(df, source_column="date", dataset_name="holidays raw data")
    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================
# Public interface
# =========================

def load_raw_data() -> Dict[str, pd.DataFrame]:
    """
    Load all raw datasets and return them in a dictionary.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing:
        - 'omip'
        - 'weather'
        - 'holidays'
    """
    raw_data = {
        "omip": load_omip_data(),
        "weather": load_weather_data(),
        "holidays": load_holidays_data(),
    }

    return raw_data


if __name__ == "__main__":
    datasets = load_raw_data()
    for name, df in datasets.items():
        print(f"Loaded '{name}' with shape: {df.shape}")

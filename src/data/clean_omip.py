"""
clean_omip.py

Cleaning utilities for the OMIP / spot market dataset.
This module standardizes column types, validates the date index,
handles duplicates, and saves a clean interim dataset.
"""

from __future__ import annotations

import pandas as pd

from src.config.paths import OMIP_CLEAN_FILE
from src.data.load_raw_data import load_omip_data


NUMERIC_OMIP_COLUMNS = [
    "Spot_Price_SPEL",
    "Future_M1_Price",
    "Future_M1_OpenInterest",
    "Future_M2_Price",
    "Future_M2_OpenInterest",
    "Future_M3_Price",
    "Future_M3_OpenInterest",
    "Future_M4_Price",
    "Future_M4_OpenInterest",
    "Future_M5_Price",
    "Future_M5_OpenInterest",
    "Future_M6_Price",
    "Future_M6_OpenInterest",
]


class OmipCleaningError(Exception):
    """Raised when the OMIP dataset cannot be cleaned correctly."""


# =========================
# Validation helpers
# =========================

def _validate_required_columns(df: pd.DataFrame) -> None:
    """Ensure the minimum required columns are present after loading."""
    required_columns = ["date", "Spot_Price_SPEL"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise OmipCleaningError(
            f"Missing required columns after OMIP load: {missing_columns}"
        )



def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert known OMIP price and open-interest columns to numeric when present.

    Missing columns are ignored to keep the cleaner robust to partially available
    futures maturities in the raw file.
    """
    df = df.copy()

    for column in NUMERIC_OMIP_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df



def _drop_duplicate_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate dates while keeping the last occurrence.

    Keeping the last row is usually safer in time-series exports because later
    extractions tend to reflect the most updated version.
    """
    df = df.copy()
    duplicated_dates = df.duplicated(subset=["date"], keep="last")
    df = df.loc[~duplicated_dates].copy()
    return df



def _validate_clean_dataframe(df: pd.DataFrame) -> None:
    """Run final checks on the cleaned OMIP dataframe."""
    if df.empty:
        raise OmipCleaningError("OMIP cleaned dataframe is empty.")

    if df["date"].isna().any():
        raise OmipCleaningError("OMIP cleaned dataframe contains null dates.")

    if df["date"].duplicated().any():
        raise OmipCleaningError("OMIP cleaned dataframe still has duplicated dates.")

    if df["Spot_Price_SPEL"].isna().all():
        raise OmipCleaningError("Spot_Price_SPEL is completely missing after cleaning.")


# =========================
# Dataframe-level cleaning function
# =========================

def clean_omip_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean an already-loaded OMIP dataframe.

    This dataframe-level API is useful for tests and for callers that already
    have the raw OMIP data in memory.
    """
    if df.empty:
        raise OmipCleaningError("OMIP input dataframe is empty.")

    cleaned_df = df.copy()
    _validate_required_columns(cleaned_df)
    cleaned_df = _coerce_numeric_columns(cleaned_df)
    cleaned_df = cleaned_df.sort_values("date").reset_index(drop=True)
    cleaned_df = _drop_duplicate_dates(cleaned_df)
    _validate_clean_dataframe(cleaned_df)
    return cleaned_df


# =========================
# Public API
# =========================

def clean_omip_data(save: bool = True) -> pd.DataFrame:
    """
    Load, clean, validate, and optionally save the OMIP dataset.

    Cleaning steps:
    1. Load raw OMIP data
    2. Validate required columns
    3. Convert numeric columns
    4. Sort by date
    5. Remove duplicated dates
    6. Run final validation
    7. Save interim dataset if requested

    Parameters
    ----------
    save : bool, optional
        Whether to save the cleaned dataframe to `data/interim/omip_clean.csv`.

    Returns
    -------
    pd.DataFrame
        Cleaned OMIP dataframe.
    """
    df = load_omip_data()
    df = clean_omip_dataframe(df)

    if save:
        df.to_csv(OMIP_CLEAN_FILE, index=False)

    return df


if __name__ == "__main__":
    cleaned_df = clean_omip_data(save=True)
    print(f"OMIP cleaned successfully. Final shape: {cleaned_df.shape}")

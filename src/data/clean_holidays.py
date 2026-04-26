"""
clean_holidays.py

Cleaning utilities for the holidays dataset.
This module standardizes the date column, validates duplicates,
lightly normalizes holiday flags, and saves a clean interim dataset.
"""

from __future__ import annotations

import pandas as pd

from src.config.paths import HOLIDAYS_CLEAN_FILE, HOLIDAYS_RAW_FILE


POSSIBLE_HOLIDAY_FLAG_COLUMNS = [
    "Is_national_holiday",
    "is_national_holiday",
    "holiday_flag",
    "is_holiday",
]

POSSIBLE_HOLIDAY_NAME_COLUMNS = [
    "holiday_name",
    "name",
    "holiday",
]


class HolidaysCleaningError(Exception):
    """Raised when the holidays dataset cannot be cleaned correctly."""


# =========================
# Loading helpers
# =========================

def _load_holidays_raw() -> pd.DataFrame:
    """Load the raw holidays file and validate its existence."""
    if not HOLIDAYS_RAW_FILE.exists():
        raise HolidaysCleaningError(f"Holidays file not found: {HOLIDAYS_RAW_FILE}")

    df = pd.read_csv(HOLIDAYS_RAW_FILE)

    if df.empty:
        raise HolidaysCleaningError("Holidays raw dataframe is empty.")

    return df


# =========================
# Cleaning helpers
# =========================

def _standardize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the holidays dataframe has a valid standardized `date` column."""
    if "date" not in df.columns:
        raise HolidaysCleaningError("The holidays file must contain a 'date' column.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        invalid_count = int(df["date"].isna().sum())
        raise HolidaysCleaningError(
            f"Found {invalid_count} invalid date values in holidays data."
        )

    return df



def _normalize_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a standardized `Is_national_holiday` column when possible.

    Priority:
    1. Reuse an existing holiday-flag column if present.
    2. If no explicit flag exists, infer holiday rows as 1.
    """
    df = df.copy()

    existing_flag = next(
        (column for column in POSSIBLE_HOLIDAY_FLAG_COLUMNS if column in df.columns),
        None,
    )

    if existing_flag is not None:
        df["Is_national_holiday"] = pd.to_numeric(
            df[existing_flag], errors="coerce"
        ).fillna(0).round().astype("Int64")
    else:
        df["Is_national_holiday"] = 1
        df["Is_national_holiday"] = df["Is_national_holiday"].astype("Int64")

    return df



def _normalize_holiday_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a standardized `holiday_name` column if a compatible source column exists.

    If no holiday-name column exists, the dataset is still valid.
    """
    df = df.copy()

    existing_name = next(
        (column for column in POSSIBLE_HOLIDAY_NAME_COLUMNS if column in df.columns),
        None,
    )

    if existing_name is not None:
        df["holiday_name"] = df[existing_name].astype("string")

    return df


def _keep_only_holiday_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows marked as national holidays.

    This produces an interim holidays file that contains holiday dates only,
    with the holiday flag explicitly set to 1.
    """
    df = df.copy()
    if "Is_national_holiday" not in df.columns:
        raise HolidaysCleaningError(
            "Cannot filter holiday rows because 'Is_national_holiday' is missing."
        )

    df = df.loc[df["Is_national_holiday"] == 1].copy()

    if "is_holiday" in df.columns:
        df["is_holiday"] = 1

    df["Is_national_holiday"] = 1
    return df



def _drop_duplicate_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate dates while keeping the last occurrence.

    This keeps the most recent version if the raw file was appended or corrected.
    """
    df = df.copy()
    duplicated_dates = df.duplicated(subset=["date"], keep="last")
    df = df.loc[~duplicated_dates].copy()
    return df



def _validate_clean_dataframe(df: pd.DataFrame) -> None:
    """Run final checks on the cleaned holidays dataframe."""
    if df.empty:
        raise HolidaysCleaningError("Holidays cleaned dataframe is empty.")

    if "date" not in df.columns:
        raise HolidaysCleaningError("Holidays cleaned dataframe has no 'date' column.")

    if df["date"].isna().any():
        raise HolidaysCleaningError("Holidays cleaned dataframe contains null dates.")

    if df["date"].duplicated().any():
        raise HolidaysCleaningError("Holidays cleaned dataframe still has duplicated dates.")

    if "Is_national_holiday" not in df.columns:
        raise HolidaysCleaningError(
            "Holidays cleaned dataframe has no 'Is_national_holiday' column."
        )


# =========================
# Public API
# =========================

def clean_holidays_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean an already-loaded holidays dataframe.

    This dataframe-level API is useful for tests and for callers that already
    have the raw holidays data in memory.
    """
    if df.empty:
        raise HolidaysCleaningError("Holidays input dataframe is empty.")

    cleaned_df = df.copy()
    cleaned_df = _standardize_date_column(cleaned_df)
    cleaned_df = _normalize_holiday_flag(cleaned_df)
    cleaned_df = _normalize_holiday_name(cleaned_df)
    cleaned_df = _keep_only_holiday_rows(cleaned_df)
    cleaned_df = cleaned_df.sort_values("date").reset_index(drop=True)
    cleaned_df = _drop_duplicate_dates(cleaned_df)
    _validate_clean_dataframe(cleaned_df)
    return cleaned_df

def clean_holidays_data(save: bool = True) -> pd.DataFrame:
    """
    Load, clean, validate, and optionally save the holidays dataset.

    Cleaning steps:
    1. Load raw holidays data
    2. Standardize the date column
    3. Normalize holiday flag column
    4. Normalize holiday name column if available
    5. Sort by date
    6. Remove duplicated dates
    7. Validate final dataframe
    8. Save interim dataset if requested

    Parameters
    ----------
    save : bool, optional
        Whether to save the cleaned dataframe to `data/interim/holidays_clean.csv`.

    Returns
    -------
    pd.DataFrame
        Cleaned holidays dataframe.
    """
    df = _load_holidays_raw()
    df = clean_holidays_dataframe(df)

    if save:
        df.to_csv(HOLIDAYS_CLEAN_FILE, index=False)

    return df


if __name__ == "__main__":
    cleaned_df = clean_holidays_data(save=True)
    print(f"Holidays cleaned successfully. Final shape: {cleaned_df.shape}")

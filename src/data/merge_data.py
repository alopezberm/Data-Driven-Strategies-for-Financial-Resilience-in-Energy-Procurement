"""
merge_data.py

Merge cleaned OMIP, weather, and holidays datasets into a single
interim modeling table indexed by date.
"""

from __future__ import annotations

import pandas as pd

from src.config.paths import HOLIDAYS_RAW_FILE, MERGED_INTERIM_FILE
from src.data.clean_omip import clean_omip_data
from src.data.clean_weather import clean_weather_data


class MergeDataError(Exception):
    """Raised when cleaned datasets cannot be merged safely."""


# =========================
# Holidays loader / cleaner
# =========================

def _load_holidays_raw() -> pd.DataFrame:
    """
    Load the holidays raw file and standardize its date column.

    Assumptions:
    - The file exists.
    - It contains at least a `date` column.
    - Any additional columns are preserved.
    """
    if not HOLIDAYS_RAW_FILE.exists():
        raise MergeDataError(f"Holidays file not found: {HOLIDAYS_RAW_FILE}")

    holidays_df = pd.read_csv(HOLIDAYS_RAW_FILE)

    if "date" not in holidays_df.columns:
        raise MergeDataError("The holidays file must contain a 'date' column.")

    holidays_df = holidays_df.copy()
    holidays_df["date"] = pd.to_datetime(holidays_df["date"], errors="coerce")

    if holidays_df["date"].isna().any():
        invalid_count = int(holidays_df["date"].isna().sum())
        raise MergeDataError(
            f"Found {invalid_count} invalid date values in holidays raw data."
        )

    holidays_df = holidays_df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    holidays_df = holidays_df.reset_index(drop=True)

    return holidays_df


# =========================
# Validation helpers
# =========================

def _validate_clean_inputs(omip_df: pd.DataFrame, weather_df: pd.DataFrame, holidays_df: pd.DataFrame) -> None:
    """Validate that cleaned inputs are ready to be merged."""
    datasets = {
        "omip": omip_df,
        "weather": weather_df,
        "holidays": holidays_df,
    }

    for name, df in datasets.items():
        if df.empty:
            raise MergeDataError(f"{name} dataframe is empty.")
        if "date" not in df.columns:
            raise MergeDataError(f"{name} dataframe does not contain a 'date' column.")
        if df["date"].isna().any():
            raise MergeDataError(f"{name} dataframe contains null dates.")
        if df["date"].duplicated().any():
            raise MergeDataError(f"{name} dataframe contains duplicated dates.")



def _validate_merged_dataframe(df: pd.DataFrame) -> None:
    """Validate the merged dataframe after joins."""
    if df.empty:
        raise MergeDataError("Merged dataframe is empty.")

    if "date" not in df.columns:
        raise MergeDataError("Merged dataframe does not contain a 'date' column.")

    if df["date"].duplicated().any():
        raise MergeDataError("Merged dataframe contains duplicated dates.")

    required_columns = ["Spot_Price_SPEL"]
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise MergeDataError(
            f"Merged dataframe is missing required columns: {missing_required}"
        )

    if df["Spot_Price_SPEL"].isna().all():
        raise MergeDataError("Spot_Price_SPEL is completely missing after merging.")


# =========================
# Merge logic
# =========================

def merge_clean_data(save: bool = True) -> pd.DataFrame:
    """
    Merge OMIP, weather, and holidays datasets into a single interim table.

    Merge policy:
    - OMIP is used as the left/base table because the project is centered on
      electricity price and hedging decisions.
    - Weather is merged on date using a left join.
    - Holidays is merged on date using a left join.

    Parameters
    ----------
    save : bool, optional
        Whether to save the merged dataframe to `data/interim/merged_interim.csv`.

    Returns
    -------
    pd.DataFrame
        Merged dataframe.
    """
    omip_df = clean_omip_data(save=True)
    weather_df = clean_weather_data(save=True)
    holidays_df = _load_holidays_raw()

    _validate_clean_inputs(omip_df, weather_df, holidays_df)

    merged_df = omip_df.merge(
        weather_df,
        on="date",
        how="left",
        suffixes=("", "_weather"),
        validate="one_to_one",
    )

    merged_df = merged_df.merge(
        holidays_df,
        on="date",
        how="left",
        suffixes=("", "_holiday"),
        validate="one_to_one",
    )

    merged_df = merged_df.sort_values("date").reset_index(drop=True)

    # Light post-merge normalization:
    # if holidays file contains an explicit holiday flag, cast it consistently.
    for column in ["Is_national_holiday", "is_national_holiday", "holiday_flag"]:
        if column in merged_df.columns:
            merged_df[column] = pd.to_numeric(merged_df[column], errors="coerce").round().astype("Int64")

    _validate_merged_dataframe(merged_df)

    if save:
        merged_df.to_csv(MERGED_INTERIM_FILE, index=False)

    return merged_df


if __name__ == "__main__":
    merged = merge_clean_data(save=True)
    print(f"Merged dataset created successfully. Final shape: {merged.shape}")
"""
merge_data.py

Merge cleaned OMIP, weather, and holidays datasets into a single
interim modeling table indexed by date.
"""

from __future__ import annotations

import pandas as pd

from src.config.constants import DATE_COLUMN, SPOT_PRICE_COLUMN
from src.config.paths import (
    DAILY_OPERATIONS_ASSUMPTIONS_FILE,
    HOLIDAYS_RAW_FILE,
    MERGED_INTERIM_FILE,
)
from src.data.clean_omip import clean_omip_data
from src.data.clean_weather import clean_weather_data


class MergeDataError(Exception):
    """Raised when cleaned datasets cannot be merged safely."""


# =========================
# Holidays loader / cleaner
# =========================

DAILY_OPERATIONS_REQUIRED_COLUMNS = {
    DATE_COLUMN,
    "units_needed_per_day",
    "energy_per_unit_mwh",
    "daily_energy_needed_mwh",
    "inventory_holding_cost_per_unit_eur_day",
    "max_capacity_units_per_day",
    "max_capacity_mwh_per_day",
}

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

    if DATE_COLUMN not in holidays_df.columns:
        raise MergeDataError(f"The holidays file must contain a '{DATE_COLUMN}' column.")

    holidays_df = holidays_df.copy()
    holidays_df[DATE_COLUMN] = pd.to_datetime(holidays_df[DATE_COLUMN], errors="coerce")

    if holidays_df[DATE_COLUMN].isna().any():
        invalid_count = int(holidays_df[DATE_COLUMN].isna().sum())
        raise MergeDataError(
            f"Found {invalid_count} invalid date values in holidays raw data."
        )

    holidays_df = holidays_df.sort_values(DATE_COLUMN).drop_duplicates(subset=[DATE_COLUMN], keep="last")
    holidays_df = holidays_df.reset_index(drop=True)

    return holidays_df


def _load_daily_operations_assumptions() -> pd.DataFrame:
    """
    Load daily operational assumptions and standardize the date column.

    Expected source:
    - data/external/daily_operations_assumptions.csv
    """
    if not DAILY_OPERATIONS_ASSUMPTIONS_FILE.exists():
        raise MergeDataError(
            f"Daily operations assumptions file not found: {DAILY_OPERATIONS_ASSUMPTIONS_FILE}"
        )

    operations_df = pd.read_csv(DAILY_OPERATIONS_ASSUMPTIONS_FILE)

    missing_columns = DAILY_OPERATIONS_REQUIRED_COLUMNS - set(operations_df.columns)
    if missing_columns:
        raise MergeDataError(
            "Daily operations assumptions file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    operations_df = operations_df.copy()
    operations_df[DATE_COLUMN] = pd.to_datetime(operations_df[DATE_COLUMN], errors="coerce")

    if operations_df[DATE_COLUMN].isna().any():
        invalid_count = int(operations_df[DATE_COLUMN].isna().sum())
        raise MergeDataError(
            f"Found {invalid_count} invalid date values in daily operations assumptions."
        )

    if operations_df[DATE_COLUMN].duplicated().any():
        raise MergeDataError("Daily operations assumptions contain duplicated dates.")

    return operations_df.sort_values(DATE_COLUMN).reset_index(drop=True)


# =========================
# Validation helpers
# =========================

def _validate_clean_inputs(
    omip_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    operations_df: pd.DataFrame,
) -> None:
    """Validate that cleaned inputs are ready to be merged."""
    datasets = {
        "omip": omip_df,
        "weather": weather_df,
        "holidays": holidays_df,
        "daily_operations": operations_df,
    }

    for name, df in datasets.items():
        if df.empty:
            raise MergeDataError(f"{name} dataframe is empty.")
        if DATE_COLUMN not in df.columns:
            raise MergeDataError(f"{name} dataframe does not contain a '{DATE_COLUMN}' column.")
        if df[DATE_COLUMN].isna().any():
            raise MergeDataError(f"{name} dataframe contains null dates.")
        if df[DATE_COLUMN].duplicated().any():
            raise MergeDataError(f"{name} dataframe contains duplicated dates.")



def _validate_merged_dataframe(df: pd.DataFrame) -> None:
    """Validate the merged dataframe after joins."""
    if df.empty:
        raise MergeDataError("Merged dataframe is empty.")

    if DATE_COLUMN not in df.columns:
        raise MergeDataError(f"Merged dataframe does not contain a '{DATE_COLUMN}' column.")

    if df[DATE_COLUMN].duplicated().any():
        raise MergeDataError("Merged dataframe contains duplicated dates.")

    required_columns = [SPOT_PRICE_COLUMN]
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise MergeDataError(
            f"Merged dataframe is missing required columns: {missing_required}"
        )

    if df[SPOT_PRICE_COLUMN].isna().all():
        raise MergeDataError(f"{SPOT_PRICE_COLUMN} is completely missing after merging.")


# =========================
# Merge logic
# =========================

def merge_datasets(
    omip_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    holidays_df: pd.DataFrame | None = None,
    daily_operations_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge already-cleaned input dataframes into a single interim modeling table.

    This dataframe-level API is useful for tests and for callers that already
    have the cleaned datasets in memory.
    """
    if omip_df.empty:
        raise MergeDataError("OMIP dataframe is empty.")
    if weather_df.empty:
        raise MergeDataError("Weather dataframe is empty.")

    merged_df = omip_df.copy()
    merged_df[DATE_COLUMN] = pd.to_datetime(merged_df[DATE_COLUMN], errors="coerce")

    weather_df = weather_df.copy()
    weather_df[DATE_COLUMN] = pd.to_datetime(weather_df[DATE_COLUMN], errors="coerce")

    merged_df = merged_df.merge(
        weather_df,
        on=DATE_COLUMN,
        how="left",
        suffixes=("", "_weather"),
        validate="one_to_one",
    )

    if holidays_df is not None:
        if holidays_df.empty:
            raise MergeDataError("Holidays dataframe is empty.")
        holidays_df = holidays_df.copy()
        holidays_df[DATE_COLUMN] = pd.to_datetime(holidays_df[DATE_COLUMN], errors="coerce")

        merged_df = merged_df.merge(
            holidays_df,
            on=DATE_COLUMN,
            how="left",
            suffixes=("", "_holiday"),
            validate="one_to_one",
        )

    if daily_operations_df is not None:
        if daily_operations_df.empty:
            raise MergeDataError("Daily operations dataframe is empty.")

        daily_operations_df = daily_operations_df.copy()
        daily_operations_df[DATE_COLUMN] = pd.to_datetime(
            daily_operations_df[DATE_COLUMN], errors="coerce"
        )

        merged_df = merged_df.merge(
            daily_operations_df,
            on=DATE_COLUMN,
            how="left",
            suffixes=("", "_ops"),
            validate="one_to_one",
        )

    if merged_df[DATE_COLUMN].isna().any():
        raise MergeDataError("Merged dataframe contains invalid dates.")

    merged_df = merged_df.sort_values(DATE_COLUMN).reset_index(drop=True)

    for column in ["Is_national_holiday", "is_national_holiday", "holiday_flag"]:
        if column in merged_df.columns:
            merged_df[column] = pd.to_numeric(
                merged_df[column], errors="coerce"
            ).round().astype("Int64")

    _validate_merged_dataframe(merged_df)
    return merged_df


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
    operations_df = _load_daily_operations_assumptions()

    _validate_clean_inputs(omip_df, weather_df, holidays_df, operations_df)
    merged_df = merge_datasets(
        omip_df,
        weather_df,
        holidays_df,
        daily_operations_df=operations_df,
    )

    if save:
        merged_df.to_csv(MERGED_INTERIM_FILE, index=False)

    return merged_df


if __name__ == "__main__":
    merged = merge_clean_data(save=True)
    print(f"Merged dataset created successfully. Final shape: {merged.shape}")
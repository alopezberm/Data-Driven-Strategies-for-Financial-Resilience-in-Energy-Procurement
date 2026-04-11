"""
split_data.py

Create chronological train / validation / test splits from the merged interim
modeling dataset.
"""

from __future__ import annotations

import pandas as pd

from src.config.paths import MERGED_INTERIM_FILE, TRAIN_FILE, VALIDATION_FILE, TEST_FILE


DEFAULT_TRAIN_END = "2023-12-31"
DEFAULT_VALIDATION_END = "2024-12-31"


class SplitDataError(Exception):
    """Raised when the merged dataset cannot be split safely."""


# =========================
# Helpers
# =========================

def _load_merged_data() -> pd.DataFrame:
    """Load the merged interim dataset and validate file existence."""
    if not MERGED_INTERIM_FILE.exists():
        raise SplitDataError(
            f"Merged interim file not found: {MERGED_INTERIM_FILE}. "
            "Run merge_data.py first."
        )

    df = pd.read_csv(MERGED_INTERIM_FILE)

    if "date" not in df.columns:
        raise SplitDataError("Merged interim dataset must contain a 'date' column.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        invalid_count = int(df["date"].isna().sum())
        raise SplitDataError(
            f"Found {invalid_count} invalid date values in merged interim dataset."
        )

    return df



def _validate_merged_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and sort the merged dataframe before splitting."""
    if df.empty:
        raise SplitDataError("Merged interim dataframe is empty.")

    if df["date"].duplicated().any():
        raise SplitDataError("Merged interim dataframe contains duplicated dates.")

    df = df.sort_values("date").reset_index(drop=True)

    return df



def _validate_split_boundaries(df: pd.DataFrame, train_end: str, validation_end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate chronological split boundaries against dataset coverage."""
    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)

    if validation_end_ts <= train_end_ts:
        raise SplitDataError("validation_end must be strictly later than train_end.")

    min_date = df["date"].min()
    max_date = df["date"].max()

    if train_end_ts < min_date:
        raise SplitDataError(
            f"train_end ({train_end_ts.date()}) is earlier than the dataset start ({min_date.date()})."
        )

    if validation_end_ts < min_date:
        raise SplitDataError(
            f"validation_end ({validation_end_ts.date()}) is earlier than the dataset start ({min_date.date()})."
        )

    if train_end_ts >= max_date:
        raise SplitDataError(
            f"train_end ({train_end_ts.date()}) leaves no data for validation/test. "
            f"Dataset ends at {max_date.date()}."
        )

    if validation_end_ts >= max_date:
        raise SplitDataError(
            f"validation_end ({validation_end_ts.date()}) leaves no data for test. "
            f"Dataset ends at {max_date.date()}."
        )

    return train_end_ts, validation_end_ts



def _split_by_dates(
    df: pd.DataFrame,
    train_end_ts: pd.Timestamp,
    validation_end_ts: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform the chronological split."""
    train_df = df.loc[df["date"] <= train_end_ts].copy()
    validation_df = df.loc[(df["date"] > train_end_ts) & (df["date"] <= validation_end_ts)].copy()
    test_df = df.loc[df["date"] > validation_end_ts].copy()

    return train_df, validation_df, test_df



def _validate_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Validate that all three splits are non-empty and chronologically correct."""
    splits = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    for split_name, split_df in splits.items():
        if split_df.empty:
            raise SplitDataError(f"The {split_name} split is empty.")
        if split_df["date"].duplicated().any():
            raise SplitDataError(f"The {split_name} split contains duplicated dates.")

    if train_df["date"].max() >= validation_df["date"].min():
        raise SplitDataError("Train and validation splits overlap or are not strictly ordered.")

    if validation_df["date"].max() >= test_df["date"].min():
        raise SplitDataError("Validation and test splits overlap or are not strictly ordered.")



def _save_splits(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save split datasets to the processed data directory."""
    train_df.to_csv(TRAIN_FILE, index=False)
    validation_df.to_csv(VALIDATION_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)


# =========================
# Public API
# =========================

def split_data(
    train_end: str = DEFAULT_TRAIN_END,
    validation_end: str = DEFAULT_VALIDATION_END,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the merged interim dataset into train, validation, and test sets.

    Parameters
    ----------
    train_end : str, optional
        Final date to include in the training split.
    validation_end : str, optional
        Final date to include in the validation split.
    save : bool, optional
        Whether to save the resulting splits to the processed data directory.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, validation_df, test_df)
    """
    df = _load_merged_data()
    df = _validate_merged_dataframe(df)
    train_end_ts, validation_end_ts = _validate_split_boundaries(df, train_end, validation_end)

    train_df, validation_df, test_df = _split_by_dates(df, train_end_ts, validation_end_ts)
    _validate_splits(train_df, validation_df, test_df)

    if save:
        _save_splits(train_df, validation_df, test_df)

    return train_df, validation_df, test_df


if __name__ == "__main__":
    train_df, validation_df, test_df = split_data(save=True)

    print("Chronological split completed successfully.")
    print(f"Train: {train_df.shape} | {train_df['date'].min().date()} -> {train_df['date'].max().date()}")
    print(
        f"Validation: {validation_df.shape} | "
        f"{validation_df['date'].min().date()} -> {validation_df['date'].max().date()}"
    )
    print(f"Test: {test_df.shape} | {test_df['date'].min().date()} -> {test_df['date'].max().date()}")

"""
split_data.py

Create chronological train / validation / test splits from the merged interim
modeling dataset.
"""

from __future__ import annotations

import pandas as pd

from src.config.constants import DATE_COLUMN
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

    if DATE_COLUMN not in df.columns:
        raise SplitDataError(
            f"Merged interim dataset must contain a '{DATE_COLUMN}' column."
        )

    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    if df[DATE_COLUMN].isna().any():
        invalid_count = int(df[DATE_COLUMN].isna().sum())
        raise SplitDataError(
            f"Found {invalid_count} invalid date values in merged interim dataset."
        )

    return df



def _validate_merged_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and sort the merged dataframe before splitting."""
    if df.empty:
        raise SplitDataError("Merged interim dataframe is empty.")

    if df[DATE_COLUMN].duplicated().any():
        raise SplitDataError("Merged interim dataframe contains duplicated dates.")

    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    return df



def _validate_split_boundaries(df: pd.DataFrame, train_end: str, validation_end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate chronological split boundaries against dataset coverage."""
    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)

    if validation_end_ts <= train_end_ts:
        raise SplitDataError("validation_end must be strictly later than train_end.")

    min_date = df[DATE_COLUMN].min()
    max_date = df[DATE_COLUMN].max()

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
    train_df = df.loc[df[DATE_COLUMN] <= train_end_ts].copy()
    validation_df = df.loc[(df[DATE_COLUMN] > train_end_ts) & (df[DATE_COLUMN] <= validation_end_ts)].copy()
    test_df = df.loc[df[DATE_COLUMN] > validation_end_ts].copy()

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
        if split_df[DATE_COLUMN].duplicated().any():
            raise SplitDataError(f"The {split_name} split contains duplicated dates.")

    if train_df[DATE_COLUMN].max() >= validation_df[DATE_COLUMN].min():
        raise SplitDataError("Train and validation splits overlap or are not strictly ordered.")

    if validation_df[DATE_COLUMN].max() >= test_df[DATE_COLUMN].min():
        raise SplitDataError("Validation and test splits overlap or are not strictly ordered.")



def _save_splits(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save split datasets to the processed data directory."""
    train_df.to_csv(TRAIN_FILE, index=False)
    validation_df.to_csv(VALIDATION_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)



# =========================
# Public API
# =========================


# Dataframe-level split function for tests and in-memory use
def chronological_train_validation_test_split(
    df: pd.DataFrame,
    train_end_date: str | None = None,
    validation_end_date: str | None = None,
    date_column: str = DATE_COLUMN,
    train_ratio: float | None = None,
    validation_ratio: float | None = None,
    test_ratio: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split an already-loaded dataframe chronologically into train, validation, and test sets.

    Supports either:
    - explicit date boundaries, or
    - ratio-based splitting.
    """
    if df.empty:
        raise SplitDataError("Input dataframe is empty.")

    if date_column not in df.columns:
        raise SplitDataError(
            f"Input dataframe must contain a '{date_column}' column."
        )

    result_df = df.copy()
    result_df[date_column] = pd.to_datetime(result_df[date_column], errors="coerce")

    if result_df[date_column].isna().any():
        invalid_count = int(result_df[date_column].isna().sum())
        raise SplitDataError(
            f"Found {invalid_count} invalid date values in split input dataframe."
        )

    if result_df[date_column].duplicated().any():
        raise SplitDataError("Input dataframe contains duplicated dates.")

    result_df = result_df.sort_values(date_column).reset_index(drop=True)

    # Mode 1: date-based split
    if train_end_date is not None and validation_end_date is not None:
        train_end_ts = pd.Timestamp(train_end_date)
        validation_end_ts = pd.Timestamp(validation_end_date)

        if train_end_ts >= validation_end_ts:
            raise SplitDataError(
                "train_end_date must be strictly earlier than validation_end_date."
            )

        train_df = result_df.loc[result_df[date_column] <= train_end_ts].copy()
        validation_df = result_df.loc[
            (result_df[date_column] > train_end_ts)
            & (result_df[date_column] <= validation_end_ts)
        ].copy()
        test_df = result_df.loc[result_df[date_column] > validation_end_ts].copy()

    # Mode 2: ratio-based split
    elif (
        train_ratio is not None
        and validation_ratio is not None
        and test_ratio is not None
    ):
        ratio_sum = train_ratio + validation_ratio + test_ratio
        if abs(ratio_sum - 1.0) > 1e-9:
            raise SplitDataError("train_ratio + validation_ratio + test_ratio must equal 1.0.")

        n = len(result_df)
        train_end_idx = int(n * train_ratio)
        validation_end_idx = train_end_idx + int(n * validation_ratio)

        train_df = result_df.iloc[:train_end_idx].copy()
        validation_df = result_df.iloc[train_end_idx:validation_end_idx].copy()
        test_df = result_df.iloc[validation_end_idx:].copy()

    else:
        raise SplitDataError(
            "Provide either (train_end_date and validation_end_date) "
            "or (train_ratio, validation_ratio, test_ratio)."
        )

    if train_df.empty or validation_df.empty or test_df.empty:
        raise SplitDataError(
            "Chronological split produced an empty train, validation, or test set."
        )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

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
    _validate_split_boundaries(df, train_end, validation_end)

    train_df, validation_df, test_df = chronological_train_validation_test_split(
        df,
        train_end_date=train_end,
        validation_end_date=validation_end,
        date_column=DATE_COLUMN,
    )
    _validate_splits(train_df, validation_df, test_df)

    if save:
        _save_splits(train_df, validation_df, test_df)

    return train_df, validation_df, test_df


if __name__ == "__main__":
    train_df, validation_df, test_df = split_data(save=True)

    print("Chronological split completed successfully.")
    print(f"Train: {train_df.shape} | {train_df[DATE_COLUMN].min().date()} -> {train_df[DATE_COLUMN].max().date()}")
    print(
        f"Validation: {validation_df.shape} | "
        f"{validation_df[DATE_COLUMN].min().date()} -> {validation_df[DATE_COLUMN].max().date()}"
    )
    print(f"Test: {test_df.shape} | {test_df[DATE_COLUMN].min().date()} -> {test_df[DATE_COLUMN].max().date()}")

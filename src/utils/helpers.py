"""
helpers.py

General-purpose helper utilities used across the project.
These are small reusable functions that don't belong to a specific module.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from src.config.constants import DATE_COLUMN
from src.utils.logger import get_logger


class HelperError(Exception):
    """Raised when a helper utility fails."""


logger = get_logger(__name__)


# =========================
# General utilities
# =========================

def ensure_list(obj: Any) -> list:
    """
    Ensure an object is returned as a list.

    Examples
    --------
    ensure_list("a") -> ["a"]
    ensure_list(["a", "b"]) -> ["a", "b"]
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, (tuple, set)):
        return list(obj)
    return [obj]



def flatten_list(nested: Iterable[Iterable[Any]]) -> list:
    """Flatten a nested iterable into a single list."""
    result: list[Any] = []
    for sub in nested:
        if isinstance(sub, (list, tuple, set)):
            result.extend(sub)
        else:
            result.append(sub)
    return result


# =========================
# DataFrame helpers
# =========================

def safe_copy(df: pd.DataFrame) -> pd.DataFrame:
    """Return a safe copy of a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise HelperError("Input must be a pandas DataFrame.")
    return df.copy()



def sort_by_date(df: pd.DataFrame, date_column: str = DATE_COLUMN) -> pd.DataFrame:
    """
    Sort a DataFrame by a date column.

    Ensures consistent ordering across the pipeline.
    """
    if date_column not in df.columns:
        raise HelperError(f"Column '{date_column}' not found in DataFrame.")

    result = df.copy()
    result[date_column] = pd.to_datetime(result[date_column], errors="coerce")

    if result[date_column].isna().any():
        raise HelperError(f"Invalid dates found in column '{date_column}'.")

    return result.sort_values(date_column).reset_index(drop=True)



def align_on_index(*dfs: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Align multiple DataFrames on their common index intersection.

    Returns
    -------
    list[pd.DataFrame]
        List of aligned DataFrames.
    """
    if not dfs:
        raise HelperError("No DataFrames provided for alignment.")

    for df in dfs:
        if not isinstance(df, pd.DataFrame):
            raise HelperError("All inputs must be pandas DataFrames.")

    common_index = dfs[0].index
    for df in dfs[1:]:
        common_index = common_index.intersection(df.index)

    return [df.loc[common_index].copy() for df in dfs]



def add_prefix_to_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Add a prefix to all column names except the configured date column."""
    if not isinstance(df, pd.DataFrame):
        raise HelperError("Input must be a pandas DataFrame.")

    if any(col.startswith(prefix) for col in df.columns if col != DATE_COLUMN):
        raise HelperError("Some columns already contain the requested prefix.")

    result = df.copy()
    result.columns = [
        col if col == DATE_COLUMN else f"{prefix}{col}"
        for col in result.columns
    ]
    return result


# =========================
# Debug / inspection helpers
# =========================

def print_df_info(df: pd.DataFrame, name: str = "dataframe") -> None:
    """Log basic info about a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise HelperError("Input must be a pandas DataFrame.")

    logger.info(f"=== {name.upper()} ===")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {len(df.columns)}")
    logger.info("Head:")
    logger.info(f"\n{df.head()}")


if __name__ == "__main__":
    # Simple smoke test
    df = pd.DataFrame(
        {
            DATE_COLUMN: ["2024-01-02", "2024-01-01"],
            "value": [2, 1],
        }
    )

    df_sorted = sort_by_date(df)
    print_df_info(df_sorted, "sorted_df")
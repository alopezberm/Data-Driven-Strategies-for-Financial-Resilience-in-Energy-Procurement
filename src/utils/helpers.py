

"""
helpers.py

General-purpose helper utilities used across the project.
These are small reusable functions that don't belong to a specific module.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


class HelperError(Exception):
    """Raised when a helper utility fails."""


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
    if isinstance(obj, tuple) or isinstance(obj, set):
        return list(obj)
    return [obj]



def flatten_list(nested: Iterable[Iterable[Any]]) -> list:
    """Flatten a nested iterable into a single list."""
    return [item for sublist in nested for item in sublist]


# =========================
# DataFrame helpers
# =========================

def safe_copy(df: pd.DataFrame) -> pd.DataFrame:
    """Return a safe copy of a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise HelperError("Input must be a pandas DataFrame.")
    return df.copy()



def sort_by_date(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
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

    common_index = dfs[0].index
    for df in dfs[1:]:
        common_index = common_index.intersection(df.index)

    return [df.loc[common_index].copy() for df in dfs]



def add_prefix_to_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Add a prefix to all column names except 'date'."""
    result = df.copy()
    result.columns = [
        col if col == "date" else f"{prefix}{col}"
        for col in result.columns
    ]
    return result


# =========================
# Debug / inspection helpers
# =========================

def print_df_info(df: pd.DataFrame, name: str = "dataframe") -> None:
    """Print basic info about a DataFrame."""
    print(f"\n=== {name.upper()} ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print("Head:")
    print(df.head())


if __name__ == "__main__":
    # Simple smoke test
    df = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-01"],
            "value": [2, 1],
        }
    )

    df_sorted = sort_by_date(df)
    print_df_info(df_sorted, "sorted_df")
"""
validation.py

Reusable validation helpers used across the project.
These utilities reduce repetition and make error messages more consistent.
"""

from __future__ import annotations

import pandas as pd

from src.config.constants import DATE_COLUMN
from src.utils.logger import get_logger


class ValidationError(Exception):
    """Raised when a reusable validation check fails."""


logger = get_logger(__name__)


def check_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    df_name: str = "dataframe",
) -> None:
    """
    Validate that a dataframe contains all required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate.
    required_columns : list[str]
        Required column names.
    df_name : str, optional
        Human-readable dataframe name for clearer errors.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"{df_name} must be a pandas DataFrame.")

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValidationError(f"Missing columns in {df_name}: {missing_columns}")


def check_non_empty_dataframe(df: pd.DataFrame, df_name: str = "dataframe") -> None:
    """Validate that a dataframe is not empty."""
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"{df_name} must be a pandas DataFrame.")

    if df.empty:
        raise ValidationError(f"{df_name} is empty.")


def check_unique_dates(
    df: pd.DataFrame,
    date_column: str = DATE_COLUMN,
    df_name: str = "dataframe",
) -> None:
    """Validate that a date column contains unique values."""
    check_required_columns(df, [date_column], df_name=df_name)

    if df[date_column].duplicated().any():
        raise ValidationError(
            f"{df_name} contains duplicated values in '{date_column}'."
        )


def check_parseable_dates(
    df: pd.DataFrame,
    date_column: str = DATE_COLUMN,
    df_name: str = "dataframe",
) -> pd.DataFrame:
    """
    Validate and parse a date column.

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with a parsed datetime column.
    """
    check_required_columns(df, [date_column], df_name=df_name)

    result_df = df.copy()
    result_df[date_column] = pd.to_datetime(result_df[date_column], errors="coerce")

    if result_df[date_column].isna().any():
        invalid_count = int(result_df[date_column].isna().sum())
        raise ValidationError(
            f"{df_name} contains {invalid_count} invalid values in '{date_column}'."
        )

    return result_df


def check_numeric_columns(
    df: pd.DataFrame,
    columns: list[str],
    df_name: str = "dataframe",
) -> pd.DataFrame:
    """
    Coerce selected columns to numeric and validate that conversion succeeded.

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with numeric columns coerced.
    """
    check_required_columns(df, columns, df_name=df_name)

    result_df = df.copy()
    for column in columns:
        original_non_null = result_df[column].notna()
        result_df[column] = pd.to_numeric(result_df[column], errors="coerce")

        if result_df[column].isna().all():
            raise ValidationError(
                f"Column '{column}' in {df_name} could not be converted to numeric values."
            )

        introduced_nans = original_non_null & result_df[column].isna()
        if introduced_nans.any():
            invalid_count = int(introduced_nans.sum())
            raise ValidationError(
                f"Column '{column}' in {df_name} contains {invalid_count} non-numeric values that could not be converted."
            )

    return result_df


def validate_and_sort_by_date(
    df: pd.DataFrame,
    date_column: str = DATE_COLUMN,
    df_name: str = "dataframe",
) -> pd.DataFrame:
    """Validate, parse, ensure uniqueness, and sort a dataframe by its date column."""
    result_df = check_parseable_dates(df, date_column=date_column, df_name=df_name)
    check_unique_dates(result_df, date_column=date_column, df_name=df_name)
    return result_df.sort_values(date_column).reset_index(drop=True)


if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            DATE_COLUMN: ["2024-01-01", "2024-01-02"],
            "value": [1.0, 2.0],
        }
    )

    checked_df = validate_and_sort_by_date(example_df, df_name="example_df")
    check_non_empty_dataframe(checked_df, df_name="example_df")
    check_required_columns(checked_df, [DATE_COLUMN, "value"], df_name="example_df")
    logger.info("Validation helpers executed successfully.")
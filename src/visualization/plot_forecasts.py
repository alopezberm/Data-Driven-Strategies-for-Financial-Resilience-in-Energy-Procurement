"""
plot_forecasts.py

Visualization utilities for point and forecast-style comparisons.
These plots are useful for showing how model predictions track the realized
spot price over time.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config.constants import DATE_COLUMN
from src.config.paths import FIGURES_DIR


class ForecastPlotError(Exception):
    """Raised when forecast plots cannot be generated safely."""


REQUIRED_COLUMNS = {DATE_COLUMN, "y_true"}


# =========================
# Validation helpers
# =========================

def _prepare_output_path(filename: str | None) -> Path | None:
    """Prepare an output path inside the configured figures directory."""
    if filename is None:
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / filename



def _validate_forecast_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a forecast dataframe and standardize its date column when present."""
    if df.empty:
        raise ForecastPlotError("Forecast dataframe is empty.")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ForecastPlotError(
            f"Forecast dataframe is missing required columns: {sorted(missing_columns)}"
        )

    result_df = df.copy()

    if DATE_COLUMN in result_df.columns:
        result_df[DATE_COLUMN] = pd.to_datetime(result_df[DATE_COLUMN], errors="coerce")
        if result_df[DATE_COLUMN].isna().any():
            invalid_count = int(result_df[DATE_COLUMN].isna().sum())
            raise ForecastPlotError(
                f"Found {invalid_count} invalid date values in forecast dataframe."
            )
        if result_df[DATE_COLUMN].duplicated().any():
            raise ForecastPlotError("Forecast dataframe contains duplicated dates.")
        result_df = result_df.sort_values(DATE_COLUMN).reset_index(drop=True)

    return result_df


# =========================
# Plot functions
# =========================

def plot_actual_vs_forecast(
    forecast_df: pd.DataFrame,
    forecast_column: str,
    title: str = "Actual vs Forecast",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot realized values against one forecast series.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Dataframe containing at least the configured date column, `y_true`, and the selected forecast column.
    forecast_column : str
        Column name for the forecast to plot.
    """
    df = _validate_forecast_df(forecast_df)

    if forecast_column not in df.columns:
        raise ForecastPlotError(
            f"Forecast column '{forecast_column}' not found in forecast dataframe."
        )

    x_axis = df[DATE_COLUMN] if DATE_COLUMN in df.columns else df.index

    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, df["y_true"], label="y_true")
    plt.plot(x_axis, df[forecast_column], label=forecast_column)
    plt.title(title)
    plt.xlabel("Date" if DATE_COLUMN in df.columns else "Index")
    plt.ylabel("Target Value")
    plt.legend()
    plt.tight_layout()

    output_path = _prepare_output_path(save_path)
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()



def plot_multiple_forecasts(
    forecast_df: pd.DataFrame,
    forecast_columns: list[str],
    title: str = "Actual vs Multiple Forecasts",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot realized values together with multiple forecast series.
    """
    if not forecast_columns:
        raise ForecastPlotError("forecast_columns cannot be empty.")

    df = _validate_forecast_df(forecast_df)

    missing_columns = [column for column in forecast_columns if column not in df.columns]
    if missing_columns:
        raise ForecastPlotError(
            f"Forecast dataframe is missing forecast columns: {missing_columns}"
        )

    x_axis = df[DATE_COLUMN] if DATE_COLUMN in df.columns else df.index

    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, df["y_true"], label="y_true")
    for column in forecast_columns:
        plt.plot(x_axis, df[column], label=column)

    plt.title(title)
    plt.xlabel("Date" if DATE_COLUMN in df.columns else "Index")
    plt.ylabel("Target Value")
    plt.legend()
    plt.tight_layout()

    output_path = _prepare_output_path(save_path)
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()



def plot_forecast_error(
    forecast_df: pd.DataFrame,
    forecast_column: str,
    title: str = "Forecast Error Over Time",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot the signed forecast error over time.

    Error definition:
    y_true - forecast
    """
    df = _validate_forecast_df(forecast_df)

    if forecast_column not in df.columns:
        raise ForecastPlotError(
            f"Forecast column '{forecast_column}' not found in forecast dataframe."
        )

    x_axis = df[DATE_COLUMN] if DATE_COLUMN in df.columns else df.index
    error_series = df["y_true"] - df[forecast_column]

    plt.figure(figsize=(12, 5))
    plt.plot(x_axis, error_series, label=f"y_true - {forecast_column}")
    plt.axhline(0, linewidth=1)
    plt.title(title)
    plt.xlabel("Date" if DATE_COLUMN in df.columns else "Index")
    plt.ylabel("Forecast Error")
    plt.legend()
    plt.tight_layout()

    output_path = _prepare_output_path(save_path)
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()



def plot_forecast_scatter(
    forecast_df: pd.DataFrame,
    forecast_column: str,
    title: str = "Forecast vs Actual Scatter",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot a scatter comparison between realized values and one forecast series.
    """
    df = _validate_forecast_df(forecast_df)

    if forecast_column not in df.columns:
        raise ForecastPlotError(
            f"Forecast column '{forecast_column}' not found in forecast dataframe."
        )

    plt.figure(figsize=(6, 6))
    plt.scatter(df[forecast_column], df["y_true"])
    plt.xlabel(forecast_column)
    plt.ylabel("y_true")
    plt.title(title)
    plt.tight_layout()

    output_path = _prepare_output_path(save_path)
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2025-01-01", periods=10, freq="D"),
            "y_true": [70, 75, 80, 78, 82, 76, 74, 79, 81, 77],
            "baseline_forecast": [72, 74, 79, 79, 81, 77, 73, 78, 80, 78],
            "quantile_q50": [71, 75, 78, 78, 80, 76, 74, 79, 80, 77],
        }
    )

    plot_actual_vs_forecast(example_df, forecast_column="baseline_forecast", show=False)
    plot_multiple_forecasts(
        example_df,
        forecast_columns=["baseline_forecast", "quantile_q50"],
        show=False,
    )
    plot_forecast_error(example_df, forecast_column="baseline_forecast", show=False)
    plot_forecast_scatter(example_df, forecast_column="baseline_forecast", show=False)
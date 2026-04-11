

"""
plot_quantiles.py

Visualization utilities for quantile forecast outputs.
These plots help inspect uncertainty bands, tail-risk forecasts,
and calibration-style behavior over time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from src.config.paths import FIGURES_DIR
from src.models.quantile_models import QuantileModelResults


class QuantilePlotError(Exception):
    """Raised when quantile forecast plots cannot be generated safely."""


# =========================
# Validation helpers
# =========================

def _prepare_output_path(filename: str | None) -> Path | None:
    """Prepare an output path inside the configured figures directory."""
    if filename is None:
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / filename



def _validate_quantile_results(results: Sequence[QuantileModelResults]) -> list[QuantileModelResults]:
    """Validate a collection of quantile model results."""
    if not results:
        raise QuantilePlotError("results cannot be empty.")

    validated_results = list(results)
    reference_index = validated_results[0].y_true.index

    for result in validated_results:
        if result.y_true.empty or result.y_pred.empty:
            raise QuantilePlotError("Quantile results contain empty prediction or target series.")

        if not result.y_true.index.equals(reference_index):
            raise QuantilePlotError(
                "All quantile results must share the same evaluation index."
            )

    quantiles = [float(result.quantile) for result in validated_results]
    if len(quantiles) != len(set(quantiles)):
        raise QuantilePlotError(
            f"Quantile values must be unique. Found: {quantiles}"
        )

    return sorted(validated_results, key=lambda x: x.quantile)



def _results_to_frame(results: Sequence[QuantileModelResults]) -> pd.DataFrame:
    """Convert quantile result objects into a single plotting dataframe."""
    validated_results = _validate_quantile_results(results)

    df = pd.DataFrame(index=validated_results[0].y_true.index)
    df["y_true"] = validated_results[0].y_true

    for result in validated_results:
        df[f"q_{result.quantile}"] = result.y_pred

    return df


# =========================
# Plot functions
# =========================

def plot_quantile_forecasts(
    results: Sequence[QuantileModelResults],
    title: str = "Quantile Forecasts vs Actual Values",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot actual values together with all available quantile forecasts.
    """
    df = _results_to_frame(results)

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["y_true"], label="y_true")

    quantile_columns = sorted(
        [column for column in df.columns if column.startswith("q_")],
        key=lambda x: float(x.replace("q_", "")),
    )
    for column in quantile_columns:
        plt.plot(df.index, df[column], label=column)

    plt.title(title)
    plt.xlabel("Evaluation Index")
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



def plot_quantile_band(
    results: Sequence[QuantileModelResults],
    lower_quantile: float,
    upper_quantile: float,
    title: str = "Prediction Band vs Actual Values",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot actual values together with a shaded prediction band.

    Example:
    - lower_quantile=0.5
    - upper_quantile=0.9
    """
    if lower_quantile >= upper_quantile:
        raise QuantilePlotError("lower_quantile must be strictly smaller than upper_quantile.")

    df = _results_to_frame(results)
    lower_column = f"q_{lower_quantile}"
    upper_column = f"q_{upper_quantile}"

    missing_columns = [col for col in [lower_column, upper_column] if col not in df.columns]
    if missing_columns:
        raise QuantilePlotError(
            f"Missing required quantile columns for band plot: {missing_columns}"
        )

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["y_true"], label="y_true")
    plt.plot(df.index, df[lower_column], label=lower_column)
    plt.plot(df.index, df[upper_column], label=upper_column)
    plt.fill_between(df.index, df[lower_column], df[upper_column], alpha=0.2)

    plt.title(title)
    plt.xlabel("Evaluation Index")
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



def plot_upper_tail_exceedances(
    results: Sequence[QuantileModelResults],
    upper_quantile: float = 0.9,
    title: str = "Upper-Tail Forecast and Exceedances",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot actual values and highlight where the realized target exceeded an upper quantile.
    """
    df = _results_to_frame(results)
    upper_column = f"q_{upper_quantile}"

    if upper_column not in df.columns:
        raise QuantilePlotError(
            f"Missing required quantile column for exceedance plot: '{upper_column}'"
        )

    exceedances = df["y_true"] > df[upper_column]

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["y_true"], label="y_true")
    plt.plot(df.index, df[upper_column], label=upper_column)
    plt.scatter(df.index[exceedances], df.loc[exceedances, "y_true"], label="exceedance")

    plt.title(title)
    plt.xlabel("Evaluation Index")
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



def plot_quantile_error_series(
    results: Sequence[QuantileModelResults],
    quantile: float,
    title: str = "Quantile Forecast Error Over Time",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot the signed forecast error for one quantile prediction series.

    Error definition:
    y_true - q_alpha
    """
    df = _results_to_frame(results)
    quantile_column = f"q_{quantile}"

    if quantile_column not in df.columns:
        raise QuantilePlotError(
            f"Missing required quantile column for error plot: '{quantile_column}'"
        )

    error_series = df["y_true"] - df[quantile_column]

    plt.figure(figsize=(12, 5))
    plt.plot(df.index, error_series, label=f"y_true - {quantile_column}")
    plt.axhline(0, linewidth=1)
    plt.title(title)
    plt.xlabel("Evaluation Index")
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


if __name__ == "__main__":
    y_true = pd.Series([70, 75, 80, 78, 82, 76])
    q50 = pd.Series([72, 74, 79, 79, 81, 77])
    q90 = pd.Series([84, 87, 92, 86, 95, 84])
    q95 = pd.Series([88, 90, 96, 89, 99, 87])

    results = [
        QuantileModelResults(
            quantile=0.5,
            model_name="gbr_quantile_0.5",
            y_true=y_true,
            y_pred=q50,
            pinball_loss=0.0,
            mae=0.0,
            rmse=0.0,
        ),
        QuantileModelResults(
            quantile=0.9,
            model_name="gbr_quantile_0.9",
            y_true=y_true,
            y_pred=q90,
            pinball_loss=0.0,
            mae=0.0,
            rmse=0.0,
        ),
        QuantileModelResults(
            quantile=0.95,
            model_name="gbr_quantile_0.95",
            y_true=y_true,
            y_pred=q95,
            pinball_loss=0.0,
            mae=0.0,
            rmse=0.0,
        ),
    ]

    plot_quantile_forecasts(results, show=False)
    plot_quantile_band(results, lower_quantile=0.5, upper_quantile=0.9, show=False)
    plot_upper_tail_exceedances(results, upper_quantile=0.9, show=False)
    plot_quantile_error_series(results, quantile=0.9, show=False)
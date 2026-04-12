"""
evaluate_model.py

Utilities to compare and evaluate forecasting models for the project.
This module is designed to work with both baseline point forecasts and
quantile-based uncertainty forecasts.
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from src.config.constants import Q50_COLUMN
from src.models.baseline_models import BaselineResults
from src.models.quantile_models import QuantileModelResults


class ModelEvaluationError(Exception):
    """Raised when model evaluation inputs are inconsistent or invalid."""


def _parse_quantile_from_column_name(column_name: str) -> float:
    """Parse a quantile value from a quantile column name like 'q_0.9'."""
    try:
        return float(column_name.replace("q_", ""))
    except ValueError as exc:
        raise ModelEvaluationError(
            f"Could not parse nominal quantile from column name '{column_name}'."
        ) from exc



def _get_sorted_quantile_columns(df: pd.DataFrame) -> list[str]:
    """Return quantile columns sorted by their numeric quantile value."""
    return sorted(
        [col for col in df.columns if col.startswith("q_")],
        key=_parse_quantile_from_column_name,
    )


# =========================
# Baseline / quantile summaries
# =========================

def summarize_baseline_results(results: Sequence[BaselineResults]) -> pd.DataFrame:
    """Create a compact summary table for baseline models."""
    if not results:
        raise ModelEvaluationError("Baseline results list cannot be empty.")

    summary_df = pd.DataFrame(
        {
            "model_type": "baseline",
            "model_name": [result.model_name for result in results],
            "quantile": pd.NA,
            "mae": [result.mae for result in results],
            "rmse": [result.rmse for result in results],
            "pinball_loss": pd.NA,
        }
    )

    return summary_df.sort_values(["rmse", "mae"]).reset_index(drop=True)



def summarize_quantile_results(results: Sequence[QuantileModelResults]) -> pd.DataFrame:
    """Create a compact summary table for quantile models."""
    if not results:
        raise ModelEvaluationError("Quantile results list cannot be empty.")

    summary_df = pd.DataFrame(
        {
            "model_type": "quantile",
            "model_name": [result.model_name for result in results],
            "quantile": [result.quantile for result in results],
            "mae": [result.mae for result in results],
            "rmse": [result.rmse for result in results],
            "pinball_loss": [result.pinball_loss for result in results],
        }
    )

    return summary_df.sort_values(["quantile", "pinball_loss", "rmse"]).reset_index(drop=True)



def compare_all_models(
    baseline_results: Sequence[BaselineResults] | None = None,
    quantile_results: Sequence[QuantileModelResults] | None = None,
) -> pd.DataFrame:
    """
    Combine baseline and quantile model summaries into a single comparison table.
    """
    frames: list[pd.DataFrame] = []

    if baseline_results:
        frames.append(summarize_baseline_results(baseline_results))

    if quantile_results:
        frames.append(summarize_quantile_results(quantile_results))

    if not frames:
        raise ModelEvaluationError(
            "At least one of baseline_results or quantile_results must be provided."
        )

    comparison_df = pd.concat(frames, axis=0, ignore_index=True)
    return comparison_df.reset_index(drop=True)


# =========================
# Quantile forecast diagnostics
# =========================

def combine_quantile_predictions(results: Sequence[QuantileModelResults]) -> pd.DataFrame:
    """
    Combine quantile predictions into one dataframe for diagnostics and plotting.

    Output columns:
    - y_true
    - q_<quantile>
    """
    if not results:
        raise ModelEvaluationError("Quantile results list cannot be empty.")

    combined_df = pd.DataFrame(index=results[0].y_true.index)
    combined_df["y_true"] = results[0].y_true

    reference_index = results[0].y_true.index
    for result in results:
        if not result.y_true.index.equals(reference_index):
            raise ModelEvaluationError(
                "All quantile results must share the same evaluation index."
            )
        quantile_column = f"q_{result.quantile:g}"
        combined_df[quantile_column] = result.y_pred
    
    # Enforce monotonicity across adjacent quantiles to avoid quantile crossing issues.
    quantile_columns = _get_sorted_quantile_columns(combined_df)

    for i in range(1, len(quantile_columns)):
        prev_col = quantile_columns[i - 1]
        curr_col = quantile_columns[i]
        combined_df[curr_col] = combined_df[[prev_col, curr_col]].max(axis=1)

    return combined_df


def evaluate_quantile_predictions(
    y_true,
    y_pred,
    quantile: float,
) -> pd.DataFrame:
    """
    Convenience wrapper used in tests for evaluating one quantile forecast.
    """
    quantile_column = f"q_{quantile:g}"
    eval_df = pd.DataFrame(
        {
            "y_true": y_true,
            quantile_column: y_pred,
        }
    )
    metrics = evaluate_quantile_coverage(eval_df, quantile_column)
    return pd.DataFrame([metrics])


def evaluate_quantile_coverage(
    combined_quantile_df: pd.DataFrame,
    quantile_column: str,
) -> dict[str, float]:
    """
    Evaluate empirical coverage for a single upper quantile.

    Example:
    For q_0.9, the empirical share of observations below the predicted quantile
    should ideally be close to 0.9.
    """
    required_columns = {"y_true", quantile_column}
    missing_columns = required_columns - set(combined_quantile_df.columns)
    if missing_columns:
        raise ModelEvaluationError(
            f"Missing required columns for coverage evaluation: {sorted(missing_columns)}"
        )

    eval_df = combined_quantile_df[["y_true", quantile_column]].dropna().copy()
    if eval_df.empty:
        raise ModelEvaluationError("Coverage evaluation dataframe is empty after dropping NaNs.")

    nominal_quantile = _parse_quantile_from_column_name(quantile_column)

    empirical_coverage = float((eval_df["y_true"] <= eval_df[quantile_column]).mean())

    return {
        "quantile": nominal_quantile,
        "empirical_coverage": empirical_coverage,
        "coverage_error": empirical_coverage - nominal_quantile,
        "n_obs": float(len(eval_df)),
    }



def evaluate_prediction_interval(
    combined_quantile_df: pd.DataFrame,
    lower_quantile_column: str,
    upper_quantile_column: str,
) -> pd.DataFrame:
    """
    Evaluate a prediction interval built from two quantile columns.

    Returns empirical coverage and average interval width.
    """
    required_columns = {"y_true", lower_quantile_column, upper_quantile_column}
    missing_columns = required_columns - set(combined_quantile_df.columns)
    if missing_columns:
        raise ModelEvaluationError(
            f"Missing required columns for interval evaluation: {sorted(missing_columns)}"
        )

    eval_df = combined_quantile_df[
        ["y_true", lower_quantile_column, upper_quantile_column]
    ].dropna().copy()

    if eval_df.empty:
        raise ModelEvaluationError("Interval evaluation dataframe is empty after dropping NaNs.")

    lower_quantile = _parse_quantile_from_column_name(lower_quantile_column)
    upper_quantile = _parse_quantile_from_column_name(upper_quantile_column)
    if lower_quantile >= upper_quantile:
        raise ModelEvaluationError(
            "lower_quantile_column must correspond to a strictly smaller quantile than upper_quantile_column."
        )

    if (eval_df[lower_quantile_column] > eval_df[upper_quantile_column]).any():
        raise ModelEvaluationError(
            "Lower quantile exceeds upper quantile for at least one observation."
        )

    within_interval = (
        (eval_df["y_true"] >= eval_df[lower_quantile_column])
        & (eval_df["y_true"] <= eval_df[upper_quantile_column])
    )
    empirical_coverage = float(within_interval.mean())
    avg_interval_width = float(
        (eval_df[upper_quantile_column] - eval_df[lower_quantile_column]).mean()
    )

    return pd.DataFrame(
        [
            {
                "lower_quantile_column": lower_quantile_column,
                "upper_quantile_column": upper_quantile_column,
                "empirical_coverage": empirical_coverage,
                "average_interval_width": avg_interval_width,
                "n_obs": float(len(eval_df)),
            }
        ]
    )



def evaluate_upper_tail_exceedance_rate(
    combined_quantile_df: pd.DataFrame,
    upper_quantile_column: str,
) -> dict[str, float]:
    """
    Compute the exceedance rate above an upper quantile forecast.

    For q_0.9, the exceedance rate should ideally be around 10%.
    """
    required_columns = {"y_true", upper_quantile_column}
    missing_columns = required_columns - set(combined_quantile_df.columns)
    if missing_columns:
        raise ModelEvaluationError(
            f"Missing required columns for exceedance evaluation: {sorted(missing_columns)}"
        )

    eval_df = combined_quantile_df[["y_true", upper_quantile_column]].dropna().copy()
    if eval_df.empty:
        raise ModelEvaluationError("Exceedance evaluation dataframe is empty after dropping NaNs.")

    exceedance_rate = float((eval_df["y_true"] > eval_df[upper_quantile_column]).mean())

    nominal_quantile = _parse_quantile_from_column_name(upper_quantile_column)

    target_exceedance = 1.0 - nominal_quantile

    return {
        "quantile": nominal_quantile,
        "empirical_exceedance_rate": exceedance_rate,
        "target_exceedance_rate": target_exceedance,
        "exceedance_error": exceedance_rate - target_exceedance,
        "n_obs": float(len(eval_df)),
    }


# =========================
# Convenience report helper
# =========================

def build_quantile_diagnostics_report(
    quantile_results: Sequence[QuantileModelResults],
) -> dict[str, Any]:
    """
    Build a compact diagnostics package for quantile forecasts.

    Includes:
    - combined prediction dataframe
    - per-quantile coverage diagnostics
    - interval diagnostics for adjacent quantile pairs
    - upper-tail exceedance diagnostics for upper quantiles
    """
    combined_df = combine_quantile_predictions(quantile_results)

    quantile_columns = _get_sorted_quantile_columns(combined_df)

    coverage_rows: list[dict[str, float]] = []
    interval_frames: list[pd.DataFrame] = []
    exceedance_rows: list[dict[str, float]] = []

    for quantile_column in quantile_columns:
        coverage_rows.append(evaluate_quantile_coverage(combined_df, quantile_column))

        quantile_value = _parse_quantile_from_column_name(quantile_column)
        if quantile_value >= 0.5:
            exceedance_rows.append(
                evaluate_upper_tail_exceedance_rate(combined_df, quantile_column)
            )

    if len(quantile_columns) >= 2:
        for lower_col, upper_col in zip(quantile_columns[:-1], quantile_columns[1:]):
            interval_frames.append(
                evaluate_prediction_interval(combined_df, lower_col, upper_col)
            )

    diagnostics = {
        "combined_predictions": combined_df,
        "coverage_summary": pd.DataFrame(coverage_rows),
        "interval_summary": (
            pd.concat(interval_frames, ignore_index=True)
            if interval_frames
            else pd.DataFrame(
                columns=[
                    "lower_quantile_column",
                    "upper_quantile_column",
                    "empirical_coverage",
                    "average_interval_width",
                    "n_obs",
                ]
            )
        ),
        "upper_tail_exceedance_summary": pd.DataFrame(exceedance_rows),
    }

    return diagnostics


if __name__ == "__main__":
    baseline_results = [
        BaselineResults(
            model_name="naive_last_value",
            y_true=pd.Series([100, 110, 120]),
            y_pred=pd.Series([98, 109, 121]),
            mae=1.33,
            rmse=1.41,
        )
    ]

    q50 = QuantileModelResults(
        quantile=0.5,
        model_name="gbr_quantile_0.5",
        y_true=pd.Series([100, 110, 120]),
        y_pred=pd.Series([99, 111, 118]),
        pinball_loss=0.9,
        mae=1.33,
        rmse=1.41,
    )
    q90 = QuantileModelResults(
        quantile=0.9,
        model_name="gbr_quantile_0.9",
        y_true=pd.Series([100, 110, 120]),
        y_pred=pd.Series([105, 116, 126]),
        pinball_loss=0.7,
        mae=5.67,
        rmse=5.72,
    )

    comparison_df = compare_all_models(
        baseline_results=baseline_results,
        quantile_results=[q50, q90],
    )
    diagnostics = build_quantile_diagnostics_report([q50, q90])
    assert Q50_COLUMN in diagnostics["combined_predictions"].columns

    print(comparison_df)
    print(diagnostics["coverage_summary"])
    print(diagnostics["interval_summary"])
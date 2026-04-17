"""
train_model.py

Training wrappers for the project's forecasting models.
This module provides a lightweight, consistent interface around the existing
baseline and quantile model utilities so the training flow is easier to reuse
from scripts, notebooks, and reporting pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config.constants import DATE_COLUMN, DEFAULT_FORECAST_HORIZON, DEFAULT_QUANTILES, TARGET_COLUMN
from src.config.settings import TrainingSettings, get_default_settings
from src.models.baseline_models import (
    BaselineResults,
    linear_regression_baseline,
    naive_last_value_baseline,
    rolling_mean_baseline,
    seasonal_naive_baseline,
    summarize_baseline_results,
)
from src.models.quantile_models import (
    QuantileModelResults,
    train_quantile_models,
)


class TrainModelError(Exception):
    """Raised when a training wrapper cannot run safely."""


@dataclass
class BaselineTrainingOutput:
    """Container for baseline training results."""

    results: list[BaselineResults]
    summary: pd.DataFrame
    trained_models: dict[str, Any]


@dataclass
class QuantileTrainingOutput:
    """Container for quantile training results."""

    results: list[QuantileModelResults]
    summary: pd.DataFrame
    trained_models: dict[float, Any]
    used_features: list[str]


def get_default_training_settings() -> TrainingSettings:
    """Return default training settings from the project configuration."""
    return get_default_settings().training


# =========================
# Validation helpers
# =========================

def _validate_train_eval_data(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    """Validate training and evaluation dataframes."""
    if train_df.empty:
        raise TrainModelError("train_df is empty.")
    if eval_df.empty:
        raise TrainModelError("eval_df is empty.")

    for df_name, df in {"train_df": train_df, "eval_df": eval_df}.items():
        if DATE_COLUMN not in df.columns:
            raise TrainModelError(
                f"{df_name} must contain a '{DATE_COLUMN}' column."
            )


# =========================
# Baseline wrappers
# =========================

def train_baseline_suite(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    horizon: int = DEFAULT_FORECAST_HORIZON,
    rolling_window: int = 7,
    seasonal_lag: int = 7,
) -> BaselineTrainingOutput:
    """
    Train and evaluate the project's baseline model suite.

    Included baselines:
    - naive last value
    - seasonal naive
    - rolling mean
    - linear regression baseline
    """
    _validate_train_eval_data(train_df, eval_df)

    naive_results = naive_last_value_baseline(
        eval_df,
        target_column=target_column,
        horizon=horizon,
    )
    seasonal_results = seasonal_naive_baseline(
        eval_df,
        target_column=target_column,
        horizon=horizon,
        seasonal_lag=seasonal_lag,
    )
    rolling_results = rolling_mean_baseline(
        eval_df,
        target_column=target_column,
        horizon=horizon,
        window=rolling_window,
    )
    linear_results, linear_model = linear_regression_baseline(
        train_df,
        eval_df,
        target_column=target_column,
        horizon=horizon,
    )

    results = [naive_results, seasonal_results, rolling_results, linear_results]
    summary = summarize_baseline_results(results)

    trained_models = {
        "naive_last_value": None,
        f"seasonal_naive_lag_{seasonal_lag}": None,
        f"rolling_mean_{rolling_window}": None,
        "linear_regression_baseline": linear_model,
    }

    return BaselineTrainingOutput(
        results=results,
        summary=summary,
        trained_models=trained_models,
    )


# =========================
# Quantile wrappers
# =========================

def train_quantile_suite(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    quantiles: list[float] | None = None,
    target_column: str = TARGET_COLUMN,
    horizon: int = DEFAULT_FORECAST_HORIZON,
    feature_columns: list[str] | None = None,
    model_params: dict | None = None,
) -> QuantileTrainingOutput:
    """
    Train and evaluate the project's quantile model suite.

    This is a thin wrapper around `train_quantile_models(...)` that returns a
    more structured output object for notebooks and pipelines.
    """
    _validate_train_eval_data(train_df, eval_df)

    training_settings = get_default_training_settings()
    quantiles = list(training_settings.quantiles) if quantiles is None else quantiles

    results, models, used_features = train_quantile_models(
        train_df=train_df,
        test_df=eval_df,
        quantiles=quantiles,
        target_column=target_column,
        horizon=horizon,
        feature_columns=feature_columns,
        model_params=model_params,
    )

    summary = pd.DataFrame(
        {
            "model_name": [result.model_name for result in results],
            "quantile": [result.quantile for result in results],
            "pinball_loss": [result.pinball_loss for result in results],
            "mae": [result.mae for result in results],
            "rmse": [result.rmse for result in results],
        }
    ).sort_values(["quantile", "pinball_loss"]).reset_index(drop=True)

    return QuantileTrainingOutput(
        results=results,
        summary=summary,
        trained_models=models,
        used_features=used_features,
    )


# =========================
# Unified training entrypoint
# =========================

SUPPORTED_MODEL_FAMILIES = ["baseline", "quantile"]

def train_model(
    model_family: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    **kwargs: Any,
) -> BaselineTrainingOutput | QuantileTrainingOutput:
    """
    Unified entrypoint for training a supported model family.

    Parameters
    ----------
    model_family : str
        Supported values:
        - "baseline"
        - "quantile"
    train_df : pd.DataFrame
        Training dataframe.
    eval_df : pd.DataFrame
        Evaluation dataframe.
    **kwargs : Any
        Additional keyword arguments forwarded to the selected training wrapper.

    Returns
    -------
    BaselineTrainingOutput | QuantileTrainingOutput
        Structured training output.
    """
    model_family_normalized = model_family.strip().lower()

    if model_family_normalized == "baseline":
        return train_baseline_suite(train_df, eval_df, **kwargs)

    if model_family_normalized == "quantile":
        return train_quantile_suite(train_df, eval_df, **kwargs)

    raise TrainModelError(
        f"Unsupported model_family. Use one of: {SUPPORTED_MODEL_FAMILIES}."
    )


if __name__ == "__main__":
    training_settings = get_default_training_settings()
    train_example = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2024-01-01", periods=80, freq="D"),
            TARGET_COLUMN: range(80),
            f"{TARGET_COLUMN}_lag_1": pd.Series(range(80)).shift(1),
            f"{TARGET_COLUMN}_lag_2": pd.Series(range(80)).shift(2),
            f"{TARGET_COLUMN}_lag_3": pd.Series(range(80)).shift(3),
            f"{TARGET_COLUMN}_lag_7": pd.Series(range(80)).shift(7),
            f"{TARGET_COLUMN}_lag_14": pd.Series(range(80)).shift(14),
            f"{TARGET_COLUMN}_lag_28": pd.Series(range(80)).shift(28),
            "Future_M1_Price_lag_1": pd.Series(range(100, 180)).shift(1),
            "Future_M1_Price_lag_7": pd.Series(range(100, 180)).shift(7),
            "Future_M1_OpenInterest_lag_1": pd.Series(range(1000, 1080)).shift(1),
            "Future_M1_OpenInterest_lag_7": pd.Series(range(1000, 1080)).shift(7),
            "day_of_week_sin": 0.0,
            "day_of_week_cos": 1.0,
            "month_sin": 0.0,
            "month_cos": 1.0,
            "day_of_year_sin": 0.0,
            "day_of_year_cos": 1.0,
        }
    )
    eval_example = train_example.iloc[40:].copy()

    baseline_output = train_model("baseline", train_example, eval_example)
    print("Baseline summary:")
    print(baseline_output.summary)
    quantile_output = train_model(
        "quantile",
        train_example,
        eval_example,
        quantiles=training_settings.quantiles,
    )
    print("\nQuantile summary:")
    print(quantile_output.summary)
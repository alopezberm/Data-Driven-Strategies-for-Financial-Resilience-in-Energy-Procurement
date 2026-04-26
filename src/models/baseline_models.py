"""
baseline_models.py

Baseline forecasting models for the project.
These baselines provide simple, interpretable reference points before moving to
more advanced uncertainty-aware models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config.constants import DATE_COLUMN, DEFAULT_FORECAST_HORIZON, TARGET_COLUMN
from src.config.settings import TrainingSettings, get_default_settings
from src.utils.validation import ValidationError, validate_and_sort_by_date


class BaselineModelError(Exception):
    """Raised when baseline model training or prediction fails."""


@dataclass
class BaselineResults:
    """Container for baseline prediction outputs and simple metrics."""

    model_name: str
    y_true: pd.Series
    y_pred: pd.Series
    mae: float
    rmse: float


def get_default_baseline_feature_columns(training_settings: TrainingSettings | None = None) -> list[str]:
    """Build the default baseline feature set from project settings."""
    if training_settings is None:
        training_settings = get_default_settings().training

    lag_steps = list(training_settings.lag_steps)

    target_lag_features = [f"{TARGET_COLUMN}_lag_{lag}" for lag in lag_steps]
    market_lag_features = [
        "Future_M1_Price_lag_1",
        "Future_M1_Price_lag_7",
        "Future_M1_OpenInterest_lag_1",
        "Future_M1_OpenInterest_lag_7",
    ]
    cyclical_features = [
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]

    return target_lag_features + market_lag_features + cyclical_features


# =========================
# Validation helpers
# =========================

def _validate_input_dataframe(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Validate and sort a time-series input dataframe."""
    try:
        result_df = validate_and_sort_by_date(df, df_name="baseline model input")
    except ValidationError as exc:
        raise BaselineModelError(str(exc)) from exc
    if target_column not in result_df.columns:
        raise BaselineModelError(
            f"Input dataframe must contain target column '{target_column}'."
        )
    return result_df


# =========================
# Target preparation
# =========================

def prepare_next_day_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    horizon: int = DEFAULT_FORECAST_HORIZON,
) -> pd.DataFrame:
    """
    Create a forward target for supervised next-step forecasting.

    Example:
    - Features at time t
    - Target at time t + horizon
    """
    if horizon <= 0:
        raise BaselineModelError("horizon must be a positive integer.")

    df = _validate_input_dataframe(df, target_column)
    df = df.copy()
    df[f"target_{target_column}_t_plus_{horizon}"] = df[target_column].shift(-horizon)

    return df


# =========================
# Utility helpers
# =========================

def _compute_regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float]:
    """Compute MAE and RMSE for a pair of prediction series."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse



def _format_results(model_name: str, y_true: pd.Series, y_pred: pd.Series) -> BaselineResults:
    """Build a BaselineResults object from predictions."""
    mae, rmse = _compute_regression_metrics(y_true, y_pred)
    return BaselineResults(
        model_name=model_name,
        y_true=y_true,
        y_pred=y_pred,
        mae=mae,
        rmse=rmse,
    )


# =========================
# Baseline 1: Naive last value
# =========================

def naive_last_value_baseline(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    horizon: int = DEFAULT_FORECAST_HORIZON,
) -> BaselineResults:
    """
    Predict tomorrow's target as today's observed target.

    This is often a surprisingly strong benchmark for time series.
    """
    prepared_df = prepare_next_day_target(df, target_column=target_column, horizon=horizon)
    target_name = f"target_{target_column}_t_plus_{horizon}"

    eval_df = prepared_df[[DATE_COLUMN, target_column, target_name]].dropna().copy()
    eval_df["prediction"] = eval_df[target_column]

    return _format_results(
        model_name="naive_last_value",
        y_true=eval_df[target_name],
        y_pred=eval_df["prediction"],
    )


# =========================
# Baseline 2: Seasonal naive
# =========================

def seasonal_naive_baseline(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    horizon: int = DEFAULT_FORECAST_HORIZON,
    seasonal_lag: int = 7,
) -> BaselineResults:
    """
    Predict using the value observed `seasonal_lag` days ago.

    For electricity prices, a weekly seasonal baseline is often informative.
    """
    if seasonal_lag <= 0:
        raise BaselineModelError("seasonal_lag must be a positive integer.")

    prepared_df = prepare_next_day_target(df, target_column=target_column, horizon=horizon)
    target_name = f"target_{target_column}_t_plus_{horizon}"

    eval_df = prepared_df[[DATE_COLUMN, target_column, target_name]].copy()
    eval_df["prediction"] = eval_df[target_column].shift(seasonal_lag)
    eval_df = eval_df.dropna().copy()

    return _format_results(
        model_name=f"seasonal_naive_lag_{seasonal_lag}",
        y_true=eval_df[target_name],
        y_pred=eval_df["prediction"],
    )


# =========================
# Baseline 3: Rolling mean
# =========================

def rolling_mean_baseline(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    horizon: int = DEFAULT_FORECAST_HORIZON,
    window: int = 7,
) -> BaselineResults:
    """
    Predict using the trailing mean of the target series.

    Uses only past information via shift(1).
    """
    if window <= 0:
        raise BaselineModelError("window must be a positive integer.")

    prepared_df = prepare_next_day_target(df, target_column=target_column, horizon=horizon)
    target_name = f"target_{target_column}_t_plus_{horizon}"

    eval_df = prepared_df[[DATE_COLUMN, target_column, target_name]].copy()
    eval_df["prediction"] = eval_df[target_column].shift(1).rolling(window=window).mean()
    eval_df = eval_df.dropna().copy()

    return _format_results(
        model_name=f"rolling_mean_{window}",
        y_true=eval_df[target_name],
        y_pred=eval_df["prediction"],
    )


# =========================
# Baseline 4: Linear regression
# =========================

def linear_regression_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    horizon: int = DEFAULT_FORECAST_HORIZON,
    feature_columns: list[str] | None = None,
) -> tuple[BaselineResults, LinearRegression]:
    """
    Train a simple linear regression baseline using engineered features.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe.
    test_df : pd.DataFrame
        Evaluation dataframe.
    target_column : str, optional
        Target variable name in the original dataframe.
    horizon : int, optional
        Forecast horizon.
    feature_columns : list[str] | None, optional
        Feature columns to use. If None, a project-specific default set is used.

    Returns
    -------
    tuple[BaselineResults, LinearRegression]
        Baseline results and the fitted sklearn model.
    """
    feature_columns = get_default_baseline_feature_columns() if feature_columns is None else feature_columns

    prepared_train = prepare_next_day_target(train_df, target_column=target_column, horizon=horizon)
    prepared_test = prepare_next_day_target(test_df, target_column=target_column, horizon=horizon)
    target_name = f"target_{target_column}_t_plus_{horizon}"

    available_features = [col for col in feature_columns if col in prepared_train.columns and col in prepared_test.columns]

    if not available_features:
        raise BaselineModelError(
            "No valid feature columns were found for linear_regression_baseline."
        )

    train_model_df = prepared_train[available_features + [target_name]].dropna().copy()
    test_model_df = prepared_test[available_features + [target_name]].dropna().copy()

    if train_model_df.empty or test_model_df.empty:
        raise BaselineModelError(
            "Train or test data became empty after dropping NaNs for linear regression baseline."
        )

    X_train = train_model_df[available_features]
    y_train = train_model_df[target_name]
    X_test = test_model_df[available_features]
    y_test = test_model_df[target_name]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_test), index=y_test.index, name="prediction")

    results = _format_results(
        model_name="linear_regression_baseline",
        y_true=y_test,
        y_pred=y_pred,
    )

    return results, model


# =========================
# Comparison helper
# =========================

def summarize_baseline_results(results: list[BaselineResults]) -> pd.DataFrame:
    """Convert a list of BaselineResults into a comparable summary table."""
    if not results:
        raise BaselineModelError("results list cannot be empty.")

    summary_df = pd.DataFrame(
        {
            "model_name": [result.model_name for result in results],
            "mae": [result.mae for result in results],
            "rmse": [result.rmse for result in results],
        }
    ).sort_values("rmse", ascending=True).reset_index(drop=True)

    return summary_df


if __name__ == "__main__":
    example_dates = pd.date_range("2024-01-01", periods=60, freq="D")
    example_target = np.linspace(50, 80, 60)

    example_df = pd.DataFrame(
        {
            DATE_COLUMN: example_dates,
            TARGET_COLUMN: example_target,
            f"{TARGET_COLUMN}_lag_1": pd.Series(example_target).shift(1),
            f"{TARGET_COLUMN}_lag_2": pd.Series(example_target).shift(2),
            f"{TARGET_COLUMN}_lag_3": pd.Series(example_target).shift(3),
            f"{TARGET_COLUMN}_lag_7": pd.Series(example_target).shift(7),
            f"{TARGET_COLUMN}_lag_14": pd.Series(example_target).shift(14),
            f"{TARGET_COLUMN}_lag_28": pd.Series(example_target).shift(28),
            "Future_M1_Price_lag_1": pd.Series(np.linspace(51, 81, 60)).shift(1),
            "Future_M1_Price_lag_7": pd.Series(np.linspace(51, 81, 60)).shift(7),
            "Future_M1_OpenInterest_lag_1": pd.Series(np.linspace(1000, 1100, 60)).shift(1),
            "Future_M1_OpenInterest_lag_7": pd.Series(np.linspace(1000, 1100, 60)).shift(7),
            "day_of_week_sin": np.sin(2 * np.pi * (example_dates.dayofweek) / 7),
            "day_of_week_cos": np.cos(2 * np.pi * (example_dates.dayofweek) / 7),
            "month_sin": np.sin(2 * np.pi * (example_dates.month) / 12),
            "month_cos": np.cos(2 * np.pi * (example_dates.month) / 12),
            "day_of_year_sin": np.sin(2 * np.pi * (example_dates.dayofyear) / 365.25),
            "day_of_year_cos": np.cos(2 * np.pi * (example_dates.dayofyear) / 365.25),
        }
    )

    naive_results = naive_last_value_baseline(example_df)
    seasonal_results = seasonal_naive_baseline(example_df)
    rolling_results = rolling_mean_baseline(example_df, window=7)
    lr_results, _ = linear_regression_baseline(example_df.iloc[:40], example_df.iloc[40:])

    summary = summarize_baseline_results(
        [naive_results, seasonal_results, rolling_results, lr_results]
    )
    print(summary)
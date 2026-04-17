"""
predict.py

Prediction wrappers for the project's forecasting models.
This module provides lightweight helper functions to generate predictions from
already-fitted models in a consistent format.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config.constants import DATE_COLUMN


class PredictError(Exception):
    """Raised when model prediction fails."""


# =========================
# Validation helpers
# =========================

def _validate_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that prediction input data is non-empty and contains the configured date column."""
    if df.empty:
        raise PredictError("Input dataframe is empty.")

    if DATE_COLUMN not in df.columns:
        raise PredictError(
            f"Input dataframe must contain a '{DATE_COLUMN}' column."
        )

    result_df = df.copy()
    result_df[DATE_COLUMN] = pd.to_datetime(result_df[DATE_COLUMN], errors="coerce")

    if result_df[DATE_COLUMN].isna().any():
        invalid_count = int(result_df[DATE_COLUMN].isna().sum())
        raise PredictError(
            f"Found {invalid_count} invalid date values in prediction input dataframe."
        )

    return result_df.sort_values(DATE_COLUMN).reset_index(drop=True)



def _validate_feature_columns(df: pd.DataFrame, feature_columns: list[str]) -> None:
    """Validate that all required feature columns are present."""
    if not feature_columns:
        raise PredictError("feature_columns cannot be empty.")

    missing_columns = [column for column in feature_columns if column not in df.columns]
    if missing_columns:
        raise PredictError(
            f"Missing required feature columns for prediction: {missing_columns}"
        )


# =========================
# Prediction helpers
# =========================

def predict_with_model(
    model: Any,
    df: pd.DataFrame,
    feature_columns: list[str],
    prediction_column_name: str = "prediction",
) -> pd.DataFrame:
    """
    Generate predictions from a fitted model using a provided feature set.

    Parameters
    ----------
    model : Any
        Fitted model object exposing a `.predict(...)` method.
    df : pd.DataFrame
        Input dataframe containing the configured date column and model features.
    feature_columns : list[str]
        Feature columns required by the model.
    prediction_column_name : str, optional
        Column name for the output predictions.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the configured date column and the prediction column.
    """
    if model is None:
        raise PredictError("Model is None. A fitted model instance is required.")

    if not hasattr(model, "predict"):
        raise PredictError("Provided model does not expose a 'predict' method.")

    validated_df = _validate_input_dataframe(df)
    _validate_feature_columns(validated_df, feature_columns)

    model_input = validated_df[feature_columns].copy()
    model_input = model_input.dropna().copy()

    if model_input.empty:
        raise PredictError(
            "All rows were dropped due to missing feature values before prediction."
        )

    predictions = model.predict(model_input)

    prediction_df = validated_df.loc[model_input.index, [DATE_COLUMN]].copy()
    prediction_df[prediction_column_name] = predictions

    return prediction_df.reset_index(drop=True)



def predict_quantile_set(
    models: dict[float, Any],
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Generate multiple quantile predictions from a dictionary of fitted models.

    Parameters
    ----------
    models : dict[float, Any]
        Dictionary mapping quantile values to fitted models.
    df : pd.DataFrame
        Input dataframe containing the configured date column and model features.
    feature_columns : list[str]
        Feature columns required by the models.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the configured date column and one column per quantile, named `q_<quantile>`.
    """
    if not models:
        raise PredictError("models dictionary cannot be empty.")

    validated_df = _validate_input_dataframe(df)
    _validate_feature_columns(validated_df, feature_columns)

    model_input = validated_df[feature_columns].copy()
    model_input = model_input.dropna().copy()

    if model_input.empty:
        raise PredictError(
            "All rows were dropped due to missing feature values before quantile prediction."
        )

    prediction_df = validated_df.loc[model_input.index, [DATE_COLUMN]].copy()

    for quantile, model in sorted(models.items(), key=lambda x: x[0]):
        if model is None:
            raise PredictError(f"Model for quantile {quantile} is None.")
        if not hasattr(model, "predict"):
            raise PredictError(f"Model for quantile {quantile} does not expose a 'predict' method.")

        prediction_df[f"q_{quantile:g}"] = model.predict(model_input)

    return prediction_df.reset_index(drop=True)


if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression

    example_df = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2024-01-01", periods=5, freq="D"),
            "x1": [1, 2, 3, 4, 5],
            "x2": [10, 11, 12, 13, 14],
            "y": [2, 4, 6, 8, 10],
        }
    )

    model = LinearRegression()
    model.fit(example_df[["x1", "x2"]], example_df["y"])

    prediction_df = predict_with_model(
        model=model,
        df=example_df,
        feature_columns=["x1", "x2"],
        prediction_column_name="y_pred",
    )

    print(prediction_df)
"""
metrics.py

Reusable metric utilities for forecasting and backtesting.
This module centralizes common evaluation functions used across the project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, mean_squared_error

from src.utils.logger import get_logger


class MetricsError(Exception):
    """Raised when metric computation fails due to invalid inputs."""


logger = get_logger(__name__)


def _validate_metric_inputs(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate, align, and sanitize metric inputs."""
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    if y_true_array.size == 0 or y_pred_array.size == 0:
        raise MetricsError("Metric inputs cannot be empty.")

    if y_true_array.shape != y_pred_array.shape:
        raise MetricsError(
            f"Metric inputs must have the same shape. Got {y_true_array.shape} and {y_pred_array.shape}."
        )

    valid_mask = ~(np.isnan(y_true_array) | np.isnan(y_pred_array))
    if not np.any(valid_mask):
        raise MetricsError("Metric inputs contain no valid paired observations.")

    y_true_array = y_true_array[valid_mask]
    y_pred_array = y_pred_array[valid_mask]

    return y_true_array, y_pred_array
def compute_bias(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Compute mean forecast bias as the average prediction error (y_pred - y_true)."""
    y_true_array, y_pred_array = _validate_metric_inputs(y_true, y_pred)
    return float(np.mean(y_pred_array - y_true_array))


def compute_mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Compute mean absolute error."""
    y_true_array, y_pred_array = _validate_metric_inputs(y_true, y_pred)
    return float(mean_absolute_error(y_true_array, y_pred_array))


def compute_rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """Compute root mean squared error."""
    y_true_array, y_pred_array = _validate_metric_inputs(y_true, y_pred)
    return float(np.sqrt(mean_squared_error(y_true_array, y_pred_array)))


def compute_mape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """
    Compute mean absolute percentage error.

    Observations with zero true values are excluded to avoid division by zero.
    """
    y_true_array, y_pred_array = _validate_metric_inputs(y_true, y_pred)

    non_zero_mask = y_true_array != 0
    if not np.any(non_zero_mask):
        raise MetricsError("MAPE cannot be computed because all true values are zero.")

    result = np.mean(
        np.abs(
            (y_true_array[non_zero_mask] - y_pred_array[non_zero_mask])
            / y_true_array[non_zero_mask]
        )
    )
    return float(result)


def compute_pinball(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    quantile: float,
) -> float:
    """Compute pinball loss for a given quantile forecast."""
    if not 0 < quantile < 1:
        raise MetricsError("quantile must be strictly between 0 and 1.")

    y_true_array, y_pred_array = _validate_metric_inputs(y_true, y_pred)
    return float(mean_pinball_loss(y_true_array, y_pred_array, alpha=quantile))


def summarize_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    quantile: float | None = None,
) -> dict[str, float]:
    """
    Build a compact metric summary dictionary.

    Includes MAE and RMSE by default, and pinball loss when a quantile is provided.
    """
    summary = {
        "mae": compute_mae(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "bias": compute_bias(y_true, y_pred),
    }

    try:
        summary["mape"] = compute_mape(y_true, y_pred)
    except MetricsError:
        summary["mape"] = np.nan

    if quantile is not None:
        summary["pinball_loss"] = compute_pinball(y_true, y_pred, quantile=quantile)

    return summary


if __name__ == "__main__":
    y_true = pd.Series([10, 12, 14, 16])
    y_pred = pd.Series([11, 11, 15, 15])

    logger.info(f"MAE: {compute_mae(y_true, y_pred)}")
    logger.info(f"RMSE: {compute_rmse(y_true, y_pred)}")
    logger.info(f"MAPE: {compute_mape(y_true, y_pred)}")
    logger.info(f"Bias: {compute_bias(y_true, y_pred)}")
    logger.info(f"Pinball: {compute_pinball(y_true, y_pred, quantile=0.9)}")
    logger.info(f"Summary: {summarize_regression_metrics(y_true, y_pred, quantile=0.9)}")
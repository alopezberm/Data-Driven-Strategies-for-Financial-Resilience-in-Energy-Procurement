

"""
feature_importance.py

Utilities for computing and summarizing feature importance for fitted models.
The module supports tree-based importance directly and permutation importance
for any fitted estimator exposing a scikit-learn style interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.inspection import permutation_importance

from src.config.constants import TARGET_COLUMN


class FeatureImportanceError(Exception):
    """Raised when feature importance cannot be computed safely."""


@dataclass
class FeatureImportanceConfig:
    """Configuration for feature-importance calculations."""

    target_column: str = TARGET_COLUMN
    scoring: str = "neg_mean_absolute_error"
    n_repeats: int = 10
    random_state: int = 42
    top_k: int | None = None


# =========================
# Validation helpers
# =========================

def _validate_model(model: Any) -> None:
    """Validate that the supplied model is fitted enough for explainability."""
    if model is None:
        raise FeatureImportanceError("Model is None.")

    if not hasattr(model, "predict"):
        raise FeatureImportanceError(
            "Model must expose a 'predict' method to compute feature importance."
        )



def _validate_feature_inputs(
    x: pd.DataFrame,
    y: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Validate feature matrix and optional target series."""
    if x.empty:
        raise FeatureImportanceError("Feature matrix X is empty.")

    if not isinstance(x, pd.DataFrame):
        raise FeatureImportanceError("X must be provided as a pandas DataFrame.")

    if y is not None and len(x) != len(y):
        raise FeatureImportanceError(
            f"X and y must have the same length. Got {len(x)} and {len(y)}."
        )

    return x.copy(), None if y is None else y.copy()


# =========================
# Model-native importance
# =========================

def get_model_native_importance(
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extract model-native feature importance.

    Supported patterns:
    - tree-based models with `feature_importances_`
    - linear models with `coef_`
    """
    _validate_model(model)

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        importance_type = "model_native"
    elif hasattr(model, "coef_"):
        coef_values = model.coef_
        if hasattr(coef_values, "ndim") and coef_values.ndim > 1:
            coef_values = coef_values[0]
        values = abs(coef_values)
        importance_type = "absolute_coefficient"
    else:
        raise FeatureImportanceError(
            "Model does not expose 'feature_importances_' or 'coef_'."
        )

    if len(values) != len(feature_names):
        raise FeatureImportanceError(
            "Number of importance values does not match number of feature names."
        )

    importance_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance": values,
            "importance_type": importance_type,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df


# =========================
# Permutation importance
# =========================

def get_permutation_importance(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    config: FeatureImportanceConfig | None = None,
) -> pd.DataFrame:
    """
    Compute permutation importance for a fitted model.
    """
    config = FeatureImportanceConfig() if config is None else config
    _validate_model(model)
    x, y = _validate_feature_inputs(x, y)

    result = permutation_importance(
        estimator=model,
        X=x,
        y=y,
        scoring=config.scoring,
        n_repeats=config.n_repeats,
        random_state=config.random_state,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature_name": list(x.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "importance_type": "permutation",
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return importance_df


# =========================
# Convenience wrappers
# =========================

def summarize_feature_importance(
    importance_df: pd.DataFrame,
    top_k: int | None = None,
) -> pd.DataFrame:
    """
    Return a cleaned and optionally truncated feature-importance table.
    """
    if importance_df.empty:
        raise FeatureImportanceError("importance_df is empty.")

    summary_df = importance_df.copy()

    sort_column = "importance_mean" if "importance_mean" in summary_df.columns else "importance"
    summary_df = summary_df.sort_values(sort_column, ascending=False).reset_index(drop=True)

    if top_k is not None:
        if top_k <= 0:
            raise FeatureImportanceError("top_k must be strictly positive when provided.")
        summary_df = summary_df.head(top_k).reset_index(drop=True)

    return summary_df



def build_feature_importance_report(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series | None = None,
    config: FeatureImportanceConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build a compact explainability report.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with available importance summaries.
    """
    config = FeatureImportanceConfig() if config is None else config
    x, y = _validate_feature_inputs(x, y)
    _validate_model(model)

    report: dict[str, pd.DataFrame] = {}

    try:
        native_importance = get_model_native_importance(model, feature_names=list(x.columns))
        report["model_native_importance"] = summarize_feature_importance(
            native_importance,
            top_k=config.top_k,
        )
    except FeatureImportanceError:
        pass

    if y is not None:
        permutation_df = get_permutation_importance(model, x, y, config=config)
        report["permutation_importance"] = summarize_feature_importance(
            permutation_df,
            top_k=config.top_k,
        )

    if not report:
        raise FeatureImportanceError(
            "No feature-importance report could be generated for the provided model."
        )

    return report


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor

    example_x = pd.DataFrame(
        {
            "Future_M1_Price": [50, 52, 54, 53, 55, 57, 58, 60],
            "temperature_2m_mean": [10, 11, 12, 12, 13, 14, 15, 15],
            "is_weekend": [0, 0, 0, 1, 1, 0, 0, 1],
        }
    )
    example_y = pd.Series([48, 51, 53, 55, 56, 58, 60, 61], name="Spot_Price_SPEL")

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(example_x, example_y)

    report = build_feature_importance_report(
        model=model,
        x=example_x,
        y=example_y,
        config=FeatureImportanceConfig(top_k=3),
    )

    for report_name, table in report.items():
        print(f"\n=== {report_name.upper()} ===")
        print(table)
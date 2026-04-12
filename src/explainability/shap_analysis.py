

"""
shap_analysis.py

Utilities for SHAP-based explainability.
The module is designed to work primarily with tree-based models and provides
safe wrappers so SHAP can be used when the dependency is available, while
failing gracefully otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config.constants import TARGET_COLUMN


class ShapAnalysisError(Exception):
    """Raised when SHAP analysis cannot be performed safely."""


@dataclass
class ShapAnalysisConfig:
    """Configuration for SHAP explainability routines."""

    target_column: str = TARGET_COLUMN
    max_background_samples: int = 200
    max_explanation_rows: int = 200
    random_state: int = 42
    top_k_features: int | None = 10


# =========================
# Dependency helpers
# =========================

def _import_shap_module():
    """Import shap lazily so the project remains usable without it."""
    try:
        import shap  # type: ignore
    except ImportError as exc:
        raise ShapAnalysisError(
            "The 'shap' package is not installed. Install it before running SHAP analysis."
        ) from exc

    return shap


# =========================
# Validation helpers
# =========================

def _validate_model(model: Any) -> None:
    """Validate that the supplied model is compatible enough for SHAP analysis."""
    if model is None:
        raise ShapAnalysisError("Model is None.")

    if not hasattr(model, "predict"):
        raise ShapAnalysisError(
            "Model must expose a 'predict' method for SHAP analysis."
        )



def _validate_inputs(
    x: pd.DataFrame,
    y: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Validate feature matrix and optional target series."""
    if not isinstance(x, pd.DataFrame):
        raise ShapAnalysisError("X must be provided as a pandas DataFrame.")

    if x.empty:
        raise ShapAnalysisError("Feature matrix X is empty.")

    if y is not None and len(x) != len(y):
        raise ShapAnalysisError(
            f"X and y must have the same length. Got {len(x)} and {len(y)}."
        )

    return x.copy(), None if y is None else y.copy()



def _sample_dataframe(
    df: pd.DataFrame,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    """Sample a dataframe only when needed."""
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=random_state).copy()


# =========================
# SHAP core utilities
# =========================

def build_tree_explainer(
    model: Any,
    background_x: pd.DataFrame,
    config: ShapAnalysisConfig | None = None,
):
    """
    Build a SHAP TreeExplainer for a fitted tree-based model.
    """
    config = ShapAnalysisConfig() if config is None else config
    shap = _import_shap_module()
    _validate_model(model)

    background_x, _ = _validate_inputs(background_x)
    background_x = _sample_dataframe(
        background_x,
        max_rows=config.max_background_samples,
        random_state=config.random_state,
    )

    try:
        explainer = shap.TreeExplainer(model, data=background_x)
    except Exception as exc:
        raise ShapAnalysisError(
            "Failed to build SHAP TreeExplainer for the provided model."
        ) from exc

    return explainer



def compute_shap_values(
    model: Any,
    x: pd.DataFrame,
    background_x: pd.DataFrame | None = None,
    config: ShapAnalysisConfig | None = None,
):
    """
    Compute SHAP values for a fitted model on a feature dataframe.

    Returns
    -------
    tuple[pd.DataFrame, Any]
        Sampled feature dataframe used for explanation and raw SHAP values.
    """
    config = ShapAnalysisConfig() if config is None else config
    _validate_model(model)
    x, _ = _validate_inputs(x)

    explain_x = _sample_dataframe(
        x,
        max_rows=config.max_explanation_rows,
        random_state=config.random_state,
    )
    background = x if background_x is None else background_x
    background, _ = _validate_inputs(background)

    explainer = build_tree_explainer(model, background, config=config)

    try:
        shap_values = explainer.shap_values(explain_x)
    except Exception as exc:
        raise ShapAnalysisError("Failed to compute SHAP values.") from exc

    # For some models SHAP can return a list; for regression we expect one matrix.
    if isinstance(shap_values, list):
        if len(shap_values) == 0:
            raise ShapAnalysisError("Received an empty SHAP values list.")
        shap_values = shap_values[0]

    return explain_x.reset_index(drop=True), shap_values


# =========================
# Summaries
# =========================

def summarize_shap_importance(
    explain_x: pd.DataFrame,
    shap_values,
    config: ShapAnalysisConfig | None = None,
) -> pd.DataFrame:
    """
    Summarize mean absolute SHAP importance by feature.
    """
    config = ShapAnalysisConfig() if config is None else config

    if len(explain_x) == 0:
        raise ShapAnalysisError("explain_x is empty.")

    shap_df = pd.DataFrame(shap_values, columns=explain_x.columns)
    importance_df = pd.DataFrame(
        {
            "feature_name": explain_x.columns,
            "mean_abs_shap": shap_df.abs().mean(axis=0).values,
            "mean_shap": shap_df.mean(axis=0).values,
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    if config.top_k_features is not None:
        if config.top_k_features <= 0:
            raise ShapAnalysisError("top_k_features must be strictly positive.")
        importance_df = importance_df.head(config.top_k_features).reset_index(drop=True)

    return importance_df



def build_local_explanation_table(
    explain_x: pd.DataFrame,
    shap_values,
    row_index: int = 0,
    top_k_features: int = 10,
) -> pd.DataFrame:
    """
    Build a local explanation table for one explained row.
    """
    if row_index < 0 or row_index >= len(explain_x):
        raise ShapAnalysisError(
            f"row_index {row_index} is out of bounds for explanation dataset of length {len(explain_x)}."
        )

    if top_k_features <= 0:
        raise ShapAnalysisError("top_k_features must be strictly positive.")

    shap_df = pd.DataFrame(shap_values, columns=explain_x.columns)
    row_features = explain_x.iloc[row_index]
    row_shap = shap_df.iloc[row_index]

    local_df = pd.DataFrame(
        {
            "feature_name": explain_x.columns,
            "feature_value": row_features.values,
            "shap_value": row_shap.values,
            "abs_shap_value": row_shap.abs().values,
        }
    ).sort_values("abs_shap_value", ascending=False).reset_index(drop=True)

    return local_df.head(top_k_features).reset_index(drop=True)


# =========================
# Report helper
# =========================

def build_shap_report(
    model: Any,
    x: pd.DataFrame,
    background_x: pd.DataFrame | None = None,
    config: ShapAnalysisConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build a compact SHAP explainability report.

    Returns
    -------
    dict[str, pd.DataFrame]
        Contains at least a global summary and a local explanation table.
    """
    config = ShapAnalysisConfig() if config is None else config

    explain_x, shap_values = compute_shap_values(
        model=model,
        x=x,
        background_x=background_x,
        config=config,
    )

    global_summary = summarize_shap_importance(
        explain_x,
        shap_values,
        config=config,
    )
    local_explanation = build_local_explanation_table(
        explain_x,
        shap_values,
        row_index=0,
        top_k_features=config.top_k_features or 10,
    )

    return {
        "global_shap_summary": global_summary,
        "local_shap_explanation": local_explanation,
    }


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor

    example_x = pd.DataFrame(
        {
            "Future_M1_Price": [50, 52, 54, 53, 55, 57, 58, 60],
            "temperature_2m_mean": [10, 11, 12, 12, 13, 14, 15, 15],
            "is_weekend": [0, 0, 0, 1, 1, 0, 0, 1],
        }
    )
    example_y = pd.Series([48, 51, 53, 55, 56, 58, 60, 61], name=TARGET_COLUMN)

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(example_x, example_y)

    try:
        report = build_shap_report(
            model=model,
            x=example_x,
            config=ShapAnalysisConfig(top_k_features=3),
        )

        for report_name, table in report.items():
            print(f"\n=== {report_name.upper()} ===")
            print(table)
    except ShapAnalysisError as exc:
        print(exc)
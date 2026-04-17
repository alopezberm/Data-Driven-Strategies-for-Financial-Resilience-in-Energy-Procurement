"""
feature_selection.py

Utilities for selecting modeling features in a transparent and reusable way.
This module helps reduce leakage risk, remove unusable columns, and optionally
keep only the most relevant numerical predictors.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config.constants import DATE_COLUMN, TARGET_COLUMN
from src.config.settings import FeatureSelectionSettings, get_default_settings


class FeatureSelectionError(Exception):
    """Raised when feature selection cannot be performed safely."""


NON_FEATURE_COLUMNS = {
    DATE_COLUMN,
    TARGET_COLUMN,
    "target_t_plus_1",
    "target_t_plus_h",
    "source_index",
    "recommended_action",
    "decision_reason",
    "action_taken",
    "strategy_name",
    "total_cost",
    "spot_cost",
    "future_cost",
    "shift_penalty_cost",
    "energy_volume_mwh",
}


def get_default_feature_selection_settings() -> FeatureSelectionSettings:
    """Return default feature-selection settings from the project configuration."""
    return get_default_settings().feature_selection


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature-selection routines."""

    target_column: str = TARGET_COLUMN
    exclude_columns: list[str] | None = None
    max_missing_share: float = 0.40
    min_non_null_rows: int = 30
    top_k_importance: int | None = None
    random_state: int = 42
    n_estimators: int = 200

    @classmethod
    def from_settings(
        cls,
        settings: FeatureSelectionSettings,
        target_column: str = TARGET_COLUMN,
        exclude_columns: list[str] | None = None,
    ) -> "FeatureSelectionConfig":
        """Build a feature-selection config from centralized project settings."""
        return cls(
            target_column=target_column,
            exclude_columns=exclude_columns,
            max_missing_share=settings.max_missing_share,
            min_non_null_rows=settings.min_non_null_rows,
            top_k_importance=settings.top_k_importance,
            random_state=settings.random_state,
            n_estimators=settings.n_estimators,
        )

    def resolved_exclude_columns(self) -> set[str]:
        """Return the full set of excluded columns."""
        extra = set() if self.exclude_columns is None else set(self.exclude_columns)
        return set(NON_FEATURE_COLUMNS).union(extra).union({self.target_column, DATE_COLUMN})


def _get_config(config: FeatureSelectionConfig | None) -> FeatureSelectionConfig:
    """Resolve an explicit config or build one from project settings."""
    if config is not None:
        return config
    return FeatureSelectionConfig.from_settings(get_default_feature_selection_settings())


# =========================
# Validation helpers
# =========================

def _validate_input_df(df: pd.DataFrame, config: FeatureSelectionConfig) -> pd.DataFrame:
    """Validate the input dataframe used for feature selection."""
    if df.empty:
        raise FeatureSelectionError("Input dataframe is empty.")

    if config.target_column not in df.columns:
        raise FeatureSelectionError(
            f"Target column '{config.target_column}' was not found in the dataframe."
        )

    return df.copy()


# =========================
# Core feature filters
# =========================

def get_candidate_feature_columns(
    df: pd.DataFrame,
    config: FeatureSelectionConfig | None = None,
) -> list[str]:
    """
    Return the initial candidate feature columns after excluding obvious
    non-feature and target columns.
    """
    config = _get_config(config)
    df = _validate_input_df(df, config)

    excluded_columns = config.resolved_exclude_columns()
    candidate_columns = [column for column in df.columns if column not in excluded_columns]

    if not candidate_columns:
        raise FeatureSelectionError("No candidate feature columns remain after exclusions.")

    return candidate_columns



def filter_numeric_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> list[str]:
    """Keep only numeric feature columns."""
    numeric_columns = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(df[column])
    ]

    if not numeric_columns:
        raise FeatureSelectionError("No numeric feature columns were found.")

    return numeric_columns



def filter_features_by_missingness(
    df: pd.DataFrame,
    feature_columns: list[str],
    max_missing_share: float = 0.40,
    min_non_null_rows: int = 30,
) -> list[str]:
    """
    Remove features with excessive missingness or too few usable observations.
    """
    kept_columns: list[str] = []

    for column in feature_columns:
        missing_share = float(df[column].isna().mean())
        non_null_rows = int(df[column].notna().sum())

        if missing_share <= max_missing_share and non_null_rows >= min_non_null_rows:
            kept_columns.append(column)

    if not kept_columns:
        raise FeatureSelectionError(
            "All candidate feature columns were removed by missingness filtering."
        )

    return kept_columns



def remove_constant_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> list[str]:
    """Remove features with one or fewer unique non-null values."""
    kept_columns = [column for column in feature_columns if df[column].nunique(dropna=True) > 1]

    if not kept_columns:
        raise FeatureSelectionError("All candidate features are constant or degenerate.")

    return kept_columns


# =========================
# Importance-based selection
# =========================

def rank_features_by_importance(
    df: pd.DataFrame,
    feature_columns: list[str],
    config: FeatureSelectionConfig | None = None,
) -> pd.DataFrame:
    """
    Rank candidate numeric features using a random-forest regressor.

    Rows with missing values in either the target or selected features are dropped
    for the purpose of fitting the ranking model.
    """
    config = _get_config(config)
    df = _validate_input_df(df, config)

    if not feature_columns:
        raise FeatureSelectionError("feature_columns cannot be empty.")

    modeling_df = df[feature_columns + [config.target_column]].dropna().copy()
    if modeling_df.empty:
        raise FeatureSelectionError(
            "No rows remain after dropping missing values for importance ranking."
        )

    x = modeling_df[feature_columns]
    y = modeling_df[config.target_column]

    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        random_state=config.random_state,
        n_jobs=-1,
    )
    model.fit(x, y)

    importance_df = pd.DataFrame(
        {
            "feature_name": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df



def select_top_k_features(
    importance_df: pd.DataFrame,
    top_k: int,
) -> list[str]:
    """Select the top-k ranked features."""
    if importance_df.empty:
        raise FeatureSelectionError("importance_df is empty.")

    if top_k <= 0:
        raise FeatureSelectionError("top_k must be strictly positive.")

    return importance_df.head(top_k)["feature_name"].tolist()


# =========================
# End-to-end convenience function
# =========================

def select_model_features(
    df: pd.DataFrame,
    config: FeatureSelectionConfig | None = None,
) -> dict[str, object]:
    """
    End-to-end feature-selection routine.

    Steps:
    1. Exclude known non-feature columns
    2. Keep only numeric columns
    3. Filter by missingness
    4. Remove constant columns
    5. Optionally rank by importance and keep top-k

    Returns
    -------
    dict[str, object]
        Dictionary with selected feature names and optional importance table.
    """
    config = _get_config(config)
    df = _validate_input_df(df, config)

    candidate_columns = get_candidate_feature_columns(df, config=config)
    numeric_columns = filter_numeric_features(df, candidate_columns)
    usable_columns = filter_features_by_missingness(
        df,
        numeric_columns,
        max_missing_share=config.max_missing_share,
        min_non_null_rows=config.min_non_null_rows,
    )
    selected_columns = remove_constant_features(df, usable_columns)

    importance_df: pd.DataFrame | None = None
    if config.top_k_importance is not None:
        importance_df = rank_features_by_importance(df, selected_columns, config=config)
        selected_columns = select_top_k_features(importance_df, config.top_k_importance)

    return {
        "selected_features": selected_columns,
        "importance_table": importance_df,
        "n_selected_features": len(selected_columns),
    }


if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2024-01-01", periods=10, freq="D"),
            TARGET_COLUMN: [50, 55, 53, 60, 62, 58, 57, 61, 65, 63],
            "Future_M1_Price": [51, 54, 54, 59, 61, 57, 58, 60, 64, 62],
            "temperature_2m_mean": [10, 11, 12, 10, 9, 8, 7, 8, 9, 10],
            "is_weekend": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            "constant_feature": [1] * 10,
        }
    )

    config = FeatureSelectionConfig(top_k_importance=3)
    result = select_model_features(example_df, config=config)

    print("Selected features:")
    print(result["selected_features"])

    if result["importance_table"] is not None:
        print("\nImportance table:")
        print(result["importance_table"])
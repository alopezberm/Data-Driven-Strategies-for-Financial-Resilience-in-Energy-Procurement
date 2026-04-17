"""
quantile_models.py

Quantile regression models for uncertainty-aware electricity price forecasting.
These models estimate conditional quantiles of the next-day spot price,
which is central to tail-risk aware procurement decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_pinball_loss

from src.config.constants import (
    DATE_COLUMN,
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_QUANTILES,
    PRIMARY_FUTURE_COLUMN,
    PRIMARY_OPEN_INTEREST_COLUMN,
    SPOT_PRICE_COLUMN,
    TARGET_COLUMN,
)
from src.config.settings import TrainingSettings, get_default_settings




def get_default_quantile_feature_columns(
    training_settings: TrainingSettings | None = None,
) -> list[str]:
    """Build the default quantile feature set from project settings."""
    if training_settings is None:
        training_settings = get_default_settings().training

    lag_steps = list(training_settings.lag_steps)
    rolling_windows = list(training_settings.rolling_windows)

    target_lag_features = [f"{TARGET_COLUMN}_lag_{lag}" for lag in lag_steps]
    target_rolling_features: list[str] = []
    for window in rolling_windows:
        target_rolling_features.extend(
            [
                f"{TARGET_COLUMN}_rolling_mean_{window}",
                f"{TARGET_COLUMN}_rolling_std_{window}",
            ]
        )

    market_features = [
        PRIMARY_FUTURE_COLUMN,
        "Future_M2_Price",
        f"{PRIMARY_FUTURE_COLUMN}_lag_1",
        f"{PRIMARY_FUTURE_COLUMN}_lag_7",
        PRIMARY_OPEN_INTEREST_COLUMN,
        f"{PRIMARY_OPEN_INTEREST_COLUMN}_lag_1",
        f"{PRIMARY_OPEN_INTEREST_COLUMN}_lag_7",
        "spread_spot_vs_future_m1",
        "spread_spot_vs_future_m1_rel",
        "spread_future_m1_vs_future_m2",
        "front_month_premium",
        "front_month_premium_rel",
    ]

    calendar_features = [
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "is_weekend",
        "Is_national_holiday",
    ]

    weather_features = [
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "wind_speed_10m_max",
        "shortwave_radiation_sum",
        "precipitation_sum",
    ]

    return target_lag_features + target_rolling_features + market_features + calendar_features + weather_features


class QuantileModelError(Exception):
    """Raised when quantile model training, prediction, or evaluation fails."""



@dataclass
class QuantileModelResults:
    """Container for a single quantile model's predictions and metrics."""

    quantile: float
    model_name: str
    y_true: pd.Series
    y_pred: pd.Series
    pinball_loss: float
    mae: float
    rmse: float


@dataclass
class QuantileModelConfig:
    """Configuration for quantile model training."""

    target_column: str = TARGET_COLUMN
    horizon: int = DEFAULT_FORECAST_HORIZON
    quantiles: list[float] | None = None
    feature_columns: list[str] | None = None
    model_params: dict | None = None

    def resolved_quantiles(self) -> list[float]:
        """Return configured quantiles or project defaults."""
        return list(DEFAULT_QUANTILES) if self.quantiles is None else list(self.quantiles)

    def resolved_feature_columns(self) -> list[str]:
        """Return configured feature columns or module defaults."""
        return (
            get_default_quantile_feature_columns()
            if self.feature_columns is None
            else list(self.feature_columns)
        )

    @classmethod
    def from_training_settings(
        cls,
        training_settings: TrainingSettings,
    ) -> "QuantileModelConfig":
        """Build quantile-model configuration from centralized training settings."""
        return cls(
            target_column=training_settings.target_column,
            horizon=training_settings.forecast_horizon,
            quantiles=list(training_settings.quantiles),
            feature_columns=get_default_quantile_feature_columns(training_settings),
            model_params=None,
        )


# =========================
# Validation helpers
# =========================


def _validate_input_dataframe(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Validate input dataframe and standardize its date column."""
    if df.empty:
        raise QuantileModelError("Input dataframe is empty.")

    if DATE_COLUMN not in df.columns:
        raise QuantileModelError(
            f"Input dataframe must contain a '{DATE_COLUMN}' column."
        )

    if target_column not in df.columns:
        raise QuantileModelError(
            f"Input dataframe must contain target column '{target_column}'."
        )

    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    if df[DATE_COLUMN].isna().any():
        invalid_count = int(df[DATE_COLUMN].isna().sum())
        raise QuantileModelError(
            f"Found {invalid_count} invalid date values in input dataframe."
        )

    if df[DATE_COLUMN].duplicated().any():
        raise QuantileModelError("Input dataframe contains duplicated dates.")

    return df.sort_values(DATE_COLUMN).reset_index(drop=True)




def _validate_quantiles(quantiles: list[float]) -> None:
    """Validate quantile values."""
    if not quantiles:
        raise QuantileModelError("quantiles list cannot be empty.")

    for quantile in quantiles:
        if not isinstance(quantile, (int, float)):
            raise QuantileModelError("All quantiles must be numeric.")
        if not 0 < float(quantile) < 1:
            raise QuantileModelError("All quantiles must be strictly between 0 and 1.")


def get_default_quantile_model_config() -> QuantileModelConfig:
    """Build the default quantile-model configuration from project settings."""
    settings = get_default_settings()
    return QuantileModelConfig.from_training_settings(settings.training)


# =========================
# Target preparation
# =========================

def prepare_quantile_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    horizon: int = DEFAULT_FORECAST_HORIZON,
) -> pd.DataFrame:
    """
    Create a forward target for supervised quantile forecasting.

    Features at time t are used to predict the target at time t + horizon.
    """
    if horizon <= 0:
        raise QuantileModelError("horizon must be a positive integer.")

    df = _validate_input_dataframe(df, target_column)
    df = df.copy()
    df[f"target_{target_column}_t_plus_{horizon}"] = df[target_column].shift(-horizon)

    return df


# =========================
# Training / prediction helpers
# =========================

def _prepare_model_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_name: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """Prepare aligned train/test matrices using the intersection of available features."""
    available_features = [
        col for col in feature_columns if col in train_df.columns and col in test_df.columns
    ]

    if not available_features:
        raise QuantileModelError(
            "No valid feature columns were found for quantile modeling."
        )

    train_model_df = train_df[available_features + [target_name]].dropna().copy()
    test_model_df = test_df[available_features + [target_name]].dropna().copy()

    if train_model_df.empty or test_model_df.empty:
        raise QuantileModelError(
            "Train or test data became empty after dropping NaNs for quantile modeling."
        )

    X_train = train_model_df[available_features]
    y_train = train_model_df[target_name]
    X_test = test_model_df[available_features]
    y_test = test_model_df[target_name]

    return X_train, y_train, X_test, y_test, available_features



def _compute_metrics(y_true: pd.Series, y_pred: pd.Series, quantile: float) -> tuple[float, float, float]:
    """Compute pinball loss, MAE, and RMSE."""
    pinball = float(mean_pinball_loss(y_true, y_pred, alpha=quantile))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return pinball, mae, rmse


# =========================
# Public API
# =========================

def train_single_quantile_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    quantile: float,
    target_column: str = TARGET_COLUMN,
    horizon: int = DEFAULT_FORECAST_HORIZON,
    feature_columns: list[str] | None = None,
    model_params: dict | None = None,
) -> tuple[QuantileModelResults, GradientBoostingRegressor, list[str]]:
    """
    Train and evaluate a single Gradient Boosting quantile regressor.

    Returns
    -------
    tuple[QuantileModelResults, GradientBoostingRegressor, list[str]]
        Results object, fitted model, and list of used features.
    """
    _validate_quantiles([quantile])

    default_config = get_default_quantile_model_config()
    feature_columns = default_config.resolved_feature_columns() if feature_columns is None else feature_columns

    default_model_params = {
        "loss": "quantile",
        "alpha": float(quantile),
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_samples_leaf": 5,
        "random_state": 42,
    }

    if model_params is not None:
        default_model_params.update(model_params)

    prepared_train = prepare_quantile_target(
        train_df, target_column=target_column, horizon=horizon
    )
    prepared_test = prepare_quantile_target(
        test_df, target_column=target_column, horizon=horizon
    )

    target_name = f"target_{target_column}_t_plus_{horizon}"
    X_train, y_train, X_test, y_test, used_features = _prepare_model_frames(
        prepared_train,
        prepared_test,
        feature_columns,
        target_name,
    )

    model = GradientBoostingRegressor(**default_model_params)
    model.fit(X_train, y_train)

    y_pred = pd.Series(model.predict(X_test), index=y_test.index, name=f"q_{quantile}_prediction")
    pinball, mae, rmse = _compute_metrics(y_test, y_pred, quantile)

    results = QuantileModelResults(
        quantile=float(quantile),
        model_name=f"gbr_quantile_{quantile}",
        y_true=y_test,
        y_pred=y_pred,
        pinball_loss=pinball,
        mae=mae,
        rmse=rmse,
    )

    return results, model, used_features




def train_quantile_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    quantiles: list[float] | None = None,
    target_column: str = TARGET_COLUMN,
    horizon: int = DEFAULT_FORECAST_HORIZON,
    feature_columns: list[str] | None = None,
    model_params: dict | None = None,
) -> tuple[list[QuantileModelResults], dict[float, GradientBoostingRegressor], list[str]]:
    """
    Train and evaluate one model per quantile.

    Returns
    -------
    tuple[list[QuantileModelResults], dict[float, GradientBoostingRegressor], list[str]]
        Results list, dictionary of fitted models, and feature list used.
    """
    default_config = get_default_quantile_model_config()
    quantiles = default_config.resolved_quantiles() if quantiles is None else quantiles
    _validate_quantiles(quantiles)

    results_list: list[QuantileModelResults] = []
    models: dict[float, GradientBoostingRegressor] = {}
    used_features_reference: list[str] | None = None

    for quantile in quantiles:
        results, model, used_features = train_single_quantile_model(
            train_df=train_df,
            test_df=test_df,
            quantile=float(quantile),
            target_column=target_column,
            horizon=horizon,
            feature_columns=feature_columns,
            model_params=model_params,
        )
        results_list.append(results)
        models[float(quantile)] = model
        used_features_reference = used_features

    return results_list, models, (used_features_reference or [])


def train_quantile_models_from_config(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: QuantileModelConfig | None = None,
) -> tuple[list[QuantileModelResults], dict[float, GradientBoostingRegressor], list[str]]:
    """
    Train quantile models using a QuantileModelConfig object.
    """
    config = get_default_quantile_model_config() if config is None else config

    return train_quantile_models(
        train_df=train_df,
        test_df=test_df,
        quantiles=config.resolved_quantiles(),
        target_column=config.target_column,
        horizon=config.horizon,
        feature_columns=config.resolved_feature_columns(),
        model_params=config.model_params,
    )



def summarize_quantile_results(results: list[QuantileModelResults]) -> pd.DataFrame:
    """Create a compact comparison table for quantile models."""
    if not results:
        raise QuantileModelError("results list cannot be empty.")

    summary_df = pd.DataFrame(
        {
            "model_name": [result.model_name for result in results],
            "quantile": [result.quantile for result in results],
            "pinball_loss": [result.pinball_loss for result in results],
            "mae": [result.mae for result in results],
            "rmse": [result.rmse for result in results],
        }
    ).sort_values(["quantile", "pinball_loss"], ascending=[True, True]).reset_index(drop=True)

    return summary_df



def combine_quantile_predictions(results: list[QuantileModelResults]) -> pd.DataFrame:
    """
    Combine multiple quantile predictions into a single dataframe for plotting / analysis.

    Assumes all results were generated on the same test index.
    """
    if not results:
        raise QuantileModelError("results list cannot be empty.")

    combined_df = pd.DataFrame(index=results[0].y_true.index)
    combined_df["y_true"] = results[0].y_true

    for result in sorted(results, key=lambda x: x.quantile):
        combined_df[f"q_{result.quantile}"] = result.y_pred

    return combined_df



if __name__ == "__main__":
    date_index = pd.date_range("2024-01-01", periods=120, freq="D")
    base_series = np.linspace(50, 90, 120) + 5 * np.sin(np.arange(120) / 7)

    example_df = pd.DataFrame(
        {
            DATE_COLUMN: date_index,
            TARGET_COLUMN: base_series,
            PRIMARY_FUTURE_COLUMN: base_series + 1.5,
            "Future_M2_Price": base_series + 2.5,
            PRIMARY_OPEN_INTEREST_COLUMN: np.linspace(1000, 1200, 120),
            "temperature_2m_mean": 15 + 10 * np.sin(np.arange(120) / 30),
            "temperature_2m_max": 20 + 10 * np.sin(np.arange(120) / 30),
            "temperature_2m_min": 10 + 10 * np.sin(np.arange(120) / 30),
            "wind_speed_10m_max": 5 + np.cos(np.arange(120) / 12),
            "shortwave_radiation_sum": 200 + 50 * np.sin(np.arange(120) / 20),
            "precipitation_sum": np.abs(np.sin(np.arange(120) / 10)),
        }
    )

    # Lightweight feature examples for standalone smoke test
    example_df[f"{TARGET_COLUMN}_lag_1"] = example_df[TARGET_COLUMN].shift(1)
    example_df[f"{TARGET_COLUMN}_lag_2"] = example_df[TARGET_COLUMN].shift(2)
    example_df[f"{TARGET_COLUMN}_lag_3"] = example_df[TARGET_COLUMN].shift(3)
    example_df[f"{TARGET_COLUMN}_lag_7"] = example_df[TARGET_COLUMN].shift(7)
    example_df[f"{TARGET_COLUMN}_lag_14"] = example_df[TARGET_COLUMN].shift(14)
    example_df[f"{TARGET_COLUMN}_lag_28"] = example_df[TARGET_COLUMN].shift(28)
    example_df[f"{PRIMARY_FUTURE_COLUMN}_lag_1"] = example_df[PRIMARY_FUTURE_COLUMN].shift(1)
    example_df[f"{PRIMARY_FUTURE_COLUMN}_lag_7"] = example_df[PRIMARY_FUTURE_COLUMN].shift(7)
    example_df[f"{PRIMARY_OPEN_INTEREST_COLUMN}_lag_1"] = example_df[PRIMARY_OPEN_INTEREST_COLUMN].shift(1)
    example_df[f"{PRIMARY_OPEN_INTEREST_COLUMN}_lag_7"] = example_df[PRIMARY_OPEN_INTEREST_COLUMN].shift(7)
    example_df[f"{TARGET_COLUMN}_rolling_mean_7"] = example_df[TARGET_COLUMN].shift(1).rolling(7).mean()
    example_df[f"{TARGET_COLUMN}_rolling_std_7"] = example_df[TARGET_COLUMN].shift(1).rolling(7).std()
    example_df[f"{TARGET_COLUMN}_rolling_mean_14"] = example_df[TARGET_COLUMN].shift(1).rolling(14).mean()
    example_df[f"{TARGET_COLUMN}_rolling_std_14"] = example_df[TARGET_COLUMN].shift(1).rolling(14).std()
    example_df["spread_spot_vs_future_m1"] = example_df[SPOT_PRICE_COLUMN] - example_df[PRIMARY_FUTURE_COLUMN]
    example_df["spread_spot_vs_future_m1_rel"] = example_df["spread_spot_vs_future_m1"] / example_df[PRIMARY_FUTURE_COLUMN]
    example_df["spread_future_m1_vs_future_m2"] = example_df[PRIMARY_FUTURE_COLUMN] - example_df["Future_M2_Price"]
    example_df["front_month_premium"] = example_df[PRIMARY_FUTURE_COLUMN] - example_df[SPOT_PRICE_COLUMN]
    example_df["front_month_premium_rel"] = example_df["front_month_premium"] / example_df[SPOT_PRICE_COLUMN]
    example_df["day_of_week_sin"] = np.sin(2 * np.pi * example_df[DATE_COLUMN].dt.dayofweek / 7)
    example_df["day_of_week_cos"] = np.cos(2 * np.pi * example_df[DATE_COLUMN].dt.dayofweek / 7)
    example_df["month_sin"] = np.sin(2 * np.pi * example_df[DATE_COLUMN].dt.month / 12)
    example_df["month_cos"] = np.cos(2 * np.pi * example_df[DATE_COLUMN].dt.month / 12)
    example_df["day_of_year_sin"] = np.sin(2 * np.pi * example_df[DATE_COLUMN].dt.dayofyear / 365.25)
    example_df["day_of_year_cos"] = np.cos(2 * np.pi * example_df[DATE_COLUMN].dt.dayofyear / 365.25)
    example_df["is_weekend"] = (example_df[DATE_COLUMN].dt.dayofweek >= 5).astype(int)
    example_df["Is_national_holiday"] = 0

    train_example = example_df.iloc[:90].copy()
    test_example = example_df.iloc[90:].copy()

    config = get_default_quantile_model_config()
    print(config)

    results_list, _, _ = train_quantile_models_from_config(train_example, test_example, config=config)
    summary_df = summarize_quantile_results(results_list)
    combined_df = combine_quantile_predictions(results_list)

    print(summary_df)
    print(combined_df.head())

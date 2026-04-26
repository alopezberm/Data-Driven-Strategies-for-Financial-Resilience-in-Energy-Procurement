

"""
test_models.py

Unit tests for the modeling layer:
- baseline_models
- quantile_models
- tail_risk_models
- evaluate_model

These tests use small synthetic datasets so they run quickly while still
checking the main expected behavior of the project's model utilities.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.baseline_models import (
    linear_regression_baseline,
    naive_last_value_baseline,
    rolling_mean_baseline,
    seasonal_naive_baseline,
    summarize_baseline_results,
)
from src.models.evaluate_model import (
    build_quantile_diagnostics_report,
    combine_quantile_predictions,
    evaluate_prediction_interval,
    evaluate_quantile_predictions,
)
from src.models.quantile_models import (
    summarize_quantile_results,
    train_quantile_models,
    train_quantile_models_from_config,
    QuantileModelConfig,
)
from src.models.tail_risk_models import (
    TailRiskConfig,
    build_tail_risk_dataset,
    compute_tail_risk_features,
    compute_tail_risk_metrics,
    flag_extreme_events,
)


@pytest.fixture
def model_df() -> pd.DataFrame:
    n = 80
    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    df = pd.DataFrame(
        {
            "date": dates,
            "Spot_Price_SPEL": [50 + i * 0.5 + (i % 7) for i in range(n)],
            "Future_M1_Price": [52 + i * 0.45 + (i % 5) for i in range(n)],
            "Future_M1_OpenInterest": [1000 + i * 3 for i in range(n)],
            "temperature_2m_mean": [10 + (i % 10) for i in range(n)],
            "wind_speed_10m_max": [20 + (i % 4) for i in range(n)],
            "is_weekend": [1 if d.weekday() >= 5 else 0 for d in dates],
            "day_of_week_sin": [0.0 for _ in range(n)],
            "day_of_week_cos": [1.0 for _ in range(n)],
            "month_sin": [0.0 for _ in range(n)],
            "month_cos": [1.0 for _ in range(n)],
            "day_of_year_sin": [0.0 for _ in range(n)],
            "day_of_year_cos": [1.0 for _ in range(n)],
        }
    )

    # Add common lag features expected by quantile/baseline utilities.
    for lag in [1, 2, 3, 7, 14, 28]:
        df[f"Spot_Price_SPEL_lag_{lag}"] = df["Spot_Price_SPEL"].shift(lag)
    for lag in [1, 7]:
        df[f"Future_M1_Price_lag_{lag}"] = df["Future_M1_Price"].shift(lag)
        df[f"Future_M1_OpenInterest_lag_{lag}"] = df["Future_M1_OpenInterest"].shift(lag)

    return df


@pytest.fixture
def train_test_df(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = model_df.iloc[:55].copy().reset_index(drop=True)
    test_df = model_df.iloc[55:].copy().reset_index(drop=True)
    return train_df, test_df


# =========================
# baseline_models tests
# =========================


def test_baseline_models_return_results_objects(train_test_df: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train_df, test_df = train_test_df

    naive_results = naive_last_value_baseline(test_df)
    seasonal_results = seasonal_naive_baseline(test_df, seasonal_lag=7)
    rolling_results = rolling_mean_baseline(test_df, window=7)
    linear_results, linear_model = linear_regression_baseline(train_df, test_df)

    assert naive_results.model_name
    assert seasonal_results.model_name
    assert rolling_results.model_name
    assert linear_results.model_name
    assert linear_model is not None

    assert len(naive_results.y_true) == len(naive_results.y_pred)
    assert len(seasonal_results.y_true) == len(seasonal_results.y_pred)
    assert len(rolling_results.y_true) == len(rolling_results.y_pred)
    assert len(linear_results.y_true) == len(linear_results.y_pred)



def test_summarize_baseline_results_returns_non_empty_table(train_test_df: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train_df, test_df = train_test_df

    results = [
        naive_last_value_baseline(test_df),
        seasonal_naive_baseline(test_df, seasonal_lag=7),
        rolling_mean_baseline(test_df, window=7),
        linear_regression_baseline(train_df, test_df)[0],
    ]

    summary_df = summarize_baseline_results(results)

    assert not summary_df.empty
    assert "model_name" in summary_df.columns
    assert any(col in summary_df.columns for col in ["mae", "rmse"])


# =========================
# quantile_models tests
# =========================


def test_train_quantile_models_returns_expected_quantiles(train_test_df: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train_df, test_df = train_test_df

    results, models, used_features = train_quantile_models(
        train_df=train_df,
        test_df=test_df,
        quantiles=[0.5, 0.9],
    )

    assert len(results) == 2
    assert set(models.keys()) == {0.5, 0.9}
    assert len(used_features) > 0
    assert {result.quantile for result in results} == {0.5, 0.9}



def test_train_quantile_models_from_config_works(train_test_df: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train_df, test_df = train_test_df

    config = QuantileModelConfig(quantiles=[0.5, 0.9])
    results, models, used_features = train_quantile_models_from_config(
        train_df=train_df,
        test_df=test_df,
        config=config,
    )

    assert len(results) == 2
    assert len(models) == 2
    assert len(used_features) > 0



def test_quantile_result_helpers_build_tables(train_test_df: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train_df, test_df = train_test_df

    results, _, _ = train_quantile_models(
        train_df=train_df,
        test_df=test_df,
        quantiles=[0.5, 0.9, 0.95],
    )

    summary_df = summarize_quantile_results(results)
    combined_df = combine_quantile_predictions(results)

    assert not summary_df.empty
    assert not combined_df.empty
    assert "y_true" in combined_df.columns
    assert "q_0.5" in combined_df.columns
    assert "q_0.9" in combined_df.columns


# =========================
# evaluate_model tests
# =========================


def test_evaluate_quantile_predictions_returns_expected_columns(train_test_df: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train_df, test_df = train_test_df
    results, _, _ = train_quantile_models(train_df=train_df, test_df=test_df, quantiles=[0.9])
    result = results[0]

    eval_df = evaluate_quantile_predictions(result.y_true, result.y_pred, quantile=0.9)

    assert isinstance(eval_df, pd.DataFrame)
    assert not eval_df.empty
    assert "quantile" in eval_df.columns
    assert "empirical_coverage" in eval_df.columns



def test_evaluate_prediction_interval_returns_interval_metrics(train_test_df: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train_df, test_df = train_test_df
    results, _, _ = train_quantile_models(train_df=train_df, test_df=test_df, quantiles=[0.5, 0.9])
    combined_df = combine_quantile_predictions(results)

    interval_df = evaluate_prediction_interval(combined_df, "q_0.5", "q_0.9")

    assert isinstance(interval_df, pd.DataFrame)
    assert not interval_df.empty
    assert "empirical_coverage" in interval_df.columns
    assert "average_interval_width" in interval_df.columns



def test_build_quantile_diagnostics_report_returns_expected_sections(train_test_df: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train_df, test_df = train_test_df
    results, _, _ = train_quantile_models(train_df=train_df, test_df=test_df, quantiles=[0.5, 0.9, 0.95])

    report = build_quantile_diagnostics_report(results)

    assert "coverage_summary" in report
    assert "interval_summary" in report
    assert "upper_tail_exceedance_summary" in report
    assert not report["coverage_summary"].empty


# =========================
# tail_risk_models tests
# =========================


def test_tail_risk_pipeline_builds_features_and_metrics(train_test_df: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train_df, test_df = train_test_df
    results, _, _ = train_quantile_models(train_df=train_df, test_df=test_df, quantiles=[0.5, 0.9, 0.95])
    combined_df = combine_quantile_predictions(results)

    config = TailRiskConfig(high_quantile=0.9, extreme_quantile=0.95)

    featured_df = compute_tail_risk_features(combined_df, config=config)
    flagged_df = flag_extreme_events(featured_df, config=config)
    metrics = compute_tail_risk_metrics(flagged_df, config=config)
    final_df, final_metrics = build_tail_risk_dataset(combined_df, config=config)

    assert "tail_spread" in featured_df.columns
    assert "extreme_spread" in featured_df.columns
    assert "tail_ratio" in featured_df.columns
    assert "is_high_risk" in flagged_df.columns
    assert "is_extreme_risk" in flagged_df.columns
    assert "mean_tail_spread" in metrics
    assert "pct_extreme_risk_days" in metrics
    assert not final_df.empty
    assert set(metrics.keys()) == set(final_metrics.keys())


# =========================
# failure-mode tests
# =========================


def test_train_quantile_models_raises_with_invalid_quantile(train_test_df: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train_df, test_df = train_test_df

    with pytest.raises(Exception):
        train_quantile_models(train_df=train_df, test_df=test_df, quantiles=[1.5])



def test_tail_risk_functions_raise_when_quantile_columns_missing() -> None:
    invalid_df = pd.DataFrame({"q_0.5": [1, 2, 3]})

    with pytest.raises(Exception):
        compute_tail_risk_features(invalid_df)
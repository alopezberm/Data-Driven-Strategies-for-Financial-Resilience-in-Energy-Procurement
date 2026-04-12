"""
test_backtesting.py

Unit tests for the backtesting layer:
- simulate_baseline
- simulate_policy
- compare_strategies
- resilience_metrics

These tests use small synthetic datasets and focus on validating the expected
structure, outputs, and a few key behavioral assumptions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtesting.compare_strategies import (
    build_strategy_comparison_report,
    build_strategy_summary_table,
)
from src.backtesting.resilience_metrics import (
    build_resilience_report,
    build_resilience_summary_table,
)
from src.backtesting.simulate_baseline import (
    BaselineSimulationConfig,
    simulate_spot_only_baseline,
    simulate_static_hedge_baseline,
)
from src.backtesting.simulate_policy import (
    PolicySimulationConfig,
    simulate_policy_strategy,
)


@pytest.fixture
def baseline_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "Spot_Price_SPEL": [50, 55, 60, 65, 70, 68, 66, 64],
            "Future_M1_Price": [52, 54, 58, 61, 63, 62, 61, 60],
            "daily_energy_mwh": [1.0] * 8,
        }
    )


@pytest.fixture
def policy_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "Spot_Price_SPEL": [50, 55, 60, 65, 70, 68, 66, 64],
            "Future_M1_Price": [52, 54, 58, 61, 63, 62, 61, 60],
            "daily_energy_mwh": [1.0] * 8,
            "recommended_action": [
                "do_nothing",
                "buy_m1_future",
                "buy_m1_future",
                "shift_production",
                "buy_m1_future",
                "do_nothing",
                "shift_production",
                "do_nothing",
            ],
            "decision_reason": [
                "No rule triggered",
                "Tail risk exceeds futures price threshold",
                "Tail risk exceeds futures price threshold",
                "Weekend + high tail risk vs central forecast",
                "Tail risk exceeds futures price threshold",
                "No rule triggered",
                "Weekend + high tail risk vs central forecast",
                "No rule triggered",
            ],
        }
    )


# =========================
# simulate_baseline tests
# =========================


def test_simulate_spot_only_baseline_returns_expected_columns(
    baseline_input_df: pd.DataFrame,
) -> None:
    result_df = simulate_spot_only_baseline(baseline_input_df)

    assert not result_df.empty
    assert len(result_df) == len(baseline_input_df)
    assert "strategy_name" in result_df.columns
    assert "spot_cost" in result_df.columns
    assert "total_cost" in result_df.columns
    assert (result_df["strategy_name"] == "spot_only").all()
    assert (result_df["total_cost"] == result_df["spot_cost"]).all()



def test_simulate_static_hedge_baseline_returns_expected_columns(
    baseline_input_df: pd.DataFrame,
) -> None:
    result_df = simulate_static_hedge_baseline(
        baseline_input_df,
        config=BaselineSimulationConfig(hedge_ratio=0.7),
    )

    assert not result_df.empty
    assert len(result_df) == len(baseline_input_df)
    assert "strategy_name" in result_df.columns
    assert "future_cost" in result_df.columns
    assert "spot_cost" in result_df.columns
    assert "total_cost" in result_df.columns
    assert (result_df["strategy_name"] == "static_hedge").all()


# =========================
# simulate_policy tests
# =========================


def test_simulate_policy_strategy_returns_expected_columns(
    policy_input_df: pd.DataFrame,
) -> None:
    result_df = simulate_policy_strategy(
        policy_input_df,
        config=PolicySimulationConfig(
            hedge_ratio_on_buy_future=1.0,
            shift_fraction=1.0,
            shift_penalty_per_mwh=2.0,
        ),
    )

    assert not result_df.empty
    assert len(result_df) == len(policy_input_df)
    assert "strategy_name" in result_df.columns
    assert "action_taken" in result_df.columns
    assert "total_cost" in result_df.columns
    assert "spot_cost" in result_df.columns
    assert "future_cost" in result_df.columns
    assert "shift_penalty_cost" in result_df.columns
    assert (result_df["strategy_name"] == "heuristic_policy").all()



def test_simulate_policy_strategy_reflects_shift_penalty(
    policy_input_df: pd.DataFrame,
) -> None:
    result_df = simulate_policy_strategy(
        policy_input_df,
        config=PolicySimulationConfig(
            hedge_ratio_on_buy_future=1.0,
            shift_fraction=1.0,
            shift_penalty_per_mwh=2.0,
        ),
    )

    shifted_rows = result_df[result_df["action_taken"] == "shift_production"]
    assert not shifted_rows.empty
    assert (shifted_rows["shift_penalty_cost"] > 0).all()


# =========================
# compare_strategies tests
# =========================


def test_build_strategy_summary_table_returns_summary(
    baseline_input_df: pd.DataFrame,
    policy_input_df: pd.DataFrame,
) -> None:
    spot_only_df = simulate_spot_only_baseline(baseline_input_df)
    static_hedge_df = simulate_static_hedge_baseline(
        baseline_input_df,
        config=BaselineSimulationConfig(hedge_ratio=0.7),
    )
    policy_df = simulate_policy_strategy(policy_input_df)

    summary_df = build_strategy_summary_table([spot_only_df, static_hedge_df, policy_df])

    assert not summary_df.empty
    assert "strategy_name" in summary_df.columns
    assert "total_cost" in summary_df.columns
    assert "n_days" in summary_df.columns
    assert set(summary_df["strategy_name"]) == {"spot_only", "static_hedge", "heuristic_policy"}



def test_build_strategy_comparison_report_returns_expected_sections(
    baseline_input_df: pd.DataFrame,
    policy_input_df: pd.DataFrame,
) -> None:
    spot_only_df = simulate_spot_only_baseline(baseline_input_df)
    static_hedge_df = simulate_static_hedge_baseline(
        baseline_input_df,
        config=BaselineSimulationConfig(hedge_ratio=0.7),
    )
    policy_df = simulate_policy_strategy(policy_input_df)

    report = build_strategy_comparison_report(
        [spot_only_df, static_hedge_df, policy_df],
        reference_strategy_name="spot_only",
    )

    assert "summary_table" in report
    assert "summary_vs_reference" in report
    assert "daily_comparison" in report
    assert not report["summary_table"].empty
    assert not report["summary_vs_reference"].empty
    assert not report["daily_comparison"].empty


# =========================
# resilience_metrics tests
# =========================


def test_build_resilience_summary_table_returns_expected_columns(
    baseline_input_df: pd.DataFrame,
    policy_input_df: pd.DataFrame,
) -> None:
    spot_only_df = simulate_spot_only_baseline(baseline_input_df)
    static_hedge_df = simulate_static_hedge_baseline(
        baseline_input_df,
        config=BaselineSimulationConfig(hedge_ratio=0.7),
    )
    policy_df = simulate_policy_strategy(policy_input_df)

    summary_df = build_resilience_summary_table(
        [spot_only_df, static_hedge_df, policy_df],
        extreme_cost_quantile=0.9,
    )

    assert not summary_df.empty
    assert "strategy_name" in summary_df.columns
    assert "n_extreme_cost_days" in summary_df.columns
    assert "share_extreme_cost_days" in summary_df.columns



def test_build_resilience_report_returns_expected_sections(
    baseline_input_df: pd.DataFrame,
    policy_input_df: pd.DataFrame,
) -> None:
    spot_only_df = simulate_spot_only_baseline(baseline_input_df)
    static_hedge_df = simulate_static_hedge_baseline(
        baseline_input_df,
        config=BaselineSimulationConfig(hedge_ratio=0.7),
    )
    policy_df = simulate_policy_strategy(policy_input_df)

    report = build_resilience_report(
        [spot_only_df, static_hedge_df, policy_df],
        reference_strategy_name="spot_only",
        extreme_cost_quantile=0.9,
    )

    assert "resilience_summary" in report
    assert "resilience_vs_reference" in report
    assert "extreme_reference_days_comparison" in report
    assert not report["resilience_summary"].empty
    assert not report["resilience_vs_reference"].empty


# =========================
# failure-mode tests
# =========================


def test_compare_strategies_raises_when_dates_do_not_match(
    baseline_input_df: pd.DataFrame,
    policy_input_df: pd.DataFrame,
) -> None:
    spot_only_df = simulate_spot_only_baseline(baseline_input_df)
    policy_df = simulate_policy_strategy(policy_input_df)

    shifted_dates_df = policy_df.copy()
    shifted_dates_df["date"] = shifted_dates_df["date"] + pd.Timedelta(days=1)

    with pytest.raises(Exception):
        build_strategy_comparison_report(
            [spot_only_df, shifted_dates_df],
            reference_strategy_name="spot_only",
        )



def test_simulate_policy_strategy_raises_when_action_column_missing(
    baseline_input_df: pd.DataFrame,
) -> None:
    invalid_df = baseline_input_df.copy()

    with pytest.raises(Exception):
        simulate_policy_strategy(invalid_df)

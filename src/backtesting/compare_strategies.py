"""
compare_strategies.py

Utilities to compare simulated procurement strategies in the backtesting layer.
This module is designed to work with outputs from:
- simulate_spot_only_baseline
- simulate_static_hedge_baseline
- simulate_policy_strategy
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.config.constants import DATE_COLUMN, DEFAULT_REFERENCE_STRATEGY, STRATEGY_HEURISTIC_POLICY, STRATEGY_SPOT_ONLY, STRATEGY_STATIC_HEDGE


class StrategyComparisonError(Exception):
    """Raised when simulated strategies cannot be compared safely."""


REQUIRED_SIMULATION_COLUMNS = {
    DATE_COLUMN,
    "strategy_name",
    "total_cost",
    "energy_volume_mwh",
}


def _validate_strategy_catalog() -> None:
    """Validate the centralized strategy catalog used by this module."""
    expected_strategies = {
        STRATEGY_SPOT_ONLY,
        STRATEGY_STATIC_HEDGE,
        STRATEGY_HEURISTIC_POLICY,
    }
    if {STRATEGY_SPOT_ONLY, STRATEGY_STATIC_HEDGE, STRATEGY_HEURISTIC_POLICY} != expected_strategies:
        raise StrategyComparisonError(
            "Centralized strategy constants are inconsistent."
        )


# =========================
# Validation helpers
# =========================

def _validate_single_simulation_df(simulation_df: pd.DataFrame) -> pd.DataFrame:
    """Validate one simulated strategy dataframe."""
    if simulation_df.empty:
        raise StrategyComparisonError("One of the simulation dataframes is empty.")

    missing_columns = REQUIRED_SIMULATION_COLUMNS - set(simulation_df.columns)
    if missing_columns:
        raise StrategyComparisonError(
            f"Simulation dataframe is missing required columns: {sorted(missing_columns)}"
        )

    validated_df = simulation_df.copy()
    validated_df[DATE_COLUMN] = pd.to_datetime(validated_df[DATE_COLUMN], errors="coerce")

    if validated_df[DATE_COLUMN].isna().any():
        invalid_count = int(validated_df[DATE_COLUMN].isna().sum())
        raise StrategyComparisonError(
            f"Found {invalid_count} invalid date values in a simulation dataframe."
        )

    if validated_df[DATE_COLUMN].duplicated().any():
        raise StrategyComparisonError(
            "A simulation dataframe contains duplicated dates."
        )

    strategy_names = validated_df["strategy_name"].dropna().unique()
    if len(strategy_names) != 1:
        raise StrategyComparisonError(
            "Each simulation dataframe must contain exactly one strategy_name."
        )

    return validated_df.sort_values(DATE_COLUMN).reset_index(drop=True)



def _validate_simulation_collection(simulation_dfs: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
    """Validate a collection of simulated strategy dataframes."""
    _validate_strategy_catalog()
    if not simulation_dfs:
        raise StrategyComparisonError("simulation_dfs cannot be empty.")

    validated_dfs = [_validate_single_simulation_df(df) for df in simulation_dfs]

    strategy_names = [df["strategy_name"].iloc[0] for df in validated_dfs]
    if len(strategy_names) != len(set(strategy_names)):
        raise StrategyComparisonError(
            f"Strategy names must be unique across simulations. Found: {strategy_names}"
        )

    reference_dates = validated_dfs[0][DATE_COLUMN]
    for df in validated_dfs[1:]:
        if not df[DATE_COLUMN].equals(reference_dates):
            raise StrategyComparisonError(
                "All simulation dataframes must share the same ordered date index."
            )

    return validated_dfs


# =========================
# Summary builders
# =========================

def summarize_strategy(simulation_df: pd.DataFrame) -> pd.DataFrame:
    """Create a one-row summary for a single simulated strategy."""
    simulation_df = _validate_single_simulation_df(simulation_df)

    strategy_name = simulation_df["strategy_name"].iloc[0]
    total_cost = float(simulation_df["total_cost"].sum())
    total_volume = float(simulation_df["energy_volume_mwh"].sum())
    average_unit_cost = total_cost / total_volume if total_volume > 0 else pd.NA
    daily_cost_volatility = float(simulation_df["total_cost"].std())
    max_daily_cost = float(simulation_df["total_cost"].max())
    min_daily_cost = float(simulation_df["total_cost"].min())

    summary = pd.DataFrame(
        {
            "strategy_name": [strategy_name],
            "n_days": [int(len(simulation_df))],
            "total_volume_mwh": [total_volume],
            "total_cost": [total_cost],
            "average_unit_cost": [average_unit_cost],
            "daily_cost_volatility": [daily_cost_volatility],
            "max_daily_cost": [max_daily_cost],
            "min_daily_cost": [min_daily_cost],
        }
    )

    # Action counts are added if available.
    if "action_taken" in simulation_df.columns:
        for action in sorted(simulation_df["action_taken"].dropna().unique()):
            summary[f"n_{action}_days"] = int((simulation_df["action_taken"] == action).sum())

    return summary



def build_strategy_summary_table(simulation_dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Create a multi-strategy summary table."""
    validated_dfs = _validate_simulation_collection(simulation_dfs)
    summary_table = pd.concat(
        [summarize_strategy(df) for df in validated_dfs],
        axis=0,
        ignore_index=True,
    )

    return summary_table.sort_values("total_cost", ascending=True).reset_index(drop=True)


# =========================
# Savings / comparison helpers
# =========================

def add_savings_vs_reference(
    summary_df: pd.DataFrame,
    reference_strategy_name: str,
) -> pd.DataFrame:
    """
    Add savings and volatility deltas relative to one reference strategy.

    Example references:
    - 'spot_only'
    - 'static_hedge'
    """
    if summary_df.empty:
        raise StrategyComparisonError("summary_df is empty.")

    if "strategy_name" not in summary_df.columns or "total_cost" not in summary_df.columns:
        raise StrategyComparisonError(
            "summary_df must contain at least 'strategy_name' and 'total_cost'."
        )

    if reference_strategy_name not in set(summary_df["strategy_name"]):
        raise StrategyComparisonError(
            f"Reference strategy '{reference_strategy_name}' was not found in summary_df."
        )

    result_df = summary_df.copy()
    reference_row = result_df.loc[
        result_df["strategy_name"] == reference_strategy_name
    ].iloc[0]

    reference_total_cost = float(reference_row["total_cost"])
    result_df[f"savings_vs_{reference_strategy_name}"] = (
        reference_total_cost - result_df["total_cost"]
    )
    result_df[f"savings_share_vs_{reference_strategy_name}"] = (
        reference_total_cost - result_df["total_cost"]
    ) / reference_total_cost if reference_total_cost != 0 else pd.NA

    if "daily_cost_volatility" in result_df.columns:
        reference_volatility = float(reference_row["daily_cost_volatility"])
        result_df[f"volatility_delta_vs_{reference_strategy_name}"] = (
            result_df["daily_cost_volatility"] - reference_volatility
        )

    return result_df



def compare_daily_costs(simulation_dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all strategies into one daily comparison table.

    Output columns:
    - date
    - total_cost_<strategy>
    - optional action columns if available
    """
    validated_dfs = _validate_simulation_collection(simulation_dfs)

    comparison_df = pd.DataFrame({DATE_COLUMN: validated_dfs[0][DATE_COLUMN]})

    for df in validated_dfs:
        strategy_name = df["strategy_name"].iloc[0]
        comparison_df[f"total_cost_{strategy_name}"] = df["total_cost"].values

        if "action_taken" in df.columns:
            comparison_df[f"action_{strategy_name}"] = df["action_taken"].values

    return comparison_df


# =========================
# Convenience report helper
# =========================

def build_strategy_comparison_report(
    simulation_dfs: Sequence[pd.DataFrame],
    reference_strategy_name: str = DEFAULT_REFERENCE_STRATEGY,
) -> dict[str, pd.DataFrame]:
    """
    Build a compact comparison package for backtesting outputs.

    Returns a dictionary with:
    - summary_table
    - summary_vs_reference
    - daily_comparison
    """
    summary_table = build_strategy_summary_table(simulation_dfs)
    summary_vs_reference = add_savings_vs_reference(
        summary_table,
        reference_strategy_name=reference_strategy_name,
    )
    daily_comparison = compare_daily_costs(simulation_dfs)

    report = {
        "summary_table": summary_table,
        "summary_vs_reference": summary_vs_reference,
        "daily_comparison": daily_comparison,
    }

    return report


if __name__ == "__main__":
    dates = pd.date_range("2025-01-01", periods=4, freq="D")

    spot_only_df = pd.DataFrame(
        {
            DATE_COLUMN: dates,
            "strategy_name": STRATEGY_SPOT_ONLY,
            "energy_volume_mwh": [10, 10, 10, 10],
            "total_cost": [800, 1200, 950, 700],
            "action_taken": ["buy_on_spot"] * 4,
        }
    )

    static_hedge_df = pd.DataFrame(
        {
            DATE_COLUMN: dates,
            "strategy_name": STRATEGY_STATIC_HEDGE,
            "energy_volume_mwh": [10, 10, 10, 10],
            "total_cost": [850, 900, 920, 880],
            "action_taken": ["static_m1_hedge"] * 4,
        }
    )

    heuristic_policy_df = pd.DataFrame(
        {
            DATE_COLUMN: dates,
            "strategy_name": STRATEGY_HEURISTIC_POLICY,
            "energy_volume_mwh": [10, 10, 10, 10],
            "total_cost": [780, 910, 890, 760],
            "action_taken": [
                "do_nothing",
                "buy_m1_future",
                "buy_m1_future",
                "shift_production",
            ],
        }
    )

    report = build_strategy_comparison_report(
        [spot_only_df, static_hedge_df, heuristic_policy_df],
        reference_strategy_name=STRATEGY_SPOT_ONLY,
    )

    print(report["summary_table"])
    print(report["summary_vs_reference"])
    print(report["daily_comparison"])
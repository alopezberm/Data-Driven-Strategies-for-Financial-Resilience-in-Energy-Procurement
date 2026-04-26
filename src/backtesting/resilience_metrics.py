"""
resilience_metrics.py

Resilience-oriented evaluation metrics for procurement strategies.
These metrics complement pure cost savings by quantifying exposure to extreme
cost days, cost stability, and worst-case outcomes.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.config.constants import (
    DATE_COLUMN,
    DEFAULT_REFERENCE_STRATEGY,
    STRATEGY_HEURISTIC_POLICY,
    STRATEGY_SPOT_ONLY,
    STRATEGY_STATIC_HEDGE,
)
from src.utils.validation import ValidationError, validate_and_sort_by_date


REQUIRED_SIMULATION_COLUMNS = {
    DATE_COLUMN,
    "strategy_name",
    "total_cost",
    "energy_volume_mwh",
}


class ResilienceMetricsError(Exception):
    """Raised when resilience metrics cannot be computed safely."""


# =========================
# Validation helpers
# =========================

def _validate_simulation_df(simulation_df: pd.DataFrame) -> pd.DataFrame:
    """Validate one simulation dataframe and standardize its date column."""
    missing_columns = REQUIRED_SIMULATION_COLUMNS - set(simulation_df.columns)
    if missing_columns:
        raise ResilienceMetricsError(
            f"Simulation dataframe is missing required columns: {sorted(missing_columns)}"
        )
    try:
        validated_df = validate_and_sort_by_date(simulation_df, df_name="simulation dataframe")
    except ValidationError as exc:
        raise ResilienceMetricsError(str(exc)) from exc

    validated_df["total_cost"] = pd.to_numeric(validated_df["total_cost"], errors="coerce")
    validated_df["energy_volume_mwh"] = pd.to_numeric(
        validated_df["energy_volume_mwh"], errors="coerce"
    )
    if validated_df["total_cost"].isna().any():
        raise ResilienceMetricsError("Simulation dataframe contains invalid total_cost values.")
    if validated_df["energy_volume_mwh"].isna().any():
        raise ResilienceMetricsError(
            "Simulation dataframe contains invalid energy_volume_mwh values."
        )
    return validated_df



def _validate_simulation_collection(simulation_dfs: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
    """Validate a collection of simulation dataframes with unique strategy names."""
    if not simulation_dfs:
        raise ResilienceMetricsError("simulation_dfs cannot be empty.")

    validated_dfs = [_validate_simulation_df(df) for df in simulation_dfs]
    strategy_names = [df["strategy_name"].iloc[0] for df in validated_dfs]

    if len(strategy_names) != len(set(strategy_names)):
        raise ResilienceMetricsError(
            f"Strategy names must be unique. Found: {strategy_names}"
        )

    return validated_dfs


# =========================
# Single-strategy resilience metrics
# =========================

def compute_resilience_metrics(
    simulation_df: pd.DataFrame,
    extreme_cost_quantile: float = 0.90,
) -> pd.DataFrame:
    """
    Compute resilience-oriented metrics for one simulated strategy.

    Parameters
    ----------
    simulation_df : pd.DataFrame
        Strategy simulation dataframe.
    extreme_cost_quantile : float, optional
        Quantile used to define internally extreme-cost days.

    Returns
    -------
    pd.DataFrame
        One-row dataframe with resilience metrics.
    """
    if not 0 < extreme_cost_quantile < 1:
        raise ResilienceMetricsError("extreme_cost_quantile must be strictly between 0 and 1.")

    df = _validate_simulation_df(simulation_df)

    strategy_name = df["strategy_name"].iloc[0]
    total_cost = float(df["total_cost"].sum())
    total_volume = float(df["energy_volume_mwh"].sum())
    average_unit_cost = total_cost / total_volume if total_volume > 0 else pd.NA

    daily_cost_volatility = float(df["total_cost"].std())
    max_daily_cost = float(df["total_cost"].max())
    min_daily_cost = float(df["total_cost"].min())
    p95_daily_cost = float(df["total_cost"].quantile(0.95))
    p99_daily_cost = float(df["total_cost"].quantile(0.99))

    extreme_threshold = float(df["total_cost"].quantile(extreme_cost_quantile))
    n_extreme_cost_days = int((df["total_cost"] >= extreme_threshold).sum())
    share_extreme_cost_days = n_extreme_cost_days / len(df)

    # Coefficient of variation as a normalized stability metric.
    if pd.notna(average_unit_cost) and average_unit_cost != 0:
        coefficient_of_variation = daily_cost_volatility / average_unit_cost
    else:
        coefficient_of_variation = pd.NA

    metrics_df = pd.DataFrame(
        {
            "strategy_name": [strategy_name],
            "n_days": [int(len(df))],
            "total_volume_mwh": [total_volume],
            "total_cost": [total_cost],
            "average_unit_cost": [average_unit_cost],
            "daily_cost_volatility": [daily_cost_volatility],
            "coefficient_of_variation": [coefficient_of_variation],
            "max_daily_cost": [max_daily_cost],
            "min_daily_cost": [min_daily_cost],
            "p95_daily_cost": [p95_daily_cost],
            "p99_daily_cost": [p99_daily_cost],
            "extreme_cost_threshold": [extreme_threshold],
            "n_extreme_cost_days": [n_extreme_cost_days],
            "share_extreme_cost_days": [share_extreme_cost_days],
        }
    )

    return metrics_df


# =========================
# Cross-strategy comparison helpers
# =========================

def build_resilience_summary_table(
    simulation_dfs: Sequence[pd.DataFrame],
    extreme_cost_quantile: float = 0.90,
) -> pd.DataFrame:
    """Compute resilience metrics for multiple strategies."""
    validated_dfs = _validate_simulation_collection(simulation_dfs)

    summary_df = pd.concat(
        [
            compute_resilience_metrics(df, extreme_cost_quantile=extreme_cost_quantile)
            for df in validated_dfs
        ],
        axis=0,
        ignore_index=True,
    )

    return summary_df.sort_values("total_cost", ascending=True).reset_index(drop=True)



def add_resilience_deltas_vs_reference(
    resilience_summary_df: pd.DataFrame,
    reference_strategy_name: str,
) -> pd.DataFrame:
    """
    Add resilience deltas relative to a reference strategy.

    Useful for statements like:
    - avoided worst-case costs
    - reduced cost volatility
    - fewer extreme-cost days
    """
    if resilience_summary_df.empty:
        raise ResilienceMetricsError("resilience_summary_df is empty.")

    if reference_strategy_name not in set(resilience_summary_df["strategy_name"]):
        raise ResilienceMetricsError(
            f"Reference strategy '{reference_strategy_name}' not found in resilience summary."
        )

    result_df = resilience_summary_df.copy()
    reference_row = result_df.loc[
        result_df["strategy_name"] == reference_strategy_name
    ].iloc[0]

    comparison_columns = [
        "total_cost",
        "daily_cost_volatility",
        "max_daily_cost",
        "p95_daily_cost",
        "p99_daily_cost",
        "n_extreme_cost_days",
        "share_extreme_cost_days",
    ]

    for column in comparison_columns:
        reference_value = reference_row[column]
        result_df[f"delta_{column}_vs_{reference_strategy_name}"] = (
            result_df[column] - reference_value
        )

        if pd.notna(reference_value) and reference_value != 0:
            result_df[f"relative_delta_{column}_vs_{reference_strategy_name}"] = (
                result_df[column] - reference_value
            ) / reference_value
        else:
            result_df[f"relative_delta_{column}_vs_{reference_strategy_name}"] = pd.NA

    # A user-friendly savings-style metric for extremes.
    result_df[f"avoided_extreme_cost_days_vs_{reference_strategy_name}"] = (
        reference_row["n_extreme_cost_days"] - result_df["n_extreme_cost_days"]
    )

    return result_df



def compare_extreme_days_against_reference(
    simulation_dfs: Sequence[pd.DataFrame],
    reference_strategy_name: str,
    extreme_cost_quantile: float = 0.90,
) -> pd.DataFrame:
    """
    Compare which days are extreme under the reference strategy and how each
    other strategy behaves on those same dates.
    """
    validated_dfs = _validate_simulation_collection(simulation_dfs)
    strategy_map = {df["strategy_name"].iloc[0]: df for df in validated_dfs}

    if reference_strategy_name not in strategy_map:
        raise ResilienceMetricsError(
            f"Reference strategy '{reference_strategy_name}' not found in simulation_dfs."
        )

    reference_df = strategy_map[reference_strategy_name].copy()
    threshold = float(reference_df["total_cost"].quantile(extreme_cost_quantile))
    extreme_reference_days = reference_df.loc[
        reference_df["total_cost"] >= threshold,
        [DATE_COLUMN, "total_cost"],
    ].rename(columns={"total_cost": f"total_cost_{reference_strategy_name}"})

    if extreme_reference_days.empty:
        raise ResilienceMetricsError(
            "No extreme reference days found. Consider lowering extreme_cost_quantile."
        )

    comparison_df = extreme_reference_days.copy()
    comparison_df["reference_extreme_threshold"] = threshold

    for strategy_name, strategy_df in strategy_map.items():
        if strategy_name == reference_strategy_name:
            continue

        strategy_subset = strategy_df[[DATE_COLUMN, "total_cost"]].rename(
            columns={"total_cost": f"total_cost_{strategy_name}"}
        )
        comparison_df = comparison_df.merge(strategy_subset, on=DATE_COLUMN, how="left")
        comparison_df[f"cost_difference_{strategy_name}_vs_{reference_strategy_name}"] = (
            comparison_df[f"total_cost_{strategy_name}"]
            - comparison_df[f"total_cost_{reference_strategy_name}"]
        )

    return comparison_df.sort_values(DATE_COLUMN).reset_index(drop=True)


# =========================
# Convenience report helper
# =========================

def build_resilience_report(
    simulation_dfs: Sequence[pd.DataFrame],
    reference_strategy_name: str = DEFAULT_REFERENCE_STRATEGY,
    extreme_cost_quantile: float = 0.90,
) -> dict[str, pd.DataFrame]:
    """
    Build a compact resilience-oriented report for multiple strategies.

    Returns:
    - resilience_summary
    - resilience_vs_reference
    - extreme_reference_days_comparison
    """
    resilience_summary = build_resilience_summary_table(
        simulation_dfs,
        extreme_cost_quantile=extreme_cost_quantile,
    )
    resilience_vs_reference = add_resilience_deltas_vs_reference(
        resilience_summary,
        reference_strategy_name=reference_strategy_name,
    )
    extreme_reference_days_comparison = compare_extreme_days_against_reference(
        simulation_dfs,
        reference_strategy_name=reference_strategy_name,
        extreme_cost_quantile=extreme_cost_quantile,
    )

    return {
        "resilience_summary": resilience_summary,
        "resilience_vs_reference": resilience_vs_reference,
        "extreme_reference_days_comparison": extreme_reference_days_comparison,
    }


if __name__ == "__main__":
    dates = pd.date_range("2025-01-01", periods=8, freq="D")

    spot_only_df = pd.DataFrame(
        {
            DATE_COLUMN: dates,
            "strategy_name": STRATEGY_SPOT_ONLY,
            "energy_volume_mwh": [10] * 8,
            "total_cost": [800, 1200, 950, 700, 1500, 880, 910, 1400],
        }
    )

    static_hedge_df = pd.DataFrame(
        {
            DATE_COLUMN: dates,
            "strategy_name": STRATEGY_STATIC_HEDGE,
            "energy_volume_mwh": [10] * 8,
            "total_cost": [850, 930, 920, 880, 980, 900, 905, 970],
        }
    )

    heuristic_policy_df = pd.DataFrame(
        {
            DATE_COLUMN: dates,
            "strategy_name": STRATEGY_HEURISTIC_POLICY,
            "energy_volume_mwh": [10] * 8,
            "total_cost": [780, 910, 890, 760, 940, 860, 875, 930],
        }
    )

    report = build_resilience_report(
        [spot_only_df, static_hedge_df, heuristic_policy_df],
        reference_strategy_name=STRATEGY_SPOT_ONLY,
        extreme_cost_quantile=0.90,
    )

    print(report["resilience_summary"])
    print(report["resilience_vs_reference"])
    print(report["extreme_reference_days_comparison"])

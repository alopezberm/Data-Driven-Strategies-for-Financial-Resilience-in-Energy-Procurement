"""
plot_backtest_results.py

Visualization utilities for procurement strategy backtesting results.
These plots are designed to support model interpretation, strategy comparison,
and reporting in the technical notebook/report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from src.config.constants import ACTIONS, DATE_COLUMN, DEFAULT_REFERENCE_STRATEGY, STRATEGY_HEURISTIC_POLICY, STRATEGY_SPOT_ONLY, STRATEGY_STATIC_HEDGE
from src.config.paths import FIGURES_DIR


class BacktestPlotError(Exception):
    """Raised when backtest plots cannot be generated safely."""


REQUIRED_SIMULATION_COLUMNS = {
    DATE_COLUMN,
    "strategy_name",
    "total_cost",
    "energy_volume_mwh",
}


ACTION_DO_NOTHING = ACTIONS[0]
ACTION_BUY_M1_FUTURE = ACTIONS[1]
ACTION_SHIFT_PRODUCTION = ACTIONS[2]



def _validate_strategy_catalog() -> None:
    """Validate the centralized strategy catalog used by this module."""
    expected_strategies = {
        STRATEGY_SPOT_ONLY,
        STRATEGY_STATIC_HEDGE,
        STRATEGY_HEURISTIC_POLICY,
    }
    if {STRATEGY_SPOT_ONLY, STRATEGY_STATIC_HEDGE, STRATEGY_HEURISTIC_POLICY} != expected_strategies:
        raise BacktestPlotError(
            "Centralized strategy constants are inconsistent."
        )



def _validate_action_catalog() -> None:
    """Validate the centralized action catalog used by this module."""
    expected_actions = {
        ACTION_DO_NOTHING,
        ACTION_BUY_M1_FUTURE,
        ACTION_SHIFT_PRODUCTION,
    }
    if len(ACTIONS) < 3 or set(ACTIONS[:3]) != expected_actions:
        raise BacktestPlotError(
            "Centralized ACTIONS constant must contain the expected action labels in the first three positions."
        )


# =========================
# Validation helpers
# =========================

def _validate_single_simulation_df(simulation_df: pd.DataFrame) -> pd.DataFrame:
    """Validate one simulation dataframe and standardize its date column."""
    if simulation_df.empty:
        raise BacktestPlotError("One of the simulation dataframes is empty.")

    missing_columns = REQUIRED_SIMULATION_COLUMNS - set(simulation_df.columns)
    if missing_columns:
        raise BacktestPlotError(
            f"Simulation dataframe is missing required columns: {sorted(missing_columns)}"
        )

    validated_df = simulation_df.copy()
    validated_df[DATE_COLUMN] = pd.to_datetime(validated_df[DATE_COLUMN], errors="coerce")
    validated_df["total_cost"] = pd.to_numeric(validated_df["total_cost"], errors="coerce")
    validated_df["energy_volume_mwh"] = pd.to_numeric(
        validated_df["energy_volume_mwh"], errors="coerce"
    )

    if validated_df[DATE_COLUMN].isna().any():
        invalid_count = int(validated_df[DATE_COLUMN].isna().sum())
        raise BacktestPlotError(
            f"Found {invalid_count} invalid date values in a simulation dataframe."
        )

    if validated_df["total_cost"].isna().any():
        raise BacktestPlotError("Simulation dataframe contains invalid total_cost values.")

    if validated_df[DATE_COLUMN].duplicated().any():
        raise BacktestPlotError("A simulation dataframe contains duplicated dates.")

    strategy_names = validated_df["strategy_name"].dropna().unique()
    if len(strategy_names) != 1:
        raise BacktestPlotError(
            "Each simulation dataframe must contain exactly one strategy_name."
        )

    return validated_df.sort_values(DATE_COLUMN).reset_index(drop=True)



def _validate_simulation_collection(simulation_dfs: Sequence[pd.DataFrame]) -> list[pd.DataFrame]:
    """Validate a collection of simulation dataframes."""
    _validate_strategy_catalog()

    if not simulation_dfs:
        raise BacktestPlotError("simulation_dfs cannot be empty.")

    validated_dfs = [_validate_single_simulation_df(df) for df in simulation_dfs]

    strategy_names = [df["strategy_name"].iloc[0] for df in validated_dfs]
    if len(strategy_names) != len(set(strategy_names)):
        raise BacktestPlotError(
            f"Strategy names must be unique across simulations. Found: {strategy_names}"
        )

    reference_dates = validated_dfs[0][DATE_COLUMN]
    for df in validated_dfs[1:]:
        if not df[DATE_COLUMN].equals(reference_dates):
            raise BacktestPlotError(
                "All simulation dataframes must share the same ordered date index."
            )

    return validated_dfs



def _prepare_output_path(filename: str | None) -> Path | None:
    """Prepare an output path inside the configured figures directory."""
    if filename is None:
        return None

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / filename


# =========================
# Plot helpers
# =========================

def plot_daily_costs(
    simulation_dfs: Sequence[pd.DataFrame],
    title: str = "Daily Procurement Costs by Strategy",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot daily total costs for multiple strategies on the same timeline."""
    validated_dfs = _validate_simulation_collection(simulation_dfs)

    plt.figure(figsize=(12, 6))
    for df in validated_dfs:
        strategy_name = df["strategy_name"].iloc[0]
        plt.plot(df[DATE_COLUMN], df["total_cost"], label=strategy_name)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Daily Total Cost")
    plt.legend()
    plt.tight_layout()

    output_path = _prepare_output_path(save_path)
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()



def plot_cumulative_costs(
    simulation_dfs: Sequence[pd.DataFrame],
    title: str = "Cumulative Procurement Costs by Strategy",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot cumulative costs over time for multiple strategies."""
    validated_dfs = _validate_simulation_collection(simulation_dfs)

    plt.figure(figsize=(12, 6))
    for df in validated_dfs:
        strategy_name = df["strategy_name"].iloc[0]
        cumulative_cost = df["total_cost"].cumsum()
        plt.plot(df[DATE_COLUMN], cumulative_cost, label=strategy_name)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Total Cost")
    plt.legend()
    plt.tight_layout()

    output_path = _prepare_output_path(save_path)
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()



def plot_daily_savings_vs_reference(
    simulation_dfs: Sequence[pd.DataFrame],
    reference_strategy_name: str = DEFAULT_REFERENCE_STRATEGY,
    title: str = "Daily Savings vs Reference Strategy",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot daily savings against a selected reference strategy."""
    validated_dfs = _validate_simulation_collection(simulation_dfs)
    strategy_map = {df["strategy_name"].iloc[0]: df for df in validated_dfs}

    if reference_strategy_name not in strategy_map:
        raise BacktestPlotError(
            f"Reference strategy '{reference_strategy_name}' not found in simulations."
        )

    reference_df = strategy_map[reference_strategy_name]
    reference_cost = reference_df["total_cost"].values

    plt.figure(figsize=(12, 6))
    for strategy_name, df in strategy_map.items():
        if strategy_name == reference_strategy_name:
            continue
        daily_savings = reference_cost - df["total_cost"].values
        plt.plot(df[DATE_COLUMN], daily_savings, label=f"{strategy_name} vs {reference_strategy_name}")

    plt.axhline(0, linewidth=1)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Daily Savings")
    plt.legend()
    plt.tight_layout()

    output_path = _prepare_output_path(save_path)
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()



def plot_total_cost_bar_chart(
    simulation_dfs: Sequence[pd.DataFrame],
    title: str = "Total Cost by Strategy",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot one bar per strategy using total backtest cost."""
    validated_dfs = _validate_simulation_collection(simulation_dfs)

    strategy_names = []
    total_costs = []
    for df in validated_dfs:
        strategy_names.append(df["strategy_name"].iloc[0])
        total_costs.append(float(df["total_cost"].sum()))

    plt.figure(figsize=(10, 6))
    plt.bar(strategy_names, total_costs)
    plt.title(title)
    plt.xlabel("Strategy")
    plt.ylabel("Total Cost")
    plt.tight_layout()

    output_path = _prepare_output_path(save_path)
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()



def plot_action_timeline(
    policy_simulation_df: pd.DataFrame,
    title: str = "Heuristic Policy Actions Over Time",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot the heuristic policy action timeline as discrete markers.

    Only intended for the policy simulation output, which must contain:
    - date
    - action_taken
    - strategy_name
    """
    validated_df = _validate_single_simulation_df(policy_simulation_df)
    _validate_action_catalog()

    if "action_taken" not in validated_df.columns:
        raise BacktestPlotError(
            "Policy simulation dataframe must contain an 'action_taken' column."
        )

    action_to_y = {
        ACTION_DO_NOTHING: 0,
        ACTION_BUY_M1_FUTURE: 1,
        ACTION_SHIFT_PRODUCTION: 2,
    }

    action_values = validated_df["action_taken"].map(action_to_y)
    if action_values.isna().any():
        unknown_actions = validated_df.loc[action_values.isna(), "action_taken"].unique().tolist()
        raise BacktestPlotError(
            f"Unknown actions found in action timeline plot: {unknown_actions}"
        )

    plt.figure(figsize=(12, 4))
    plt.scatter(validated_df[DATE_COLUMN], action_values)
    plt.yticks(list(action_to_y.values()), list(action_to_y.keys()))
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Recommended Action")
    plt.tight_layout()

    output_path = _prepare_output_path(save_path)
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    dates = pd.date_range("2025-01-01", periods=6, freq="D")

    spot_only_df = pd.DataFrame(
        {
            DATE_COLUMN: dates,
            "strategy_name": STRATEGY_SPOT_ONLY,
            "energy_volume_mwh": [10] * 6,
            "total_cost": [800, 1200, 950, 700, 1400, 900],
        }
    )

    static_hedge_df = pd.DataFrame(
        {
            DATE_COLUMN: dates,
            "strategy_name": STRATEGY_STATIC_HEDGE,
            "energy_volume_mwh": [10] * 6,
            "total_cost": [850, 930, 920, 880, 980, 905],
        }
    )

    heuristic_policy_df = pd.DataFrame(
        {
            DATE_COLUMN: dates,
            "strategy_name": STRATEGY_HEURISTIC_POLICY,
            "energy_volume_mwh": [10] * 6,
            "total_cost": [780, 910, 890, 760, 940, 875],
            "action_taken": [
                ACTION_DO_NOTHING,
                ACTION_BUY_M1_FUTURE,
                ACTION_BUY_M1_FUTURE,
                ACTION_SHIFT_PRODUCTION,
                ACTION_DO_NOTHING,
                ACTION_SHIFT_PRODUCTION,
            ],
        }
    )

    all_simulations = [spot_only_df, static_hedge_df, heuristic_policy_df]

    plot_daily_costs(all_simulations, show=False)
    plot_cumulative_costs(all_simulations, show=False)
    plot_daily_savings_vs_reference(all_simulations, reference_strategy_name=STRATEGY_SPOT_ONLY, show=False)
    plot_total_cost_bar_chart(all_simulations, show=False)
    plot_action_timeline(heuristic_policy_df, show=False)
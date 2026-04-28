"""
generate_exec_summary.py

Generates two C-Level executive summary PNGs for the Canva presentation:
  1. exec_summary_business_case.png  — "The Bottom Line" cost comparison
  2. exec_summary_resilience_overlay.png — "The Protection Map"

Output directory: data/outputs/figures/
Run from the project root:
    python -m src.visualization.generate_exec_summary
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.constants import MDP_D, STRATEGY_SPOT_ONLY, STRATEGY_HEURISTIC_POLICY
from src.visualization.plots import (
    plot_exec_summary_business_case,
    plot_exec_summary_resilience_overlay,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FIGURES_DIR = _PROJECT_ROOT / "data" / "outputs" / "figures"

_SUMMARY_CSV  = _PROJECT_ROOT / "data" / "outputs" / "backtests" / "strategy_summary_vs_spot_only.csv"
_DAILY_CSV    = _PROJECT_ROOT / "data" / "outputs" / "backtests" / "strategy_daily_comparison.csv"
_MODELING_CSV = _PROJECT_ROOT / "data" / "processed" / "modeling_dataset.csv"


def _load_results() -> pd.DataFrame:
    df = pd.read_csv(_SUMMARY_CSV)
    return df[["strategy_name", "total_cost", "n_days"]]


def _compute_kpis(df_results: pd.DataFrame) -> tuple[float, float]:
    """Return (total_savings_eur, per_unit_savings_eur)."""
    cost_map = df_results.set_index("strategy_name")["total_cost"].to_dict()
    spot_cost = float(cost_map[STRATEGY_SPOT_ONLY])
    heur_cost = float(cost_map[STRATEGY_HEURISTIC_POLICY])
    total_savings = spot_cost - heur_cost

    n_days = int(df_results.loc[
        df_results["strategy_name"] == STRATEGY_SPOT_ONLY, "n_days"
    ].iloc[0])
    per_unit = total_savings / (MDP_D * n_days)
    return total_savings, per_unit


def _load_simulation() -> pd.DataFrame:
    """
    Merge 2025 spot prices (modeling_dataset.csv) with action decisions
    (strategy_daily_comparison.csv) on date.
    """
    full = pd.read_csv(_MODELING_CSV, usecols=["date", "Spot_Price_SPEL"])
    full["date"] = pd.to_datetime(full["date"])
    spot = full[full["date"] >= "2025-01-01"].copy()

    actions = pd.read_csv(
        _DAILY_CSV,
        usecols=["date", "action_heuristic_policy", "action_rl_policy"],
    )
    actions["date"] = pd.to_datetime(actions["date"])

    merged = pd.merge(spot, actions, on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def main() -> None:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    out1 = _FIGURES_DIR / "exec_summary_business_case.png"
    out2 = _FIGURES_DIR / "exec_summary_resilience_overlay.png"

    print("Loading strategy summary data...")
    df_results = _load_results()
    print(f"  Strategies: {df_results['strategy_name'].tolist()}")
    print(f"  Costs: {df_results.set_index('strategy_name')['total_cost'].to_dict()}")

    total_savings, per_unit = _compute_kpis(df_results)
    print(f"\n  KPI — Total EUR saved    : EUR {total_savings:,.2f}")
    print(f"  KPI — EUR saved per unit : EUR {per_unit:.4f}")

    print("\nGenerating Plot 1: 'The Bottom Line'...")
    plot_exec_summary_business_case(
        df_results,
        save_path=out1,
        show=False,
        per_unit_savings=per_unit,
    )
    print(f"  Saved -> {out1}")

    print("\nLoading 2025 simulation data...")
    df_sim = _load_simulation()
    print(f"  Rows: {len(df_sim)}  |  Date range: {df_sim['date'].min().date()} to {df_sim['date'].max().date()}")
    hedge_days = (df_sim["action_heuristic_policy"] == "buy_m1_future").sum()
    print(f"  Hedged days (heuristic DSS): {hedge_days}/{len(df_sim)}")

    print("\nGenerating Plot 2: 'The Protection Map'...")
    plot_exec_summary_resilience_overlay(df_sim, save_path=out2, show=False)
    print(f"  Saved -> {out2}")

    print("\nDone. Both PNGs written to:")
    print(f"  {out1}")
    print(f"  {out2}")


if __name__ == "__main__":
    main()

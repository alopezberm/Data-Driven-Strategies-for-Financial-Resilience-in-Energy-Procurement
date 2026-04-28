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

from src.visualization.plots import (
    plot_exec_summary_business_case,
    plot_exec_summary_resilience_overlay,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FIGURES_DIR = _PROJECT_ROOT / "data" / "outputs" / "figures"

_SUMMARY_CSV = _PROJECT_ROOT / "data" / "outputs" / "backtests" / "strategy_summary_vs_spot_only.csv"
_DAILY_CSV   = _PROJECT_ROOT / "data" / "outputs" / "backtests" / "strategy_daily_comparison.csv"
_TEST_CSV    = _PROJECT_ROOT / "data" / "processed" / "test.csv"


def _load_results() -> pd.DataFrame:
    df = pd.read_csv(_SUMMARY_CSV)
    return df[["strategy_name", "total_cost", "n_days"]]


def _load_simulation() -> pd.DataFrame:
    """
    Merge the 2025 spot prices (test.csv) with action decisions
    (strategy_daily_comparison.csv) on date.

    test.csv uses 'Date' (capital D); daily comparison uses 'date' (lower).
    """
    spot = pd.read_csv(_TEST_CSV, usecols=["Date", "Spot_Price_SPEL"])
    spot = spot.rename(columns={"Date": "date"})
    spot["date"] = pd.to_datetime(spot["date"])

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

    print("\nGenerating Plot 1: 'The Bottom Line'...")
    plot_exec_summary_business_case(df_results, save_path=out1, show=False)
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

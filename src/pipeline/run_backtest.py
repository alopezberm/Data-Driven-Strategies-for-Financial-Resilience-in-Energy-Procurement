

"""
run_backtest.py

End-to-end backtesting pipeline for the energy procurement DSS.

This script:
1. Loads engineered train/validation datasets
2. Trains quantile models
3. Prepares policy inputs
4. Applies the heuristic policy
5. Simulates baseline and policy strategies
6. Compares strategies and resilience metrics
7. Saves outputs and figures

Usage
-----
python -m src.pipeline.run_backtest
"""

from __future__ import annotations

import pandas as pd

from src.config.paths import BACKTESTS_DIR, FIGURES_DIR, POLICIES_DIR, PROCESSED_DATA_DIR
from src.models.quantile_models import train_quantile_models, summarize_quantile_results
from src.models.evaluate_model import build_quantile_diagnostics_report
from src.decision.policy_inputs import prepare_policy_inputs
from src.decision.heuristic_policy import apply_heuristic_policy
from src.backtesting.simulate_baseline import (
    BaselineSimulationConfig,
    simulate_spot_only_baseline,
    simulate_static_hedge_baseline,
)
from src.backtesting.simulate_policy import (
    PolicySimulationConfig,
    simulate_policy_strategy,
)
from src.backtesting.compare_strategies import build_strategy_comparison_report
from src.backtesting.resilience_metrics import build_resilience_report
from src.visualization.plot_quantiles import (
    plot_quantile_band,
    plot_upper_tail_exceedances,
)
from src.visualization.plot_backtest_results import (
    plot_action_timeline,
    plot_cumulative_costs,
    plot_daily_costs,
    plot_daily_savings_vs_reference,
    plot_total_cost_bar_chart,
)


TRAIN_FEATURES_FILE = PROCESSED_DATA_DIR / "train_features.csv"
VALIDATION_FEATURES_FILE = PROCESSED_DATA_DIR / "validation_features.csv"


class BacktestPipelineError(Exception):
    """Raised when the backtest pipeline cannot run safely."""


# =========================
# Helpers
# =========================

def _check_required_inputs() -> None:
    """Ensure required processed feature files exist before running."""
    required_files = [TRAIN_FEATURES_FILE, VALIDATION_FEATURES_FILE]
    missing_files = [str(file_path) for file_path in required_files if not file_path.exists()]

    if missing_files:
        raise BacktestPipelineError(
            "Missing required input files. Please generate feature files first:\n"
            + "\n".join(missing_files)
        )



def _prepare_output_directories() -> None:
    """Create output directories if they do not already exist."""
    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
    POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Main pipeline
# =========================

def run_backtest_pipeline() -> dict[str, pd.DataFrame]:
    """
    Execute the full validation backtest pipeline.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with key output tables for downstream use.
    """
    _check_required_inputs()
    _prepare_output_directories()

    # =========================
    # 1. Load datasets
    # =========================
    train = pd.read_csv(TRAIN_FEATURES_FILE)
    val = pd.read_csv(VALIDATION_FEATURES_FILE)

    # =========================
    # 2. Train quantile models
    # =========================
    quantile_results, _, _ = train_quantile_models(
        train,
        val,
        quantiles=[0.5, 0.9, 0.95],
    )

    quantile_summary = summarize_quantile_results(quantile_results)
    quantile_diagnostics = build_quantile_diagnostics_report(quantile_results)

    # =========================
    # 3. Prepare policy inputs
    # =========================
    policy_inputs_df = prepare_policy_inputs(val, quantile_results)

    # =========================
    # 4. Apply heuristic policy
    # =========================
    decisions_df = apply_heuristic_policy(policy_inputs_df)

    # =========================
    # 5. Align validation subset
    # =========================
    if "source_index" not in policy_inputs_df.columns:
        raise BacktestPipelineError(
            "Policy inputs dataframe must contain 'source_index' for alignment."
        )

    val_aligned = val.loc[policy_inputs_df["source_index"]].copy()
    val_aligned = val_aligned.sort_index().reset_index(drop=True)
    decisions_df = decisions_df.sort_values("source_index").reset_index(drop=True)

    # =========================
    # 6. Simulate strategies
    # =========================
    spot_only_df = simulate_spot_only_baseline(val_aligned)

    static_hedge_df = simulate_static_hedge_baseline(
        val_aligned,
        config=BaselineSimulationConfig(
            hedge_ratio=0.7,
        ),
    )

    policy_sim_df = simulate_policy_strategy(
        decisions_df,
        config=PolicySimulationConfig(
            hedge_ratio_on_buy_future=1.0,
            shift_fraction=1.0,
            shift_penalty_per_mwh=2.0,
        ),
    )

    simulation_dfs = [spot_only_df, static_hedge_df, policy_sim_df]

    # =========================
    # 7. Comparison + resilience
    # =========================
    comparison_report = build_strategy_comparison_report(
        simulation_dfs,
        reference_strategy_name="spot_only",
    )

    resilience_report = build_resilience_report(
        simulation_dfs,
        reference_strategy_name="spot_only",
        extreme_cost_quantile=0.90,
    )

    # =========================
    # 8. Save tables
    # =========================
    decisions_df.to_csv(POLICIES_DIR / "validation_policy_decisions.csv", index=False)

    spot_only_df.to_csv(BACKTESTS_DIR / "validation_spot_only.csv", index=False)
    static_hedge_df.to_csv(BACKTESTS_DIR / "validation_static_hedge.csv", index=False)
    policy_sim_df.to_csv(BACKTESTS_DIR / "validation_heuristic_policy.csv", index=False)

    quantile_summary.to_csv(BACKTESTS_DIR / "quantile_model_summary.csv", index=False)
    quantile_diagnostics["coverage_summary"].to_csv(
        BACKTESTS_DIR / "quantile_coverage_summary.csv", index=False
    )
    quantile_diagnostics["interval_summary"].to_csv(
        BACKTESTS_DIR / "quantile_interval_summary.csv", index=False
    )
    quantile_diagnostics["upper_tail_exceedance_summary"].to_csv(
        BACKTESTS_DIR / "quantile_upper_tail_exceedance_summary.csv", index=False
    )

    comparison_report["summary_table"].to_csv(
        BACKTESTS_DIR / "strategy_summary_table.csv", index=False
    )
    comparison_report["summary_vs_reference"].to_csv(
        BACKTESTS_DIR / "strategy_summary_vs_spot_only.csv", index=False
    )
    comparison_report["daily_comparison"].to_csv(
        BACKTESTS_DIR / "strategy_daily_comparison.csv", index=False
    )

    resilience_report["resilience_summary"].to_csv(
        BACKTESTS_DIR / "resilience_summary.csv", index=False
    )
    resilience_report["resilience_vs_reference"].to_csv(
        BACKTESTS_DIR / "resilience_vs_spot_only.csv", index=False
    )
    resilience_report["extreme_reference_days_comparison"].to_csv(
        BACKTESTS_DIR / "extreme_days_vs_spot_only.csv", index=False
    )

    # =========================
    # 9. Save figures
    # =========================
    plot_quantile_band(
        quantile_results,
        lower_quantile=0.5,
        upper_quantile=0.9,
        title="Prediction Band: Q50 to Q90",
        save_path="quantile_band_q50_q90.png",
        show=False,
    )

    plot_upper_tail_exceedances(
        quantile_results,
        upper_quantile=0.9,
        title="Upper-Tail Forecast and Exceedances (Q90)",
        save_path="upper_tail_exceedances_q90.png",
        show=False,
    )

    plot_daily_costs(
        simulation_dfs,
        title="Daily Procurement Costs by Strategy",
        save_path="daily_costs_by_strategy.png",
        show=False,
    )

    plot_cumulative_costs(
        simulation_dfs,
        title="Cumulative Procurement Costs by Strategy",
        save_path="cumulative_costs_by_strategy.png",
        show=False,
    )

    plot_daily_savings_vs_reference(
        simulation_dfs,
        reference_strategy_name="spot_only",
        title="Daily Savings vs Spot-Only Strategy",
        save_path="daily_savings_vs_spot_only.png",
        show=False,
    )

    plot_total_cost_bar_chart(
        simulation_dfs,
        title="Total Backtest Cost by Strategy",
        save_path="total_cost_bar_chart.png",
        show=False,
    )

    plot_action_timeline(
        policy_sim_df,
        title="Heuristic Policy Actions Over Time",
        save_path="heuristic_policy_action_timeline.png",
        show=False,
    )

    # =========================
    # 10. Return outputs
    # =========================
    return {
        "policy_decisions": decisions_df,
        "quantile_summary": quantile_summary,
        "strategy_summary": comparison_report["summary_table"],
        "strategy_summary_vs_reference": comparison_report["summary_vs_reference"],
        "resilience_summary": resilience_report["resilience_summary"],
        "resilience_vs_reference": resilience_report["resilience_vs_reference"],
    }


if __name__ == "__main__":
    outputs = run_backtest_pipeline()

    print("Backtest pipeline completed successfully.\n")

    print("Saved policy outputs to:")
    print(f"  - {POLICIES_DIR}")

    print("\nSaved backtest tables to:")
    print(f"  - {BACKTESTS_DIR}")

    print("\nSaved figures to:")
    print(f"  - {FIGURES_DIR}")

    print("\n=== Strategy Summary ===")
    print(outputs["strategy_summary"])

    print("\n=== Strategy Summary vs Spot Only ===")
    print(outputs["strategy_summary_vs_reference"])

    print("\n=== Resilience Summary ===")
    print(outputs["resilience_summary"])
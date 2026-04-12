

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

from src.config.constants import (
    DEFAULT_QUANTILES,
    DEFAULT_REFERENCE_STRATEGY,
    STRATEGY_RL_POLICY,
)
from src.config.paths import BACKTESTS_DIR, FIGURES_DIR, POLICIES_DIR, PROCESSED_DATA_DIR
from src.utils.logger import get_logger
from src.models.quantile_models import train_quantile_models, summarize_quantile_results
from src.models.evaluate_model import build_quantile_diagnostics_report
from src.decision.policy_inputs import prepare_policy_inputs
from src.decision.heuristic_policy import apply_heuristic_policy
from src.decision.rl_policy import apply_rl_policy
from src.rl.train_rl_agent import train_q_learning_agent
from src.backtesting.simulate_baseline import (
    BaselineSimulationConfig,
    simulate_spot_only_baseline,
    simulate_static_hedge_baseline,
)
from src.backtesting.simulate_policy import (
    PolicySimulationConfig,
    simulate_policy_strategy,
)
from src.backtesting.simulate_rl_policy import simulate_rl_policy_strategy
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

POLICY_DECISIONS_OUTPUT_FILE = POLICIES_DIR / "validation_policy_decisions.csv"
SPOT_ONLY_OUTPUT_FILE = BACKTESTS_DIR / "validation_spot_only.csv"
STATIC_HEDGE_OUTPUT_FILE = BACKTESTS_DIR / "validation_static_hedge.csv"

HEURISTIC_POLICY_OUTPUT_FILE = BACKTESTS_DIR / "validation_heuristic_policy.csv"
RL_POLICY_DECISIONS_OUTPUT_FILE = POLICIES_DIR / "validation_rl_policy_decisions.csv"
RL_POLICY_OUTPUT_FILE = BACKTESTS_DIR / "validation_rl_policy.csv"

QUANTILE_SUMMARY_OUTPUT_FILE = BACKTESTS_DIR / "quantile_model_summary.csv"
QUANTILE_COVERAGE_OUTPUT_FILE = BACKTESTS_DIR / "quantile_coverage_summary.csv"
QUANTILE_INTERVAL_OUTPUT_FILE = BACKTESTS_DIR / "quantile_interval_summary.csv"
QUANTILE_TAIL_EXCEEDANCE_OUTPUT_FILE = BACKTESTS_DIR / "quantile_upper_tail_exceedance_summary.csv"

STRATEGY_SUMMARY_OUTPUT_FILE = BACKTESTS_DIR / "strategy_summary_table.csv"
STRATEGY_VS_REFERENCE_OUTPUT_FILE = BACKTESTS_DIR / "strategy_summary_vs_spot_only.csv"
STRATEGY_DAILY_COMPARISON_OUTPUT_FILE = BACKTESTS_DIR / "strategy_daily_comparison.csv"

RESILIENCE_SUMMARY_OUTPUT_FILE = BACKTESTS_DIR / "resilience_summary.csv"
RESILIENCE_VS_REFERENCE_OUTPUT_FILE = BACKTESTS_DIR / "resilience_vs_spot_only.csv"
EXTREME_DAYS_VS_REFERENCE_OUTPUT_FILE = BACKTESTS_DIR / "extreme_days_vs_spot_only.csv"

QUANTILE_BAND_FIGURE_FILE = FIGURES_DIR / "quantile_band_q50_q90.png"
TAIL_EXCEEDANCE_FIGURE_FILE = FIGURES_DIR / "upper_tail_exceedances_q90.png"
DAILY_COSTS_FIGURE_FILE = FIGURES_DIR / "daily_costs_by_strategy.png"
CUMULATIVE_COSTS_FIGURE_FILE = FIGURES_DIR / "cumulative_costs_by_strategy.png"
DAILY_SAVINGS_FIGURE_FILE = FIGURES_DIR / "daily_savings_vs_spot_only.png"
TOTAL_COST_BAR_FIGURE_FILE = FIGURES_DIR / "total_cost_bar_chart.png"
ACTION_TIMELINE_FIGURE_FILE = FIGURES_DIR / "heuristic_policy_action_timeline.png"
RL_ACTION_TIMELINE_FIGURE_FILE = FIGURES_DIR / "rl_policy_action_timeline.png"


class BacktestPipelineError(Exception):
    """Raised when the backtest pipeline cannot run safely."""

logger = get_logger(__name__)


# =========================
# Helpers
# =========================

def _check_required_inputs() -> None:
    """Ensure required processed feature files exist before running."""
    logger.info("Checking required input files for backtest pipeline...")
    required_files = [TRAIN_FEATURES_FILE, VALIDATION_FEATURES_FILE]
    missing_files = [str(file_path) for file_path in required_files if not file_path.exists()]
    if missing_files:
        raise BacktestPipelineError(
            "Missing required input files. Please generate feature files first:\n"
            + "\n".join(missing_files)
        )



def _prepare_output_directories() -> None:
    """Create output directories if they do not already exist."""
    logger.info("Preparing output directories for backtest artifacts...")
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
    logger.info("Starting end-to-end backtest pipeline...")
    _check_required_inputs()
    _prepare_output_directories()

    # =========================
    # 1. Load datasets
    # =========================
    train = pd.read_csv(TRAIN_FEATURES_FILE)
    val = pd.read_csv(VALIDATION_FEATURES_FILE)
    logger.info(f"Loaded train features: {train.shape}")
    logger.info(f"Loaded validation features: {val.shape}")

    # =========================
    # 2. Train quantile models
    # =========================
    quantile_results, _, _ = train_quantile_models(
        train,
        val,
        quantiles=DEFAULT_QUANTILES,
    )

    quantile_summary = summarize_quantile_results(quantile_results)
    quantile_diagnostics = build_quantile_diagnostics_report(quantile_results)
    logger.info("Quantile models trained and diagnostics computed.")

    # =========================
    # 3. Prepare policy inputs
    # =========================
    policy_inputs_df = prepare_policy_inputs(val, quantile_results)
    logger.info(f"Prepared policy inputs: {policy_inputs_df.shape}")

    # =========================
    # 4. Apply heuristic policy
    # =========================
    decisions_df = apply_heuristic_policy(policy_inputs_df)
    logger.info(f"Applied heuristic policy: {decisions_df.shape}")

    rl_training_artifacts = train_q_learning_agent(policy_inputs_df)
    logger.info(
        f"Trained RL agent with {len(rl_training_artifacts.agent.q_table)} learned states."
    )

    rl_policy_artifacts = apply_rl_policy(
        agent=rl_training_artifacts.agent,
        policy_inputs_df=policy_inputs_df,
        include_q_values=True,
    )
    rl_decisions_df = rl_policy_artifacts.decisions_df.copy()
    logger.info(f"Applied RL policy: {rl_decisions_df.shape}")

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

    rl_decisions_df = rl_decisions_df.sort_values("row_id").reset_index(drop=True)

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

    rl_policy_input_df = val_aligned.copy().reset_index(drop=True)
    rl_policy_input_df["recommended_action"] = rl_decisions_df["recommended_action"].values
    rl_policy_input_df["action_source"] = STRATEGY_RL_POLICY

    rl_policy_sim_df = simulate_rl_policy_strategy(rl_policy_input_df)

    simulation_dfs = [spot_only_df, static_hedge_df, policy_sim_df, rl_policy_sim_df]
    logger.info(f"Spot-only simulation shape: {spot_only_df.shape}")
    logger.info(f"Static-hedge simulation shape: {static_hedge_df.shape}")
    logger.info(f"Heuristic-policy simulation shape: {policy_sim_df.shape}")
    logger.info(f"RL-policy simulation shape: {rl_policy_sim_df.shape}")

    # =========================
    # 7. Comparison + resilience
    # =========================
    comparison_report = build_strategy_comparison_report(
        simulation_dfs,
        reference_strategy_name=DEFAULT_REFERENCE_STRATEGY,
    )

    resilience_report = build_resilience_report(
        simulation_dfs,
        reference_strategy_name=DEFAULT_REFERENCE_STRATEGY,
        extreme_cost_quantile=0.90,
    )

    # =========================
    # 8. Save tables
    # =========================
    decisions_df.to_csv(POLICY_DECISIONS_OUTPUT_FILE, index=False)
    rl_decisions_df.to_csv(RL_POLICY_DECISIONS_OUTPUT_FILE, index=False)

    spot_only_df.to_csv(SPOT_ONLY_OUTPUT_FILE, index=False)
    static_hedge_df.to_csv(STATIC_HEDGE_OUTPUT_FILE, index=False)
    policy_sim_df.to_csv(HEURISTIC_POLICY_OUTPUT_FILE, index=False)
    rl_policy_sim_df.to_csv(RL_POLICY_OUTPUT_FILE, index=False)

    quantile_summary.to_csv(QUANTILE_SUMMARY_OUTPUT_FILE, index=False)
    quantile_diagnostics["coverage_summary"].to_csv(
        QUANTILE_COVERAGE_OUTPUT_FILE, index=False
    )
    quantile_diagnostics["interval_summary"].to_csv(
        QUANTILE_INTERVAL_OUTPUT_FILE, index=False
    )
    quantile_diagnostics["upper_tail_exceedance_summary"].to_csv(
        QUANTILE_TAIL_EXCEEDANCE_OUTPUT_FILE, index=False
    )

    comparison_report["summary_table"].to_csv(
        STRATEGY_SUMMARY_OUTPUT_FILE, index=False
    )
    comparison_report["summary_vs_reference"].to_csv(
        STRATEGY_VS_REFERENCE_OUTPUT_FILE, index=False
    )
    comparison_report["daily_comparison"].to_csv(
        STRATEGY_DAILY_COMPARISON_OUTPUT_FILE, index=False
    )

    resilience_report["resilience_summary"].to_csv(
        RESILIENCE_SUMMARY_OUTPUT_FILE, index=False
    )
    resilience_report["resilience_vs_reference"].to_csv(
        RESILIENCE_VS_REFERENCE_OUTPUT_FILE, index=False
    )
    resilience_report["extreme_reference_days_comparison"].to_csv(
        EXTREME_DAYS_VS_REFERENCE_OUTPUT_FILE, index=False
    )

    # =========================
    # 9. Save figures
    # =========================
    plot_quantile_band(
        quantile_results,
        lower_quantile=0.5,
        upper_quantile=0.9,
        title="Prediction Band: Q50 to Q90",
        save_path=str(QUANTILE_BAND_FIGURE_FILE),
        show=False,
    )

    plot_upper_tail_exceedances(
        quantile_results,
        upper_quantile=0.9,
        title="Upper-Tail Forecast and Exceedances (Q90)",
        save_path=str(TAIL_EXCEEDANCE_FIGURE_FILE),
        show=False,
    )

    plot_daily_costs(
        simulation_dfs,
        title="Daily Procurement Costs by Strategy",
        save_path=str(DAILY_COSTS_FIGURE_FILE),
        show=False,
    )

    plot_cumulative_costs(
        simulation_dfs,
        title="Cumulative Procurement Costs by Strategy",
        save_path=str(CUMULATIVE_COSTS_FIGURE_FILE),
        show=False,
    )

    plot_daily_savings_vs_reference(
        simulation_dfs,
        reference_strategy_name=DEFAULT_REFERENCE_STRATEGY,
        title="Daily Savings vs Spot-Only Strategy",
        save_path=str(DAILY_SAVINGS_FIGURE_FILE),
        show=False,
    )

    plot_total_cost_bar_chart(
        simulation_dfs,
        title="Total Backtest Cost by Strategy",
        save_path=str(TOTAL_COST_BAR_FIGURE_FILE),
        show=False,
    )

    plot_action_timeline(
        policy_sim_df,
        title="Heuristic Policy Actions Over Time",
        save_path=str(ACTION_TIMELINE_FIGURE_FILE),
        show=False,
    )

    plot_action_timeline(
        rl_policy_sim_df,
        title="RL Policy Actions Over Time",
        save_path=str(RL_ACTION_TIMELINE_FIGURE_FILE),
        show=False,
    )

    logger.info(f"Saved policy decisions to {POLICY_DECISIONS_OUTPUT_FILE}")
    logger.info(f"Saved backtest tables to {BACKTESTS_DIR}")
    logger.info(f"Saved figures to {FIGURES_DIR}")

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
        "rl_policy_decisions": rl_decisions_df,
        "rl_rewards_summary": rl_training_artifacts.rewards_summary_df,
    }


if __name__ == "__main__":
    outputs = run_backtest_pipeline()

    logger.info("Backtest pipeline completed successfully.")
    logger.info(f"Saved policy outputs to: {POLICIES_DIR}")
    logger.info(f"Saved backtest tables to: {BACKTESTS_DIR}")
    logger.info(f"Saved figures to: {FIGURES_DIR}")

    logger.info("=== Strategy Summary ===")
    logger.info(f"\n{outputs['strategy_summary']}")

    logger.info("=== Strategy Summary vs Reference ===")
    logger.info(f"\n{outputs['strategy_summary_vs_reference']}")

    logger.info("=== Resilience Summary ===")
    logger.info(f"\n{outputs['resilience_summary']}")
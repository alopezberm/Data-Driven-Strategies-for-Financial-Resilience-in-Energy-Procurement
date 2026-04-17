

"""Tests for reinforcement-learning training, policy generation, and simulation."""

from __future__ import annotations

import pandas as pd

from src.backtesting.simulate_rl_policy import (
    STRATEGY_RL_POLICY,
    simulate_rl_policy_strategy,
    summarize_rl_policy_simulation,
)
from src.config.constants import DATE_COLUMN, PRIMARY_FUTURE_COLUMN, Q50_COLUMN, Q90_COLUMN
from src.decision.rl_policy import apply_rl_policy
from src.rl.train_rl_agent import train_q_learning_agent


# =========================
# Fixtures / helpers
# =========================


def _build_example_rl_df() -> pd.DataFrame:
    """Create a small but realistic RL-ready dataframe for tests."""
    return pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2025-01-01", periods=12, freq="D"),
            Q50_COLUMN: [50, 52, 55, 57, 60, 58, 62, 64, 63, 61, 59, 56],
            Q90_COLUMN: [60, 63, 68, 70, 75, 72, 78, 80, 79, 76, 73, 69],
            PRIMARY_FUTURE_COLUMN: [52, 53, 56, 58, 61, 60, 63, 65, 64, 62, 60, 57],
            "Spot_Price_SPEL": [51, 54, 58, 60, 66, 61, 68, 70, 67, 64, 62, 58],
            "tail_vs_future_abs": [8, 10, 12, 12, 14, 12, 15, 15, 15, 14, 13, 12],
            "tail_vs_central_abs": [10, 11, 13, 13, 15, 14, 16, 16, 16, 15, 14, 13],
            "is_weekend": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
            "Is_national_holiday": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "daily_energy_mwh": [10] * 12,
        }
    )


# =========================
# RL training tests
# =========================


def test_train_q_learning_agent_returns_expected_artifacts() -> None:
    df = _build_example_rl_df()

    artifacts = train_q_learning_agent(df)

    assert artifacts.agent is not None
    assert isinstance(artifacts.episode_rewards, list)
    assert len(artifacts.episode_rewards) > 0
    assert isinstance(artifacts.rewards_history_df, pd.DataFrame)
    assert isinstance(artifacts.rewards_summary_df, pd.DataFrame)
    assert not artifacts.rewards_history_df.empty
    assert not artifacts.rewards_summary_df.empty
    assert "episode" in artifacts.rewards_history_df.columns
    assert "total_reward" in artifacts.rewards_history_df.columns
    assert "reward_mean" in artifacts.rewards_summary_df.columns


# =========================
# RL policy tests
# =========================


def test_apply_rl_policy_returns_valid_decisions() -> None:
    df = _build_example_rl_df()
    training_artifacts = train_q_learning_agent(df)

    policy_artifacts = apply_rl_policy(
        agent=training_artifacts.agent,
        policy_inputs_df=df,
        include_q_values=True,
    )

    decisions_df = policy_artifacts.decisions_df

    assert isinstance(decisions_df, pd.DataFrame)
    assert not decisions_df.empty
    assert decisions_df.shape[0] == df.shape[0]
    assert DATE_COLUMN in decisions_df.columns
    assert "action_id" in decisions_df.columns
    assert "recommended_action" in decisions_df.columns
    assert "action_source" in decisions_df.columns
    assert (decisions_df["action_source"] == "rl_policy").all()

    if policy_artifacts.q_values_df is not None:
        assert isinstance(policy_artifacts.q_values_df, pd.DataFrame)
        assert policy_artifacts.q_values_df.shape[0] == df.shape[0]


# =========================
# RL simulation tests
# =========================


def test_simulate_rl_policy_strategy_returns_expected_columns() -> None:
    df = _build_example_rl_df()
    training_artifacts = train_q_learning_agent(df)
    policy_artifacts = apply_rl_policy(
        agent=training_artifacts.agent,
        policy_inputs_df=df,
        include_q_values=False,
    )

    policy_df = df.copy()
    policy_df["recommended_action"] = policy_artifacts.decisions_df["recommended_action"].values
    policy_df["action_source"] = "rl_policy"

    simulation_df = simulate_rl_policy_strategy(policy_df)

    assert isinstance(simulation_df, pd.DataFrame)
    assert not simulation_df.empty
    assert simulation_df.shape[0] == df.shape[0]
    assert DATE_COLUMN in simulation_df.columns
    assert "action_taken" in simulation_df.columns
    assert "future_cost" in simulation_df.columns
    assert "spot_cost" in simulation_df.columns
    assert "shift_cost" in simulation_df.columns
    assert "total_cost" in simulation_df.columns
    assert "strategy_name" in simulation_df.columns
    assert (simulation_df["strategy_name"] == STRATEGY_RL_POLICY).all()



def test_summarize_rl_policy_simulation_returns_one_row_summary() -> None:
    df = _build_example_rl_df()
    training_artifacts = train_q_learning_agent(df)
    policy_artifacts = apply_rl_policy(
        agent=training_artifacts.agent,
        policy_inputs_df=df,
        include_q_values=False,
    )

    policy_df = df.copy()
    policy_df["recommended_action"] = policy_artifacts.decisions_df["recommended_action"].values
    policy_df["action_source"] = "rl_policy"

    simulation_df = simulate_rl_policy_strategy(policy_df)
    summary_df = summarize_rl_policy_simulation(simulation_df)

    assert isinstance(summary_df, pd.DataFrame)
    assert summary_df.shape[0] == 1
    assert "strategy_name" in summary_df.columns
    assert "total_cost" in summary_df.columns
    assert "average_daily_cost" in summary_df.columns
    assert summary_df.loc[0, "strategy_name"] == STRATEGY_RL_POLICY
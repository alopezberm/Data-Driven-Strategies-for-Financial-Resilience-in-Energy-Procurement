

"""
evaluate_rl_agent.py

Evaluation helpers for trained tabular reinforcement-learning agents.

This module focuses on post-training analysis rather than training itself.
It provides utilities to:
- roll a trained RL agent forward over a dataframe of policy inputs
- summarize chosen actions
- summarize state-action values when needed
- produce artifacts that can later be compared against heuristic and baseline strategies
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config.constants import DATE_COLUMN
from src.decision.rl_agent import QLearningAgent, RLAgentError
from src.rl.rl_environment import RLEnvironmentConfig
from src.rl.utils_rl import RLUtilsError, compound_action_label, decode_compound_action
from src.utils.logger import get_logger


logger = get_logger(__name__)


class RLEvaluationError(Exception):
    """Raised when RL evaluation cannot be executed safely."""


@dataclass
class RLEvaluationArtifacts:
    """Container for RL evaluation outputs."""

    decisions_df: pd.DataFrame
    action_summary_df: pd.DataFrame
    state_action_values_df: pd.DataFrame | None = None


# =========================
# Helpers
# =========================


def _validate_policy_inputs_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the dataframe used to evaluate a trained RL agent."""
    if not isinstance(df, pd.DataFrame):
        raise RLEvaluationError("RL evaluation input must be a pandas DataFrame.")
    if df.empty:
        raise RLEvaluationError("RL evaluation dataframe is empty.")
    return df.reset_index(drop=True)



def _resolve_date_column(df: pd.DataFrame) -> str | None:
    """Return the configured date column if present, otherwise None."""
    return DATE_COLUMN if DATE_COLUMN in df.columns else None



def _build_action_summary(decisions_df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact action frequency summary dataframe."""
    if decisions_df.empty:
        raise RLEvaluationError("Cannot summarize actions from an empty decisions dataframe.")

    summary_df = (
        decisions_df["recommended_action"]
        .value_counts(dropna=False)
        .rename_axis("recommended_action")
        .reset_index(name="n_days")
    )
    summary_df["share_days"] = summary_df["n_days"] / summary_df["n_days"].sum()
    return summary_df


# =========================
# Core evaluation API
# =========================


def evaluate_trained_rl_agent(
    agent: QLearningAgent,
    policy_inputs_df: pd.DataFrame,
    env_config: RLEnvironmentConfig | None = None,
    include_action_values: bool = True,
) -> RLEvaluationArtifacts:
    """
    Evaluate a trained RL agent over a dataframe of policy-style state inputs.

    Parameters
    ----------
    agent : QLearningAgent
        Trained tabular RL agent.
    policy_inputs_df : pd.DataFrame
        Dataframe containing the policy input state variables.
    env_config : RLEnvironmentConfig | None, optional
        Optional environment config used to resolve expected state columns.
    include_action_values : bool, optional
        Whether to also return state-action Q-values for each evaluated row.

    Returns
    -------
    RLEvaluationArtifacts
        RL decisions, compact action summary, and optional state-action values.
    """
    if not isinstance(agent, QLearningAgent):
        raise RLEvaluationError("agent must be an instance of QLearningAgent.")

    evaluation_df = _validate_policy_inputs_df(policy_inputs_df)
    date_column = _resolve_date_column(evaluation_df)

    decisions_rows: list[dict[str, Any]] = []
    state_action_rows: list[dict[str, Any]] = []

    c = env_config or RLEnvironmentConfig()

    for idx, (_, row) in enumerate(evaluation_df.iterrows()):
        m1_price = float(row[c.future_m1_column]) if c.future_m1_column in row.index else 0.0
        spot_price = float(row[c.spot_column]) if c.spot_column in row.index else m1_price

        state: dict[str, Any] = {
            "forecast_central": float(row[c.q50_column]) if c.q50_column in row.index else 0.0,
            "forecast_tail": float(row[c.q90_column]) if c.q90_column in row.index else 0.0,
            "forecast_central_h3": float(row[c.q50_h3_column]) if c.q50_h3_column in row.index else 0.0,
            "forecast_tail_h3": float(row[c.q90_h3_column]) if c.q90_h3_column in row.index else 0.0,
            "m1_price": m1_price,
            "spot_price": spot_price,
            "spot_m1_spread": (
                float(row[c.spot_m1_spread_column])
                if c.spot_m1_spread_column in row.index and pd.notna(row[c.spot_m1_spread_column])
                else spot_price - m1_price
            ),
            "inventory": float(c.initial_inventory),
            "inventory_bin": 1.0,
        }

        try:
            action_id = agent.select_action(state)
            prod, m1, m2, m3 = decode_compound_action(action_id)
            action_label = compound_action_label(action_id)
        except (RLAgentError, RLUtilsError, KeyError, ValueError, TypeError) as exc:
            raise RLEvaluationError(f"Failed to evaluate RL action on row {idx}: {exc}") from exc

        decision_row: dict[str, Any] = {
            "row_id": idx,
            "action_id": int(action_id),
            "recommended_action": action_label,
            "production_units": int(prod),
            "m1_block_mwh": int(m1),
            "m2_block_mwh": int(m2),
            "m3_block_mwh": int(m3),
            "action_source": "rl_policy",
        }
        if date_column is not None:
            decision_row[DATE_COLUMN] = row[date_column]
        decisions_rows.append(decision_row)

        if include_action_values:
            try:
                action_values = agent.get_action_values(state)
            except RLAgentError as exc:
                raise RLEvaluationError(
                    f"Failed to retrieve action values on row {idx}: {exc}"
                ) from exc

            value_row: dict[str, Any] = {"row_id": idx}
            if date_column is not None:
                value_row[DATE_COLUMN] = row[date_column]
            for action_key, value in sorted(action_values.items(), key=lambda item: item[0]):
                value_row[f"q_value_action_{int(action_key)}"] = float(value)
            state_action_rows.append(value_row)

    decisions_df = pd.DataFrame(decisions_rows)
    action_summary_df = _build_action_summary(decisions_df)
    state_action_values_df = (
        pd.DataFrame(state_action_rows) if include_action_values else None
    )

    logger.info("RL evaluation completed successfully.")
    logger.info(f"Evaluated rows: {len(decisions_df)}")
    logger.info(f"Action summary:\n{action_summary_df}")

    return RLEvaluationArtifacts(
        decisions_df=decisions_df,
        action_summary_df=action_summary_df,
        state_action_values_df=state_action_values_df,
    )


# =========================
# Smoke test
# =========================

if __name__ == "__main__":
    from src.config.constants import PRIMARY_FUTURE_COLUMN, Q50_COLUMN, Q90_COLUMN
    from src.rl.train_rl_agent import train_q_learning_agent

    example_df = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2025-01-01", periods=6, freq="D"),
            Q50_COLUMN: [50, 55, 60, 58, 62, 64],
            Q90_COLUMN: [60, 68, 78, 66, 80, 82],
            PRIMARY_FUTURE_COLUMN: [52, 57, 63, 60, 65, 67],
            "Spot_Price_SPEL": [51, 58, 66, 61, 69, 70],
            "tail_vs_future_abs": [8, 11, 15, 6, 15, 15],
            "tail_vs_central_abs": [10, 13, 18, 8, 18, 18],
            "is_weekend": [0, 0, 1, 0, 0, 1],
            "Is_national_holiday": [0, 0, 0, 0, 0, 0],
        }
    )

    training_artifacts = train_q_learning_agent(example_df)
    evaluation_artifacts = evaluate_trained_rl_agent(
        agent=training_artifacts.agent,
        policy_inputs_df=example_df,
    )

    logger.info(f"RL decisions head:\n{evaluation_artifacts.decisions_df.head()}")
    if evaluation_artifacts.state_action_values_df is not None:
        logger.info(
            f"State-action values head:\n{evaluation_artifacts.state_action_values_df.head()}"
        )
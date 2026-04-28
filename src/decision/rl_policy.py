

"""
rl_policy.py

Policy-level wrapper for using a trained RL agent inside the project decision
pipeline.

This module sits in `src/decision/` because its responsibility is not training
or diagnostics, but transforming policy-style state inputs into project actions
that can be consumed by downstream evaluation and backtesting modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config.constants import (
    DATE_COLUMN,
    PRIMARY_FUTURE_COLUMN,
    Q50_COLUMN,
    Q50_H3_COLUMN,
    Q90_COLUMN,
    Q90_H3_COLUMN,
    SPOT_M1_SPREAD_COLUMN,
)
from src.decision.rl_agent import QLearningAgent, RLAgentError
from src.rl.rl_environment import RLEnvironmentConfig
from src.rl.utils_rl import RLUtilsError, compound_action_label, decode_compound_action
from src.utils.logger import get_logger


logger = get_logger(__name__)


class RLPolicyError(Exception):
    """Raised when an RL policy cannot generate valid actions."""


@dataclass
class RLPolicyArtifacts:
    """Container for RL policy decisions and optional Q-value diagnostics."""

    decisions_df: pd.DataFrame
    action_summary_df: pd.DataFrame
    q_values_df: pd.DataFrame | None = None


# =========================
# Helpers
# =========================


def _validate_policy_inputs_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a dataframe used as RL policy input."""
    if not isinstance(df, pd.DataFrame):
        raise RLPolicyError("policy_inputs_df must be a pandas DataFrame.")
    if df.empty:
        raise RLPolicyError("policy_inputs_df is empty.")
    return df.reset_index(drop=True)



def _resolve_env_config(env_config: RLEnvironmentConfig | None) -> RLEnvironmentConfig:
    """Resolve environment config or build a default one."""
    return RLEnvironmentConfig() if env_config is None else env_config



def _build_state_from_row(
    row: pd.Series,
    env_config: RLEnvironmentConfig,
) -> dict[str, float]:
    """Build an RL state dictionary from one policy-input row."""
    required_columns = [
        env_config.q50_column,
        env_config.q90_column,
        env_config.q50_h3_column,
        env_config.q90_h3_column,
        env_config.future_m1_column,
        env_config.spot_column,
    ]
    missing_columns = [col for col in required_columns if col not in row.index]
    if missing_columns:
        raise RLPolicyError(
            f"Policy input row is missing required RL state columns: {missing_columns}"
        )

    m1_price = float(row[env_config.future_m1_column])
    spot_price = float(row[env_config.spot_column])

    state: dict[str, float] = {
        "forecast_central": float(row[env_config.q50_column]),
        "forecast_tail": float(row[env_config.q90_column]),
        "forecast_central_h3": float(row[env_config.q50_h3_column]),
        "forecast_tail_h3": float(row[env_config.q90_h3_column]),
        "m1_price": m1_price,
        "spot_price": spot_price,
        "spot_m1_spread": (
            float(row[env_config.spot_m1_spread_column])
            if env_config.spot_m1_spread_column in row.index
            and pd.notna(row[env_config.spot_m1_spread_column])
            else spot_price - m1_price
        ),
        "inventory": float(env_config.initial_inventory),
        "inventory_bin": 1.0,  # default mid-bin when no live inventory is tracked
    }
    return state



def _build_action_summary(decisions_df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact action frequency summary for RL policy outputs."""
    if decisions_df.empty:
        raise RLPolicyError("Cannot summarize actions from an empty decisions dataframe.")

    summary_df = (
        decisions_df["recommended_action"]
        .value_counts(dropna=False)
        .rename_axis("recommended_action")
        .reset_index(name="n_days")
    )
    summary_df["share_days"] = summary_df["n_days"] / summary_df["n_days"].sum()
    return summary_df


# =========================
# Core policy API
# =========================


def apply_rl_policy(
    agent: QLearningAgent,
    policy_inputs_df: pd.DataFrame,
    env_config: RLEnvironmentConfig | None = None,
    include_q_values: bool = True,
) -> RLPolicyArtifacts:
    """
    Apply a trained RL agent to a policy-input dataframe.

    Parameters
    ----------
    agent : QLearningAgent
        Trained tabular RL agent.
    policy_inputs_df : pd.DataFrame
        Dataframe containing policy-style RL state inputs.
    env_config : RLEnvironmentConfig | None, optional
        Configuration describing which columns are used to build the state.
    include_q_values : bool, optional
        Whether to also return row-level Q-values for all encoded actions.

    Returns
    -------
    RLPolicyArtifacts
        RL decisions plus optional Q-value diagnostics.
    """
    if not isinstance(agent, QLearningAgent):
        raise RLPolicyError("agent must be an instance of QLearningAgent.")

    validated_df = _validate_policy_inputs_df(policy_inputs_df)
    resolved_env_config = _resolve_env_config(env_config)

    decision_rows: list[dict[str, Any]] = []
    q_value_rows: list[dict[str, Any]] = []

    for idx, (_, row) in enumerate(validated_df.iterrows()):
        try:
            state = _build_state_from_row(row, resolved_env_config)
            action_id = agent.select_action(state)
            prod, m1, m2, m3 = decode_compound_action(action_id)
            action_label = compound_action_label(action_id)
        except (RLAgentError, RLUtilsError, ValueError, TypeError, KeyError) as exc:
            raise RLPolicyError(f"Failed to generate RL action on row {idx}: {exc}") from exc

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
        if DATE_COLUMN in row.index:
            decision_row[DATE_COLUMN] = row[DATE_COLUMN]
        decision_rows.append(decision_row)

        if include_q_values:
            try:
                action_values = agent.get_action_values(state)
            except RLAgentError as exc:
                raise RLPolicyError(
                    f"Failed to retrieve Q-values on row {idx}: {exc}"
                ) from exc

            q_value_row: dict[str, Any] = {"row_id": idx}
            if DATE_COLUMN in row.index:
                q_value_row[DATE_COLUMN] = row[DATE_COLUMN]
            for action_key, value in sorted(action_values.items(), key=lambda item: item[0]):
                q_value_row[f"q_value_action_{int(action_key)}"] = float(value)
            q_value_rows.append(q_value_row)

    decisions_df = pd.DataFrame(decision_rows)
    action_summary_df = _build_action_summary(decisions_df)
    q_values_df = pd.DataFrame(q_value_rows) if include_q_values else None

    logger.info("RL policy applied successfully.")
    logger.info(f"Generated RL decisions for {len(decisions_df)} rows.")
    logger.info(f"Action summary:\n{action_summary_df}")

    return RLPolicyArtifacts(
        decisions_df=decisions_df,
        action_summary_df=action_summary_df,
        q_values_df=q_values_df,
    )


# =========================
# Smoke test
# =========================

if __name__ == "__main__":
    from src.config.constants import SECONDARY_FUTURE_COLUMN
    from src.rl.train_rl_agent import train_q_learning_agent

    example_df = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2025-01-01", periods=6, freq="D"),
            Q50_COLUMN: [50.0, 55.0, 60.0, 58.0, 62.0, 64.0],
            Q90_COLUMN: [60.0, 68.0, 78.0, 66.0, 80.0, 82.0],
            Q50_H3_COLUMN: [52.0, 57.0, 62.0, 59.0, 64.0, 66.0],
            Q90_H3_COLUMN: [65.0, 73.0, 83.0, 71.0, 85.0, 87.0],
            PRIMARY_FUTURE_COLUMN: [52.0, 57.0, 63.0, 60.0, 65.0, 67.0],
            SECONDARY_FUTURE_COLUMN: [53.0, 58.0, 64.0, 61.0, 66.0, 68.0],
            "Future_M3_Price": [54.0, 59.0, 65.0, 62.0, 67.0, 69.0],
            "Spot_Price_SPEL": [51.0, 58.0, 66.0, 61.0, 69.0, 70.0],
        }
    )

    training_artifacts = train_q_learning_agent(example_df)
    policy_artifacts = apply_rl_policy(
        agent=training_artifacts.agent,
        policy_inputs_df=example_df,
        include_q_values=False,
    )
    logger.info(f"RL decisions head:\n{policy_artifacts.decisions_df.head()}")

    logger.info(f"RL decisions head:\n{policy_artifacts.decisions_df.head()}")
    if policy_artifacts.q_values_df is not None:
        logger.info(f"RL Q-values head:\n{policy_artifacts.q_values_df.head()}")
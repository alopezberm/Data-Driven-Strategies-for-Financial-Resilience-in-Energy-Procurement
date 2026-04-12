"""
action_rules.py

Encapsulates the decision rules used by the heuristic policy.
This separates *what* the rules are from *how* they are applied,
while allowing defaults to be loaded from centralized project settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.config.constants import (
    ALLOW_SHIFT_ON_WEEKENDS_RULE,
    TAIL_VS_CENTRAL_ABS_THRESHOLD,
    TAIL_VS_FUTURE_ABS_THRESHOLD,
)
from src.config.settings import PolicySettings, get_default_settings


class ActionRuleError(Exception):
    """Raised when a rule cannot be evaluated."""


# =========================
# Configuration
# =========================
# Defaults are defined in constants.py and mapped from settings.py when used implicitly.

@dataclass
class ActionRuleConfig:
    """Configuration for heuristic action-rule evaluation."""

    # Thresholds
    tail_vs_future_abs_threshold: float = TAIL_VS_FUTURE_ABS_THRESHOLD
    tail_vs_central_abs_threshold: float = TAIL_VS_CENTRAL_ABS_THRESHOLD

    # Flags
    allow_shift_on_weekends: bool = ALLOW_SHIFT_ON_WEEKENDS_RULE

    @classmethod
    def from_policy_settings(cls, policy_settings: PolicySettings) -> "ActionRuleConfig":
        """Build action-rule configuration from centralized policy settings."""
        return cls(
            tail_vs_future_abs_threshold=policy_settings.min_abs_risk_premium_to_hedge,
            tail_vs_central_abs_threshold=policy_settings.min_abs_risk_premium_to_shift,
            allow_shift_on_weekends=policy_settings.allow_shift_on_weekends,
        )


# =========================
# Validation
# =========================

REQUIRED_COLUMNS = {
    "tail_vs_future_abs",
    "tail_vs_central_abs",
    "is_weekend",
}


def _validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that the input dataframe contains all required rule columns."""
    if df.empty:
        raise ActionRuleError("Input dataframe is empty.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ActionRuleError(f"Missing required columns: {sorted(missing)}")

    return df.copy()


# =========================
# Rule definitions
# =========================

def rule_buy_m1_future(row: pd.Series, config: ActionRuleConfig) -> bool:
    """
    Buy future if tail risk is clearly above current future price.
    """
    return row["tail_vs_future_abs"] >= config.tail_vs_future_abs_threshold



def rule_shift_production(row: pd.Series, config: ActionRuleConfig) -> bool:
    """
    Shift production if high tail risk AND flexible day (weekend).
    """
    return (
        config.allow_shift_on_weekends
        and row["is_weekend"] == 1
        and row["tail_vs_central_abs"] >= config.tail_vs_central_abs_threshold
    )



# =========================
# Rule engine
# =========================

def _get_config(config: Optional[ActionRuleConfig]) -> ActionRuleConfig:
    """Return the provided config or a default ActionRuleConfig instance."""
    return get_default_action_rule_config() if config is None else config


def get_default_action_rule_config() -> ActionRuleConfig:
    """Build the default action-rule configuration from project settings."""
    settings = get_default_settings()
    return ActionRuleConfig.from_policy_settings(settings.policy)


def evaluate_action(row: pd.Series, config: Optional[ActionRuleConfig] = None) -> str:
    """
    Evaluate all rules in priority order and return the chosen action.

    Priority:
    1. buy_m1_future
    2. shift_production
    3. do_nothing
    """
    config = _get_config(config)

    if rule_buy_m1_future(row, config):
        return "buy_m1_future"

    if rule_shift_production(row, config):
        return "shift_production"

    return "do_nothing"



def evaluate_action_with_reason(
    row: pd.Series,
    config: Optional[ActionRuleConfig] = None,
) -> tuple[str, str]:
    """
    Same as evaluate_action but returns (action, reason).
    """
    config = _get_config(config)

    if rule_buy_m1_future(row, config):
        return "buy_m1_future", "Tail risk exceeds futures price threshold"

    if rule_shift_production(row, config):
        return "shift_production", "Weekend + high tail risk vs central forecast"

    return "do_nothing", "No rule triggered"


# =========================
# Batch application
# =========================

def apply_action_rules(
    df: pd.DataFrame,
    config: Optional[ActionRuleConfig] = None,
) -> pd.DataFrame:
    """
    Apply action rules to a full dataframe.

    Returns
    -------
    pd.DataFrame
        Original dataframe + recommended_action + decision_reason
    """
    validated_df = _validate_input(df)

    actions = []
    reasons = []

    for _, row in validated_df.iterrows():
        action, reason = evaluate_action_with_reason(row, config)
        actions.append(action)
        reasons.append(reason)

    result_df = validated_df.copy()
    result_df["recommended_action"] = actions
    result_df["decision_reason"] = reasons

    return result_df


def summarize_action_rules(result_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize action counts after rule application."""
    if "recommended_action" not in result_df.columns:
        raise ActionRuleError(
            "Result dataframe must contain 'recommended_action' to build a summary."
        )

    summary_df = (
        result_df["recommended_action"]
        .value_counts(dropna=False)
        .rename_axis("recommended_action")
        .reset_index(name="n_rows")
    )
    summary_df["share"] = summary_df["n_rows"] / len(result_df)
    return summary_df


# =========================
# Test block
# =========================

if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            "tail_vs_future_abs": [2, 6, 4, 8],
            "tail_vs_central_abs": [1, 4, 5, 7],
            "is_weekend": [0, 0, 1, 1],
        }
    )

    config = get_default_action_rule_config()
    print(config)

    result = apply_action_rules(example_df, config=config)
    print(result)
    print(summarize_action_rules(result))
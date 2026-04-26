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
    ACTIONS,
    ALLOW_SHIFT_ON_WEEKENDS,
    DEFAULT_PRODUCTION_LEVEL,
    MIN_ABS_RISK_PREMIUM_TO_BUY_M2,
    MIN_ABS_RISK_PREMIUM_TO_BUY_M3,
    MIN_ABS_RISK_PREMIUM_TO_DECREASE,
    PRODUCTION_LEVELS,
    TAIL_VS_CENTRAL_ABS_THRESHOLD,
    TAIL_VS_FUTURE_ABS_THRESHOLD,
    validate_action_catalog,
)
from src.config.settings import PolicySettings, get_default_settings


class ActionRuleError(Exception):
    """Raised when a rule cannot be evaluated."""


ACTION_DO_NOTHING = ACTIONS[0]
ACTION_BUY_M1_FUTURE = ACTIONS[1]
ACTION_SHIFT_PRODUCTION = ACTIONS[2]
ACTION_INCREASE_PRODUCTION = ACTIONS[3]
ACTION_DECREASE_PRODUCTION = ACTIONS[4]
ACTION_BUY_M2_FUTURE = ACTIONS[5]
ACTION_BUY_M3_FUTURE = ACTIONS[6]


# =========================
# Configuration
# =========================
# Defaults are defined in constants.py and mapped from settings.py when used implicitly.

@dataclass
class ActionRuleConfig:
    """Configuration for heuristic action-rule evaluation."""

    # Thresholds — original actions
    tail_vs_future_abs_threshold: float = TAIL_VS_FUTURE_ABS_THRESHOLD
    tail_vs_central_abs_threshold: float = TAIL_VS_CENTRAL_ABS_THRESHOLD

    # Thresholds — extended actions (production + M2/M3 futures)
    decrease_production_threshold: float = MIN_ABS_RISK_PREMIUM_TO_DECREASE
    buy_m2_future_threshold: float = MIN_ABS_RISK_PREMIUM_TO_BUY_M2
    buy_m3_future_threshold: float = MIN_ABS_RISK_PREMIUM_TO_BUY_M3

    # Flags
    allow_shift_on_weekends: bool = ALLOW_SHIFT_ON_WEEKENDS

    # Extended action set: when True the rule engine considers production
    # adjustment and M+2/M+3 futures in addition to the original three actions.
    use_extended_actions: bool = False

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
# Extended rules — production adjustment
# =========================

def rule_decrease_production(row: pd.Series, config: ActionRuleConfig) -> bool:
    """
    Cut production by one step when tail-risk is high OR inventory is dangerously full.

    Two triggers:
    1. Tail-risk premium over futures exceeds threshold → energy too expensive to run flat out.
    2. Inventory near capacity (bin=2) → risk of overflow; no point producing more.
    inventory_bin is optional (defaults to 1 = medium) so rules work without factory MDP.
    """
    current_level = float(row.get("production_level", DEFAULT_PRODUCTION_LEVEL))
    at_minimum = current_level <= PRODUCTION_LEVELS[0]
    high_tail_risk = row["tail_vs_future_abs"] >= config.decrease_production_threshold
    high_inventory = float(row.get("inventory_bin", 1)) >= 2
    return not at_minimum and (high_tail_risk or high_inventory)


def rule_increase_production(row: pd.Series, config: ActionRuleConfig) -> bool:
    """
    Raise production by one step when energy is cheap OR inventory is dangerously low.

    Two triggers:
    1. Tail risk below futures (prices are favourable) → produce more now at low cost.
    2. Inventory in low bin (bin=0) → risk of stockout; must build safety stock.
    inventory_bin is optional (defaults to 1 = medium) so rules work without factory MDP.
    """
    current_level = float(row.get("production_level", DEFAULT_PRODUCTION_LEVEL))
    at_maximum = current_level >= PRODUCTION_LEVELS[-1]
    cheap_energy = row.get("tail_vs_future_abs", 0.0) < 0
    low_inventory = float(row.get("inventory_bin", 1)) <= 0
    return not at_maximum and (cheap_energy or low_inventory)


# =========================
# Extended rules — M+2 / M+3 futures
# =========================

def rule_buy_m2_future(row: pd.Series, config: ActionRuleConfig) -> bool:
    """
    Buy M+2 futures when mid-term tail risk exceeds the M+1 hedge and M+2
    is not already fully committed.

    This hedges beyond the front month when uncertainty is elevated over a
    two-month horizon.
    """
    return row["tail_vs_future_abs"] >= config.buy_m2_future_threshold


def rule_buy_m3_future(row: pd.Series, config: ActionRuleConfig) -> bool:
    """
    Buy M+3 futures only under severe long-horizon risk.

    Reserved for cases where the tail-risk premium is so large that locking in
    prices three months out is economically justified despite lower liquidity.
    """
    return row["tail_vs_future_abs"] >= config.buy_m3_future_threshold


# =========================
# Rule engine
# =========================

def _get_config(config: Optional[ActionRuleConfig]) -> ActionRuleConfig:
    """Return the provided config or a default ActionRuleConfig instance."""
    validate_action_catalog()
    return get_default_action_rule_config() if config is None else config


def get_default_action_rule_config() -> ActionRuleConfig:
    """Build the default action-rule configuration from project settings."""
    settings = get_default_settings()
    return ActionRuleConfig.from_policy_settings(settings.policy)


def evaluate_action(row: pd.Series, config: Optional[ActionRuleConfig] = None) -> str:
    """
    Evaluate all rules in priority order and return the chosen action.

    Original priority (always active):
    1. buy_m1_future   — hedge when tail risk exceeds futures price
    2. shift_production — reduce load on flexible + high-risk days
    3. do_nothing

    Extended priority (active when config.use_extended_actions is True):
    1. buy_m3_future   — severe long-horizon risk
    2. buy_m2_future   — elevated mid-horizon risk
    3. buy_m1_future   — front-month tail risk
    4. decrease_production — high spot/tail cost, output sacrifice
    5. increase_production — cheap energy window, stock building
    6. shift_production
    7. do_nothing
    """
    config = _get_config(config)

    if config.use_extended_actions:
        if rule_buy_m3_future(row, config):
            return ACTION_BUY_M3_FUTURE
        if rule_buy_m2_future(row, config):
            return ACTION_BUY_M2_FUTURE
        if rule_buy_m1_future(row, config):
            return ACTION_BUY_M1_FUTURE
        if rule_decrease_production(row, config):
            return ACTION_DECREASE_PRODUCTION
        if rule_increase_production(row, config):
            return ACTION_INCREASE_PRODUCTION
        if rule_shift_production(row, config):
            return ACTION_SHIFT_PRODUCTION
        return ACTION_DO_NOTHING

    # Original (backward-compatible) path
    if rule_buy_m1_future(row, config):
        return ACTION_BUY_M1_FUTURE
    if rule_shift_production(row, config):
        return ACTION_SHIFT_PRODUCTION
    return ACTION_DO_NOTHING



def evaluate_action_with_reason(
    row: pd.Series,
    config: Optional[ActionRuleConfig] = None,
) -> tuple[str, str]:
    """
    Same as evaluate_action but returns (action, reason).
    """
    config = _get_config(config)

    if config.use_extended_actions:
        if rule_buy_m3_future(row, config):
            return ACTION_BUY_M3_FUTURE, "Severe long-horizon tail risk — buy M+3 futures"
        if rule_buy_m2_future(row, config):
            return ACTION_BUY_M2_FUTURE, "Elevated mid-horizon tail risk — buy M+2 futures"
        if rule_buy_m1_future(row, config):
            return ACTION_BUY_M1_FUTURE, "Tail risk exceeds futures price threshold"
        if rule_decrease_production(row, config):
            inv_bin = float(row.get("inventory_bin", 1))
            reason = (
                "Inventory near capacity — reduce production by 10%"
                if inv_bin >= 2
                else "High expected cost — reduce production by 10%"
            )
            return ACTION_DECREASE_PRODUCTION, reason
        if rule_increase_production(row, config):
            inv_bin = float(row.get("inventory_bin", 1))
            reason = (
                "Inventory low — increase production by 10% to rebuild safety stock"
                if inv_bin <= 0
                else "Low expected cost — increase production by 10%"
            )
            return ACTION_INCREASE_PRODUCTION, reason
        if rule_shift_production(row, config):
            return ACTION_SHIFT_PRODUCTION, "Weekend + high tail risk vs central forecast"
        return ACTION_DO_NOTHING, "No rule triggered"

    # Original (backward-compatible) path
    if rule_buy_m1_future(row, config):
        return ACTION_BUY_M1_FUTURE, "Tail risk exceeds futures price threshold"
    if rule_shift_production(row, config):
        return ACTION_SHIFT_PRODUCTION, "Weekend + high tail risk vs central forecast"
    return ACTION_DO_NOTHING, "No rule triggered"


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
    config = _get_config(config)

    actions = []
    reasons = []

    for _, row in validated_df.iterrows():
        action, reason = evaluate_action_with_reason(row, config=config)
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
    assert set(result["recommended_action"]).issubset(set(ACTIONS))
    print(result)
    print(summarize_action_rules(result))
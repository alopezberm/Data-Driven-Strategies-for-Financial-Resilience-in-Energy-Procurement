"""
policy_evaluation.py

Evaluation utilities for the heuristic decision policy.
This module summarizes how often the policy acts, how actions are distributed,
how decisions relate to key risk signals, and how simulated policy actions
translate into cost outcomes.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.config.constants import ACTIONS, DATE_COLUMN, validate_action_catalog
from src.utils.validation import ValidationError, validate_and_sort_by_date


class PolicyEvaluationError(Exception):
    """Raised when policy evaluation cannot be performed safely."""


REQUIRED_POLICY_COLUMNS = {
    DATE_COLUMN,
    "recommended_action",
}

OPTIONAL_SIMULATION_COLUMNS = {
    "action_taken",
    "total_cost",
    "spot_cost",
    "future_cost",
    "shift_penalty_cost",
    "energy_volume_mwh",
}


ACTION_DO_NOTHING = ACTIONS[0]
ACTION_BUY_M1_FUTURE = ACTIONS[1]
ACTION_SHIFT_PRODUCTION = ACTIONS[2]


# =========================
# Validation helpers
# =========================

def _validate_policy_df(policy_df: pd.DataFrame) -> pd.DataFrame:
    """Validate a policy dataframe and standardize its date column."""
    missing_columns = REQUIRED_POLICY_COLUMNS - set(policy_df.columns)
    if missing_columns:
        raise PolicyEvaluationError(
            f"Policy dataframe is missing required columns: {sorted(missing_columns)}"
        )
    try:
        return validate_and_sort_by_date(policy_df, df_name="policy dataframe")
    except ValidationError as exc:
        raise PolicyEvaluationError(str(exc)) from exc


def _resolve_action_column(df: pd.DataFrame) -> str:
    """Resolve the action column name used in the provided dataframe."""
    validate_action_catalog()
    if "recommended_action" in df.columns:
        return "recommended_action"
    if "action_taken" in df.columns:
        return "action_taken"
    raise PolicyEvaluationError(
        "Policy dataframe must contain either 'recommended_action' or 'action_taken'."
    )


# =========================
# Summary builders
# =========================

def summarize_policy_actions(policy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact summary of policy action frequency.

    Returns
    -------
    pd.DataFrame
        One row per action with count and share.
    """
    df = _validate_policy_df(policy_df)
    action_column = _resolve_action_column(df)
    output_action_column = "recommended_action"

    summary_df = (
        df[action_column]
        .value_counts(dropna=False)
        .rename_axis(output_action_column)
        .reset_index(name="n_days")
    )
    summary_df["share"] = summary_df["n_days"] / len(df)

    return summary_df



def summarize_policy_time_coverage(policy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact date-range summary for the policy evaluation subset.
    """
    df = _validate_policy_df(policy_df)
    action_column = _resolve_action_column(df)

    return pd.DataFrame(
        {
            "n_days": [int(len(df))],
            "start_date": [df[DATE_COLUMN].min()],
            "end_date": [df[DATE_COLUMN].max()],
            "n_unique_actions": [int(df[action_column].nunique(dropna=True))],
        }
    )



def summarize_actions_by_calendar_context(policy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize how actions are distributed across weekends and holidays when available.
    """
    df = _validate_policy_df(policy_df)
    action_column = _resolve_action_column(df)

    group_columns = [action_column]
    available_context_columns = [
        column for column in ["is_weekend", "Is_national_holiday"] if column in df.columns
    ]

    if not available_context_columns:
        raise PolicyEvaluationError(
            "Policy dataframe must contain at least one of ['is_weekend', 'Is_national_holiday'] "
            "for calendar-context evaluation."
        )

    group_columns.extend(available_context_columns)

    summary_df = (
        df.groupby(group_columns, dropna=False)
        .size()
        .reset_index(name="n_days")
        .sort_values([action_column, "n_days"], ascending=[True, False])
        .reset_index(drop=True)
    )

    if action_column != "recommended_action":
        summary_df = summary_df.rename(columns={action_column: "recommended_action"})

    return summary_df



def summarize_actions_vs_risk_signals(
    policy_df: pd.DataFrame,
    risk_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Summarize average risk-signal levels by recommended action.

    Parameters
    ----------
    policy_df : pd.DataFrame
        Policy dataframe containing action labels and risk signals.
    risk_columns : Sequence[str] | None, optional
        Risk columns to summarize. If None, a sensible default set is used
        whenever available.
    """
    df = _validate_policy_df(policy_df)
    action_column = _resolve_action_column(df)

    default_risk_columns = [
        "tail_vs_future_abs",
        "tail_vs_future_rel",
        "tail_vs_central_abs",
        "tail_vs_central_rel",
        "forecast_central",
        "forecast_tail",
        "current_m1_future",
    ]
    selected_risk_columns = default_risk_columns if risk_columns is None else list(risk_columns)
    selected_risk_columns = [column for column in selected_risk_columns if column in df.columns]

    if not selected_risk_columns:
        raise PolicyEvaluationError(
            "No valid risk columns were found in the policy dataframe."
        )

    summary_df = (
        df.groupby(action_column, dropna=False)[selected_risk_columns]
        .mean(numeric_only=True)
        .reset_index()
    )

    if action_column != "recommended_action":
        summary_df = summary_df.rename(columns={action_column: "recommended_action"})

    return summary_df



def summarize_policy_reasons(policy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize decision reasons when the policy explanation column is available.
    """
    df = _validate_policy_df(policy_df)
    action_column = _resolve_action_column(df)

    if "decision_reason" not in df.columns:
        raise PolicyEvaluationError(
            "Policy dataframe must contain a 'decision_reason' column."
        )

    summary_df = (
        df.groupby([action_column, "decision_reason"], dropna=False)
        .size()
        .reset_index(name="n_days")
        .sort_values([action_column, "n_days"], ascending=[True, False])
        .reset_index(drop=True)
    )

    if action_column != "recommended_action":
        summary_df = summary_df.rename(columns={action_column: "recommended_action"})

    return summary_df


def summarize_simulated_action_costs(policy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize cost outcomes by action when simulated backtest columns are available.

    This is intended for policy simulation outputs that include columns such as:
    - action_taken
    - total_cost
    - spot_cost
    - future_cost
    - shift_penalty_cost
    - energy_volume_mwh
    """
    df = _validate_policy_df(policy_df)
    action_column = _resolve_action_column(df)

    required_cost_columns = ["total_cost"]
    missing_cost_columns = [column for column in required_cost_columns if column not in df.columns]
    if missing_cost_columns:
        raise PolicyEvaluationError(
            "Policy dataframe must contain simulated cost columns to summarize action costs. "
            f"Missing: {missing_cost_columns}"
        )

    numeric_columns = [
        column
        for column in [
            "total_cost",
            "spot_cost",
            "future_cost",
            "shift_penalty_cost",
            "energy_volume_mwh",
        ]
        if column in df.columns
    ]

    summary_df = (
        df.groupby(action_column, dropna=False)[numeric_columns]
        .agg(["count", "mean", "sum"])
    )
    summary_df.columns = [f"{column}_{agg}" for column, agg in summary_df.columns]
    summary_df = summary_df.reset_index()

    if action_column != "recommended_action":
        summary_df = summary_df.rename(columns={action_column: "recommended_action"})

    return summary_df


# =========================
# Convenience report helper
# =========================

def build_policy_evaluation_report(policy_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build a compact policy evaluation package.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing several policy summary tables.
    """
    validate_action_catalog()
    report: dict[str, pd.DataFrame] = {
        "action_summary": summarize_policy_actions(policy_df),
        "time_coverage_summary": summarize_policy_time_coverage(policy_df),
        "risk_signal_summary": summarize_actions_vs_risk_signals(policy_df),
    }

    try:
        report["calendar_context_summary"] = summarize_actions_by_calendar_context(policy_df)
    except PolicyEvaluationError:
        pass

    try:
        report["reason_summary"] = summarize_policy_reasons(policy_df)
    except PolicyEvaluationError:
        pass

    try:
        report["simulated_action_cost_summary"] = summarize_simulated_action_costs(policy_df)
    except PolicyEvaluationError:
        pass

    return report


if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2025-01-01", periods=8, freq="D"),
            "recommended_action": [
                ACTION_DO_NOTHING,
                ACTION_BUY_M1_FUTURE,
                ACTION_BUY_M1_FUTURE,
                ACTION_SHIFT_PRODUCTION,
                ACTION_DO_NOTHING,
                ACTION_BUY_M1_FUTURE,
                ACTION_SHIFT_PRODUCTION,
                ACTION_DO_NOTHING,
            ],
            "decision_reason": [
                "No rule triggered.",
                "Tail risk above futures price.",
                "Tail risk above futures price.",
                "Flexible day and high tail risk.",
                "No rule triggered.",
                "Tail risk above futures price.",
                "Flexible day and high tail risk.",
                "No rule triggered.",
            ],
            "is_weekend": [0, 0, 0, 1, 1, 0, 1, 0],
            "Is_national_holiday": [0, 0, 0, 0, 0, 0, 0, 1],
            "tail_vs_future_abs": [2.0, 10.5, 12.2, 15.8, 3.1, 9.7, 18.0, 1.5],
            "tail_vs_future_rel": [0.02, 0.11, 0.13, 0.18, 0.03, 0.10, 0.20, 0.01],
            "tail_vs_central_abs": [1.0, 7.0, 8.0, 12.0, 1.5, 6.0, 14.0, 0.5],
            "tail_vs_central_rel": [0.01, 0.08, 0.09, 0.15, 0.02, 0.07, 0.18, 0.01],
            "forecast_central": [70, 72, 74, 76, 71, 73, 78, 69],
            "forecast_tail": [72, 82.5, 86.2, 91.8, 74.1, 82.7, 96.0, 70.5],
            "current_m1_future": [70, 72, 74, 76, 71, 73, 78, 69],
            "total_cost": [70.0, 72.0, 74.0, 20.0, 71.0, 73.0, 25.0, 69.0],
            "spot_cost": [70.0, 0.0, 0.0, 0.0, 71.0, 0.0, 0.0, 69.0],
            "future_cost": [0.0, 72.0, 74.0, 0.0, 0.0, 73.0, 0.0, 0.0],
            "shift_penalty_cost": [0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 25.0, 0.0],
            "energy_volume_mwh": [1.0] * 8,
        }
    )

    report = build_policy_evaluation_report(example_df)

    for name, table in report.items():
        print(f"\n=== {name.upper()} ===")
        print(table)
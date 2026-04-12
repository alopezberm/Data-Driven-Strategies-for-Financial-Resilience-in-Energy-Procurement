

"""
scenario_explanations.py

Utilities for generating human-readable explanations for specific forecast,
risk, and policy scenarios. These explanations are intended to support the
technical report, notebooks, and business-facing storytelling.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.constants import DEFAULT_FUTURE_COLUMN, DEFAULT_Q50_COLUMN, DEFAULT_Q90_COLUMN


class ScenarioExplanationError(Exception):
    """Raised when a scenario explanation cannot be generated safely."""


@dataclass
class ScenarioExplanationConfig:
    """Configuration for scenario-level textual explanations."""

    q50_column: str = DEFAULT_Q50_COLUMN
    q90_column: str = DEFAULT_Q90_COLUMN
    future_column: str = DEFAULT_FUTURE_COLUMN
    action_column: str = "recommended_action"
    reason_column: str = "decision_reason"
    date_column: str = "date"
    high_risk_label_threshold: float = 10.0
    moderate_risk_label_threshold: float = 5.0


# =========================
# Validation helpers
# =========================

def _validate_dataframe(
    df: pd.DataFrame,
    config: ScenarioExplanationConfig,
) -> pd.DataFrame:
    """Validate the scenario dataframe and standardize the date column when present."""
    if df.empty:
        raise ScenarioExplanationError("Scenario dataframe is empty.")

    required_columns = {
        config.q50_column,
        config.q90_column,
        config.future_column,
    }
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ScenarioExplanationError(
            f"Scenario dataframe is missing required columns: {missing_columns}"
        )

    result_df = df.copy()
    if config.date_column in result_df.columns:
        result_df[config.date_column] = pd.to_datetime(
            result_df[config.date_column],
            errors="coerce",
        )
    return result_df



def _get_row(df: pd.DataFrame, row_index: int) -> pd.Series:
    """Return a single scenario row by positional index."""
    if row_index < 0 or row_index >= len(df):
        raise ScenarioExplanationError(
            f"row_index {row_index} is out of bounds for dataframe of length {len(df)}."
        )
    return df.iloc[row_index]


# =========================
# Low-level explanation helpers
# =========================

def classify_risk_level(
    tail_vs_future_abs: float,
    config: ScenarioExplanationConfig | None = None,
) -> str:
    """Classify scenario risk intensity based on the tail-vs-future spread."""
    config = ScenarioExplanationConfig() if config is None else config

    if tail_vs_future_abs >= config.high_risk_label_threshold:
        return "high"
    if tail_vs_future_abs >= config.moderate_risk_label_threshold:
        return "moderate"
    return "low"



def compute_scenario_signals(
    row: pd.Series,
    config: ScenarioExplanationConfig | None = None,
) -> dict[str, float | str | None]:
    """Compute interpretable signals for one scenario row."""
    config = ScenarioExplanationConfig() if config is None else config

    q50 = float(row[config.q50_column])
    q90 = float(row[config.q90_column])
    future_price = float(row[config.future_column])

    tail_vs_future_abs = q90 - future_price
    tail_vs_future_rel = tail_vs_future_abs / future_price if future_price != 0 else float("nan")
    tail_vs_central_abs = q90 - q50
    tail_vs_central_rel = tail_vs_central_abs / q50 if q50 != 0 else float("nan")
    risk_level = classify_risk_level(tail_vs_future_abs, config=config)

    return {
        "forecast_central": q50,
        "forecast_tail": q90,
        "current_m1_future": future_price,
        "tail_vs_future_abs": tail_vs_future_abs,
        "tail_vs_future_rel": tail_vs_future_rel,
        "tail_vs_central_abs": tail_vs_central_abs,
        "tail_vs_central_rel": tail_vs_central_rel,
        "risk_level": risk_level,
        "recommended_action": row.get(config.action_column),
        "decision_reason": row.get(config.reason_column),
        "date": row.get(config.date_column),
    }


# =========================
# Text explanation builders
# =========================

def explain_scenario_row(
    row: pd.Series,
    config: ScenarioExplanationConfig | None = None,
) -> str:
    """Generate a human-readable explanation for one scenario row."""
    config = ScenarioExplanationConfig() if config is None else config
    signals = compute_scenario_signals(row, config=config)

    date_value = signals["date"]
    if pd.notna(date_value):
        if hasattr(date_value, "strftime"):
            date_text = date_value.strftime("%Y-%m-%d")
        else:
            date_text = str(date_value)
    else:
        date_text = "unknown date"

    action_text = signals["recommended_action"]
    if pd.isna(action_text):
        action_text = "no explicit action"

    reason_text = signals["decision_reason"]
    if pd.isna(reason_text):
        reason_text = "No explicit decision reason was provided."

    return (
        f"On {date_text}, the central forecast was {signals['forecast_central']:.2f} while the "
        f"upper-tail forecast reached {signals['forecast_tail']:.2f}. The current M1 futures price "
        f"was {signals['current_m1_future']:.2f}, implying a tail-vs-future spread of "
        f"{signals['tail_vs_future_abs']:.2f} ({signals['tail_vs_future_rel']:.1%}). "
        f"Relative to the central scenario, the tail forecast was higher by "
        f"{signals['tail_vs_central_abs']:.2f} ({signals['tail_vs_central_rel']:.1%}). "
        f"This is classified as a {signals['risk_level']} risk scenario. The policy therefore "
        f"recommended '{action_text}'. Reason: {reason_text}"
    )



def explain_dataframe_scenarios(
    df: pd.DataFrame,
    config: ScenarioExplanationConfig | None = None,
    max_rows: int = 5,
) -> pd.DataFrame:
    """
    Generate scenario explanations for the first `max_rows` rows of a dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with one explanation per selected row.
    """
    config = ScenarioExplanationConfig() if config is None else config
    validated_df = _validate_dataframe(df, config)

    if max_rows <= 0:
        raise ScenarioExplanationError("max_rows must be strictly positive.")

    selected_df = validated_df.head(max_rows).copy()
    explanations = [
        explain_scenario_row(row, config=config)
        for _, row in selected_df.iterrows()
    ]

    result_df = selected_df.copy()
    result_df["scenario_explanation"] = explanations
    return result_df



def explain_extreme_scenarios(
    df: pd.DataFrame,
    config: ScenarioExplanationConfig | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Generate explanations for the most extreme tail-vs-future scenarios.
    """
    config = ScenarioExplanationConfig() if config is None else config
    validated_df = _validate_dataframe(df, config)

    if top_n <= 0:
        raise ScenarioExplanationError("top_n must be strictly positive.")

    temp_df = validated_df.copy()
    temp_df["tail_vs_future_abs"] = temp_df[config.q90_column] - temp_df[config.future_column]
    extreme_df = temp_df.sort_values("tail_vs_future_abs", ascending=False).head(top_n).copy()

    explanations = [
        explain_scenario_row(row, config=config)
        for _, row in extreme_df.iterrows()
    ]
    extreme_df["scenario_explanation"] = explanations
    return extreme_df.reset_index(drop=True)


# =========================
# Convenience report helper
# =========================

def build_scenario_explanation_report(
    df: pd.DataFrame,
    config: ScenarioExplanationConfig | None = None,
    preview_rows: int = 5,
    top_extreme_rows: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    Build a compact scenario explanation report.
    """
    config = ScenarioExplanationConfig() if config is None else config
    validated_df = _validate_dataframe(df, config)

    return {
        "scenario_preview": explain_dataframe_scenarios(
            validated_df,
            config=config,
            max_rows=preview_rows,
        ),
        "extreme_scenarios": explain_extreme_scenarios(
            validated_df,
            config=config,
            top_n=top_extreme_rows,
        ),
    }


if __name__ == "__main__":
    example_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=6, freq="D"),
            "q_0.5": [50, 55, 60, 62, 58, 57],
            "q_0.9": [58, 66, 78, 80, 68, 65],
            "Future_M1_Price": [52, 57, 63, 64, 60, 58],
            "recommended_action": [
                "do_nothing",
                "buy_m1_future",
                "buy_m1_future",
                "shift_production",
                "buy_m1_future",
                "do_nothing",
            ],
            "decision_reason": [
                "No rule triggered",
                "Tail risk exceeds futures price threshold",
                "Tail risk exceeds futures price threshold",
                "Weekend + high tail risk vs central forecast",
                "Tail risk exceeds futures price threshold",
                "No rule triggered",
            ],
        }
    )

    report = build_scenario_explanation_report(example_df)

    for name, table in report.items():
        print(f"\n=== {name.upper()} ===")
        print(table[["date", "scenario_explanation"]].head())
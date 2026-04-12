"""
policy_inputs.py

Utilities to prepare policy-ready inputs from market data and quantile model outputs.
This module acts as the bridge between forecasting and decision-making.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.config.constants import DATE_COLUMN, POLICY_REQUIRED_COLUMNS, PRIMARY_FUTURE_COLUMN, Q50_COLUMN, Q90_COLUMN, SPOT_PRICE_COLUMN
from src.models.quantile_models import QuantileModelResults


class PolicyInputsError(Exception):
    """Raised when policy input tables cannot be prepared safely."""


REQUIRED_BASE_COLUMNS = set(POLICY_REQUIRED_COLUMNS[:3])

OPTIONAL_USEFUL_COLUMNS = [
    "is_weekend",
    "Is_national_holiday",
    "daily_energy_mwh",
    "forecast_central",
    "forecast_tail",
    "front_month_premium",
    "front_month_premium_rel",
    "temperature_2m_mean",
    "wind_speed_10m_max",
]


def _format_quantile_column(quantile: float) -> str:
    """Format a quantile value into the project's quantile-column naming convention."""
    return f"q_{quantile:g}"


# =========================
# Validation helpers
# =========================

def _validate_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the base market dataframe used to feed the policy."""
    if df.empty:
        raise PolicyInputsError("Base dataframe is empty.")

    missing_columns = REQUIRED_BASE_COLUMNS - set(df.columns)
    if missing_columns:
        raise PolicyInputsError(
            f"Base dataframe is missing required columns: {sorted(missing_columns)}"
        )

    validated_df = df.copy()
    validated_df[DATE_COLUMN] = pd.to_datetime(validated_df[DATE_COLUMN], errors="coerce")

    if validated_df[DATE_COLUMN].isna().any():
        invalid_count = int(validated_df[DATE_COLUMN].isna().sum())
        raise PolicyInputsError(
            f"Found {invalid_count} invalid date values in the base dataframe."
        )

    if validated_df[DATE_COLUMN].duplicated().any():
        raise PolicyInputsError("Base dataframe contains duplicated dates.")

    return validated_df.sort_values(DATE_COLUMN).reset_index(drop=False)



def _validate_quantile_results(results: Sequence[QuantileModelResults]) -> None:
    """Validate that quantile model outputs are present and structurally coherent."""
    if not results:
        raise PolicyInputsError("Quantile results sequence cannot be empty.")

    reference_index = results[0].y_true.index
    for result in results:
        if not result.y_true.index.equals(reference_index):
            raise PolicyInputsError(
                "All quantile results must share the same evaluation index."
            )


# =========================
# Quantile prediction formatting
# =========================

def quantile_results_to_frame(results: Sequence[QuantileModelResults]) -> pd.DataFrame:
    """
    Convert quantile model outputs into a single dataframe.

    Output columns:
    - y_true
    - q_<quantile>
    """
    _validate_quantile_results(results)

    quantile_df = pd.DataFrame(index=results[0].y_true.index)
    quantile_df["y_true"] = results[0].y_true

    for result in sorted(results, key=lambda x: x.quantile):
        quantile_df[_format_quantile_column(result.quantile)] = result.y_pred

    return quantile_df


# =========================
# Policy input preparation
# =========================

def prepare_policy_inputs(
    base_df: pd.DataFrame,
    quantile_results: Sequence[QuantileModelResults],
    include_optional_columns: bool = True,
) -> pd.DataFrame:
    """
    Build a policy-ready dataframe by combining market data with quantile forecasts.

    Parameters
    ----------
    base_df : pd.DataFrame
        Test/evaluation dataframe containing actual market inputs.
    quantile_results : Sequence[QuantileModelResults]
        Quantile model outputs generated on the same evaluation rows.
    include_optional_columns : bool, optional
        Whether to preserve a set of optional but useful business/context columns.

    Returns
    -------
    pd.DataFrame
        Policy-ready dataframe ready for `apply_heuristic_policy(...)`.
    """
    validated_base_df = _validate_base_dataframe(base_df)
    quantile_df = quantile_results_to_frame(quantile_results)

    # Align by shared source index rather than requiring equal dataframe length.
    shared_index = quantile_df.index.intersection(validated_base_df["index"])

    if len(shared_index) == 0:
        raise PolicyInputsError(
            "No shared index values were found between base_df and quantile results."
        )

    aligned_base_df = validated_base_df[validated_base_df["index"].isin(shared_index)].copy()
    quantile_df = quantile_df.loc[shared_index].copy()

    aligned_base_df = aligned_base_df.rename(columns={"index": "source_index"})
    quantile_df = quantile_df.reset_index(drop=False).rename(columns={"index": "source_index"})

    policy_df = aligned_base_df.merge(
        quantile_df,
        on="source_index",
        how="inner",
        validate="one_to_one",
    )

    # Keep a clean and predictable column order.
    core_columns = [
        DATE_COLUMN,
        SPOT_PRICE_COLUMN,
        PRIMARY_FUTURE_COLUMN,
    ]
    quantile_columns = sorted(
        [column for column in policy_df.columns if column.startswith("q_")],
        key=lambda x: float(x.replace("q_", "")),
    )

    selected_columns = core_columns + quantile_columns

    if include_optional_columns:
        selected_columns.extend(
            [column for column in OPTIONAL_USEFUL_COLUMNS if column in policy_df.columns]
        )

    # Preserve y_true for diagnostics if available.
    if "y_true" in policy_df.columns:
        selected_columns.append("y_true")

    # Preserve source index so downstream debugging remains easy.
    if "source_index" in policy_df.columns:
        selected_columns.append("source_index")

    # Remove duplicates while preserving order.
    selected_columns = list(dict.fromkeys(selected_columns))

    missing_selected = [column for column in selected_columns if column not in policy_df.columns]
    if missing_selected:
        raise PolicyInputsError(
            f"The prepared policy dataframe is missing expected columns: {missing_selected}"
        )

    policy_df = policy_df[selected_columns].sort_values(DATE_COLUMN).reset_index(drop=True)

    return policy_df



def summarize_policy_inputs(policy_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact overview of the policy input dataframe."""
    if policy_df.empty:
        raise PolicyInputsError("policy_df is empty.")

    summary = pd.DataFrame(
        {
            "n_rows": [int(len(policy_df))],
            "date_min": [policy_df[DATE_COLUMN].min()],
            "date_max": [policy_df[DATE_COLUMN].max()],
            "n_quantile_columns": [
                int(sum(column.startswith("q_") for column in policy_df.columns))
            ],
            "has_weekend_flag": ["is_weekend" in policy_df.columns],
            "has_holiday_flag": ["Is_national_holiday" in policy_df.columns],
            "has_volume_column": ["daily_energy_mwh" in policy_df.columns],
        }
    )

    return summary


if __name__ == "__main__":
    base_df = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2025-01-01", periods=4, freq="D"),
            SPOT_PRICE_COLUMN: [70, 75, 80, 78],
            PRIMARY_FUTURE_COLUMN: [72, 73, 74, 75],
            "is_weekend": [0, 0, 1, 1],
            "Is_national_holiday": [0, 0, 0, 0],
            "daily_energy_mwh": [10, 10, 10, 10],
        },
        index=[100, 101, 102, 103],
    )

    q50 = QuantileModelResults(
        quantile=0.5,
        model_name="gbr_quantile_0.5",
        y_true=pd.Series([75, 80, 78, 82], index=[100, 101, 102, 103]),
        y_pred=pd.Series([74, 79, 80, 81], index=[100, 101, 102, 103]),
        pinball_loss=0.0,
        mae=0.0,
        rmse=0.0,
    )
    q90 = QuantileModelResults(
        quantile=0.9,
        model_name="gbr_quantile_0.9",
        y_true=pd.Series([75, 80, 78, 82], index=[100, 101, 102, 103]),
        y_pred=pd.Series([85, 92, 88, 95], index=[100, 101, 102, 103]),
        pinball_loss=0.0,
        mae=0.0,
        rmse=0.0,
    )

    policy_df = prepare_policy_inputs(base_df, [q50, q90])
    assert Q50_COLUMN in policy_df.columns and Q90_COLUMN in policy_df.columns
    print(policy_df)
    print(summarize_policy_inputs(policy_df))

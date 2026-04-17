"""
run_full_pipeline.py

End-to-end pipeline execution script.

This script runs the full workflow:
1. Build modeling dataset
2. Build feature dictionary
3. Run backtest

Usage:
    python -m src.pipeline.run_full_pipeline
"""

from __future__ import annotations

import time
import traceback

from src.pipeline.build_modeling_dataset import build_modeling_dataset
from src.pipeline.build_feature_dictionary import build_feature_dictionary
from src.pipeline.run_backtest import run_backtest_pipeline
from src.utils.logger import get_logger


logger = get_logger(__name__)


# =========================
# Main pipeline
# =========================


def run_full_pipeline() -> dict | None:
    """
    Execute the full project pipeline.

    Steps:
    1. Build modeling dataset
    2. Build feature dictionary
    3. Run backtest
    """

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("STARTING FULL PIPELINE")
    logger.info("=" * 60)

    try:
        # -------------------------
        # Step 1: Modeling dataset
        # -------------------------
        step_start = time.time()
        logger.info("[1/3] Building modeling dataset...")
        build_modeling_dataset()
        logger.info(f"[1/3] Done in {time.time() - step_start:.2f}s")

        # -------------------------
        # Step 2: Feature dictionary
        # -------------------------
        step_start = time.time()
        logger.info("[2/3] Building feature dictionary...")
        build_feature_dictionary()
        logger.info(f"[2/3] Done in {time.time() - step_start:.2f}s")

        # -------------------------
        # Step 3: Backtest
        # -------------------------
        step_start = time.time()
        logger.info("[3/3] Running backtest pipeline...")
        outputs = run_backtest_pipeline()
        logger.info(f"[3/3] Done in {time.time() - step_start:.2f}s")

        total_time = time.time() - start_time

        logger.info("=" * 60)
        logger.info(f"FULL PIPELINE COMPLETED SUCCESSFULLY in {total_time:.2f} seconds")
        logger.info("=" * 60)

        return outputs

    except Exception as exc:
        logger.error("FULL PIPELINE FAILED")
        logger.error(str(exc))
        logger.error(traceback.format_exc())
        raise


# =========================
# CLI entrypoint
# =========================


if __name__ == "__main__":
    run_full_pipeline()

"""
paths.py

Centralized path configuration for the project.
Ensures consistent and reproducible file handling across modules.
"""

from pathlib import Path

from src.config.constants import (
    OUTPUT_BACKTESTS_DIRNAME,
    OUTPUT_FIGURES_DIRNAME,
    OUTPUT_POLICIES_DIRNAME,
)

# =========================
# ROOT DIRECTORY
# =========================

# Root of the repository (2 levels up from this file)
BASE_DIR = Path(__file__).resolve().parents[2]

# =========================
# DATA DIRECTORIES
# =========================

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Raw subfolders
RAW_OMIP_DIR = RAW_DATA_DIR / "omip"
RAW_WEATHER_DIR = RAW_DATA_DIR / "weather"
RAW_HOLIDAYS_DIR = RAW_DATA_DIR / "holidays"

# =========================
# FILE PATHS (RAW)
# =========================

OMIP_RAW_FILE = RAW_OMIP_DIR / "omip_prices_raw.csv"
WEATHER_RAW_FILE = RAW_WEATHER_DIR / "openmeteo_raw.csv"
HOLIDAYS_RAW_FILE = RAW_HOLIDAYS_DIR / "holidays_raw.csv"

# =========================
# FILE PATHS (INTERIM)
# =========================

OMIP_CLEAN_FILE = INTERIM_DATA_DIR / "omip_clean.csv"
WEATHER_CLEAN_FILE = INTERIM_DATA_DIR / "weather_clean.csv"
MERGED_INTERIM_FILE = INTERIM_DATA_DIR / "merged_interim.csv"

# =========================
# FILE PATHS (PROCESSED)
# =========================

MODELING_DATASET_FILE = PROCESSED_DATA_DIR / "modeling_dataset.csv"

TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
VALIDATION_FILE = PROCESSED_DATA_DIR / "validation.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

FEATURE_DICTIONARY_FILE = PROCESSED_DATA_DIR / "feature_dictionary.csv"

# =========================
# OUTPUT DIRECTORIES
# =========================

FORECASTS_DIR = OUTPUTS_DIR / "forecasts"  # keep as-is (not yet in constants)
BACKTESTS_DIR = OUTPUTS_DIR / OUTPUT_BACKTESTS_DIRNAME
POLICIES_DIR = OUTPUTS_DIR / OUTPUT_POLICIES_DIRNAME
FIGURES_DIR = OUTPUTS_DIR / OUTPUT_FIGURES_DIRNAME

# =========================
# UTILITY FUNCTION
# =========================

def create_directories():
    """
    Create all necessary directories if they do not exist.
    This should be run once at the beginning of the pipeline.
    """
    directories = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUTS_DIR,
        RAW_OMIP_DIR,
        RAW_WEATHER_DIR,
        RAW_HOLIDAYS_DIR,
        FORECASTS_DIR,
        BACKTESTS_DIR,
        POLICIES_DIR,
        FIGURES_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Utility to get output path by category
def get_output_path(category: str) -> Path:
    """Return a path inside the outputs directory based on category name."""
    mapping = {
        "forecasts": FORECASTS_DIR,
        OUTPUT_BACKTESTS_DIRNAME: BACKTESTS_DIR,
        OUTPUT_POLICIES_DIRNAME: POLICIES_DIR,
        OUTPUT_FIGURES_DIRNAME: FIGURES_DIR,
    }
    if category not in mapping:
        raise ValueError(f"Unknown output category: {category}")
    return mapping[category]


if __name__ == "__main__":
    create_directories()
    print("Project directories initialized successfully.")

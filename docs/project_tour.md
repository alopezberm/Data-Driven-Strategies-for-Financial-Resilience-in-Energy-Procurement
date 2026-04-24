# Project Tour

This document walks the repository in the order you would actually run it.
It is meant to give a fast mental model of the project, the data flow, and
where the reusable logic lives.

## 1. What The Project Does

The project is a decision support system for industrial electricity procurement.
It combines:

- market and weather data ingestion
- feature engineering on daily time series
- quantile forecasting for tail-risk awareness
- a rule-based procurement policy
- an experimental RL policy
- counterfactual backtesting and strategy comparison

The main idea is simple: predict uncertainty, turn that signal into a daily
procurement action, and evaluate the result against baseline strategies.

## 2. The Order To Read Or Run It

If you want to follow the project end to end, this is the natural sequence:

1. Read the overview in [README.md](../README.md)
2. Inspect the source data under [data/raw](../data/raw)
3. Clean and align the data into [data/interim](../data/interim)
4. Build the modeling table in [data/processed/modeling_dataset.csv](../data/processed/modeling_dataset.csv)
5. Split the data chronologically into train, validation, and test sets
6. Train the quantile models
7. Convert predictions into policy inputs
8. Apply the heuristic and RL policies
9. Run the backtesting simulations
10. Review tables and figures in [data/outputs](../data/outputs)

The orchestration entry point is [src/pipeline/run_full_pipeline.py](../src/pipeline/run_full_pipeline.py).

## 3. Folder By Folder Tour

### Repository Root

- [README.md](../README.md): project summary, run instructions, and output overview
- [requirements.txt](../requirements.txt): Python dependencies for local runs
- [environment.yml](../environment.yml): Conda environment definition

### data/

This folder holds the datasets at each stage of the pipeline.

#### data/raw/

Source inputs before any cleaning or alignment.

- [data/raw/holidays](../data/raw/holidays): holiday data inputs
- [data/raw/omip](../data/raw/omip): OMIP futures inputs
- [data/raw/weather](../data/raw/weather): weather inputs

#### data/interim/

Cleaned and merged intermediate files.

- `holidays_clean.csv`: cleaned holiday calendar
- `omip_clean.csv`: cleaned futures prices and open interest
- `weather_clean.csv`: cleaned weather table
- `merged_interim.csv`: aligned daily dataset used to build features

#### data/processed/

Model-ready artifacts.

- `modeling_dataset.csv`: engineered feature matrix used for modeling
- `train.csv`, `validation.csv`, `test.csv`: chronological target-level splits
- `train_features.csv`, `validation_features.csv`, `test_features.csv`: feature-level splits used by modeling and backtesting
- `feature_dictionary.csv`: column documentation for the modeling dataset

#### data/outputs/

Generated results.

- `backtests/`: strategy metrics, quantile diagnostics, resilience tables
- `policies/`: policy decision outputs
- `figures/`: plots for forecasts, costs, actions, and comparisons

### notebooks/

The notebooks are organized by workflow stage.

#### 01_data_extraction/

- `01_extract_omip_data.ipynb`: pull or prepare OMIP market data
- `02_extract_weather_data.ipynb`: pull or prepare weather data

#### 02_data_understanding/

- exploratory analysis of OMIP, weather, and the merged dataset

#### 03_preprocessing/

- cleaning the raw inputs
- merging them into a single daily table
- building the first engineered features

#### 04_feature_engineering_and_analysis/

- deeper feature analysis, selection, and diagnostics

#### 05_modeling/

- baseline forecasting
- quantile regression
- tail-risk modeling
- model comparison

#### 06_decision_engine/

- heuristic policy logic
- RL prototype
- policy evaluation

#### 07_backtesting/

- counterfactual backtesting
- strategy comparison
- sensitivity analysis

#### 08_reporting/

- technical report support
- executive summary support

Important note: the reusable implementation now lives primarily in [src/](../src), so the notebooks are best treated as exploratory or reporting layers rather than the canonical implementation.

### src/

This is the real implementation layer.

#### src/config/

- centralized constants and paths
- project settings used across the pipeline

#### src/data/

- raw-data loading
- cleaning
- merging
- splitting into train/validation/test

#### src/features/

- calendar/time features
- lag features
- rolling-window features
- futures and market-structure features
- the feature-matrix builder that combines everything

#### src/models/

- baseline models
- quantile models
- tail-risk models
- training, prediction, and evaluation helpers

#### src/decision/

- policy inputs
- heuristic rules
- RL agent and RL policy inference
- policy evaluation helpers

#### src/rl/

- RL environment
- training loop
- evaluation utilities
- persistence and diagnostics

#### src/backtesting/

- baseline simulations
- policy simulations
- RL simulation
- strategy comparison
- resilience metrics

#### src/explainability/

- feature importance and scenario explanation helpers

#### src/visualization/

- forecast plots
- quantile plots
- backtest plots
- policy action plots

#### src/pipeline/

- `run_full_pipeline.py`: end-to-end orchestration
- `build_modeling_dataset.py`: create the modeling dataset from merged interim data
- `build_feature_dictionary.py`: document the final feature set
- `run_backtest.py`: train models, build policy inputs, simulate strategies, and save outputs

#### src/utils/

- logging
- metrics
- helper functions
- validation utilities

### tests/

Automated checks for the main subsystems.

- data pipeline
- feature engineering
- models
- backtesting
- RL

## 4. How The Pipeline Runs

The shortest accurate summary is:

1. Raw OMIP, weather, and holiday data are cleaned.
2. The cleaned tables are merged into one daily interim dataset.
3. Feature engineering expands that table into the modeling dataset.
4. The dataset is split chronologically to avoid leakage.
5. Quantile models are trained on the engineered features.
6. Policy inputs are built from model outputs and market context.
7. The heuristic policy and RL policy generate daily actions.
8. Baseline and policy simulations are compared in backtesting.
9. Outputs are written to `data/outputs/backtests`, `data/outputs/policies`, and `data/outputs/figures`.

The top-level command for the full workflow is:

```bash
python -m src.pipeline.run_full_pipeline
```

## 5. What To Look At First

If you only have a few minutes, start here:

- [README.md](../README.md) for the project summary
- [src/pipeline/run_full_pipeline.py](../src/pipeline/run_full_pipeline.py) for the orchestration flow
- [src/features/build_feature_matrix.py](../src/features/build_feature_matrix.py) for the feature-engineering logic
- [src/decision/heuristic_policy.py](../src/decision/heuristic_policy.py) for the core decision rule
- [src/backtesting/compare_strategies.py](../src/backtesting/compare_strategies.py) for strategy comparison

## 6. Practical Mental Model

Think of the repository in three layers:

- data preparation: raw -> interim -> processed
- decision making: quantile forecasts -> policy inputs -> actions
- evaluation: simulations -> summaries -> figures

That is the whole project in one loop.

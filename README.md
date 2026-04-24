# Data-Driven Strategies for Financial Resilience in Energy Procurement

A Data-Driven Decision Support System (DSS) to optimize industrial energy procurement and mitigate tail risks in the Spanish electricity market (Spot/Futures hedging). Developed for the Advanced Business Analytics course (42578) at DTU.

---

## 📌 Executive Summary

Industrial manufacturers in the Spanish electricity market (MIBEL) face extreme price volatility. Sudden Spot price spikes can significantly impact profitability. Traditional procurement strategies force a rigid choice between fixed contracts and Spot exposure, failing to dynamically manage **tail risks**.

This project presents an **end-to-end Decision Support System (DSS)** that acts as a virtual consultant, providing daily recommendations for:

- Financial hedging (futures contracts)
- Operational decisions (e.g., production shifting)

The goal is to **minimize expected costs under uncertainty while reducing exposure to extreme price events**.

---

## ⚙️ System Architecture

The solution follows a modular pipeline:

```
Raw Data → Cleaning → Feature Engineering → Modeling → Decision → Backtesting
```

### 1. Risk Prediction Engine

- Quantile forecasting (e.g., q50, q90)
- Tail-risk awareness instead of point prediction
- Captures extreme price scenarios

### 2. Decision Engine

Transforms risk signals into actions:

- Heuristic policy (baseline)
- RL-based policy (experimental extension)

### 3. Backtesting Engine

- Counterfactual simulation
- Strategy comparison:
  - Spot-only
  - Static hedge
  - Heuristic policy
	- RL policy

---

## 📊 Data Sources

- Spot prices (SPEL)
- OMIP Futures (M+1 to M+6)
- Weather data (OpenMeteo)
- Calendar features (holidays, seasonality)

Key principle: **strict chronological splits (no leakage)**.

---

## 📈 Outputs

After running the full pipeline, the system generates:

### Processed Data

- `data/processed/modeling_dataset.csv`
- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `data/processed/train_features.csv`
- `data/processed/validation_features.csv`
- `data/processed/test_features.csv`
- `data/processed/feature_dictionary.csv`

### Backtesting Results
- Strategy simulations (`data/outputs/backtests/`)
- Policy decisions (`data/outputs/policies/`)

### Visualizations
- Cost comparison
- Quantile forecasts
- Tail exceedances
- Policy timelines

---

## 🚀 How to Run the Project

### Option 1 — Using virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option 2 — Using Conda

```bash
conda env create -f environment.yml
conda activate tailrisk-env
```

---

### Run tests (sanity check)

```bash
python -m pytest tests/
```

Expected:

```
39 passed
```

---

### Regenerate holidays raw data (2020-2025)

If you want to rebuild the holidays source file used by preprocessing, run:

```bash
python -m src.data.generate_holidays_raw
```

This generates:

- `data/raw/holidays/holidays_raw.csv`

with Spanish national holidays for years 2020 through 2025.

You can also run the equivalent reproducible notebook:

- `notebooks/03_preprocessing/00_generate_holidays_raw.ipynb`

---

### Run full pipeline (end-to-end)

```bash
python -m src.pipeline.run_full_pipeline
```

This will:

1. Build modeling dataset
2. Generate features
3. Train quantile models
4. Apply decision policy
5. Run backtesting
6. Save results and figures

---

## 🤖 Reinforcement Learning Strategy

A fourth strategy based on tabular Reinforcement Learning (Q-learning) is implemented and fully integrated into the pipeline.

### Overview

The RL agent learns a daily decision policy over three actions:
- do_nothing
- buy_m1_future
- shift_production

The objective is to minimize realized energy procurement cost over time.

### Implementation

The RL module is structured as follows:
- `src/rl/rl_environment.py`: environment definition, reward function, and transitions
- `src/rl/train_rl_agent.py`: training pipeline, diagnostics, and artifact generation
- `src/rl/utils_rl.py`: RL utilities, summaries, and persistence helpers
- `src/decision/rl_agent.py`: tabular Q-learning agent
- `src/decision/rl_policy.py`: policy inference using the trained Q-table
- `src/backtesting/simulate_rl_policy.py`: RL strategy simulation in backtesting

### Integration in Backtesting

The RL strategy is evaluated alongside:
- spot_only
- static_hedge
- heuristic_policy

All strategies are compared on:
- total cost
- savings vs spot-only
- volatility
- resilience metrics

### Current Status

The RL strategy is fully functional and integrated, but should currently be considered experimental:
- it learns non-trivial policies and adapts to market conditions
- it is sensitive to reward design and state representation
- it may exploit simplifications in the simulation, especially around frequent shift_production

At the current stage:
- RL is implemented, tested, and included as a fourth comparable strategy
- RL can outperform other strategies under some configurations
- however, its behavior is not yet considered fully robust or fully interpretable
- the heuristic policy remains the most reliable benchmark for presentation and discussion

Future improvements may include:
- improved reward design
- tighter operational constraints
- refined state representation
- more realistic treatment of production shifting
- moving beyond tabular RL

---

## 📂 Project Structure

For a guided, step-by-step walkthrough of the repository, see [docs/project_tour.md](docs/project_tour.md).

```
.
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── holidays/
│   │   │   └── holidays_raw.csv
│   │   ├── omip/
│   │   │   └── omip_prices_raw.csv
│   │   └── weather/
│   │       └── openmeteo_raw.csv
│   │
│   ├── interim/
│   │   ├── holidays_clean.csv
│   │   ├── omip_clean.csv
│   │   ├── weather_clean.csv
│   │   └── merged_interim.csv
│   │
│   ├── processed/
│   │   ├── modeling_dataset.csv
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   ├── test.csv
│   │   ├── train_features.csv
│   │   ├── validation_features.csv
│   │   ├── test_features.csv
│   │   └── feature_dictionary.csv
│   │
│   └── outputs/
│       ├── backtests/
│       │   ├── extreme_days_vs_spot_only.csv
│       │   ├── quantile_coverage_summary.csv
│       │   ├── quantile_interval_summary.csv
│       │   ├── quantile_model_summary.csv
│       │   ├── quantile_upper_tail_exceedance_summary.csv
│       │   ├── resilience_summary.csv
│       │   ├── resilience_vs_spot_only.csv
│       │   ├── strategy_daily_comparison.csv
│       │   ├── strategy_summary_table.csv
│       │   ├── strategy_summary_vs_spot_only.csv
│       │   ├── validation_heuristic_policy.csv
│       │   ├── validation_rl_policy.csv
│       │   ├── validation_spot_only.csv
│       │   └── validation_static_hedge.csv
│       │
│       ├── policies/
│       │   ├── validation_policy_decisions.csv
│       │   └── validation_rl_policy_decisions.csv
│       │
│       └── figures/
│           ├── cumulative_costs_by_strategy.png
│           ├── daily_costs_by_strategy.png
│           ├── daily_savings_vs_spot_only.png
│           ├── heuristic_policy_action_timeline.png
│           ├── rl_policy_action_timeline.png
│           ├── policy_action_frequency.png
│           ├── policy_action_share.png
│           ├── policy_actions_timeline.png
│           ├── policy_actions_vs_tail_risk.png
│           ├── quantile_band_q50_q90.png
│           ├── quantile_error_q90.png
│           ├── quantile_forecasts.png
│           ├── total_cost_bar_chart.png
│           └── upper_tail_exceedances_q90.png
│
├── notebooks/
│   ├── 01_data_extraction/
│   │   ├── 01_extract_omip_data.ipynb
│   │   └── 02_extract_weather_data.ipynb
│   │
│   ├── 02_data_understanding/
│   │   ├── 01_eda_omip.ipynb
│   │   ├── 02_eda_weather.ipynb
│   │   └── 03_eda_merged_dataset.ipynb
│   │
│   ├── 03_preprocessing/
│   │   ├── 01_clean_omip.ipynb
│   │   ├── 02_clean_weather.ipynb
│   │   ├── 03_merge_datasets.ipynb
│   │   └── 04_feature_engineering.ipynb
│   │
│   ├── 04_modeling/
│   │   ├── 01_baseline_forecast.ipynb
│   │   ├── 02_quantile_regression.ipynb
│   │   ├── 03_tail_risk_model.ipynb
│   │   └── 04_model_comparison.ipynb
│   │
│   ├── 05_decision_engine/
│   │   ├── 01_heuristic_policy.ipynb
│   │   ├── 02_rl_prototype.ipynb
│   │   └── 03_policy_evaluation.ipynb
│   │
│   ├── 06_backtesting/
│   │   ├── 01_counterfactual_backtest.ipynb
│   │   ├── 02_strategy_comparison.ipynb
│   │   └── 03_sensitivity_analysis.ipynb
│   │
│   └── 07_reporting/
│       ├── 01_technical_report.ipynb
│       └── 02_executive_summary_support.ipynb
│
├── src/
│   ├── config/
│   │   ├── paths.py
│   │   ├── settings.py
│   │   └── constants.py
│   │
│   ├── data/
│   │   ├── load_raw_data.py
│   │   ├── clean_omip.py
│   │   ├── clean_weather.py
│   │   ├── clean_holidays.py
│   │   ├── merge_data.py
│   │   └── split_data.py
│   │
│   ├── features/
│   │   ├── build_time_features.py
│   │   ├── build_lag_features.py
│   │   ├── build_rolling_features.py
│   │   ├── build_future_features.py
│   │   ├── build_feature_matrix.py
│   │   └── feature_selection.py
│   │
│   ├── models/
│   │   ├── baseline_models.py
│   │   ├── quantile_models.py
│   │   ├── tail_risk_models.py
│   │   ├── train_model.py
│   │   ├── predict.py
│   │   └── evaluate_model.py
│   │
│   ├── decision/
│   │   ├── policy_inputs.py
│   │   ├── heuristic_policy.py
│   │   ├── rl_agent.py
│   │   ├── rl_policy.py
│   │   ├── action_rules.py
│   │   └── policy_evaluation.py
│   │
│   ├── rl/
│   │   ├── rl_environment.py
│   │   ├── train_rl_agent.py
│   │   ├── evaluate_rl_agent.py
│   │   └── utils_rl.py
│   │
│   ├── backtesting/
│   │   ├── simulate_baseline.py
│   │   ├── simulate_policy.py
│   │   ├── simulate_rl_policy.py
│   │   ├── compare_strategies.py
│   │   └── resilience_metrics.py
│   │
│   ├── explainability/
│   │   ├── shap_analysis.py
│   │   ├── feature_importance.py
│   │   └── scenario_explanations.py
│   │
│   ├── visualization/
│   │   ├── plot_forecasts.py
│   │   ├── plot_quantiles.py
│   │   ├── plot_backtest_results.py
│   │   └── plot_policy_actions.py
│   │
│   ├── pipeline/
│   │   ├── run_full_pipeline.py
│   │   ├── run_backtest.py
│   │   ├── build_modeling_dataset.py
│   │   └── build_feature_dictionary.py
│   │
│   └── utils/
│       ├── logger.py
│       ├── metrics.py
│       ├── helpers.py
│       └── validation.py
│
├── reports/
│   ├── figures/
│   └── technical_report/
│       ├── technical_report.ipynb
│       └── technical_report.html
│
├── docs/
│   ├── project_plan.md
│   ├── methodology.md
│   ├── data_description.md
│   ├── feature_definitions.md
│   ├── modeling_decisions.md
│   └── backtesting_design.md
│
└── tests/
    ├── test_data_pipeline.py
    ├── test_feature_engineering.py
    ├── test_models.py
    ├── test_backtesting.py
    └── test_rl.py
```

---

## ✅ Current Status

- Full pipeline implemented ✅
- Modular architecture ✅
- Reproducible environment ✅
- All tests passing (39/39) ✅
- End-to-end execution working ✅
- RL strategy integrated as fourth strategy ✅

---

## 🎯 Key Contributions

- Tail-risk-aware electricity price forecasting
- Integration of financial and operational decisions
- End-to-end DSS pipeline
- Counterfactual backtesting framework
- Reproducible ML system
- Experimental RL extension integrated into the full pipeline

---

## 📌 Notes

- Notebooks are used for exploration and reporting
- Core logic is fully implemented in `/src/`
- Pipeline is production-style and reproducible
- RL is currently included as an experimental extension and benchmark, not as the primary production-ready decision policy

---

## 👥 Authors

DTU – MSc Business Analytics (Group 17)

s242875 - Ignacio Ripoll González | s253159 - Pablo Baurier Gasch | s253272 - Alejandro López Bermejo

---

## 📄 License

Academic project – for educational purposes only

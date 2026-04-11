# Data-Driven Strategies for Financial Resilience in Energy Procurement

A Data-Driven Decision Support System (DSS) to optimize industrial energy procurement and mitigate tail risks in the Spanish electricity market (Spot/Futures hedging). Developed for the Advanced Business Analytics course (42578) at DTU.

---

## 📌 Executive Summary

Industrial manufacturers in the Spanish electricity market (MIBEL) face extreme price volatility, where sudden Spot price spikes can erase monthly profit margins. Traditional procurement strategies force a rigid choice between expensive fixed contracts and highly volatile Spot market exposure, failing to dynamically manage **tail risks**.

This repository presents an **Advanced Decision Support System (DSS)** designed as a virtual consultant. The system provides daily, data-driven recommendations to optimize both:

- **Financial hedging** (via futures contracts)
- **Operational decisions** (e.g., shifting production based on external signals)

The objective is to **minimize expected procurement costs under uncertainty while reducing exposure to extreme price spikes (tail risk)**.

---

## ⚙️ Architecture & Methodology

The solution follows a two-stage analytics framework:

### 1. Risk Prediction Engine (Machine Learning)

Instead of predicting only expected prices, the system focuses on **uncertainty-aware forecasting**:

- Predicts **conditional quantiles** (e.g., 90th/95th percentile)
- Captures **tail risk exposure**
- Provides probabilistic insights into extreme price scenarios

This enables a shift from point forecasting to **risk-aware decision making**.

---

### 2. Prescriptive Decision Engine

A decision-making module transforms risk signals into actions:

- **Financial Actions**
  - Example: *Buy M+1 futures to lock in prices*
- **Operational Actions**
  - Example: *Shift production to avoid high-cost periods*

Two approaches are considered:

- Data-driven **heuristic policies** (baseline)
- **Reinforcement Learning (RL)** (advanced extension)

The system effectively solves:

\[
\text{Optimal Decision} = \arg\min \mathbb{E}[\text{Cost} \mid \text{Uncertainty}]
\]

---

## 📊 Data Strategy

The model relies on a time-aware, multi-source dataset:

### 🔹 Spot Market & External Drivers
- Daily Spot electricity prices (SPEL)
- Weather variables (temperature, wind, radiation, etc.)
- Calendar features (seasonality, holidays)

### 🔹 Hedging Alternatives
- OMIP Futures (M+1 to M+6)
- Prices and Open Interest

### 🔹 Key Design Principle
- Strict **chronological split** (no data leakage)
- Designed for **real-world deployment conditions**

---

## 📈 Business Impact Validation

To demonstrate real-world value, the project includes a:

### 🔁 Counterfactual Backtesting Framework

Simulates decision-making over unseen data:

- Applies DSS recommendations day-by-day
- Compares against baseline strategies:
  - Spot-only procurement
  - Static hedging

### 📏 Evaluation Metrics

- 💰 Total procurement cost
- 📉 Cost savings vs baseline
- ⚡ Exposure to extreme price events
- 📊 Stability of energy costs (resilience)

---

## 🎯 Key Contributions

- Tail-risk-aware forecasting of electricity prices
- Integration of financial hedging and operational flexibility
- Prescriptive analytics for decision optimization
- Counterfactual evaluation of strategies under uncertainty
- End-to-end Decision Support System (DSS)

---

## 🔁 Reproducibility

The project is fully modular and reproducible:

- Pipeline:  
  `Raw Data → Cleaning → Feature Engineering → Modeling → Decision → Backtesting`

- All experiments:
  - Respect temporal ordering
  - Avoid data leakage
  - Are reproducible via `/src/` scripts or `/notebooks/`

---

## 📂 Repository Structure

```bash
group17_tailrisk_solutions/
│
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── omip/
│   │   │   └── omip_prices_raw.csv
│   │   ├── weather/
│   │   │   └── openmeteo_raw.csv
│   │   └── holidays/
│   │       └── holidays_raw.csv
│   │
│   ├── interim/
│   │   ├── omip_clean.csv
│   │   ├── weather_clean.csv
│   │   └── merged_interim.csv
│   │
│   ├── processed/
│   │   ├── modeling_dataset.csv
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   ├── test.csv
│   │   └── feature_dictionary.csv
│   │
│   └── outputs/
│       ├── forecasts/
│       ├── backtests/
│       ├── policies/
│       └── figures/
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
│       ├── technical_report.ipynb
│       └── executive_summary_support.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── paths.py
│   │   ├── settings.py
│   │   └── constants.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_raw_data.py
│   │   ├── load_processed_data.py
│   │   ├── clean_omip.py
│   │   ├── clean_weather.py
│   │   ├── merge_data.py
│   │   └── split_data.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_time_features.py
│   │   ├── build_lag_features.py
│   │   ├── build_rolling_features.py
│   │   ├── build_future_features.py
│   │   └── feature_selection.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_models.py
│   │   ├── quantile_models.py
│   │   ├── tail_risk_models.py
│   │   ├── train_model.py
│   │   ├── predict.py
│   │   └── evaluate_model.py
│   │
│   ├── decision/
│   │   ├── __init__.py
│   │   ├── policy_inputs.py
│   │   ├── heuristic_policy.py
│   │   ├── rl_environment.py
│   │   ├── rl_agent.py
│   │   ├── action_rules.py
│   │   └── policy_evaluation.py
│   │
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── simulate_baseline.py
│   │   ├── simulate_policy.py
│   │   ├── compare_strategies.py
│   │   └── resilience_metrics.py
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_analysis.py
│   │   ├── feature_importance.py
│   │   └── scenario_explanations.py
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plot_forecasts.py
│   │   ├── plot_quantiles.py
│   │   ├── plot_backtest_results.py
│   │   └── plot_policy_actions.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── metrics.py
│       ├── helpers.py
│       └── validation.py
│
├── reports/
│   ├── figures/
│   ├── tables/
│   ├── executive_summary/
│   │   └── executive_summary.pdf
│   ├── technical_report/
│   │   ├── technical_report.ipynb
│   │   ├── technical_report.html
│   │   └── technical_report.pdf
│   └── contributions/
│       └── statement_of_contributions.pdf
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
    └── test_backtesting.py
```
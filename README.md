# Data-Driven-Strategies-for-Financial-Resilience-in-Energy-Procurement

A Data-Driven Decision Support System to optimize industrial energy procurement and mitigate tail risks in the Spanish electricity market (Spot/Futures hedging). Developed for the Advanced Business Analytics course (42578) at DTU.

## 📌 Executive Summary
Industrial manufacturers in the Spanish electricity market (MIBEL) face extreme price volatility, where sudden Spot energy spikes can erase monthly profit margins. Traditional procurement methods force a rigid choice between expensive fixed contracts and highly vulnerable Spot market exposure, struggling to dynamically manage "tail risks".

This repository contains the codebase for an **Advanced Decision Support System (DSS)**. Acting as a virtual consultant, this engine provides factory managers with daily, data-driven recommendations to optimize both financial hedging (Future contracts) and short-term operational schedules (shifting production based on weather), effectively minimizing energy costs under high uncertainty.

## ⚙️ Architecture & Methodology
The pipeline is structured into a two-step analytics framework:

1. **Risk Prediction Engine (Machine Learning):** Instead of merely predicting an average price, we train advanced predictive models to focus on uncertainty quantification. By forecasting the upper bounds of market prices (tail risks), the system assesses short- and mid-term financial exposure.
2. **Prescriptive Decision Engine:** A decision-making algorithm (utilizing Reinforcement Learning or data-driven heuristics) ingests the risk signals. It evaluates constraints and triggers either:
   * **Financial Actions:** e.g., "Buy M+1 futures today to lock in costs".
   * **Operational Actions:** e.g., "Postpone production based on short-term weather forecasts".

## 📊 Data Strategy
The model trains on a chronological split to ensure robust out-of-sample evaluation, utilizing two main data streams:
* **Spot Market & Operations:** Daily average Spot prices (baseload energy cost), combined with short-term weather and renewable generation forecasts.
* **Hedging Alternatives:** Daily Settlement Prices for "Spanish Power Base" Monthly Futures (OMIP), collected via automated Web Scraping.

## 📈 Business Impact Validation
To prove tangible financial value, the system includes a **Counterfactual Backtest** module. It simulates procurement and production operations over an unseen testing period, comparing the costs incurred by our DSS recommendations against a standard baseline strategy. Success is quantified by total monetary savings and profit margin stabilization.

---

## 📂 Repository Structure

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
│   │   ├── df_extraction.ipynb
│   │   └── 260404_OpenMeteo_Provincias.ipynb
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
│       ├── group17_TechnicalReport.ipynb
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
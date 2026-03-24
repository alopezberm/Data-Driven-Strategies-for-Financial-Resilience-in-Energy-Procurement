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

```text
├── data/
│   ├── raw/               # Raw market prices (OMIP) and weather forecasts
│   └── processed/         # Cleaned and merged datasets ready for modeling
├── notebooks/             # Jupyter notebooks for EDA and model prototyping
├── src/                   # Core Python scripts
│   ├── data_scraper.py    # Automated collection of futures and spot prices
│   ├── risk_model.py      # Probabilistic ML models for uncertainty quantification
│   ├── decision_engine.py # RL agent / Heuristics logic for daily recommendations
│   └── backtester.py      # Counterfactual simulation environment
├── requirements.txt       # Project dependencies
└── README.md              # You are here

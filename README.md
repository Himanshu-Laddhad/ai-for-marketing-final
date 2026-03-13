# AI for Marketing: Airbnb Host Attrition Prediction & Retention Optimization

End-to-end ML pipeline combining **XGBoost** and **LightGBM** with a **custom profit-scoring function** to predict Airbnb host churn and optimize a cost-constrained retention gift strategy.

## Overview

Airbnb loses revenue whenever a host deactivates. A $1,000 retention gift can prevent churn — but only for hosts where the expected revenue benefit exceeds the cost. This project builds a full ML system to:

1. **Understand** the data through EDA (distributions, class imbalance, geographic patterns, feature correlations)
2. **Engineer** revenue and profit features from reservation history
3. **Train and tune** XGBoost and LightGBM classifiers using Optuna
4. **Score** predictions with a custom profit function instead of generic AUC
5. **Apply** a decision-theoretic expected value rule to generate targeted gift recommendations

## Business Problem

> *Give a $1,000 gift if and only if:*
> **`p × annual_profit > $1,000`**
>
> This rule ensures every gift dollar has positive expected ROI, targeting only hosts where the predicted attrition risk × their value to Airbnb justifies the intervention cost.

## Dataset

| Field | Description |
|---|---|
| `reservationdays1–12` | Monthly booking days (12 months) |
| `averagedailyrateusd` | Host's average nightly rate |
| `rating_overall` | Average guest rating |
| `nmon` | Active months on platform |
| `latitude / longitude` | Host location |
| `attrition` | Target: 1 = churned, 0 = retained |

**Training file:** `abb.csv` | **Scoring file:** `abb_new.csv`

## Pipeline

```
abb.csv
  │
  ├── EDA (distributions, correlation heatmap, geo scatter)
  │
  ├── Feature Engineering
  │     total_reservationdays → totalRevenue → profit
  │
  ├── Preprocessing Pipeline
  │     Numeric: median impute + standard scale
  │     Categorical: constant impute + one-hot encode
  │
  ├── Optuna Hyperparameter Search
  │     XGBoost ──┐
  │               ├── Optimized for custom profit score
  │     LightGBM─┘
  │
  ├── Train Final Models on Full Dataset
  │
  ├── Apply to abb_new.csv
  │     → attrition_prob
  │     → expected_benefit
  │     → recommend_gift
  │
  └── Save: abb_new_updated.csv
```

## Results

| Model | Evaluated On |
|---|---|
| XGBoost (Optuna-tuned) | Validation set AUC + custom profit score |
| LightGBM (Optuna-tuned) | Validation set AUC + custom profit score |

Both models are compared on the **custom profit metric** rather than AUC alone — ensuring model selection is aligned with the actual business objective.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-gray)
![LightGBM](https://img.shields.io/badge/LightGBM-gray)
![Optuna](https://img.shields.io/badge/Optuna-gray)
![scikit--learn](https://img.shields.io/badge/scikit--learn-gray)
![pandas](https://img.shields.io/badge/pandas-gray)
![seaborn](https://img.shields.io/badge/seaborn-gray)
![matplotlib](https://img.shields.io/badge/matplotlib-gray)

## How to Run

```bash
git clone https://github.com/<your-username>/<repo-name>.git

pip install xgboost lightgbm optuna optuna-integration[sklearn] scikit-learn pandas numpy matplotlib seaborn

jupyter notebook AI_for_Marketing_final_final.ipynb
```

> **Note:** Place `abb.csv` and `abb_new.csv` in the same directory before running.

## File Structure

```
final-project-ai-marketing/
├── AI_for_Marketing_final_final.ipynb   # Main notebook
├── abb.csv                              # Training data (not included)
├── abb_new.csv                          # Scoring data (not included)
├── abb_new_predictions.csv             # Generated: initial predictions
├── abb_new_updated.csv                 # Generated: final output with gift flags
└── README.md
```

## Concepts Demonstrated

- EDA with class imbalance analysis and geographic visualization
- Feature engineering from time-series aggregates
- Custom ML scoring functions aligned to business objectives
- Multi-model comparison with Optuna hyperparameter tuning
- Expected value decision theory applied to ML outputs

---
*Course: MGMT 52610 — Data and AI-Driven Marketing*

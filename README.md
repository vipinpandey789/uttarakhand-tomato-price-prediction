# Uttarakhand Tomato Price Prediction

A machine learning project to predict daily tomato prices across all 13 districts of Uttarakhand using 6 years of mandi data (2020–2025). Built an ensemble of XGBoost, LightGBM, and Random Forest models with a Streamlit dashboard for live price prediction.

---

## Results

| Model | MAE (₹/qtl) | RMSE | R² | MAPE |
|---|---|---|---|---|
| **Ensemble (Final)** | **₹11.57** | **24.98** | **0.9992** | **0.61%** |
| XGBoost | ₹12.41 | 24.01 | 0.9992 | 0.67% |
| LightGBM | ₹12.63 | 25.65 | 0.9991 | 0.67% |
| Random Forest | ₹16.42 | 35.30 | 0.9984 | 0.86% |
| Ridge (Baseline) | ₹80.94 | 114.18 | 0.9828 | 5.06% |

The weighted ensemble (XGB 36.1% + LGB 35.8% + RF 28.1%) gave the best generalization across all districts.

---

## Project Structure

```
tomato_project/
│
├── Uttarakhand_Tomato_FINAL_Master.csv   ← Master dataset (28,496 rows)
├── predictions_2025.csv                  ← Model predictions on 2025 test set
├── model_comparison.csv                  ← All model metrics
├── per_district_performance.csv          ← Per-district MAE, MAPE, R²
│
├── Tomato_Price_Prediction.ipynb         ← Full analysis notebook
├── tomato_dashboard.html                 ← Standalone browser dashboard
│
├── app.py                                ← Streamlit interactive app
├── requirements.txt
│
└── models/
    ├── lgb_model.pkl
    ├── xgb_model.pkl
    ├── rf_model.pkl
    ├── ridge_model.pkl
    ├── ridge_scaler.pkl
    ├── encoders.pkl
    ├── features.pkl
    ├── ensemble_weights.pkl
    └── top3_names.pkl
```

---

## Dataset

- **Source:** Mandi price records for Uttarakhand (2020–2025)
- **Size:** 28,496 rows × 39 columns
- **Districts:** All 13 — Almora, Bageshwar, Chamoli, Champawat, Dehradun, Haridwar, Nainital, Pauri Garhwal, Pithoragarh, Rudraprayag, Tehri Garhwal, UdhamSinghNagar, Uttarkashi
- **Target variable:** `Price_Modal_Avg` (₹ per quintal)

Key engineered features: 1-day/7-day/30-day lag prices, rolling volatility, price momentum, log-transformed lags, seasonal multipliers, and district zone encoding.
---
---

## Dashboard Pages

**Overview** — Season-wise average prices, year-wise trend by district, monthly seasonality pattern

**Price Analysis** — Actual vs predicted (2025 test set), 6-year price history, district performance table with grades

**Model Results** — MAPE and MAE comparison across all models, LightGBM feature importances, full metrics table

**Predict Price** — Enter district, month, yesterday's price, 7-day avg, 30-day avg, and rainfall to get a live ensemble prediction with confidence range

---

## Key Findings

- Tomato prices in Uttarakhand follow a strong seasonal pattern — monsoon months (July–September) consistently show the highest prices (avg ₹2,728/qtl), while winter months (Nov–Feb) are the lowest (avg ₹1,014/qtl)
- Lag features (especially 1-day and 7-day price) dominate feature importance — short-term price momentum is a stronger predictor than rainfall or static district encoding
- Hill districts like Rudraprayag, Chamoli, and Pithoragarh show higher and more volatile prices compared to plains districts like Haridwar and UdhamSinghNagar
- The ensemble reduced MAE by 6.8% compared to the best single model (XGBoost alone)

---

## Tech Stack

- **Python** — pandas, numpy, scikit-learn
- **Models** — XGBoost, LightGBM, Random Forest, Ridge
- **Visualization** — Plotly, Matplotlib, Seaborn
- **Dashboard** — Streamlit
- **Storage** — joblib (.pkl model files), SQLite (notebook)

---

## Author

**Vipin Pandey**
MCA (Data Science) — UPES Dehradun

[LinkedIn](https://linkedin.com/in/vipin-pandey-21810328a) · [GitHub](https://github.com/vipinpandey789)

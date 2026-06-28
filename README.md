# Uttarakhand Tomato Price Prediction

Daily tomato modal price forecasting for all 13 districts of Uttarakhand, using
six years of district-level price and weather data.

## Overview

This project predicts the next day's tomato mandi price for a given district using
recent price history, seasonal patterns, and rainfall. Three models — Random Forest,
XGBoost, and LightGBM — are trained independently and combined into a single ensemble
prediction.

## Dashboard pages (app.py)

- **Overview** — KPI cards, a district map (approximate headquarters locations
  with the selected district highlighted), a price ranking panel, seasonal and
  yearly trend charts, and a multi-district comparison chart.
- **Price Analysis** — actual vs predicted prices on the 2025 test set, plus a
  full per-district performance table.
- **Model Results** — metrics and feature importances for all four models.
- **Predict Price** — single-day prediction with confidence bounds, using the
  real trained models.
- **Forecast** — multi-day forecast for any horizon up to 90 days, with quick
  presets (7/15/30/60/90 days) or a custom start and end date, a confidence
  band chart, a forecast table, and CSV export.

## Dataset

- 28,496 daily rows across 13 districts
- Period: January 2020 to December 2025
- Districts span plains (Dehradun, Haridwar, UdhamSinghNagar), Kumaon hills (Nainital,
  Almora, Pithoragarh, Champawat, Bageshwar), and Garhwal hills (Pauri Garhwal, Tehri
  Garhwal, Chamoli, Rudraprayag, Uttarkashi)
- Each row carries price, rainfall, and calendar information for one district on one day

## Model performance (2025 held-out test year)

| Model | MAE (Rs/quintal) | RMSE | R-squared | MAPE |
|---|---|---|---|---|
| Random Forest | 171.04 | 240.14 | 0.924 | 9.25% |
| XGBoost | 172.49 | 241.62 | 0.923 | 9.39% |
| LightGBM | 173.22 | 242.70 | 0.922 | 9.43% |
| **Ensemble (average)** | **171.77** | **240.72** | **0.924** | **9.32%** |

All splits are chronological: training on 2020–2023, validation on 2024 for early
stopping, and testing on 2025, with no shuffling across the time axis at any point.

Every district scores between 0.91 and 0.93 R-squared on the test set, with MAPE
between 8.70% and 10.02%. Pithoragarh and Nainital are the most predictable districts;
Champawat and Tehri Garhwal are the least, though the gap between best and worst is
small.

## Features

The model uses only information that would genuinely be available before the
prediction date:

- Price lagged by 1, 3, 7, 14, and 30 days
- Rolling price average and rolling price standard deviation over 7, 14, and 30 days,
  each computed on data shifted by one day so the window never touches the
  prediction day itself
- Rainfall lagged by 1 and 7 days, and a 7-day rolling rainfall average
- Calendar signals: month, week, quarter, day of week, season, monsoon flag, winter
  flag, and cyclical sine/cosine encodings of month and week
- District metadata: zone, elevation, district type, number of reporting markets

Same-day price minimum, maximum, and same-day weather readings are intentionally
excluded, since they would not be confirmed and available at the moment a real
prediction is needed.

## Project structure

```
Uttarakhand_Tomato_FINAL_Master.csv   source dataset
price_model.py                        shared feature engineering, prediction, and forecasting logic (not runnable directly — imported by app.py and the notebook)
Tomato_Price_Prediction.ipynb         full training notebook, runs end to end
app.py                                Streamlit dashboard with live prediction and multi-day forecast
.streamlit/config.toml                dark theme settings for the dashboard
tomato_dashboard.html                 static HTML dashboard (charts and metrics only)
models/
    random_forest.pkl
    xgboost_model.pkl
    lightgbm_model.pkl
    label_encoders.pkl
    predictor_columns.pkl
model_comparison.csv                  metrics for all four models
district_performance.csv              per-district MAE, R-squared, MAPE
test_predictions.csv                  row-level actual vs predicted, 2025
forecast_residual_bounds.csv          per-district 5th/95th percentile residuals, used for confidence bounds
district_lookup.json                  latest known values per district, used for auto-fill
dashboard_data.json                   data consumed by tomato_dashboard.html
```

## Running it

**Notebook**

```
pip install pandas numpy scikit-learn xgboost lightgbm joblib matplotlib seaborn
jupyter notebook Tomato_Price_Prediction.ipynb
```

Run all cells from top to bottom. The notebook trains all three models, evaluates
them, saves the `.pkl` files into `models/`, and writes `model_comparison.csv`,
`district_performance.csv`, and `test_predictions.csv`.

**Streamlit dashboard with live prediction**

```
pip install streamlit plotly pandas numpy scikit-learn xgboost lightgbm joblib
streamlit run app.py
```

The Predict Price page in `app.py` loads the trained `.pkl` files directly and runs
a genuine forward pass through all three models for any district, date, and price
input, then averages the three outputs into the displayed ensemble prediction.

**Static dashboard**

Open `tomato_dashboard.html` directly in a browser. It displays the dataset overview,
price analysis, and model results using pre-computed data in `dashboard_data.json`.
Since it is a static file with no Python backend, it cannot run the trained models
itself — its Predict Price page links to the Streamlit app for live predictions.

## Forecasting and confidence intervals

`app.py` includes a Forecast page supporting any horizon up to 90 days: quick picks
for 7, 15, 30, 60, and 90 days, or a custom start and end date. Multi-day forecasts
use recursive forecasting — each day's ensemble prediction is fed back in as the
next day's price input, since no real future price exists yet to use instead.
Accuracy decreases the further out the horizon goes, as later days depend on the
model's own earlier predictions rather than observed prices.

Every prediction, whether single-day or multi-day, includes a lower bound, upper
bound, and confidence percentage. These bounds come from the actual residuals
(actual minus predicted) recorded in `test_predictions.csv` for the 2025 test
year. For each district, the 5th and 95th percentile of that district's real
residuals are added to the point prediction, giving a 90% empirical confidence
range grounded in observed model error rather than an arbitrary fixed percentage.

## Scope and limitations

- Random Forest uses 200 trees at max depth 12, chosen specifically to keep
  `random_forest.pkl` small (about 22 MB) so the dashboard loads in a couple of
  seconds rather than the 5+ seconds a deeper, larger forest would take. Accuracy
  at this size matches the larger configuration tested during development.
- Multi-day forecasts compound uncertainty: a 30-day forecast is meaningfully less
  reliable than a 1-day forecast, since each step uses the previous step's
  prediction rather than a real observed price.
- Confidence bounds are derived from 2025 test-set residuals per district (365
  observations each). They describe typical historical error, not a guarantee for
  any specific future day.
- No geographically accurate map of Uttarakhand districts is included in either
  dashboard.

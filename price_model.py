import pandas as pd
import numpy as np
import warnings
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")


def load_raw_prices(path):
    frame = pd.read_csv(path)
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values(["District", "Date"]).reset_index(drop=True)
    return frame


def build_lagged_features(frame):
    base = frame[[
        "Date", "District", "Zone", "Dist_Type", "Elevation_m", "Num_Markets",
        "Price_Modal_Avg", "Rainfall_mm", "Year", "Month", "Quarter",
        "Week_of_Year", "Day_of_Week", "Season"
    ]].copy()

    price_by_district = base.groupby("District")["Price_Modal_Avg"]

    for lag_days in (1, 3, 7, 14, 30):
        base[f"price_lag_{lag_days}d"] = price_by_district.shift(lag_days)

    for window in (7, 14, 30):
        base[f"price_rollavg_{window}d"] = price_by_district.shift(1).transform(
            lambda series: series.rolling(window, min_periods=3).mean()
        )

    for window in (7, 30):
        base[f"price_rollstd_{window}d"] = price_by_district.shift(1).transform(
            lambda series: series.rolling(window, min_periods=3).std()
        )

    base["price_pct_change_7d"] = (
        (base["price_lag_1d"] - base["price_lag_7d"]) / base["price_lag_7d"]
    ) * 100

    rainfall_by_district = base.groupby("District")["Rainfall_mm"]
    base["rainfall_lag_1d"] = rainfall_by_district.shift(1)
    base["rainfall_lag_7d"] = rainfall_by_district.shift(7)
    base["rainfall_rollavg_7d"] = rainfall_by_district.shift(1).transform(
        lambda series: series.rolling(7, min_periods=3).mean()
    )

    base["month_sin"] = np.sin(2 * np.pi * base["Month"] / 12)
    base["month_cos"] = np.cos(2 * np.pi * base["Month"] / 12)
    base["week_sin"] = np.sin(2 * np.pi * base["Week_of_Year"] / 52)
    base["week_cos"] = np.cos(2 * np.pi * base["Week_of_Year"] / 52)
    base["is_monsoon"] = base["Month"].isin([6, 7, 8, 9]).astype(int)
    base["is_winter"] = base["Month"].isin([11, 12, 1, 2]).astype(int)
    base["price_momentum_7d"] = base["price_lag_1d"] - base["price_lag_7d"]
    base["lag1_over_roll30"] = base["price_lag_1d"] / (base["price_rollavg_30d"] + 1)

    base = base.drop(columns=["Rainfall_mm"])
    base = base.dropna(subset=["price_lag_30d", "price_rollavg_30d", "price_rollstd_30d"])
    return base.reset_index(drop=True)


def encode_categories(frame, columns):
    encoders = {}
    for column in columns:
        encoder = LabelEncoder()
        frame[column] = encoder.fit_transform(frame[column].astype(str))
        encoders[column] = encoder
    return frame, encoders


def time_based_split(frame, train_until, valid_year, test_year):
    train_rows = frame[frame["Year"] <= train_until]
    valid_rows = frame[frame["Year"] == valid_year]
    test_rows = frame[frame["Year"] == test_year]
    return train_rows, valid_rows, test_rows


def evaluate(label, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"model": label, "mae": round(mae, 2), "rmse": round(rmse, 2),
             "r2": round(r2, 4), "mape": round(mape, 2)}


def train_random_forest(x_train, y_train):
    model = RandomForestRegressor(
        n_estimators=400, max_depth=14, min_samples_leaf=6,
        max_features=0.6, n_jobs=-1, random_state=42
    )
    model.fit(x_train, y_train)
    return model


def train_xgboost(x_train, y_train, x_valid, y_valid):
    model = xgb.XGBRegressor(
        n_estimators=1200, learning_rate=0.03, max_depth=6,
        min_child_weight=8, subsample=0.8, colsample_bytree=0.75,
        reg_lambda=1.5, random_state=42, n_jobs=-1,
        early_stopping_rounds=50, eval_metric="rmse"
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    return model


def train_lightgbm(x_train, y_train, x_valid, y_valid):
    model = lgb.LGBMRegressor(
        n_estimators=1500, learning_rate=0.02, num_leaves=40,
        max_depth=7, min_child_samples=25, subsample=0.8,
        colsample_bytree=0.75, reg_lambda=1.0, random_state=42,
        n_jobs=-1, verbose=-1
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def build_prediction_row(district, month, year, week_of_year, day_of_week, quarter,
                          price_lag_1d, price_lag_3d, price_lag_7d, price_lag_14d, price_lag_30d,
                          price_rollavg_7d, price_rollavg_14d, price_rollavg_30d,
                          price_rollstd_7d, price_rollstd_30d,
                          rainfall_lag_1d, rainfall_lag_7d, rainfall_rollavg_7d,
                          district_meta, label_encoders, season_name):
    meta = district_meta[district]
    row = {
        "District": label_encoders["District"].transform([district])[0],
        "Zone": label_encoders["Zone"].transform([meta["zone"]])[0],
        "Dist_Type": label_encoders["Dist_Type"].transform([meta["dist_type"]])[0],
        "Elevation_m": meta["elevation_m"],
        "Num_Markets": meta["num_markets"],
        "Year": year,
        "Month": month,
        "Quarter": quarter,
        "Week_of_Year": week_of_year,
        "Day_of_Week": day_of_week,
        "Season": label_encoders["Season"].transform([season_name])[0],
        "price_lag_1d": price_lag_1d,
        "price_lag_3d": price_lag_3d,
        "price_lag_7d": price_lag_7d,
        "price_lag_14d": price_lag_14d,
        "price_lag_30d": price_lag_30d,
        "price_rollavg_7d": price_rollavg_7d,
        "price_rollavg_14d": price_rollavg_14d,
        "price_rollavg_30d": price_rollavg_30d,
        "price_rollstd_7d": price_rollstd_7d,
        "price_rollstd_30d": price_rollstd_30d,
        "price_pct_change_7d": ((price_lag_1d - price_lag_7d) / price_lag_7d) * 100,
        "rainfall_lag_1d": rainfall_lag_1d,
        "rainfall_lag_7d": rainfall_lag_7d,
        "rainfall_rollavg_7d": rainfall_rollavg_7d,
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "week_sin": np.sin(2 * np.pi * week_of_year / 52),
        "week_cos": np.cos(2 * np.pi * week_of_year / 52),
        "is_monsoon": 1 if month in (6, 7, 8, 9) else 0,
        "is_winter": 1 if month in (11, 12, 1, 2) else 0,
        "price_momentum_7d": price_lag_1d - price_lag_7d,
        "lag1_over_roll30": price_lag_1d / (price_rollavg_30d + 1),
    }
    return row


def season_for_month(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4):
        return "Spring"
    if month in (5, 6):
        return "Summer"
    if month in (7, 8, 9):
        return "Monsoon"
    return "Post_Monsoon"


def load_residual_bounds(path="forecast_residual_bounds.csv"):
    bounds = pd.read_csv(path)
    return bounds.set_index("district").to_dict("index")


def apply_confidence_bounds(district, predicted_price, residual_bounds, confidence_level=90):
    bounds = residual_bounds[district]
    lower_bound = max(0, predicted_price + bounds["residual_p05"])
    upper_bound = predicted_price + bounds["residual_p95"]
    return {
        "predicted_price": round(predicted_price, 0),
        "lower_bound": round(lower_bound, 0),
        "upper_bound": round(upper_bound, 0),
        "confidence_level": confidence_level,
    }


def forecast_recursive(district, start_date, horizon_days, price_history, rainfall_history,
                        forest_model, xgboost_model, lightgbm_model,
                        label_encoders, predictor_columns, district_meta, residual_bounds):
    """
    Produces a day-by-day forecast by feeding each predicted price back in as the
    next day's lag-1 input. price_history and rainfall_history must each contain
    at least the last 30 days of known values, most recent value last.
    """
    prices = list(price_history)
    rainfall = list(rainfall_history)
    current_date = start_date
    forecasts = []

    for step in range(horizon_days):
        price_lag_1d = prices[-1]
        price_lag_3d = prices[-3] if len(prices) >= 3 else prices[-1]
        price_lag_7d = prices[-7] if len(prices) >= 7 else prices[-1]
        price_lag_14d = prices[-14] if len(prices) >= 14 else prices[-1]
        price_lag_30d = prices[-30] if len(prices) >= 30 else prices[-1]

        recent_7 = prices[-7:] if len(prices) >= 7 else prices
        recent_14 = prices[-14:] if len(prices) >= 14 else prices
        recent_30 = prices[-30:] if len(prices) >= 30 else prices

        price_rollavg_7d = np.mean(recent_7)
        price_rollavg_14d = np.mean(recent_14)
        price_rollavg_30d = np.mean(recent_30)
        price_rollstd_7d = np.std(recent_7) if len(recent_7) >= 2 else 0.0
        price_rollstd_30d = np.std(recent_30) if len(recent_30) >= 2 else 0.0

        rain_recent_7 = rainfall[-7:] if len(rainfall) >= 7 else rainfall
        rainfall_lag_1d = rainfall[-1]
        rainfall_lag_7d = rainfall[-7] if len(rainfall) >= 7 else rainfall[-1]
        rainfall_rollavg_7d = np.mean(rain_recent_7)

        week_of_year = current_date.isocalendar()[1]
        quarter = (current_date.month - 1) // 3 + 1
        season_name = season_for_month(current_date.month)

        row = build_prediction_row(
            district=district, month=current_date.month, year=current_date.year,
            week_of_year=week_of_year, day_of_week=current_date.weekday(), quarter=quarter,
            price_lag_1d=price_lag_1d, price_lag_3d=price_lag_3d, price_lag_7d=price_lag_7d,
            price_lag_14d=price_lag_14d, price_lag_30d=price_lag_30d,
            price_rollavg_7d=price_rollavg_7d, price_rollavg_14d=price_rollavg_14d,
            price_rollavg_30d=price_rollavg_30d,
            price_rollstd_7d=price_rollstd_7d, price_rollstd_30d=price_rollstd_30d,
            rainfall_lag_1d=rainfall_lag_1d, rainfall_lag_7d=rainfall_lag_7d,
            rainfall_rollavg_7d=rainfall_rollavg_7d,
            district_meta=district_meta, label_encoders=label_encoders, season_name=season_name
        )

        input_frame = pd.DataFrame([row])[predictor_columns]
        forest_pred = forest_model.predict(input_frame)[0]
        xgboost_pred = xgboost_model.predict(input_frame)[0]
        lightgbm_pred = lightgbm_model.predict(input_frame)[0]
        ensemble_pred = (forest_pred + xgboost_pred + lightgbm_pred) / 3

        bounds = apply_confidence_bounds(district, ensemble_pred, residual_bounds)

        forecasts.append({
            "date": current_date,
            "day_ahead": step + 1,
            "predicted_price": bounds["predicted_price"],
            "lower_bound": bounds["lower_bound"],
            "upper_bound": bounds["upper_bound"],
            "confidence_level": bounds["confidence_level"],
        })

        prices.append(ensemble_pred)
        rainfall.append(rainfall[-1])
        current_date = current_date + pd.Timedelta(days=1)

    return forecasts


DISTRICT_METADATA = {
    "Almora":          {"zone": "Kumaon_Hills",     "dist_type": "Semi_Rural", "elevation_m": 1638, "num_markets": 2, "lat": 29.60, "lon": 79.66},
    "Bageshwar":       {"zone": "Kumaon_Hills",     "dist_type": "Rural",      "elevation_m": 960,  "num_markets": 1, "lat": 29.84, "lon": 79.77},
    "Chamoli":         {"zone": "Garhwal_Hills",    "dist_type": "Rural",      "elevation_m": 2412, "num_markets": 1, "lat": 30.41, "lon": 79.32},
    "Champawat":       {"zone": "Kumaon_Hills",     "dist_type": "Rural",      "elevation_m": 1615, "num_markets": 1, "lat": 29.34, "lon": 80.10},
    "Dehradun":        {"zone": "Plains_Foothills", "dist_type": "Urban",      "elevation_m": 435,  "num_markets": 4, "lat": 30.32, "lon": 78.03},
    "Haridwar":        {"zone": "Plains",           "dist_type": "Urban",     "elevation_m": 314,  "num_markets": 5, "lat": 29.95, "lon": 78.16},
    "Nainital":        {"zone": "Kumaon_Hills",     "dist_type": "Semi_Urban", "elevation_m": 2084, "num_markets": 2, "lat": 29.38, "lon": 79.46},
    "Pauri Garhwal":   {"zone": "Garhwal_Hills",    "dist_type": "Rural",      "elevation_m": 1814, "num_markets": 1, "lat": 30.15, "lon": 78.78},
    "Pithoragarh":     {"zone": "Kumaon_Hills",     "dist_type": "Semi_Rural", "elevation_m": 1814, "num_markets": 2, "lat": 29.58, "lon": 80.22},
    "Rudraprayag":     {"zone": "Garhwal_Hills",    "dist_type": "Rural",      "elevation_m": 895,  "num_markets": 1, "lat": 30.28, "lon": 78.98},
    "Tehri Garhwal":   {"zone": "Garhwal_Hills",    "dist_type": "Rural",      "elevation_m": 1750, "num_markets": 2, "lat": 30.39, "lon": 78.48},
    "UdhamSinghNagar": {"zone": "Terai_Plains",     "dist_type": "Semi_Urban", "elevation_m": 244,  "num_markets": 7, "lat": 28.97, "lon": 79.40},
    "Uttarkashi":      {"zone": "Garhwal_Hills",    "dist_type": "Rural",      "elevation_m": 1158, "num_markets": 1, "lat": 30.73, "lon": 78.45},
}


if __name__ == "__main__":
    raise SystemExit(
        "price_model.py is a backend module imported by app.py and the training "
        "notebook. It has no dashboard of its own and is not meant to be run "
        "directly or launched with streamlit. Run the dashboard with:\n\n"
        "    streamlit run app.py\n"
    )

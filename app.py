import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import json
import os
import datetime

from price_model import (build_prediction_row, season_for_month, DISTRICT_METADATA,
                          forecast_recursive, load_residual_bounds, apply_confidence_bounds)

st.set_page_config(
    page_title="Uttarakhand Tomato Price Prediction",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1a3c5e; }
    [data-testid="stSidebar"] * { color: white !important; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    h1 { color: #1a3c5e; font-size: 1.4rem !important; }
    h2 { color: #1a3c5e; font-size: 1.15rem !important; font-weight: 700; }
    h3 { color: #1a3c5e; font-size: 1rem !important; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

NAVY, ORANGE, GREEN, STEEL, MUTED = "#1a3c5e", "#e05c2a", "#3aaa6d", "#4a6fa5", "#95a5b3"
MONTHS_LABEL = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
DIST_ZONES = {name: meta["zone"].replace("_", " ") for name, meta in DISTRICT_METADATA.items()}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_dataset():
    frame = pd.read_csv(os.path.join(BASE_DIR, "Uttarakhand_Tomato_FINAL_Master.csv"))
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


@st.cache_data
def load_predictions():
    frame = pd.read_csv(os.path.join(BASE_DIR, "test_predictions.csv"))
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


@st.cache_data
def load_model_comparison():
    return pd.read_csv(os.path.join(BASE_DIR, "model_comparison.csv"))


@st.cache_data
def load_district_performance():
    return pd.read_csv(os.path.join(BASE_DIR, "district_performance.csv"))


@st.cache_data
def load_district_lookup():
    with open(os.path.join(BASE_DIR, "district_lookup.json")) as handle:
        return json.load(handle)


@st.cache_data
def load_residual_bounds_cached():
    return load_residual_bounds(os.path.join(BASE_DIR, "forecast_residual_bounds.csv"))


@st.cache_resource
def load_trained_models():
    model_dir = os.path.join(BASE_DIR, "models")
    forest_model = joblib.load(os.path.join(model_dir, "random_forest.pkl"))
    xgboost_model = joblib.load(os.path.join(model_dir, "xgboost_model.pkl"))
    lightgbm_model = joblib.load(os.path.join(model_dir, "lightgbm_model.pkl"))
    label_encoders = joblib.load(os.path.join(model_dir, "label_encoders.pkl"))
    predictor_columns = joblib.load(os.path.join(model_dir, "predictor_columns.pkl"))
    return forest_model, xgboost_model, lightgbm_model, label_encoders, predictor_columns


df = load_dataset()
preds_df = load_predictions()
mc_df = load_model_comparison()
dp_df = load_district_performance()
district_lookup = load_district_lookup()
residual_bounds = load_residual_bounds_cached()
forest_model, xgboost_model, lightgbm_model, label_encoders, predictor_columns = load_trained_models()

DISTRICTS = sorted(df["District"].unique())
ENSEMBLE_ROW = mc_df[mc_df["model"] == "Ensemble"].iloc[0]


def predict_price(district, target_date, price_lag_1d, price_lag_7d, price_lag_30d, rainfall_today):
    lookup = district_lookup[district]
    price_lag_3d = (price_lag_1d + price_lag_7d) / 2
    price_lag_14d = (price_lag_7d + price_lag_30d) / 2
    rollavg_7d = (price_lag_1d + price_lag_7d) / 2
    rollavg_14d = (rollavg_7d + price_lag_30d) / 2
    rollavg_30d = price_lag_30d
    rollstd_7d = lookup["price_rollstd_7d"]
    rollstd_30d = lookup["price_rollstd_30d"]

    week_of_year = target_date.isocalendar()[1]
    quarter = (target_date.month - 1) // 3 + 1
    season_name = season_for_month(target_date.month)

    row = build_prediction_row(
        district=district, month=target_date.month, year=target_date.year,
        week_of_year=week_of_year, day_of_week=target_date.weekday(), quarter=quarter,
        price_lag_1d=price_lag_1d, price_lag_3d=price_lag_3d, price_lag_7d=price_lag_7d,
        price_lag_14d=price_lag_14d, price_lag_30d=price_lag_30d,
        price_rollavg_7d=rollavg_7d, price_rollavg_14d=rollavg_14d, price_rollavg_30d=rollavg_30d,
        price_rollstd_7d=rollstd_7d, price_rollstd_30d=rollstd_30d,
        rainfall_lag_1d=rainfall_today, rainfall_lag_7d=lookup["rainfall_lag_7d"],
        rainfall_rollavg_7d=lookup["rainfall_rollavg_7d"],
        district_meta=DISTRICT_METADATA, label_encoders=label_encoders, season_name=season_name
    )

    input_frame = pd.DataFrame([row])[predictor_columns]
    forest_pred = forest_model.predict(input_frame)[0]
    xgboost_pred = xgboost_model.predict(input_frame)[0]
    lightgbm_pred = lightgbm_model.predict(input_frame)[0]
    ensemble_pred = (forest_pred + xgboost_pred + lightgbm_pred) / 3
    bounds = apply_confidence_bounds(district, ensemble_pred, residual_bounds)

    return {
        "random_forest": round(forest_pred, 0),
        "xgboost": round(xgboost_pred, 0),
        "lightgbm": round(lightgbm_pred, 0),
        "ensemble": round(ensemble_pred, 0),
        "lower_bound": bounds["lower_bound"],
        "upper_bound": bounds["upper_bound"],
        "confidence_level": bounds["confidence_level"],
        "season": season_name,
    }


def get_recent_history(district, num_days=30):
    district_rows = df[df["District"] == district].sort_values("Date").tail(num_days)
    price_history = district_rows["Price_Modal_Avg"].tolist()
    rainfall_history = district_rows["Rainfall_mm"].tolist()
    last_date = district_rows["Date"].iloc[-1]
    return price_history, rainfall_history, last_date


def run_forecast(district, horizon_days):
    price_history, rainfall_history, last_known_date = get_recent_history(district)
    start_date = last_known_date + pd.Timedelta(days=1)
    forecasts = forecast_recursive(
        district=district, start_date=start_date, horizon_days=horizon_days,
        price_history=price_history, rainfall_history=rainfall_history,
        forest_model=forest_model, xgboost_model=xgboost_model, lightgbm_model=lightgbm_model,
        label_encoders=label_encoders, predictor_columns=predictor_columns,
        district_meta=DISTRICT_METADATA, residual_bounds=residual_bounds
    )
    return pd.DataFrame(forecasts)


def get_forecast_start_date(district):
    _, _, last_known_date = get_recent_history(district)
    return (last_known_date + pd.Timedelta(days=1)).date()


with st.sidebar:
    st.markdown("## 🍅 Tomato Price\nPrediction")
    st.markdown("**Uttarakhand · 2020–2025**")
    st.markdown("---")
    page = st.radio("Navigation", ["Overview", "Price Analysis", "Model Results", "Predict Price", "Forecast"],
                     label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<p style="font-size:11px;color:rgba(255,255,255,0.4)">MCA · Data Science<br>Uttarakhand Agri Intelligence</p>',
                unsafe_allow_html=True)


if page == "Overview":
    st.title("Uttarakhand Tomato Price Intelligence")
    st.markdown(
        f'<span style="background:rgba(58,170,109,0.15);color:#3aaa6d;font-size:12px;font-weight:600;padding:5px 14px;'
        f'border-radius:20px;border:1px solid rgba(58,170,109,0.3)">● Ensemble R² {ENSEMBLE_ROW["r2"]:.3f} &nbsp;·&nbsp; MAPE {ENSEMBLE_ROW["mape"]:.2f}% &nbsp;·&nbsp; Updated through {df["Date"].max().strftime("%b %Y")}</span>',
        unsafe_allow_html=True
    )
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    latest_year = df["Year"].max()
    prev_year_avg = df[df["Year"] == latest_year - 1]["Price_Modal_Avg"].mean()
    this_year_avg = df[df["Year"] == latest_year]["Price_Modal_Avg"].mean()
    yoy_change = ((this_year_avg - prev_year_avg) / prev_year_avg) * 100

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi_style = ("background:linear-gradient(145deg,#162840,#1a3c5e);border-radius:14px;padding:18px 20px;"
                 "border:1px solid rgba(255,255,255,0.06);height:118px")

    def kpi_card(label, value, sub, accent):
        return f"""
<div style="{kpi_style}">
  <div style="font-size:11px;color:rgba(255,255,255,0.5);text-transform:uppercase;letter-spacing:.5px">{label}</div>
  <div style="font-size:26px;font-weight:800;color:white;margin-top:6px">{value}</div>
  <div style="font-size:11.5px;color:{accent};margin-top:4px">{sub}</div>
</div>"""

    with kpi1:
        st.markdown(kpi_card("Ensemble Accuracy", f"{100 - ENSEMBLE_ROW['mape']:.1f}%", "RF · XGB · LGB average", "#3aaa6d"), unsafe_allow_html=True)
    with kpi2:
        st.markdown(kpi_card("Districts Tracked", "13", "Full state coverage", "#4a6fa5"), unsafe_allow_html=True)
    with kpi3:
        st.markdown(kpi_card("Dataset Size", f"{len(df):,}", "Daily rows · 2020–2025", "#4a6fa5"), unsafe_allow_html=True)
    with kpi4:
        arrow = "▲" if yoy_change >= 0 else "▼"
        change_color = "#e05c2a" if yoy_change >= 0 else "#3aaa6d"
        st.markdown(kpi_card(f"{latest_year} Avg Price", f"₹{this_year_avg:,.0f}", f"{arrow} {abs(yoy_change):.1f}% vs {latest_year-1}", change_color), unsafe_allow_html=True)
    with kpi5:
        peak_season_row = df.groupby("Season")["Price_Modal_Avg"].mean().idxmax()
        peak_val = df.groupby("Season")["Price_Modal_Avg"].mean().max()
        st.markdown(kpi_card("Peak Season", peak_season_row.replace("_", " "), f"₹{peak_val:,.0f}/qtl average", "#e05c2a"), unsafe_allow_html=True)

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    map_col, rank_col = st.columns([1.3, 1])

    with map_col:
        st.markdown("#### District Map — select a district to highlight it")
        map_district = st.selectbox("Highlight district", DISTRICTS, key="map_dist",
                                     index=DISTRICTS.index("Dehradun") if "Dehradun" in DISTRICTS else 0)

        district_avg_price = df.groupby("District")["Price_Modal_Avg"].mean()
        map_lats, map_lons, map_names, map_sizes, map_colors, map_text = [], [], [], [], [], []
        for d in DISTRICTS:
            meta = DISTRICT_METADATA[d]
            map_lats.append(meta["lat"])
            map_lons.append(meta["lon"])
            map_names.append(d)
            avg_price = district_avg_price.get(d, 0)
            map_sizes.append(34 if d == map_district else 20)
            map_colors.append(ORANGE if d == map_district else STEEL)
            map_text.append(f"{d}<br>Avg price: ₹{avg_price:,.0f}/qtl")

        map_fig = go.Figure(go.Scattergeo(
            lon=map_lons, lat=map_lats, text=map_text, hoverinfo="text",
            mode="markers+text", textposition="top center",
            textfont=dict(size=10, color="#e8edf3"),
            marker=dict(size=map_sizes, color=map_colors, line=dict(width=1.5, color="white"), opacity=0.92)
        ))
        map_fig.update_geos(
            lataxis_range=[28.5, 31.5], lonaxis_range=[77.5, 81.2],
            showland=True, landcolor="#1a2533", showcountries=True,
            countrycolor="rgba(255,255,255,0.2)", showsubunits=True,
            subunitcolor="rgba(255,255,255,0.15)", bgcolor="rgba(0,0,0,0)",
            showocean=False, showlakes=False
        )
        map_fig.update_layout(height=420, margin=dict(t=10, b=10, l=10, r=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(map_fig, use_container_width=True)
        st.caption("Marker positions are approximate district headquarters coordinates, shown for relative "
                   "geographic context rather than precise administrative boundaries.")

    with rank_col:
        st.markdown("#### District Ranking — by average price")
        rank_df = df.groupby("District")["Price_Modal_Avg"].mean().sort_values(ascending=False).reset_index()
        rank_df.columns = ["District", "Avg Price"]
        rank_df.index = rank_df.index + 1
        max_price = rank_df["Avg Price"].max()

        rows_html = ""
        for idx, row in rank_df.iterrows():
            bar_pct = (row["Avg Price"] / max_price) * 100
            highlight = "background:rgba(224,92,42,0.12)" if row["District"] == map_district else ""
            rows_html += f"""
<div style="display:flex;align-items:center;gap:10px;padding:7px 10px;border-radius:8px;{highlight}">
  <div style="width:20px;font-size:11px;color:rgba(255,255,255,0.4)">{idx}</div>
  <div style="width:120px;font-size:12.5px;color:#e8edf3">{row['District']}</div>
  <div style="flex:1;background:rgba(255,255,255,0.07);border-radius:4px;height:8px;overflow:hidden">
    <div style="width:{bar_pct}%;background:{'#e05c2a' if row['District']==map_district else '#4a6fa5'};height:100%"></div>
  </div>
  <div style="width:65px;text-align:right;font-size:12px;font-weight:600;color:#e8edf3">₹{row['Avg Price']:,.0f}</div>
</div>"""
        st.markdown(f'<div style="max-height:420px;overflow-y:auto">{rows_html}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Avg Price by Season")
        season_order = ["Winter", "Spring", "Summer", "Monsoon", "Post_Monsoon"]
        season_avg = df.groupby("Season")["Price_Modal_Avg"].mean().reindex(season_order).reset_index()
        colors = [STEEL, GREEN, "#f0a500", ORANGE, NAVY]
        fig = go.Figure(go.Bar(
            x=season_avg["Season"].str.replace("_", " "), y=season_avg["Price_Modal_Avg"].round(0),
            marker_color=colors, text=season_avg["Price_Modal_Avg"].round(0),
            texttemplate="₹%{text:,.0f}", textposition="outside"
        ))
        fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8edf3"),
                           yaxis=dict(title="₹/Quintal", gridcolor="rgba(255,255,255,0.12)"),
                           xaxis=dict(showgrid=False), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Year-wise Price Trend")
        dist_sel = st.selectbox("District", DISTRICTS, key="ov_dist",
                                 index=DISTRICTS.index(map_district))
        yearly = df[df["District"] == dist_sel].groupby("Year")["Price_Modal_Avg"].mean().reset_index()
        fig = go.Figure(go.Scatter(
            x=yearly["Year"], y=yearly["Price_Modal_Avg"].round(0),
            mode="lines+markers", line=dict(color=ORANGE, width=2.5),
            marker=dict(size=8, color=ORANGE), fill="tozeroy", fillcolor="rgba(224,92,42,0.12)"
        ))
        fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8edf3"),
                           yaxis=dict(title="₹/Quintal", gridcolor="rgba(255,255,255,0.12)"),
                           xaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### District Comparison — multi-select")
    compare_districts = st.multiselect("Compare districts", DISTRICTS,
                                        default=[map_district, "Dehradun"] if map_district != "Dehradun" else [map_district, "Haridwar"],
                                        key="compare_dist")
    if compare_districts:
        compare_fig = go.Figure()
        palette = [ORANGE, STEEL, GREEN, "#f0a500", NAVY, "#a35dd6"]
        for i, d in enumerate(compare_districts):
            monthly = df[df["District"] == d].groupby("Month")["Price_Modal_Avg"].mean().reindex(range(1, 13))
            compare_fig.add_trace(go.Scatter(
                x=MONTHS_LABEL, y=monthly.round(0), mode="lines+markers", name=d,
                line=dict(color=palette[i % len(palette)], width=2.3),
                marker=dict(size=6, color=palette[i % len(palette)])
            ))
        compare_fig.update_layout(height=360, margin=dict(t=10, b=10, l=10, r=10),
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8edf3"),
                                   yaxis=dict(title="₹/Quintal", gridcolor="rgba(255,255,255,0.12)"),
                                   xaxis=dict(showgrid=False),
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(compare_fig, use_container_width=True)
    else:
        st.caption("Select at least one district above to see its monthly price pattern.")


elif page == "Price Analysis":
    st.title("Price Analysis — All 13 Districts")
    st.markdown("---")

    st.markdown("### Actual vs Predicted — 2025 Test Set")
    dist_avp = st.selectbox("Select District", DISTRICTS, key="avp_dist")
    sub_avp = preds_df[preds_df["District"] == dist_avp].sort_values("Date")
    if len(sub_avp):
        sub_avp_m = sub_avp.copy()
        sub_avp_m["month_year"] = sub_avp_m["Date"].dt.to_period("M").astype(str)
        sub_avp_m = sub_avp_m.groupby("month_year").agg(
            actual=("actual", "mean"), predicted=("predicted", "mean")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub_avp_m["month_year"], y=sub_avp_m["actual"].round(0),
                                  mode="lines", name="Actual", line=dict(color=NAVY, width=2),
                                  fill="tozeroy", fillcolor="rgba(26,60,94,0.06)"))
        fig.add_trace(go.Scatter(x=sub_avp_m["month_year"], y=sub_avp_m["predicted"].round(0),
                                  mode="lines", name="Predicted", line=dict(color=ORANGE, width=2, dash="dash")))
        district_mae = (sub_avp["actual"] - sub_avp["predicted"]).abs().mean()
        fig.update_layout(height=320, margin=dict(t=20, b=10, l=10, r=10),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8edf3"),
                           title=f"{dist_avp} · MAE ₹{district_mae:.0f}/qtl",
                           yaxis=dict(title="₹/Quintal", gridcolor="rgba(255,255,255,0.12)"),
                           xaxis=dict(showgrid=False),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### District Performance Summary (2025 Test)")
    dp_show = dp_df.copy()
    dp_show["zone"] = dp_show["district"].map(DIST_ZONES)
    dp_show["grade"] = dp_show["mape"].apply(
        lambda x: "Good" if x < 9.0 else "Fair" if x < 9.5 else "Acceptable")
    dp_show["mape"] = dp_show["mape"].round(2).astype(str) + "%"
    dp_show["mae"] = "₹" + dp_show["mae"].round(0).astype(int).astype(str)
    dp_show = dp_show.rename(columns={"district": "District", "zone": "Zone",
                                       "mae": "MAE (₹/qtl)", "r2": "R²",
                                       "mape": "MAPE (%)", "grade": "Grade"})
    st.dataframe(dp_show[["District", "Zone", "MAPE (%)", "MAE (₹/qtl)", "R²", "Grade"]],
                 use_container_width=True, hide_index=True)


elif page == "Model Results":
    st.title("Model Results")
    st.markdown("---")

    best_row = mc_df.sort_values("mape").iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Best Model", best_row["model"], "Lowest test MAPE")
    with c2: st.metric("Mean Abs Error", f"₹{ENSEMBLE_ROW['mae']:.2f}", "Ensemble, per quintal")
    with c3: st.metric("Ensemble R²", f"{ENSEMBLE_ROW['r2']:.4f}", "2025 held-out year")
    with c4: st.metric("Train Rows", "18,603", "2020–2023")
    st.markdown("---")

    mc_sorted = mc_df.sort_values("mape").reset_index(drop=True)
    colors_bar = [ORANGE if m == "Ensemble" else NAVY for m in mc_sorted["model"]]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### MAPE Comparison (%)")
        fig = go.Figure(go.Bar(
            x=mc_sorted["mape"], y=mc_sorted["model"], orientation="h",
            marker_color=colors_bar, text=mc_sorted["mape"].apply(lambda x: f"{x:.2f}%"),
            textposition="outside"
        ))
        fig.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=60),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8edf3"),
                           xaxis=dict(title="%", gridcolor="rgba(255,255,255,0.12)"), yaxis=dict(showgrid=False),
                           showlegend=False)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### MAE Comparison (₹/qtl)")
        fig = go.Figure(go.Bar(
            x=mc_sorted["mae"], y=mc_sorted["model"], orientation="h",
            marker_color=colors_bar, text=mc_sorted["mae"].apply(lambda x: f"₹{x:.1f}"),
            textposition="outside"
        ))
        fig.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=60),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8edf3"),
                           xaxis=dict(title="₹", gridcolor="rgba(255,255,255,0.12)"), yaxis=dict(showgrid=False),
                           showlegend=False)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top Features (LightGBM)")
    feature_importance = pd.read_csv(os.path.join(BASE_DIR, "feature_importance.csv"))
    feature_importance.columns = ["feature", "importance"]
    feature_importance = feature_importance.sort_values("importance", ascending=True)
    fig = go.Figure(go.Bar(
        x=feature_importance["importance"], y=feature_importance["feature"], orientation="h",
        marker_color=NAVY
    ))
    fig.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10),
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8edf3"),
                       xaxis=dict(title="Importance Score", gridcolor="rgba(255,255,255,0.12)"),
                       yaxis=dict(showgrid=False, tickfont=dict(size=11)), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Full Model Comparison Table")
    table = mc_sorted.copy()
    table["mae"] = "₹" + table["mae"].astype(str)
    table["rmse"] = "₹" + table["rmse"].astype(str)
    table["mape"] = table["mape"].astype(str) + "%"
    table = table.rename(columns={"model": "Model", "mae": "MAE", "rmse": "RMSE", "r2": "R²", "mape": "MAPE"})
    st.dataframe(table[["Model", "MAE", "RMSE", "R²", "MAPE"]], use_container_width=True, hide_index=True)


elif page == "Predict Price":
    st.title("Price Prediction Engine")
    st.markdown("---")

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.markdown("### Input Parameters")
        st.info("Select a district to auto-fill the last known values from the dataset. "
                "Predictions come directly from the trained Random Forest, XGBoost and "
                "LightGBM models, averaged into a single ensemble prediction.")

        district = st.selectbox("District", DISTRICTS, key="p_dist")
        lookup = district_lookup[district]
        st.caption(f"Last known price for {district}: ₹{lookup['last_price']:.0f}/qtl "
                   f"on {lookup['last_date']}")

        target_date = st.date_input("Prediction date", value=datetime.date.today())

        pc1, pc2 = st.columns(2)
        with pc1:
            price_lag_1d = st.number_input("Yesterday's price (₹/qtl)", min_value=300, max_value=8000,
                                            value=int(lookup["price_lag_1d"]), step=10)
        with pc2:
            price_lag_7d = st.number_input("7-day average price (₹/qtl)", min_value=300, max_value=8000,
                                            value=int(lookup["price_lag_7d"]), step=10)

        pc3, pc4 = st.columns(2)
        with pc3:
            price_lag_30d = st.number_input("30-day average price (₹/qtl)", min_value=300, max_value=8000,
                                             value=int(lookup["price_lag_30d"]), step=10)
        with pc4:
            rainfall_today = st.number_input("Rainfall today (mm)", min_value=0.0, max_value=250.0,
                                              value=float(lookup["rainfall_lag_1d"]), step=0.5)

        if st.button("Predict Price", type="primary", use_container_width=True):
            result = predict_price(district, target_date, price_lag_1d, price_lag_7d,
                                    price_lag_30d, rainfall_today)

            st.markdown("---")
            st.markdown(f"""
<div style="background:linear-gradient(135deg,#1a3c5e 0%,#2a5298 100%);color:white;border-radius:12px;padding:22px 24px">
  <div style="font-size:11px;color:rgba(255,255,255,0.55);letter-spacing:.5px;text-transform:uppercase">Ensemble Prediction</div>
  <div style="font-size:42px;font-weight:800;letter-spacing:-1px;line-height:1.1">₹{result['ensemble']:,.0f}</div>
  <div style="font-size:13px;color:rgba(255,255,255,0.65);margin-top:4px">Per quintal (100 kg) · ₹{result['ensemble']/100:.2f} per kg · {result['season']} season</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:16px">
    <div style="background:rgba(255,255,255,0.1);border-radius:8px;padding:10px 12px">
      <div style="font-size:9.5px;color:rgba(255,255,255,0.5);text-transform:uppercase">Lower Bound</div>
      <div style="font-size:14px;font-weight:700;margin-top:3px">₹{result['lower_bound']:,.0f}</div>
    </div>
    <div style="background:rgba(255,255,255,0.1);border-radius:8px;padding:10px 12px">
      <div style="font-size:9.5px;color:rgba(255,255,255,0.5);text-transform:uppercase">Upper Bound</div>
      <div style="font-size:14px;font-weight:700;margin-top:3px">₹{result['upper_bound']:,.0f}</div>
    </div>
    <div style="background:rgba(255,255,255,0.1);border-radius:8px;padding:10px 12px">
      <div style="font-size:9.5px;color:rgba(255,255,255,0.5);text-transform:uppercase">Confidence</div>
      <div style="font-size:14px;font-weight:700;margin-top:3px">{result['confidence_level']}%</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:10px">
    <div style="background:rgba(255,255,255,0.07);border-radius:8px;padding:8px 12px">
      <div style="font-size:9px;color:rgba(255,255,255,0.45);text-transform:uppercase">Random Forest</div>
      <div style="font-size:12px;font-weight:600;margin-top:2px">₹{result['random_forest']:,.0f}</div>
    </div>
    <div style="background:rgba(255,255,255,0.07);border-radius:8px;padding:8px 12px">
      <div style="font-size:9px;color:rgba(255,255,255,0.45);text-transform:uppercase">XGBoost</div>
      <div style="font-size:12px;font-weight:600;margin-top:2px">₹{result['xgboost']:,.0f}</div>
    </div>
    <div style="background:rgba(255,255,255,0.07);border-radius:8px;padding:8px 12px">
      <div style="font-size:9px;color:rgba(255,255,255,0.45);text-transform:uppercase">LightGBM</div>
      <div style="font-size:12px;font-weight:600;margin-top:2px">₹{result['lightgbm']:,.0f}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
            st.caption(f"The lower and upper bound come from the 5th and 95th percentile of "
                       f"actual prediction errors observed for {district} on the 2025 test set, "
                       f"giving a {result['confidence_level']}% empirical confidence range.")

            district_mape = dp_df[dp_df["district"] == district]["mape"].values
            if len(district_mape):
                st.caption(f"Typical prediction error for {district} on the 2025 test set: "
                           f"around {district_mape[0]:.1f}% of the predicted price.")

    with col2:
        st.markdown("### Monthly Reference for Selected District")
        monthly_r = df[df["District"] == district].groupby("Month")["Price_Modal_Avg"].mean().reset_index()
        bar_colors = [ORANGE if 6 <= m <= 9 else NAVY for m in monthly_r["Month"]]
        fig = go.Figure(go.Bar(
            x=MONTHS_LABEL, y=monthly_r["Price_Modal_Avg"].round(0), marker_color=bar_colors
        ))
        fig.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8edf3"),
                           title=f"{district} · Jul–Sep highlighted (monsoon)",
                           yaxis=dict(title="₹/Quintal", gridcolor="rgba(255,255,255,0.12)"),
                           xaxis=dict(showgrid=False), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Last Known Values — All Districts")
        lookup_rows = []
        for d in DISTRICTS:
            entry = district_lookup[d]
            lookup_rows.append({
                "District": d,
                "Price (₹/qtl)": f"₹{entry['last_price']:.0f}",
                "7D Avg": f"₹{entry['price_lag_7d']:.0f}",
                "30D Avg": f"₹{entry['price_lag_30d']:.0f}",
            })
        st.dataframe(pd.DataFrame(lookup_rows), use_container_width=True, hide_index=True)


elif page == "Forecast":
    st.title("Multi-Day Forecast")
    st.markdown("---")
    st.info("Each day's forecast is produced by feeding the previous day's predicted price "
             "back in as the next day's input, starting from the most recent known price "
             "in the dataset. Forecast accuracy decreases the further out you look, since "
             "later days depend on earlier predictions rather than observed prices.")

    forecast_district = st.selectbox("District", DISTRICTS, key="forecast_dist")
    forecast_start = get_forecast_start_date(forecast_district)
    max_forecast_end = forecast_start + datetime.timedelta(days=90)

    st.markdown(f"Forecasting begins the day after the most recent known price for "
                f"**{forecast_district}**, which is **{forecast_start.strftime('%d %b %Y')}**.")

    quick_pick = st.radio("Quick select", ["Custom range", "7 days", "15 days", "30 days", "60 days", "90 days"],
                          horizontal=True, key="quick_horizon")
    quick_days_map = {"7 days": 7, "15 days": 15, "30 days": 30, "60 days": 60, "90 days": 90}

    if quick_pick == "Custom range":
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            range_start = st.date_input("Start date", value=forecast_start,
                                        min_value=forecast_start, max_value=max_forecast_end,
                                        key="forecast_start_date")
        with date_col2:
            range_end = st.date_input("End date", value=forecast_start + datetime.timedelta(days=6),
                                      min_value=forecast_start, max_value=max_forecast_end,
                                      key="forecast_end_date")
        if range_end < range_start:
            st.error("End date must be on or after the start date.")
            st.stop()
        lead_days = (range_start - forecast_start).days
        horizon_days = (range_end - forecast_start).days + 1
    else:
        lead_days = 0
        horizon_days = quick_days_map[quick_pick]

    horizon_days = min(horizon_days, 90)

    with st.spinner(f"Running recursive forecast for {horizon_days} day(s)..."):
        forecast_df = run_forecast(forecast_district, horizon_days)
    forecast_df["date_label"] = pd.to_datetime(forecast_df["date"]).dt.strftime("%d %b %Y")

    display_df = forecast_df.iloc[lead_days:].reset_index(drop=True) if lead_days else forecast_df

    if len(display_df) == 1:
        row = display_df.iloc[0]
        st.markdown(f"""
<div style="background:linear-gradient(135deg,#1a3c5e 0%,#2a5298 100%);color:white;border-radius:12px;padding:22px 24px;margin-top:10px">
  <div style="font-size:11px;color:rgba(255,255,255,0.55);letter-spacing:.5px;text-transform:uppercase">Forecast · {row['date_label']}</div>
  <div style="font-size:42px;font-weight:800;letter-spacing:-1px;line-height:1.1">₹{row['predicted_price']:,.0f}</div>
  <div style="font-size:13px;color:rgba(255,255,255,0.65);margin-top:4px">{forecast_district} · Per quintal (100 kg)</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:16px">
    <div style="background:rgba(255,255,255,0.1);border-radius:8px;padding:10px 12px">
      <div style="font-size:9.5px;color:rgba(255,255,255,0.5);text-transform:uppercase">Lower Bound</div>
      <div style="font-size:14px;font-weight:700;margin-top:3px">₹{row['lower_bound']:,.0f}</div>
    </div>
    <div style="background:rgba(255,255,255,0.1);border-radius:8px;padding:10px 12px">
      <div style="font-size:9.5px;color:rgba(255,255,255,0.5);text-transform:uppercase">Upper Bound</div>
      <div style="font-size:14px;font-weight:700;margin-top:3px">₹{row['upper_bound']:,.0f}</div>
    </div>
    <div style="background:rgba(255,255,255,0.1);border-radius:8px;padding:10px 12px">
      <div style="font-size:9.5px;color:rgba(255,255,255,0.5);text-transform:uppercase">Confidence</div>
      <div style="font-size:14px;font-weight:700;margin-top:3px">{row['confidence_level']}%</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    elif len(display_df) > 1:
        col1, col2 = st.columns([1.3, 1])
        with col1:
            st.markdown(f"### {display_df.iloc[0]['date_label']} – {display_df.iloc[-1]['date_label']} · {forecast_district}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=display_df["date_label"], y=display_df["upper_bound"],
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=display_df["date_label"], y=display_df["lower_bound"],
                mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor="rgba(224,92,42,0.18)", name="Confidence Range"
            ))
            fig.add_trace(go.Scatter(
                x=display_df["date_label"], y=display_df["predicted_price"],
                mode="lines+markers", line=dict(color=ORANGE, width=2.5),
                marker=dict(size=5, color=ORANGE), name="Predicted Price"
            ))
            fig.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8edf3"),
                               yaxis=dict(title="₹/Quintal", gridcolor="rgba(255,255,255,0.12)"),
                               xaxis=dict(showgrid=False, tickangle=-45),
                               legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Forecast Summary")
            st.metric("First Day Prediction", f"₹{display_df.iloc[0]['predicted_price']:,.0f}")
            st.metric("Last Day Prediction", f"₹{display_df.iloc[-1]['predicted_price']:,.0f}")
            price_change = display_df.iloc[-1]["predicted_price"] - display_df.iloc[0]["predicted_price"]
            st.metric("Change over horizon", f"₹{price_change:+,.0f}")
            st.caption("Confidence bounds widen toward the end of the horizon because the "
                       "model's own predictions are used as inputs for later days rather than "
                       "observed prices, compounding uncertainty day over day.")
    else:
        st.warning("No forecast days fall inside the selected range. Adjust the dates above.")

    st.markdown("### Forecast Table")
    table_view = display_df[["date_label", "day_ahead", "predicted_price", "lower_bound",
                              "upper_bound", "confidence_level"]].copy()
    table_view.columns = ["Date", "Day Ahead", "Predicted Price (₹/qtl)", "Lower Bound (₹/qtl)",
                          "Upper Bound (₹/qtl)", "Confidence (%)"]
    table_view["Predicted Price (₹/qtl)"] = table_view["Predicted Price (₹/qtl)"].apply(lambda x: f"₹{x:,.0f}")
    table_view["Lower Bound (₹/qtl)"] = table_view["Lower Bound (₹/qtl)"].apply(lambda x: f"₹{x:,.0f}")
    table_view["Upper Bound (₹/qtl)"] = table_view["Upper Bound (₹/qtl)"].apply(lambda x: f"₹{x:,.0f}")
    st.dataframe(table_view, use_container_width=True, hide_index=True)

    csv_bytes = display_df[["date_label", "day_ahead", "predicted_price", "lower_bound",
                            "upper_bound", "confidence_level"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download forecast as CSV", csv_bytes,
                       file_name=f"{forecast_district}_forecast.csv", mime="text/csv")

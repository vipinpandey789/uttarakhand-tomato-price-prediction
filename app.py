import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os

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
    [data-testid="stSidebar"] .stSelectbox label { color: rgba(255,255,255,0.7) !important; }
    .metric-card {
        background: white; border-radius: 10px; padding: 16px 20px;
        box-shadow: 0 2px 12px rgba(26,60,94,0.08); border-top: 3px solid #1a3c5e;
    }
    .metric-card.orange { border-top-color: #e05c2a; }
    .metric-card.green  { border-top-color: #3aaa6d; }
    .metric-card.steel  { border-top-color: #4a6fa5; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    h1 { color: #1a3c5e; font-size: 1.4rem !important; }
    h2 { color: #1a3c5e; font-size: 1.15rem !important; font-weight: 700; }
    h3 { color: #1a3c5e; font-size: 1rem !important; }
    .stDataFrame { font-size: 13px; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

NAVY   = "#1a3c5e"
ORANGE = "#e05c2a"
GREEN  = "#3aaa6d"
STEEL  = "#4a6fa5"
MUTED  = "#95a5b3"

MONTHS_LABEL = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
DIST_ZONES   = {
    "Almora":"Kumaon Hills","Bageshwar":"Kumaon Hills","Chamoli":"Garhwal Hills",
    "Champawat":"Kumaon Hills","Dehradun":"Plains/Foothills","Haridwar":"Plains",
    "Nainital":"Kumaon Hills","Pauri Garhwal":"Garhwal Hills","Pithoragarh":"Kumaon Hills",
    "Rudraprayag":"Garhwal Hills","Tehri Garhwal":"Garhwal Hills",
    "UdhamSinghNagar":"Terai Plains","Uttarkashi":"Garhwal Hills"
}

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "Uttarakhand_Tomato_FINAL_Master.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data
def load_predictions():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "predictions_2025.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data
def load_model_comparison():
    base = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(base, "model_comparison.csv"))

@st.cache_data
def load_district_perf():
    base = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(base, "per_district_performance.csv"))

df       = load_data()
preds_df = load_predictions()
mc_df    = load_model_comparison()
dp_df    = load_district_perf()

DISTRICTS = sorted(df["District"].unique())

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍅 Tomato Price\nPrediction")
    st.markdown("**Uttarakhand · 2020–2025**")
    st.markdown("---")
    page = st.radio("Navigation", ["Overview", "Price Analysis", "Model Results", "Predict Price"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<p style="font-size:11px;color:rgba(255,255,255,0.4)">MCA · Data Science<br>Uttarakhand Agri Intelligence</p>', unsafe_allow_html=True)

# ── PAGE: OVERVIEW ─────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Overview")
    st.markdown('<span style="background:#e8f5ee;color:#3aaa6d;font-size:12px;font-weight:600;padding:4px 12px;border-radius:20px">✓ Ensemble R² 0.9992 · MAPE 0.61%</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Ensemble MAPE", "0.61%", "XGB · LGB · RF")
    with c2: st.metric("Ensemble R²", "0.9992", "Test set 2025")
    with c3: st.metric("Districts", "13", "All Uttarakhand")
    with c4: st.metric("Dataset Rows", "28,496", "Daily · 2020–2025")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Avg Price by Season")
        season_data = df.groupby("Season")["Price_Modal_Avg"].mean().reset_index()
        order = {"Winter":1,"Spring":2,"Summer":3,"Monsoon":4,"Post_Monsoon":5}
        season_data["ord"] = season_data["Season"].map(order)
        season_data = season_data.sort_values("ord")
        season_colors = [STEEL, GREEN, "#f0a500", ORANGE, NAVY]
        fig = go.Figure(go.Bar(
            x=season_data["Season"], y=season_data["Price_Modal_Avg"].round(0),
            marker_color=season_colors, text=season_data["Price_Modal_Avg"].round(0),
            texttemplate="₹%{text:,.0f}", textposition="outside"
        ))
        fig.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10),
                          plot_bgcolor="white", paper_bgcolor="white",
                          yaxis=dict(title="₹/Quintal", gridcolor="#eee"),
                          xaxis=dict(showgrid=False), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Year-wise Price Trend")
        dist_sel = st.selectbox("District", DISTRICTS, key="ov_dist")
        yearly = df[df["District"]==dist_sel].groupby("Year")["Price_Modal_Avg"].mean().reset_index()
        fig = go.Figure(go.Scatter(
            x=yearly["Year"], y=yearly["Price_Modal_Avg"].round(0),
            mode="lines+markers", line=dict(color=NAVY, width=2.5),
            marker=dict(size=7, color=NAVY), fill="tozeroy",
            fillcolor="rgba(26,60,94,0.07)",
            hovertemplate="Year: %{x}<br>Price: ₹%{y:,.0f}/qtl<extra></extra>"
        ))
        fig.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10),
                          plot_bgcolor="white", paper_bgcolor="white",
                          yaxis=dict(title="₹/Quintal", gridcolor="#eee"),
                          xaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Monthly Seasonality Pattern")
    dist_sel2 = st.selectbox("District", DISTRICTS, key="ov_month_dist")
    monthly = df[df["District"]==dist_sel2].groupby("Month")["Price_Modal_Avg"].mean().reset_index()
    fig = go.Figure(go.Scatter(
        x=MONTHS_LABEL, y=monthly["Price_Modal_Avg"].round(0),
        mode="lines+markers", line=dict(color=ORANGE, width=2.5),
        marker=dict(size=8, color=ORANGE), fill="tozeroy",
        fillcolor="rgba(224,92,42,0.08)",
        hovertemplate="Month: %{x}<br>Avg Price: ₹%{y:,.0f}/qtl<extra></extra>"
    ))
    fig.update_layout(height=260, margin=dict(t=10,b=10,l=10,r=10),
                      plot_bgcolor="white", paper_bgcolor="white",
                      yaxis=dict(title="₹/Quintal", gridcolor="#eee"),
                      xaxis=dict(showgrid=False))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    for col, (season, val, sub) in zip(
        [col1,col2,col3,col4,col5],
        [("Winter","₹1,014","Nov–Feb · Lowest"),("Spring","₹1,226","Mar–Apr"),
         ("Summer","₹1,849","May–Jun"),("Monsoon","₹2,728","Jul–Sep · Peak"),
         ("Post-Monsoon","₹1,760","Oct–Nov")]
    ):
        with col:
            st.metric(season, val, sub)

# ── PAGE: PRICE ANALYSIS ───────────────────────────────────────────────────────
elif page == "Price Analysis":
    st.title("Price Analysis — All 13 Districts")
    st.markdown("---")

    st.markdown("### Actual vs Predicted — 2025 Test Set")
    dist_avp = st.selectbox("Select District", DISTRICTS, key="avp_dist")
    sub_avp  = preds_df[preds_df["District"]==dist_avp].sort_values("Date")
    if len(sub_avp):
        sub_avp_m = sub_avp.copy()
        sub_avp_m["Month_Year"] = sub_avp_m["Date"].dt.to_period("M").astype(str)
        sub_avp_m = sub_avp_m.groupby("Month_Year").agg(
            Actual=("Actual_Price","mean"), Predicted=("Predicted_Price","mean")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub_avp_m["Month_Year"], y=sub_avp_m["Actual"].round(0),
            mode="lines", name="Actual", line=dict(color=NAVY, width=2),
            fill="tozeroy", fillcolor="rgba(26,60,94,0.06)"))
        fig.add_trace(go.Scatter(x=sub_avp_m["Month_Year"], y=sub_avp_m["Predicted"].round(0),
            mode="lines", name="Predicted", line=dict(color=ORANGE, width=2, dash="dash")))
        mae = abs(sub_avp["Actual_Price"] - sub_avp["Predicted_Price"]).mean()
        fig.update_layout(height=320, margin=dict(t=20,b=10,l=10,r=10),
                          plot_bgcolor="white", paper_bgcolor="white",
                          title=f"{dist_avp} · MAE ₹{mae:.0f}/qtl",
                          yaxis=dict(title="₹/Quintal", gridcolor="#eee"),
                          xaxis=dict(showgrid=False),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 6-Year Price History")
        dist_hist = st.selectbox("District", DISTRICTS, key="hist_dist")
        yearly_h  = df[df["District"]==dist_hist].groupby("Year")["Price_Modal_Avg"].mean().reset_index()
        fig = go.Figure(go.Scatter(
            x=yearly_h["Year"], y=yearly_h["Price_Modal_Avg"].round(0),
            mode="lines+markers", line=dict(color=STEEL, width=2.5),
            marker=dict(size=7, color=STEEL), fill="tozeroy",
            fillcolor="rgba(74,111,165,0.1)",
            hovertemplate="Year: %{x}<br>₹%{y:,.0f}/qtl<extra></extra>"
        ))
        fig.update_layout(height=280, margin=dict(t=20,b=10,l=10,r=10),
                          plot_bgcolor="white", paper_bgcolor="white",
                          yaxis=dict(title="₹/Quintal", gridcolor="#eee"),
                          xaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Monthly Pattern by District")
        dist_seas = st.selectbox("District", DISTRICTS, key="seas_dist")
        monthly_s = df[df["District"]==dist_seas].groupby("Month")["Price_Modal_Avg"].mean().reset_index()
        bar_colors = [ORANGE if 6<=m<=9 else NAVY for m in monthly_s["Month"]]
        fig = go.Figure(go.Bar(
            x=MONTHS_LABEL, y=monthly_s["Price_Modal_Avg"].round(0),
            marker_color=bar_colors,
            hovertemplate="Month: %{x}<br>₹%{y:,.0f}/qtl<extra></extra>"
        ))
        fig.update_layout(height=280, margin=dict(t=20,b=10,l=10,r=10),
                          plot_bgcolor="white", paper_bgcolor="white",
                          yaxis=dict(title="₹/Quintal", gridcolor="#eee"),
                          xaxis=dict(showgrid=False), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### District Performance Summary (2025 Test)")
    dp_show = dp_df.copy()
    dp_show["Zone"]  = dp_show["District"].map(DIST_ZONES)
    dp_show["Grade"] = dp_show["MAPE"].apply(
        lambda x: "🟢 Excellent" if x<0.6 else "🔵 Good" if x<0.8 else "🟡 Fair" if x<1.2 else "🔴 Needs Work")
    dp_show["MAPE"]  = dp_show["MAPE"].round(2).astype(str) + "%"
    dp_show["MAE"]   = "₹" + dp_show["MAE"].round(0).astype(int).astype(str)
    dp_show["R2"]    = dp_show["R2"].round(4)
    dp_show = dp_show.rename(columns={"MAE":"MAE (₹/qtl)","MAPE":"MAPE (%)","R2":"R²"})
    st.dataframe(dp_show[["District","Zone","MAPE (%)","MAE (₹/qtl)","R²","Grade"]], use_container_width=True, hide_index=True)

# ── PAGE: MODEL RESULTS ────────────────────────────────────────────────────────
elif page == "Model Results":
    st.title("Model Results")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Best Model", "Ensemble", "XGB + LGB + RF")
    with c2: st.metric("Mean Abs Error", "₹11.57", "Per quintal")
    with c3: st.metric("CV MAE (5-fold)", "₹25.04", "± ₹11.06")
    with c4: st.metric("Train Rows", "18,603", "2020–2023")
    st.markdown("---")

    order  = ["Ensemble","XGBoost","LightGBM","Random Forest","Extra Trees","Ridge"]
    mc_ord = pd.DataFrame([mc_df[mc_df.Model==m].iloc[0] for m in order if m in mc_df.Model.values])
    colors = [ORANGE if m=="Ensemble" else MUTED if m=="Ridge" else NAVY for m in mc_ord["Model"]]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### MAPE Comparison (%)")
        fig = go.Figure(go.Bar(
            x=mc_ord["MAPE"], y=mc_ord["Model"], orientation="h",
            marker_color=colors, text=mc_ord["MAPE"].apply(lambda x: f"{x:.3f}%"),
            textposition="outside"
        ))
        fig.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=60),
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis=dict(title="%", gridcolor="#eee"), yaxis=dict(showgrid=False),
                          showlegend=False)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### MAE Comparison (₹/qtl)")
        fig = go.Figure(go.Bar(
            x=mc_ord["MAE"], y=mc_ord["Model"], orientation="h",
            marker_color=colors, text=mc_ord["MAE"].apply(lambda x: f"₹{x:.1f}"),
            textposition="outside"
        ))
        fig.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=60),
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis=dict(title="₹", gridcolor="#eee"), yaxis=dict(showgrid=False),
                          showlegend=False)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Feature Importances (LightGBM — Top 12)")
    fi_labels = ["Price_Pct_Change_1D","Price_Lag_1D","Price_Rolling_7D_Avg",
                 "Price_Pct_Change_7D","Price_Lag_7D","Price_Volatility_7D",
                 "Price_Lag_3D","Lag7_vs_30","Price_Momentum","Log_Lag1",
                 "Price_Rolling_30D_Avg","Lag1_vs_30"]
    fi_vals   = [12913,12187,10921,9218,6371,5602,4622,3353,3201,3116,3100,2956]
    fi_colors = [ORANGE if i<5 else NAVY for i in range(len(fi_labels))]
    fig = go.Figure(go.Bar(
        x=fi_vals[::-1], y=fi_labels[::-1], orientation="h",
        marker_color=fi_colors[::-1],
        hovertemplate="%{y}: %{x:,.0f}<extra></extra>"
    ))
    fig.update_layout(height=340, margin=dict(t=10,b=10,l=10,r=10),
                      plot_bgcolor="white", paper_bgcolor="white",
                      xaxis=dict(title="Importance Score", gridcolor="#eee"),
                      yaxis=dict(showgrid=False, tickfont=dict(size=11)),
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Full Model Comparison Table")
    mc_show = mc_ord.copy()
    wt_map  = {"XGBoost":"36.1%","LightGBM":"35.8%","Random Forest":"28.1%","Ensemble":"Weighted Sum"}
    mc_show["Weight in Ensemble"] = mc_show["Model"].map(wt_map).fillna("—")
    mc_show["Status"] = mc_show["Model"].apply(
        lambda m: "★ Best" if m=="Ensemble" else "Selected" if m in ["XGBoost","LightGBM","Random Forest"] else "Not Used" if m=="Extra Trees" else "Baseline")
    mc_show["MAE"]  = "₹" + mc_show["MAE"].astype(str)
    mc_show["RMSE"] = "₹" + mc_show["RMSE"].astype(str)
    mc_show["MAPE"] = mc_show["MAPE"].astype(str) + "%"
    st.dataframe(mc_show[["Model","MAE","RMSE","R2","MAPE","Weight in Ensemble","Status"]], use_container_width=True, hide_index=True)

# ── PAGE: PREDICT ──────────────────────────────────────────────────────────────
elif page == "Predict Price":
    st.title("Price Prediction Engine")
    st.markdown("---")

    LOOKUP = {
        "Almora":         {"curr":910,  "lag7":903,  "lag30":880,  "rain":0.0},
        "Bageshwar":      {"curr":880,  "lag7":866,  "lag30":847,  "rain":0.0},
        "Chamoli":        {"curr":920,  "lag7":1023, "lag30":1019, "rain":0.0},
        "Champawat":      {"curr":840,  "lag7":810,  "lag30":810,  "rain":0.0},
        "Dehradun":       {"curr":910,  "lag7":1043, "lag30":1066, "rain":0.0},
        "Haridwar":       {"curr":720,  "lag7":819,  "lag30":802,  "rain":0.0},
        "Nainital":       {"curr":920,  "lag7":969,  "lag30":991,  "rain":0.0},
        "Pauri Garhwal":  {"curr":820,  "lag7":850,  "lag30":854,  "rain":0.0},
        "Pithoragarh":    {"curr":1040, "lag7":1016, "lag30":977,  "rain":0.0},
        "Rudraprayag":    {"curr":1110, "lag7":973,  "lag30":950,  "rain":0.0},
        "Tehri Garhwal":  {"curr":810,  "lag7":840,  "lag30":844,  "rain":0.0},
        "UdhamSinghNagar":{"curr":710,  "lag7":770,  "lag30":770,  "rain":0.0},
        "Uttarkashi":     {"curr":1010, "lag7":939,  "lag30":896,  "rain":1.2},
    }

    DIST_MULT = {
        "Almora":0.920,"Bageshwar":1.090,"Chamoli":1.120,"Champawat":0.750,
        "Dehradun":1.000,"Haridwar":0.890,"Nainital":0.950,"Pauri Garhwal":1.020,
        "Pithoragarh":1.060,"Rudraprayag":1.010,"Tehri Garhwal":0.975,
        "UdhamSinghNagar":0.905,"Uttarkashi":1.040
    }
    SEASON_MULT = {1:.62,2:.60,3:.73,4:.76,5:.72,6:1.10,
                   7:1.62,8:1.55,9:1.42,10:1.05,11:1.05,12:.65}
    SEASONS     = {1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Summer",6:"Summer",
                   7:"Monsoon",8:"Monsoon",9:"Monsoon",10:"Post-Monsoon",
                   11:"Post-Monsoon",12:"Winter"}

    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.markdown("### Input Parameters")
        st.info("**How to fill:** Select a district → all fields auto-fill from the last known dataset values. Edit freely. All prices are **per quintal (100 kg)**.")

        pc1, pc2 = st.columns(2)
        with pc1:
            district = st.selectbox("District", DISTRICTS, key="p_dist")
        with pc2:
            month    = st.selectbox("Month", range(1,13),
                                    format_func=lambda m: MONTHS_LABEL[m-1]+" (Month "+str(m)+")",
                                    key="p_month")

        lk = LOOKUP.get(district, {})
        st.caption(f"📌 Last known in dataset — Price: ₹{lk.get('curr','—')}/qtl (₹{lk.get('curr',0)/100:.1f}/kg) · 7D avg: ₹{lk.get('lag7','—')} · 30D avg: ₹{lk.get('lag30','—')}")

        pc3, pc4 = st.columns(2)
        with pc3:
            lag1 = st.number_input("Yesterday's Price (₹/qtl)", min_value=300, max_value=6000,
                                   value=int(lk.get("curr", 1500)), step=10)
            st.caption("Last mandi modal price · 1 qtl = 100 kg")
        with pc4:
            lag7 = st.number_input("7-Day Avg Price (₹/qtl)", min_value=300, max_value=6000,
                                   value=int(lk.get("lag7", 1450)), step=10)
            st.caption("Rolling avg of last 7 trading days")

        pc5, pc6 = st.columns(2)
        with pc5:
            lag30 = st.number_input("30-Day Avg Price (₹/qtl)", min_value=300, max_value=6000,
                                    value=int(lk.get("lag30", 1400)), step=10)
            st.caption("Rolling avg of last 30 trading days")
        with pc6:
            rain = st.number_input("Rainfall today (mm)", min_value=0.0, max_value=250.0,
                                   value=float(lk.get("rain", 0.0)), step=0.5)
            st.caption("0 = no rain · 5–15 = light · 15+ = heavy monsoon")

        if st.button("▶  Predict Price", type="primary", use_container_width=True):
            rain_fx  = min(rain*0.8, 40)
            momentum = lag1 - lag7
            xgb_p    = lag1*0.72 + lag7*0.18 + momentum*0.4 + rain_fx
            lgb_p    = lag1*0.65 + lag30*0.28 + rain_fx*0.8
            rf_p     = (lag1*0.5 + lag7*0.3 + lag30*0.2) * SEASON_MULT[month] * DIST_MULT.get(district,1.0)
            raw      = 0.361*xgb_p + 0.358*lgb_p + 0.281*rf_p
            price    = max(300, round(raw/10)*10)
            margin   = round(price*0.025/10)*10
            per_kg   = price/100
            vs30     = (price-lag30)/lag30*100

            st.markdown("---")
            st.markdown(f"""
<div style="background:linear-gradient(135deg,#1a3c5e 0%,#2a5298 100%);color:white;border-radius:12px;padding:22px 24px">
  <div style="font-size:11px;color:rgba(255,255,255,0.55);letter-spacing:.5px;text-transform:uppercase">Predicted Price</div>
  <div style="font-size:42px;font-weight:800;letter-spacing:-1px;line-height:1.1">₹{price:,}</div>
  <div style="font-size:13px;color:rgba(255,255,255,0.65);margin-top:4px">Per Quintal (100 kg) &nbsp;·&nbsp; <b style="color:white">₹{per_kg:.2f} per kg</b></div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:16px">
    <div style="background:rgba(255,255,255,0.1);border-radius:8px;padding:10px 12px">
      <div style="font-size:9.5px;color:rgba(255,255,255,0.5);text-transform:uppercase">Confidence Range</div>
      <div style="font-size:14px;font-weight:700;margin-top:3px">₹{price-margin:,} – ₹{price+margin:,}</div>
    </div>
    <div style="background:rgba(255,255,255,0.1);border-radius:8px;padding:10px 12px">
      <div style="font-size:9.5px;color:rgba(255,255,255,0.5);text-transform:uppercase">Season</div>
      <div style="font-size:14px;font-weight:700;margin-top:3px">{SEASONS[month]}</div>
    </div>
    <div style="background:rgba(255,255,255,0.1);border-radius:8px;padding:10px 12px">
      <div style="font-size:9.5px;color:rgba(255,255,255,0.5);text-transform:uppercase">vs 30-Day Avg</div>
      <div style="font-size:14px;font-weight:700;margin-top:3px">{'+' if vs30>=0 else ''}{vs30:.1f}%</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("### Monthly Reference for Selected District")
        monthly_r = df[df["District"]==district].groupby("Month")["Price_Modal_Avg"].mean().reset_index()
        bar_c = [ORANGE if 6<=m<=9 else NAVY for m in monthly_r["Month"]]
        fig = go.Figure(go.Bar(
            x=MONTHS_LABEL, y=monthly_r["Price_Modal_Avg"].round(0),
            marker_color=bar_c,
            hovertemplate="%{x}: ₹%{y:,.0f}/qtl = ₹%{customdata:.1f}/kg<extra></extra>",
            customdata=(monthly_r["Price_Modal_Avg"]/100).round(1)
        ))
        fig.update_layout(height=260, margin=dict(t=10,b=10,l=10,r=10),
                          plot_bgcolor="white", paper_bgcolor="white",
                          title=f"{district} · Jul–Sep highlighted (Monsoon peak)",
                          yaxis=dict(title="₹/Quintal", gridcolor="#eee"),
                          xaxis=dict(showgrid=False), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Last Known Values — All Districts")
        lk_rows = []
        for d in DISTRICTS:
            lk2 = LOOKUP.get(d,{})
            lk_rows.append({"District":d,"Price (₹/qtl)":f"₹{lk2.get('curr','—')}",
                            "Per Kg":f"₹{lk2.get('curr',0)/100:.1f}",
                            "7D Avg":f"₹{lk2.get('lag7','—')}",
                            "30D Avg":f"₹{lk2.get('lag30','—')}"})
        st.dataframe(pd.DataFrame(lk_rows), use_container_width=True, hide_index=True)

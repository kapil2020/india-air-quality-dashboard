import os
import json
from io import StringIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# ============================================================
# 0) PAGE CONFIG (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="IIT KGP Air Quality Intelligence",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 1) DESIGN SYSTEM (theme tokens)
#    Toggle: Dark / Light (award style)
# ============================================================
THEMES = {
    "Dark (Pro)": {
        "bg": "#0B0F14",
        "surface": "#0F1620",
        "surface2": "#111B26",
        "border": "rgba(255,255,255,0.08)",
        "text": "rgba(255,255,255,0.92)",
        "muted": "rgba(255,255,255,0.66)",
        "muted2": "rgba(255,255,255,0.50)",
        "accent": "#00D4FF",
        "accent2": "#00FFA3",
        "danger": "#FF4D6D",
        "warn": "#FFB020",
        "ok": "#2DD4BF",
        "shadow": "0 12px 40px rgba(0,0,0,0.40)",
        "map_style": "carto-darkmatter",
        "plotly_template": "plotly_dark",
    },
    "Light (Editorial)": {
        "bg": "#F7F9FC",
        "surface": "#FFFFFF",
        "surface2": "#FFFFFF",
        "border": "rgba(10,20,30,0.10)",
        "text": "rgba(10,20,30,0.92)",
        "muted": "rgba(10,20,30,0.64)",
        "muted2": "rgba(10,20,30,0.45)",
        "accent": "#005BFF",
        "accent2": "#12B981",
        "danger": "#E11D48",
        "warn": "#F59E0B",
        "ok": "#10B981",
        "shadow": "0 14px 45px rgba(10,20,30,0.12)",
        "map_style": "carto-positron",
        "plotly_template": "plotly_white",
    },
}

CATEGORY_COLORS = {
    "Severe": "#E11D48",
    "Very Poor": "#F97316",
    "Poor": "#FB923C",
    "Moderate": "#FBBF24",
    "Satisfactory": "#34D399",
    "Good": "#22C55E",
    "Unknown": "#94A3B8",
}

POLLUTANT_COLORS = {
    "PM2.5": "#FF4D6D",
    "PM10": "#00D4FF",
    "NO2": "#60A5FA",
    "SO2": "#F59E0B",
    "CO": "#FB7185",
    "O3": "#34D399",
    "Other": "#94A3B8",
}

HEALTH_RECOMMENDATIONS = {
    "Good": "Perfect day for outdoor activities.",
    "Satisfactory": "Sensitive groups: reduce prolonged heavy exertion.",
    "Moderate": "Sensitive groups: reduce outdoor activity, prefer indoor.",
    "Poor": "Everyone: reduce prolonged heavy exertion outdoors.",
    "Very Poor": "Avoid outdoor activities, especially sensitive groups.",
    "Severe": "Avoid all outdoor activity. Keep windows closed, use purifier.",
    "Unknown": "Data unavailable. Use precautionary protection.",
}


def get_category(aqi_val):
    if pd.isna(aqi_val):
        return "Unknown"
    if aqi_val <= 50:
        return "Good"
    if aqi_val <= 100:
        return "Satisfactory"
    if aqi_val <= 200:
        return "Moderate"
    if aqi_val <= 300:
        return "Poor"
    if aqi_val <= 400:
        return "Very Poor"
    return "Severe"


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# ============================================================
# 2) SIDEBAR CONTROLS (theme, refresh, filters)
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    theme_name = st.selectbox(
        "Theme",
        list(THEMES.keys()),
        index=0,
        help="Switch between a premium dark theme and an editorial light theme.",
    )
    T = THEMES[theme_name]

    # Global Plotly template
    pio.templates.default = T["plotly_template"]

    st.markdown("---")

    st.caption("Data source: CPCB files (local).")
    force_refresh = st.button("🔄 Refresh data", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎛️ Analysis filters")

# ============================================================
# 3) CSS (typography + responsive + card system)
# ============================================================
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
  --bg: {T["bg"]};
  --surface: {T["surface"]};
  --surface2: {T["surface2"]};
  --border: {T["border"]};
  --text: {T["text"]};
  --muted: {T["muted"]};
  --muted2: {T["muted2"]};
  --accent: {T["accent"]};
  --accent2: {T["accent2"]};
  --danger: {T["danger"]};
  --warn: {T["warn"]};
  --ok: {T["ok"]};
  --shadow: {T["shadow"]};
}}

html, body, [data-testid="stAppViewContainer"] {{
  background: var(--bg) !important;
}}

* {{
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}}

.main .block-container {{
  padding-top: 1.25rem;
  padding-bottom: 2.5rem;
  max-width: 1400px;
}}

a {{
  color: var(--accent);
  text-decoration: none;
}}
a:hover {{
  opacity: 0.9;
}}

h1, h2, h3 {{
  letter-spacing: -0.02em;
}}

.hero {{
  background: radial-gradient(1200px 500px at 20% 0%, rgba(0,212,255,0.22), transparent 55%),
              radial-gradient(900px 450px at 80% 10%, rgba(0,255,163,0.18), transparent 55%),
              linear-gradient(180deg, var(--surface), rgba(255,255,255,0.00));
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 22px 22px 18px 22px;
  box-shadow: var(--shadow);
  margin-bottom: 18px;
}}

.hero-title {{
  font-size: 2.15rem;
  font-weight: 900;
  line-height: 1.1;
  color: var(--text);
  margin: 0;
}}

.hero-sub {{
  margin-top: 8px;
  color: var(--muted);
  font-size: 1.02rem;
  line-height: 1.6;
}}

.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 14px;
}}

.kpi {{
  border: 1px solid var(--border);
  background: var(--surface2);
  border-radius: 16px;
  padding: 14px 14px;
  box-shadow: 0 10px 28px rgba(0,0,0,0.18);
}}

.kpi .label {{
  color: var(--muted2);
  font-size: 0.86rem;
  font-weight: 600;
}}

.kpi .value {{
  margin-top: 6px;
  color: var(--text);
  font-weight: 900;
  font-size: 1.7rem;
  letter-spacing: -0.02em;
}}

.kpi .hint {{
  margin-top: 6px;
  color: var(--muted);
  font-size: 0.92rem;
}}

.card {{
  border: 1px solid var(--border);
  background: var(--surface);
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: var(--shadow);
}}

.section-title {{
  font-size: 1.25rem;
  font-weight: 850;
  color: var(--text);
  margin: 0 0 10px 0;
}}

.section-sub {{
  margin-top: -4px;
  margin-bottom: 12px;
  color: var(--muted);
  font-size: 0.95rem;
}}

.badge {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.02);
  color: var(--muted);
  font-size: 0.86rem;
  font-weight: 650;
}}

hr {{
  border: none;
  border-top: 1px solid var(--border);
  margin: 18px 0;
}}

@media (max-width: 1100px) {{
  .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
@media (max-width: 560px) {{
  .kpi-grid {{ grid-template-columns: 1fr; }}
  .hero-title {{ font-size: 1.65rem; }}
}}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 4) DATA LOADING (same logic as your current app, improved checks)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_data_and_metadata(_refresh_key: int = 0):
    """
    Loads:
      1) data/YYYY-MM-DD.csv (today) if exists and has 'date'
      2) else combined_air_quality.txt (tab-separated) with parse_dates=['date']

    Expected minimal columns:
      date, city, index
    Optional:
      level, pollutant
    """
    today = pd.to_datetime("today").date()
    csv_path = f"data/{today}.csv"
    fallback_file = "combined_air_quality.txt"

    df_loaded = None
    msg_parts = []
    last_update_time = None

    if os.path.exists(csv_path):
        try:
            tmp = pd.read_csv(csv_path)
            if "date" in tmp.columns:
                tmp["date"] = pd.to_datetime(tmp["date"])
                df_loaded = tmp
                msg_parts.append(f"Live: {today}.csv")
                last_update_time = pd.Timestamp(os.path.getmtime(csv_path), unit="s")
            else:
                msg_parts.append(f"Found {today}.csv but missing 'date'. Fallback used.")
        except Exception as e:
            msg_parts.append(f"Error reading {today}.csv: {e}. Fallback used.")

    if df_loaded is None:
        if not os.path.exists(fallback_file):
            return pd.DataFrame(), "ERROR: combined_air_quality.txt not found.", None
        try:
            df_loaded = pd.read_csv(fallback_file, sep="\t", parse_dates=["date"])
            msg_parts.append("Archive: combined_air_quality.txt")
            last_update_time = pd.Timestamp(os.path.getmtime(fallback_file), unit="s")
        except Exception as e:
            return pd.DataFrame(), f"ERROR reading fallback: {e}", None

    # Normalize columns
    for col, default_val in [("pollutant", np.nan), ("level", "Unknown")]:
        if col not in df_loaded.columns:
            df_loaded[col] = default_val

    # Clean pollutant
    df_loaded["pollutant"] = (
        df_loaded["pollutant"]
        .astype(str)
        .str.split(",").str[0].str.strip()
        .replace(["nan", "NaN", "None", ""], np.nan)
        .fillna("Other")
    )

    # Ensure level exists and matches computed category if missing/invalid
    df_loaded["level"] = df_loaded["level"].astype(str).fillna("Unknown")
    # If level is nonsense, recompute from AQI
    valid_levels = set(CATEGORY_COLORS.keys())
    bad = ~df_loaded["level"].isin(valid_levels)
    if bad.any():
        df_loaded.loc[bad, "level"] = df_loaded.loc[bad, "index"].apply(get_category)

    # Basic required columns
    required = ["date", "city", "index"]
    for c in required:
        if c not in df_loaded.columns:
            return pd.DataFrame(), f"ERROR: Missing required column '{c}'", last_update_time

    # Optional filter from your original logic: limit 2025 months > 5
    if 2025 in df_loaded["date"].dt.year.unique():
        df_loaded = df_loaded[~((df_loaded["date"].dt.year == 2025) & (df_loaded["date"].dt.month > 5))]

    # Enforce types
    df_loaded["date"] = pd.to_datetime(df_loaded["date"])
    df_loaded["index"] = pd.to_numeric(df_loaded["index"], errors="coerce")

    msg = " • ".join(msg_parts) if msg_parts else "Data loaded"
    return df_loaded, msg, last_update_time


_refresh_key = int(datetime.now().timestamp()) if force_refresh else 0
df, load_message, data_last_updated = load_data_and_metadata(_refresh_key=_refresh_key)

if df.empty:
    st.error("Dashboard cannot operate without data. Please check your files.")
    st.stop()

# ============================================================
# 5) SIDEBAR FILTERS (now that data is loaded)
# ============================================================
with st.sidebar:
    unique_cities = sorted(df["city"].dropna().unique().tolist())
    default_city = "Delhi" if "Delhi" in unique_cities else (unique_cities[0] if unique_cities else None)

    selected_cities = st.multiselect(
        "Cities",
        options=unique_cities,
        default=[default_city] if default_city else [],
        help="Choose one or more cities for deep dive and comparisons.",
    )

    years = sorted(df["date"].dt.year.dropna().unique().tolist())
    default_year = 2024 if 2024 in years else (max(years) if years else None)
    year = st.selectbox("Year", options=years, index=years.index(default_year) if default_year in years else 0)

    months_map = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
    }

    if year == 2025:
        month_options = ["All Months"] + [months_map[i] for i in range(1, 6)]
    else:
        month_options = ["All Months"] + list(months_map.values())

    selected_month_name = st.selectbox("Month", options=month_options, index=0)

    month_number = None
    if selected_month_name != "All Months":
        month_number = [k for k, v in months_map.items() if v == selected_month_name][0]

    st.markdown("---")
    st.markdown("### 🧠 Smart options")
    show_story = st.toggle("Story mode (guided insights)", value=True)
    compact_mode = st.toggle("Compact mode", value=False)

# ============================================================
# 6) FILTER DATA
# ============================================================
df_period = df[df["date"].dt.year == year].copy()
if month_number is not None:
    df_period = df_period[df_period["date"].dt.month == month_number].copy()

# If user picked cities, keep a city-filtered copy too
df_city = df_period[df_period["city"].isin(selected_cities)].copy() if selected_cities else pd.DataFrame()

# ============================================================
# 7) PLOTLY LAYOUT HELPERS
# ============================================================
def layout_base(title=None, height=None):
    return dict(
        title=dict(
            text=title or "",
            x=0.02,
            xanchor="left",
            font=dict(size=18, family="Inter", color=T["text"]),
        ),
        font=dict(family="Inter", size=13, color=T["text"]),
        paper_bgcolor=T["surface"],
        plot_bgcolor=T["surface"],
        margin=dict(l=40, r=20, t=60 if title else 30, b=40),
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
        hoverlabel=dict(bgcolor=T["surface2"], font_size=12, font_family="Inter"),
    )

PLOT_CONFIG = {
    "displayModeBar": True,
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
}


# ============================================================
# 8) HERO HEADER
# ============================================================
period_label = f"{selected_month_name} {year}" if selected_month_name != "All Months" else f"Full Year {year}"
last_update_txt = data_last_updated.strftime("%Y-%m-%d %H:%M:%S") if data_last_updated else "Unknown"

st.markdown(
    f"""
<div class="hero">
  <div class="badge">🌬️ IIT KGP • Air Quality Intelligence</div>
  <h1 class="hero-title">India AQI Dashboard (award-style)</h1>
  <div class="hero-sub">
    High-resolution monitoring, comparisons, anomaly signals, and decision-ready insights.
    <br/>Period: <b>{period_label}</b> • Data: <b>{load_message}</b> • Last update: <b>{last_update_txt}</b>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 9) NATIONAL KPIs (premium cards)
# ============================================================
cities_count = int(df_period["city"].nunique())
days_count = int(df_period["date"].nunique())
avg_aqi = float(df_period["index"].mean()) if df_period["index"].notna().any() else np.nan
p90_aqi = float(df_period["index"].quantile(0.90)) if df_period["index"].notna().any() else np.nan
cat_avg = get_category(avg_aqi) if np.isfinite(avg_aqi) else "Unknown"
cat_color = CATEGORY_COLORS.get(cat_avg, CATEGORY_COLORS["Unknown"])
dominant_poll = (
    df_period["pollutant"].mode().iloc[0] if (not df_period.empty and df_period["pollutant"].notna().any()) else "Other"
)
poll_color = POLLUTANT_COLORS.get(dominant_poll, POLLUTANT_COLORS["Other"])

kpi_html = f"""
<div class="kpi-grid">
  <div class="kpi">
    <div class="label">Coverage</div>
    <div class="value">{cities_count}</div>
    <div class="hint">Cities monitored</div>
  </div>
  <div class="kpi">
    <div class="label">Observations</div>
    <div class="value">{days_count}</div>
    <div class="hint">Days in the selected period</div>
  </div>
  <div class="kpi">
    <div class="label">National mean AQI</div>
    <div class="value" style="color:{cat_color};">{avg_aqi:.1f}</div>
    <div class="hint">{cat_avg}</div>
  </div>
  <div class="kpi">
    <div class="label">Upper-tail AQI (P90)</div>
    <div class="value" style="color:{T["warn"]};">{p90_aqi:.1f}</div>
    <div class="hint">Extreme-days signal</div>
  </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

if show_story:
    st.markdown(
        f"""
<div class="card" style="margin-top:12px;">
  <div class="section-title">What this says (story mode)</div>
  <div class="section-sub">
    National mean is <b style="color:{cat_color};">{avg_aqi:.1f} ({cat_avg})</b>.
    P90 is <b style="color:{T["warn"]};">{p90_aqi:.1f}</b>, which captures high-pollution tail risk.
    Dominant pollutant across records is <b style="color:{poll_color};">{dominant_poll}</b>.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# ============================================================
# 10) CITY RANKINGS (cleanest vs most polluted)
# ============================================================
st.markdown("## 🏆 City rankings")

if df_period.empty or df_period["index"].dropna().empty:
    st.info("No AQI values available for this selection.")
else:
    city_avg = df_period.groupby("city")["index"].mean().dropna().sort_values()
    max_cities = len(city_avg)

    c1, c2 = st.columns([1.2, 2.8], gap="large")
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Ranking controls</div>', unsafe_allow_html=True)
        n_show = st.slider(
            "Cities to show",
            min_value=3,
            max_value=min(20, max_cities) if max_cities >= 3 else 3,
            value=min(8, max_cities) if max_cities >= 8 else max(3, max_cities),
        )
        st.markdown(
            f"<div class='section-sub'>Showing <b>{n_show}</b> cleanest and <b>{n_show}</b> most polluted cities.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    top_clean = city_avg.head(n_show).reset_index()
    top_dirty = city_avg.tail(n_show).reset_index()

    top_clean.columns = ["City", "AvgAQI"]
    top_dirty.columns = ["City", "AvgAQI"]
    top_clean["Category"] = top_clean["AvgAQI"].apply(get_category)
    top_dirty["Category"] = top_dirty["AvgAQI"].apply(get_category)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Cleanest vs most polluted</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Side-by-side view, consistent color mapping by AQI category.</div>', unsafe_allow_html=True)

        colA, colB = st.columns(2, gap="medium")

        with colA:
            fig_clean = px.bar(
                top_clean.sort_values("AvgAQI", ascending=False),
                x="AvgAQI",
                y="City",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
                orientation="h",
                text="AvgAQI",
            )
            fig_clean.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
            fig_clean.update_layout(**layout_base("🥇 Cleanest cities", height=310 if compact_mode else 380))
            fig_clean.update_layout(showlegend=False)
            fig_clean.update_xaxes(title_text="Average AQI", gridcolor=T["border"])
            fig_clean.update_yaxes(title_text=None)
            st.plotly_chart(fig_clean, use_container_width=True, config=PLOT_CONFIG)

        with colB:
            fig_dirty = px.bar(
                top_dirty.sort_values("AvgAQI", ascending=True),
                x="AvgAQI",
                y="City",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
                orientation="h",
                text="AvgAQI",
            )
            fig_dirty.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
            fig_dirty.update_layout(**layout_base("⚠️ Most polluted cities", height=310 if compact_mode else 380))
            fig_dirty.update_layout(showlegend=False)
            fig_dirty.update_xaxes(title_text="Average AQI", gridcolor=T["border"])
            fig_dirty.update_yaxes(title_text=None)
            st.plotly_chart(fig_dirty, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ============================================================
# 11) DEEP DIVE (per selected city)
# ============================================================
st.markdown("## 🏙️ City deep dive")

if not selected_cities:
    st.info("Select at least one city from the sidebar.")
else:
    # Multi-tabs for product-like experience
    tab_overview, tab_trends, tab_patterns, tab_pollution, tab_forecast, tab_quality = st.tabs(
        ["Overview", "Trends", "Patterns", "Pollutants", "Forecast", "Data quality"]
    )

    # ---------- OVERVIEW ----------
    with tab_overview:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Current snapshot</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Latest day available in the selected period (per city), plus quick health guidance.</div>',
            unsafe_allow_html=True,
        )

        cols = st.columns(min(4, max(1, len(selected_cities))), gap="medium")
        for i, city in enumerate(selected_cities[: len(cols)]):
            city_df = df_period[df_period["city"] == city].copy()
            if city_df.empty or city_df["index"].dropna().empty:
                cols[i].warning(f"{city}: no data")
                continue

            latest = city_df.sort_values("date", ascending=False).iloc[0]
            aqi_val = float(latest["index"]) if np.isfinite(latest["index"]) else np.nan
            level = str(latest.get("level", get_category(aqi_val)))
            pollutant = str(latest.get("pollutant", "Other"))
            rec = HEALTH_RECOMMENDATIONS.get(level, HEALTH_RECOMMENDATIONS["Unknown"])
            color = CATEGORY_COLORS.get(level, CATEGORY_COLORS["Unknown"])

            cols[i].markdown(
                f"""
<div class="kpi" style="height:100%;">
  <div class="label">{city}</div>
  <div class="value" style="color:{color};">{aqi_val:.0f}</div>
  <div class="hint"><b>{level}</b> • Dominant: <span style="color:{POLLUTANT_COLORS.get(pollutant, POLLUTANT_COLORS["Other"])};"><b>{pollutant}</b></span></div>
  <div class="hint" style="margin-top:8px;">{rec}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        if len(selected_cities) > 1:
            st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Cross-city comparison (time series)</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Spline line for readability, unified hover for fast comparison.</div>', unsafe_allow_html=True)

            comp = df_period[df_period["city"].isin(selected_cities)].sort_values("date").copy()
            fig = px.line(
                comp,
                x="date",
                y="index",
                color="city",
                markers=True,
                line_shape="spline",
                labels={"index": "AQI", "date": "Date", "city": "City"},
            )
            fig.update_layout(**layout_base(title="AQI trends comparison", height=380 if compact_mode else 500))
            fig.update_layout(hovermode="x unified")
            fig.update_xaxes(title_text=None, gridcolor=T["border"])
            fig.update_yaxes(title_text="AQI", gridcolor=T["border"])
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------- TRENDS ----------
    with tab_trends:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Trends with anomaly signals</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Daily series, 7-day rolling, and high-pollution anomalies (2σ above 30-day rolling mean).</div>',
            unsafe_allow_html=True,
        )

        city_sel = st.selectbox("City for trend view", selected_cities, index=0)
        d = df_period[df_period["city"] == city_sel].sort_values("date").copy()
        d = d[d["index"].notna()].copy()

        if len(d) < 10:
            st.info("Not enough data points to render trend with anomalies.")
        else:
            d["roll7"] = d["index"].rolling(7, min_periods=1).mean()
            d["roll30"] = d["index"].rolling(30, min_periods=1).mean()
            d["std30"] = d["index"].rolling(30, min_periods=2).std()
            d["thr"] = d["roll30"] + 2.0 * d["std30"].fillna(0)
            d["is_anom"] = d["index"] > d["thr"]

            anom = d[d["is_anom"]].copy()

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=d["date"],
                    y=d["index"],
                    mode="lines+markers",
                    name="Daily AQI",
                    line=dict(width=1.6, color=T["muted2"]),
                    marker=dict(size=4, opacity=0.8, color=T["muted2"]),
                    customdata=np.stack([d["level"].astype(str).values], axis=1),
                    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>AQI: %{y:.0f}<br>Category: %{customdata[0]}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=d["date"],
                    y=d["roll7"],
                    mode="lines",
                    name="7-day rolling",
                    line=dict(width=2.8, color=T["accent"]),
                    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>7-day mean: %{y:.1f}<extra></extra>",
                )
            )

            if not anom.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anom["date"],
                        y=anom["index"],
                        mode="markers",
                        name="Anomaly",
                        marker=dict(size=11, color=T["danger"], line=dict(width=1, color=T["bg"])),
                        hovertemplate="<b>High event</b><br>%{x|%Y-%m-%d}<br>AQI: %{y:.0f}<extra></extra>",
                    )
                )

            fig.update_layout(**layout_base(title=f"{city_sel}: AQI trend", height=380 if compact_mode else 520))
            fig.update_layout(hovermode="x unified")
            fig.update_xaxes(title_text=None, gridcolor=T["border"])
            fig.update_yaxes(title_text="AQI", gridcolor=T["border"])

            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

            if show_story:
                if not anom.empty:
                    st.warning(f"Detected {len(anom)} high-pollution event day(s). Open the table below to inspect.")
                    with st.expander("View anomaly days"):
                        show_cols = ["date", "index", "level", "pollutant"]
                        st.dataframe(anom[show_cols].rename(columns={"index": "AQI"}), use_container_width=True)
                else:
                    st.success("No high-pollution anomalies detected under this rule (2σ).")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- PATTERNS ----------
    with tab_patterns:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Patterns</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Weekday structure and month-day heatmap for fast pattern discovery.</div>',
            unsafe_allow_html=True,
        )

        city_sel = st.selectbox("City for pattern view", selected_cities, index=0, key="pattern_city")
        d = df_period[df_period["city"] == city_sel].copy()
        d = d[d["index"].notna()].copy()

        cA, cB = st.columns([1.2, 1.8], gap="large")

        with cA:
            if d.empty:
                st.info("No data.")
            else:
                d["weekday"] = d["date"].dt.day_name()
                weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                d["weekday"] = pd.Categorical(d["weekday"], categories=weekday_order, ordered=True)

                fig = px.box(
                    d.sort_values("weekday"),
                    x="weekday",
                    y="index",
                    points="outliers",
                    labels={"weekday": "Weekday", "index": "AQI"},
                )
                fig.update_layout(**layout_base(title="Weekday distribution", height=340 if compact_mode else 420))
                fig.update_xaxes(title_text=None, gridcolor=T["border"])
                fig.update_yaxes(title_text="AQI", gridcolor=T["border"])
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        with cB:
            if d.empty:
                st.info("No data.")
            else:
                d["month"] = d["date"].dt.month_name()
                d["day"] = d["date"].dt.day
                # Order months as calendar order
                months_order = [months_map[i] for i in range(1, 13)]
                d["month"] = pd.Categorical(d["month"], categories=months_order, ordered=True)

                pv = d.pivot_table(index="month", columns="day", values="index", aggfunc="mean", observed=False)
                pv = pv.dropna(how="all")

                if pv.empty:
                    st.info("No heatmap values.")
                else:
                    fig = px.imshow(
                        pv,
                        aspect="auto",
                        labels=dict(x="Day of month", y="Month", color="AQI"),
                        color_continuous_scale="Inferno" if theme_name.startswith("Dark") else "Magma",
                    )
                    fig.update_layout(**layout_base(title="Month × Day heatmap", height=340 if compact_mode else 420))
                    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- POLLUTANTS ----------
    with tab_pollution:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dominant pollutant intelligence</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Composition for the selected period and long-run changes across years.</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2, gap="large")

        with col1:
            city_sel = st.selectbox("City (period composition)", selected_cities, index=0, key="poll_city_period")
            d = df_period[df_period["city"] == city_sel].copy()
            if d.empty:
                st.info("No data.")
            else:
                g = d.groupby("pollutant").size().reset_index(name="days")
                fig = px.pie(
                    g,
                    values="days",
                    names="pollutant",
                    hole=0.5,
                    color="pollutant",
                    color_discrete_map=POLLUTANT_COLORS,
                )
                fig.update_layout(**layout_base(title="Period pollutant mix", height=360 if compact_mode else 460))
                fig.update_layout(legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"))
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        with col2:
            city_sel = st.selectbox("City (multi-year)", selected_cities, index=0, key="poll_city_years")
            d = df[df["city"] == city_sel].copy()
            if d.empty:
                st.info("No data.")
            else:
                d["year"] = d["date"].dt.year
                gp = d.groupby(["year", "pollutant"]).size().unstack(fill_value=0)
                pct = gp.apply(lambda x: (x / x.sum() * 100) if x.sum() > 0 else x, axis=1).fillna(0)
                long = pct.reset_index().melt(id_vars="year", var_name="pollutant", value_name="pct")

                fig = px.bar(
                    long,
                    x="year",
                    y="pct",
                    color="pollutant",
                    barmode="stack",
                    color_discrete_map=POLLUTANT_COLORS,
                    labels={"pct": "Days (%)", "year": "Year"},
                )
                fig.update_layout(**layout_base(title="Pollutants over years (share of days)", height=360 if compact_mode else 460))
                fig.update_xaxes(type="category", title_text=None, gridcolor=T["border"])
                fig.update_yaxes(title_text="Days (%)", ticksuffix="%", gridcolor=T["border"])
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- FORECAST ----------
    with tab_forecast:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Short-horizon AQI forecast (polynomial baseline)</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">A transparent baseline forecast (degree-2). Good for a dashboard preview, not a scientific predictor.</div>',
            unsafe_allow_html=True,
        )

        city_sel = st.selectbox("City for forecast", selected_cities, index=0, key="forecast_city")
        horizon = st.slider("Forecast horizon (days)", 7, 30, 15)

        d = df_period[df_period["city"] == city_sel].sort_values("date")[["date", "index"]].dropna()
        if len(d) < 15:
            st.warning(f"Need at least 15 valid points for forecast. Found {len(d)}.")
        else:
            d["t"] = (d["date"] - d["date"].min()).dt.days.astype(int)

            X = d["t"].values.reshape(-1, 1)
            y = d["index"].values

            poly = PolynomialFeatures(degree=2)
            Xp = poly.fit_transform(X)

            model = LinearRegression().fit(Xp, y)

            last_t = int(d["t"].max())
            future_t = np.arange(0, last_t + horizon + 1)
            future_dates = [d["date"].min() + pd.Timedelta(days=int(i)) for i in future_t]

            future_pred = model.predict(poly.transform(future_t.reshape(-1, 1)))
            future_pred = np.maximum(0, future_pred)

            obs = pd.DataFrame({"date": d["date"], "AQI": y})
            fc = pd.DataFrame({"date": future_dates, "AQI": future_pred})
            fc["is_future"] = fc["date"] > obs["date"].max()

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=obs["date"],
                    y=obs["AQI"],
                    mode="lines+markers",
                    name="Observed",
                    line=dict(width=1.6, color=T["muted2"]),
                    marker=dict(size=4, opacity=0.85, color=T["muted2"]),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=fc["date"],
                    y=fc["AQI"],
                    mode="lines",
                    name="Forecast",
                    line=dict(width=3.0, dash="dash", color=T["accent"]),
                )
            )

            forecast_start = obs["date"].max() + pd.Timedelta(days=1)
            forecast_end = fc["date"].max()
            fig.add_vrect(
                x0=forecast_start,
                x1=forecast_end,
                fillcolor="rgba(255, 77, 109, 0.10)" if theme_name.startswith("Dark") else "rgba(225, 29, 72, 0.08)",
                layer="below",
                line_width=0,
                annotation_text="Forecast window",
                annotation_position="top left",
                annotation_font_color=T["danger"],
            )

            fig.update_layout(**layout_base(title=f"{city_sel}: forecast", height=380 if compact_mode else 520))
            fig.update_layout(hovermode="x unified")
            fig.update_xaxes(title_text=None, gridcolor=T["border"])
            fig.update_yaxes(title_text="AQI", gridcolor=T["border"])
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

            if show_story:
                future_only = fc[fc["is_future"]].copy()
                if not future_only.empty:
                    worst = float(future_only["AQI"].max())
                    worst_cat = get_category(worst)
                    st.info(
                        f"Forecast max over next {horizon} days: {worst:.0f} ({worst_cat}). "
                        f"Guidance: {HEALTH_RECOMMENDATIONS.get(worst_cat, HEALTH_RECOMMENDATIONS['Unknown'])}"
                    )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- DATA QUALITY ----------
    with tab_quality:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Data quality and coverage</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Missingness, duplicates, and coverage gaps for the selected slice.</div>',
            unsafe_allow_html=True,
        )

        if df_period.empty:
            st.info("No records in this selection.")
        else:
            total_rows = len(df_period)
            missing_aqi = int(df_period["index"].isna().sum())
            missing_city = int(df_period["city"].isna().sum())
            dup_rows = int(df_period.duplicated(subset=["city", "date"]).sum())

            c1, c2, c3, c4 = st.columns(4, gap="medium")
            c1.metric("Rows", f"{total_rows:,}")
            c2.metric("Missing AQI", f"{missing_aqi:,}")
            c3.metric("Missing city", f"{missing_city:,}")
            c4.metric("Duplicate (city,date)", f"{dup_rows:,}")

            # Coverage by city: number of days present
            cov = df_period.groupby("city")["date"].nunique().reset_index(name="days_present")
            cov = cov.sort_values("days_present", ascending=False)

            fig = px.bar(
                cov.head(25),
                x="days_present",
                y="city",
                orientation="h",
                labels={"days_present": "Days present", "city": "City"},
            )
            fig.update_layout(**layout_base(title="Top 25 cities by coverage (days)", height=360 if compact_mode else 460))
            fig.update_xaxes(gridcolor=T["border"])
            fig.update_yaxes(title_text=None)
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

            if show_story:
                st.caption("Tip: If you see many missing values, check your upstream CPCB scraping/export step.")

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 12) HOTSPOTS MAP (uses lat_long.txt like your original app)
# ============================================================
st.markdown("## 📍 AQI hotspots map")

map_left, map_right = st.columns([3, 1], gap="large")

with map_left:
    coords_file_path = "lat_long.txt"
    if df_period.empty:
        st.warning("No data for map in this selection.")
    else:
        grouped = df_period.groupby("city").agg(
            avg_aqi=("index", "mean"),
            dominant_pollutant=("pollutant", lambda x: x.mode().iloc[0] if not x.mode().empty else "Other"),
        ).reset_index().dropna(subset=["avg_aqi"])

        if not os.path.exists(coords_file_path):
            st.warning(f"Missing coordinates file: {coords_file_path}")
        else:
            city_coords_data = {}
            try:
                with open(coords_file_path, "r", encoding="utf-8") as f:
                    exec(f.read(), {}, city_coords_data)
                city_coords = city_coords_data.get("city_coords", {})
            except Exception as e:
                st.error(f"Error reading lat_long.txt: {e}")
                city_coords = {}

            if not city_coords:
                st.warning("No city_coords found in lat_long.txt")
            else:
                latlong = pd.DataFrame([{"city": k, "lat": v[0], "lon": v[1]} for k, v in city_coords.items()])
                m = pd.merge(grouped, latlong, on="city", how="inner")

                if m.empty:
                    st.warning("Could not merge AQI data with coordinates.")
                else:
                    m["AQI Category"] = m["avg_aqi"].apply(get_category)
                    # Scaled bubble size
                    m["size"] = np.maximum(m["avg_aqi"] / 10.0, 5.0)

                    fig = px.scatter_mapbox(
                        m,
                        lat="lat",
                        lon="lon",
                        size="size",
                        size_max=26,
                        color="AQI Category",
                        color_discrete_map=CATEGORY_COLORS,
                        hover_name="city",
                        custom_data=["city", "avg_aqi", "dominant_pollutant", "AQI Category"],
                        zoom=4.2,
                        center={"lat": 23.5, "lon": 82.0},
                    )
                    fig.update_traces(
                        hovertemplate=(
                            "<b style='font-size:1.1em;'>%{customdata[0]}</b><br>"
                            "Avg AQI: %{customdata[1]:.1f} (%{customdata[3]})<br>"
                            "Dominant pollutant: %{customdata[2]}<extra></extra>"
                        )
                    )
                    fig.update_layout(**layout_base(title=f"Average AQI hotspots • {period_label}", height=520 if compact_mode else 700))
                    fig.update_layout(mapbox_style=T["map_style"], margin=dict(l=10, r=10, t=60, b=10))
                    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

with map_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How to read this map</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Bubble size = severity, color = AQI category, hover = details.</div>',
        unsafe_allow_html=True,
    )
    for cat, col in CATEGORY_COLORS.items():
        if cat == "Unknown":
            continue
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin:8px 0;'>"
            f"<div style='width:14px;height:14px;border-radius:4px;background:{col};'></div>"
            f"<div style='color:{T['text']};font-weight:700;'>{cat}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='badge'>Dominant pollutant: <b style='color:{poll_color};'>{dominant_poll}</b></div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ============================================================
# 13) DOWNLOAD (filtered export like your original app)
# ============================================================
st.markdown("## 📥 Export")

export_df = df_city.copy() if not df_city.empty else pd.DataFrame()
if export_df.empty:
    st.info("Select city/cities to enable export of filtered rows.")
else:
    export_df = export_df.sort_values(["city", "date"])
    csv_buf = StringIO()
    export_df.to_csv(csv_buf, index=False)
    st.download_button(
        "Download filtered data (CSV)",
        data=csv_buf.getvalue(),
        file_name=f"IITKGP_AQI_{year}_{selected_month_name.replace(' ', '')}_{len(selected_cities)}cities.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================
# 14) FOOTER (clean, premium)
# ============================================================
st.markdown(
    f"""
<div class="card" style="margin-top:14px;">
  <div class="section-title">About</div>
  <div class="section-sub">
    Built for a product-quality experience: consistent typography, clear hierarchy, responsive layout,
    and decision-focused visuals. Data is loaded from your local CPCB exports.
  </div>
  <div class="badge">Made with Streamlit + Plotly • Theme: <b>{theme_name}</b> • Period: <b>{period_label}</b></div>
</div>
""",
    unsafe_allow_html=True,
)

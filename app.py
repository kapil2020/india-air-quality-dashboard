import os
import json
from io import StringIO
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# ============================================================
# 0) PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="IIT KGP Air Quality Intelligence",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 1) THEME + TOKENS
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
    "Good": "Good day for outdoor activities.",
    "Satisfactory": "Sensitive groups: reduce heavy exertion outdoors.",
    "Moderate": "Sensitive groups: prefer indoor, reduce outdoor time.",
    "Poor": "Everyone: reduce prolonged heavy exertion outdoors.",
    "Very Poor": "Avoid outdoor activity, especially sensitive groups.",
    "Severe": "Avoid outdoor activity. Close windows. Use mask and purifier.",
    "Unknown": "Data unavailable. Use precautions.",
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


# ============================================================
# 2) SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    theme_name = st.selectbox("Theme", list(THEMES.keys()), index=0)
    T = THEMES[theme_name]
    pio.templates.default = T["plotly_template"]

    st.markdown("---")
    force_refresh = st.button("🔄 Refresh data", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧠 Options")
    show_story = st.toggle("Story mode", value=True)
    compact_mode = st.toggle("Compact mode", value=False)

# ============================================================
# 3) CSS
# ============================================================
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

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
# 4) ROBUST DATA LOADER (matches your repo names)
#    Files used:
#      combined_air_quality.txt
#      data/YYYY-MM-DD.csv (optional)
# ============================================================
def _candidate_paths(script_dir: Path, cwd_dir: Path, rel_path: str):
    return [cwd_dir / rel_path, script_dir / rel_path]


@st.cache_data(ttl=3600, show_spinner=False)
def load_data_and_metadata(_refresh_key: int = 0):
    script_dir = Path(__file__).resolve().parent
    cwd_dir = Path.cwd()

    today = pd.to_datetime("today").date()
    rel_today_csv = f"data/{today}.csv"
    rel_fallback_txt = "combined_air_quality.txt"

    searched = []
    for p in _candidate_paths(script_dir, cwd_dir, rel_today_csv):
        searched.append(str(p))
    for p in _candidate_paths(script_dir, cwd_dir, rel_fallback_txt):
        searched.append(str(p))

    # Find files
    today_csv = next((p for p in _candidate_paths(script_dir, cwd_dir, rel_today_csv) if p.exists()), None)
    fallback_txt = next((p for p in _candidate_paths(script_dir, cwd_dir, rel_fallback_txt) if p.exists()), None)

    df_loaded = None
    msg_parts = []
    last_update_time = None

    if today_csv is not None:
        try:
            tmp = pd.read_csv(today_csv)
            if "date" in tmp.columns:
                tmp["date"] = pd.to_datetime(tmp["date"])
                df_loaded = tmp
                msg_parts.append(f"Live CSV: {today_csv.name}")
                last_update_time = pd.Timestamp(today_csv.stat().st_mtime, unit="s")
            else:
                msg_parts.append(f"Found {today_csv.name} but missing 'date', using fallback.")
        except Exception as e:
            msg_parts.append(f"CSV read error: {e}, using fallback.")

    if df_loaded is None and fallback_txt is not None:
        try:
            df_loaded = pd.read_csv(fallback_txt, sep="\t", parse_dates=["date"])
            msg_parts.append("Archive: combined_air_quality.txt")
            last_update_time = pd.Timestamp(fallback_txt.stat().st_mtime, unit="s")
        except Exception as e:
            return pd.DataFrame(), f"ERROR reading combined_air_quality.txt: {e}", None, {
                "cwd": str(cwd_dir),
                "script_dir": str(script_dir),
                "searched": searched,
                "today_csv_found": str(today_csv) if today_csv else None,
                "fallback_txt_found": str(fallback_txt) if fallback_txt else None,
            }

    diag = {
        "cwd": str(cwd_dir),
        "script_dir": str(script_dir),
        "searched": searched,
        "today_csv_found": str(today_csv) if today_csv else None,
        "fallback_txt_found": str(fallback_txt) if fallback_txt else None,
    }

    if df_loaded is None:
        return pd.DataFrame(), "NO_DATA", None, diag

    # Required columns
    required = ["date", "city", "index"]
    for c in required:
        if c not in df_loaded.columns:
            return pd.DataFrame(), f"ERROR missing column: {c}", last_update_time, diag

    # Optional columns
    if "pollutant" not in df_loaded.columns:
        df_loaded["pollutant"] = np.nan
    if "level" not in df_loaded.columns:
        df_loaded["level"] = "Unknown"

    # Cleanup
    df_loaded["date"] = pd.to_datetime(df_loaded["date"])
    df_loaded["index"] = pd.to_numeric(df_loaded["index"], errors="coerce")

    df_loaded["pollutant"] = (
        df_loaded["pollutant"]
        .astype(str)
        .str.split(",").str[0].str.strip()
        .replace(["nan", "NaN", "None", ""], np.nan)
        .fillna("Other")
    )

    valid_levels = set(CATEGORY_COLORS.keys())
    df_loaded["level"] = df_loaded["level"].astype(str).fillna("Unknown")
    bad = ~df_loaded["level"].isin(valid_levels)
    if bad.any():
        df_loaded.loc[bad, "level"] = df_loaded.loc[bad, "index"].apply(get_category)

    msg = " • ".join(msg_parts) if msg_parts else "Data loaded"
    return df_loaded, msg, last_update_time, diag


_refresh_key = int(datetime.now().timestamp()) if force_refresh else 0
df, load_message, data_last_updated, diag = load_data_and_metadata(_refresh_key=_refresh_key)

# Upload fallback (so dashboard never dies without telling you why)
if isinstance(load_message, str) and load_message == "NO_DATA":
    with st.sidebar:
        st.error("No data files found in repo path. Upload combined_air_quality.txt to continue.")
        uploaded = st.file_uploader("Upload combined_air_quality.txt", type=["txt"])
    st.markdown("### Data diagnostics")
    st.code(json.dumps(diag, indent=2))

    if uploaded is None:
        st.stop()

    try:
        df = pd.read_csv(uploaded, sep="\t", parse_dates=["date"])
        load_message = "Uploaded: combined_air_quality.txt"
        data_last_updated = pd.to_datetime("now")
    except Exception as e:
        st.error(f"Upload read failed: {e}")
        st.stop()

if df.empty:
    st.error("Dashboard cannot operate without data.")
    st.markdown("### Data diagnostics")
    st.code(json.dumps(diag, indent=2))
    st.stop()

# ============================================================
# 5) FILTERS (after data)
# ============================================================
months_map = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}

with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎛️ Filters")

    unique_cities = sorted(df["city"].dropna().unique().tolist())
    default_city = "Delhi" if "Delhi" in unique_cities else (unique_cities[0] if unique_cities else None)

    selected_cities = st.multiselect(
        "Cities",
        options=unique_cities,
        default=[default_city] if default_city else [],
    )

    years = sorted(df["date"].dt.year.dropna().unique().tolist())
    default_year = max(years) if years else None
    year = st.selectbox("Year", options=years, index=years.index(default_year) if default_year in years else 0)

    month_options = ["All Months"] + [months_map[i] for i in range(1, 13)]
    selected_month_name = st.selectbox("Month", options=month_options, index=0)

month_number = None
if selected_month_name != "All Months":
    month_number = [k for k, v in months_map.items() if v == selected_month_name][0]

df_period = df[df["date"].dt.year == year].copy()
if month_number is not None:
    df_period = df_period[df_period["date"].dt.month == month_number].copy()

df_city = df_period[df_period["city"].isin(selected_cities)].copy() if selected_cities else pd.DataFrame()

# ============================================================
# 6) PLOT HELPERS
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
# 7) HERO
# ============================================================
period_label = f"{selected_month_name} {year}" if selected_month_name != "All Months" else f"Full Year {year}"
last_update_txt = data_last_updated.strftime("%Y-%m-%d %H:%M:%S") if data_last_updated is not None else "Unknown"

st.markdown(
    f"""
<div class="hero">
  <div class="badge">🌬️ IIT KGP • Air Quality Intelligence</div>
  <h1 class="hero-title">India AQI Dashboard</h1>
  <div class="hero-sub">
    Period: <b>{period_label}</b> • Data: <b>{load_message}</b> • Last update: <b>{last_update_txt}</b>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 8) KPIs
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

st.markdown(
    f"""
<div class="kpi-grid">
  <div class="kpi">
    <div class="label">Coverage</div>
    <div class="value">{cities_count}</div>
    <div class="hint">Cities</div>
  </div>
  <div class="kpi">
    <div class="label">Observations</div>
    <div class="value">{days_count}</div>
    <div class="hint">Days</div>
  </div>
  <div class="kpi">
    <div class="label">Mean AQI</div>
    <div class="value" style="color:{cat_color};">{avg_aqi:.1f}</div>
    <div class="hint">{cat_avg}</div>
  </div>
  <div class="kpi">
    <div class="label">P90 AQI</div>
    <div class="value" style="color:{T["warn"]};">{p90_aqi:.1f}</div>
    <div class="hint">Extreme days signal</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if show_story:
    st.markdown(
        f"""
<div class="card" style="margin-top:12px;">
  <div class="section-title">Quick insight</div>
  <div class="section-sub">
    Mean is <b style="color:{cat_color};">{avg_aqi:.1f} ({cat_avg})</b>.
    P90 is <b style="color:{T["warn"]};">{p90_aqi:.1f}</b>.
    Dominant pollutant is <b style="color:{poll_color};">{dominant_poll}</b>.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# ============================================================
# 9) CITY RANKINGS
# ============================================================
st.markdown("## 🏆 City rankings")

if df_period["index"].dropna().empty:
    st.info("No AQI values for this selection.")
else:
    city_avg = df_period.groupby("city")["index"].mean().dropna().sort_values()
    max_cities = len(city_avg)

    c1, c2 = st.columns([1.2, 2.8], gap="large")
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Controls</div>', unsafe_allow_html=True)
        n_show = st.slider("Cities to show", 3, min(20, max_cities) if max_cities >= 3 else 3, min(8, max_cities) if max_cities else 3)
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
# 10) CITY DEEP DIVE
# ============================================================
st.markdown("## 🏙️ City deep dive")

if not selected_cities:
    st.info("Select at least one city.")
else:
    tab_overview, tab_trends, tab_pollutants, tab_forecast = st.tabs(
        ["Overview", "Trends", "Pollutants", "Forecast"]
    )

    with tab_overview:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Snapshot</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Latest available day per city in the selected period.</div>', unsafe_allow_html=True)

        cols = st.columns(min(4, max(1, len(selected_cities))), gap="medium")
        for i, city in enumerate(selected_cities[: len(cols)]):
            city_df = df_period[df_period["city"] == city].copy()
            city_df = city_df.dropna(subset=["index"]).sort_values("date", ascending=False)

            if city_df.empty:
                cols[i].warning(f"{city}: no data")
                continue

            latest = city_df.iloc[0]
            aqi_val = float(latest["index"])
            level = str(latest.get("level", get_category(aqi_val)))
            pollutant = str(latest.get("pollutant", "Other"))
            rec = HEALTH_RECOMMENDATIONS.get(level, HEALTH_RECOMMENDATIONS["Unknown"])
            color = CATEGORY_COLORS.get(level, CATEGORY_COLORS["Unknown"])

            cols[i].markdown(
                f"""
<div class="kpi" style="height:100%;">
  <div class="label">{city}</div>
  <div class="value" style="color:{color};">{aqi_val:.0f}</div>
  <div class="hint"><b>{level}</b> • Dominant: <b style="color:{POLLUTANT_COLORS.get(pollutant, POLLUTANT_COLORS["Other"])};">{pollutant}</b></div>
  <div class="hint" style="margin-top:8px;">{rec}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        if len(selected_cities) > 1:
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
            fig.update_layout(**layout_base("AQI trends comparison", height=380 if compact_mode else 500))
            fig.update_layout(hovermode="x unified")
            fig.update_xaxes(title_text=None, gridcolor=T["border"])
            fig.update_yaxes(title_text="AQI", gridcolor=T["border"])
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("</div>", unsafe_allow_html=True)

    with tab_trends:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Trends</div>', unsafe_allow_html=True)

        city_sel = st.selectbox("City", selected_cities, index=0, key="trend_city")
        d = df_period[df_period["city"] == city_sel].sort_values("date").copy()
        d = d.dropna(subset=["index"])

        if len(d) < 10:
            st.info("Not enough points.")
        else:
            d["roll7"] = d["index"].rolling(7, min_periods=1).mean()

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=d["date"], y=d["index"],
                    mode="lines+markers",
                    name="Daily",
                    line=dict(width=1.6, color=T["muted2"]),
                    marker=dict(size=4, opacity=0.85, color=T["muted2"]),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=d["date"], y=d["roll7"],
                    mode="lines",
                    name="7-day mean",
                    line=dict(width=3.0, color=T["accent"]),
                )
            )

            fig.update_layout(**layout_base(f"{city_sel}: AQI trend", height=380 if compact_mode else 520))
            fig.update_layout(hovermode="x unified")
            fig.update_xaxes(title_text=None, gridcolor=T["border"])
            fig.update_yaxes(title_text="AQI", gridcolor=T["border"])
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("</div>", unsafe_allow_html=True)

    with tab_pollutants:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Pollutants</div>', unsafe_allow_html=True)

        city_sel = st.selectbox("City", selected_cities, index=0, key="poll_city")
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
            fig.update_layout(**layout_base("Period pollutant mix", height=360 if compact_mode else 460))
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("</div>", unsafe_allow_html=True)

    with tab_forecast:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Forecast (baseline)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Polynomial regression (degree 2). Only for dashboard preview.</div>', unsafe_allow_html=True)

        city_sel = st.selectbox("City", selected_cities, index=0, key="fc_city")
        horizon = st.slider("Forecast horizon (days)", 7, 30, 15)

        d = df_period[df_period["city"] == city_sel].sort_values("date")[["date", "index"]].dropna()
        if len(d) < 15:
            st.warning(f"Need at least 15 valid points. Found {len(d)}.")
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

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=obs["date"], y=obs["AQI"], mode="lines+markers", name="Observed",
                                     line=dict(width=1.6, color=T["muted2"]),
                                     marker=dict(size=4, opacity=0.85, color=T["muted2"])))
            fig.add_trace(go.Scatter(x=fc["date"], y=fc["AQI"], mode="lines", name="Forecast",
                                     line=dict(width=3.0, dash="dash", color=T["accent"])))

            fig.update_layout(**layout_base(f"{city_sel}: forecast", height=380 if compact_mode else 520))
            fig.update_layout(hovermode="x unified")
            fig.update_xaxes(title_text=None, gridcolor=T["border"])
            fig.update_yaxes(title_text="AQI", gridcolor=T["border"])
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ============================================================
# 11) MAP (lat_long.txt in repo root)
# ============================================================
st.markdown("## 📍 AQI hotspots map")

script_dir = Path(__file__).resolve().parent
cwd_dir = Path.cwd()

lat_candidates = _candidate_paths(script_dir, cwd_dir, "lat_long.txt")
coords_path = next((p for p in lat_candidates if p.exists()), None)

map_left, map_right = st.columns([3, 1], gap="large")

with map_left:
    grouped = df_period.groupby("city").agg(
        avg_aqi=("index", "mean"),
        dominant_pollutant=("pollutant", lambda x: x.mode().iloc[0] if not x.mode().empty else "Other"),
    ).reset_index().dropna(subset=["avg_aqi"])

    if coords_path is None:
        st.warning("lat_long.txt not found. Map will not show. Check repo root.")
    else:
        city_coords_data = {}
        try:
            content = coords_path.read_text(encoding="utf-8", errors="ignore")
            exec(content, {}, city_coords_data)
            city_coords = city_coords_data.get("city_coords", {})
        except Exception as e:
            st.error(f"lat_long.txt read error: {e}")
            city_coords = {}

        if not city_coords:
            st.warning("city_coords not found inside lat_long.txt")
        else:
            latlong = pd.DataFrame([{"city": k, "lat": v[0], "lon": v[1]} for k, v in city_coords.items()])
            m = pd.merge(grouped, latlong, on="city", how="inner")

            if m.empty:
                st.warning("No coordinate match between city names.")
            else:
                m["AQI Category"] = m["avg_aqi"].apply(get_category)
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
                        "<b>%{customdata[0]}</b><br>"
                        "Avg AQI: %{customdata[1]:.1f} (%{customdata[3]})<br>"
                        "Dominant pollutant: %{customdata[2]}<extra></extra>"
                    )
                )
                fig.update_layout(**layout_base(f"Average AQI hotspots • {period_label}", height=520 if compact_mode else 700))
                fig.update_layout(mapbox_style=T["map_style"], margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

with map_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Legend</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Color is AQI category. Bubble size is severity.</div>', unsafe_allow_html=True)
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
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ============================================================
# 12) EXPORT
# ============================================================
st.markdown("## 📥 Export")

if df_city.empty:
    st.info("Select cities to enable export.")
else:
    export_df = df_city.sort_values(["city", "date"]).copy()
    buf = StringIO()
    export_df.to_csv(buf, index=False)
    st.download_button(
        "Download filtered data (CSV)",
        data=buf.getvalue(),
        file_name=f"IITKGP_AQI_{year}_{selected_month_name.replace(' ', '')}_{len(selected_cities)}cities.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================
# 13) FOOTER
# ============================================================
st.markdown(
    f"""
<div class="card" style="margin-top:14px;">
  <div class="section-title">About</div>
  <div class="section-sub">
    Uses repo files: combined_air_quality.txt, lat_long.txt, and optional daily CSV inside data/.
    If your app shows "NO_DATA", open the diagnostics box and check paths.
  </div>
  <div class="badge">Theme: <b>{theme_name}</b> • Period: <b>{period_label}</b></div>
</div>
""",
    unsafe_allow_html=True,
)

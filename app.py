import os
import json
from pathlib import Path
from io import StringIO
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="India Air Quality Intelligence",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# THEME TOKENS
# =========================
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
        "cont_scale": "Turbo",
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
        "cont_scale": "Cividis",
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
    "NH3": "#A78BFA",
    "Other": "#94A3B8",
}

AQI_BANDS = [
    ("Good", 0, 50),
    ("Satisfactory", 51, 100),
    ("Moderate", 101, 200),
    ("Poor", 201, 300),
    ("Very Poor", 301, 400),
    ("Severe", 401, 1000),
]

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
    v = float(aqi_val)
    if v <= 50:
        return "Good"
    if v <= 100:
        return "Satisfactory"
    if v <= 200:
        return "Moderate"
    if v <= 300:
        return "Poor"
    if v <= 400:
        return "Very Poor"
    return "Severe"


def safe_mode(series: pd.Series, fallback="Other"):
    try:
        m = series.dropna().mode()
        if len(m) == 0:
            return fallback
        return str(m.iloc[0])
    except Exception:
        return fallback


MONTHS_MAP = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}
MONTH_ORDER = [MONTHS_MAP[i] for i in range(1, 13)]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# =========================
# SIDEBAR CONTROLS
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    theme_name = st.selectbox("Theme", list(THEMES.keys()), index=0)
    T = THEMES[theme_name]
    pio.templates.default = T["plotly_template"]

    st.markdown("---")
    force_refresh = st.button("🔄 Refresh data", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧠 Power options")
    show_story = st.toggle("Story insights", value=True)
    compact_mode = st.toggle("Compact mode", value=False)
    smooth_lines = st.toggle("Smooth trends", value=True)
    show_uncertainty = st.toggle("Show percentile band (P10–P90)", value=True)

    st.markdown("---")
    st.markdown("### 🔎 Advanced signals")
    anomaly_sigma = st.slider("Anomaly sensitivity (σ)", 1.5, 3.5, 2.0, 0.1)
    episode_threshold = st.selectbox("Episode threshold", [200, 300, 400], index=1, help="Episode days are contiguous days with AQI >= threshold")


# =========================
# CSS (modern, responsive)
# =========================
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [data-testid="stAppViewContainer"] {{ background: {T["bg"]} !important; }}
* {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}

.main .block-container {{
  padding-top: 1.25rem;
  padding-bottom: 2.5rem;
  max-width: 1480px;
}}

.hero {{
  background: radial-gradient(1200px 520px at 18% 0%, rgba(0,212,255,0.22), transparent 55%),
              radial-gradient(900px 460px at 80% 12%, rgba(0,255,163,0.18), transparent 55%),
              linear-gradient(180deg, {T["surface"]}, rgba(255,255,255,0.00));
  border: 1px solid {T["border"]};
  border-radius: 20px;
  padding: 22px 22px 18px 22px;
  box-shadow: {T["shadow"]};
  margin-bottom: 14px;
}}
.hero-title {{
  font-size: 2.15rem;
  font-weight: 900;
  line-height: 1.1;
  color: {T["text"]};
  margin: 0;
}}
.hero-sub {{
  margin-top: 8px;
  color: {T["muted"]};
  font-size: 1.02rem;
  line-height: 1.55;
}}

.badge {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  border-radius: 999px;
  border: 1px solid {T["border"]};
  background: rgba(255,255,255,0.02);
  color: {T["muted"]};
  font-size: 0.86rem;
  font-weight: 650;
}}

.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-top: 12px;
}}
.kpi {{
  border: 1px solid {T["border"]};
  background: {T["surface2"]};
  border-radius: 16px;
  padding: 14px 14px;
  box-shadow: 0 10px 28px rgba(0,0,0,0.18);
}}
.kpi .label {{
  color: {T["muted2"]};
  font-size: 0.86rem;
  font-weight: 700;
}}
.kpi .value {{
  margin-top: 6px;
  color: {T["text"]};
  font-weight: 900;
  font-size: 1.75rem;
  letter-spacing: -0.02em;
}}
.kpi .hint {{
  margin-top: 6px;
  color: {T["muted"]};
  font-size: 0.92rem;
}}

.card {{
  border: 1px solid {T["border"]};
  background: {T["surface"]};
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: {T["shadow"]};
}}

.section-title {{
  font-size: 1.22rem;
  font-weight: 900;
  color: {T["text"]};
  margin: 0 0 10px 0;
}}
.section-sub {{
  margin-top: -4px;
  margin-bottom: 12px;
  color: {T["muted"]};
  font-size: 0.95rem;
}}

hr {{
  border: none;
  border-top: 1px solid {T["border"]};
  margin: 16px 0;
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


# =========================
# ROBUST FILE PATHS (repo structure)
# =========================
def candidate_paths(rel_path: str):
    script_dir = Path(__file__).resolve().parent
    cwd_dir = Path.cwd()
    return [cwd_dir / rel_path, script_dir / rel_path]


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(_refresh_key: int = 0):
    today = pd.to_datetime("today").date()
    rel_today_csv = f"data/{today}.csv"
    rel_fallback_txt = "combined_air_quality.txt"

    today_csv = next((p for p in candidate_paths(rel_today_csv) if p.exists()), None)
    fallback_txt = next((p for p in candidate_paths(rel_fallback_txt) if p.exists()), None)

    diag = {
        "cwd": str(Path.cwd()),
        "script_dir": str(Path(__file__).resolve().parent),
        "today_csv": str(today_csv) if today_csv else None,
        "fallback_txt": str(fallback_txt) if fallback_txt else None,
        "searched": [str(p) for p in (candidate_paths(rel_today_csv) + candidate_paths(rel_fallback_txt))],
    }

    df_loaded = None
    msg = []

    # 1) Try daily CSV first
    if today_csv is not None:
        try:
            tmp = pd.read_csv(today_csv)
            if "date" in tmp.columns:
                tmp["date"] = pd.to_datetime(tmp["date"])
                df_loaded = tmp
                msg.append(f"Live: {today_csv.name}")
            else:
                msg.append("Live CSV missing 'date', using archive")
        except Exception as e:
            msg.append(f"Live CSV error: {e}, using archive")

    # 2) Fallback archive
    if df_loaded is None and fallback_txt is not None:
        try:
            df_loaded = pd.read_csv(fallback_txt, sep="\t")
            msg.append("Archive: combined_air_quality.txt")
        except Exception as e:
            return pd.DataFrame(), f"ERROR reading archive: {e}", None, diag

    if df_loaded is None:
        return pd.DataFrame(), "NO_DATA", None, diag

    # Normalize expected columns
    # Required: date, city, index
    # Optional: level, pollutant, datetime, hour
    if "date" not in df_loaded.columns:
        return pd.DataFrame(), "ERROR: 'date' column missing", None, diag

    # Try parse date robustly
    df_loaded["date"] = pd.to_datetime(df_loaded["date"], errors="coerce")
    df_loaded = df_loaded.dropna(subset=["date"]).copy()

    if "city" not in df_loaded.columns:
        return pd.DataFrame(), "ERROR: 'city' column missing", None, diag
    if "index" not in df_loaded.columns:
        return pd.DataFrame(), "ERROR: 'index' column missing", None, diag

    df_loaded["index"] = pd.to_numeric(df_loaded["index"], errors="coerce")

    if "pollutant" not in df_loaded.columns:
        df_loaded["pollutant"] = np.nan
    if "level" not in df_loaded.columns:
        df_loaded["level"] = "Unknown"

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

    # Derive time features
    df_loaded["year"] = df_loaded["date"].dt.year
    df_loaded["month"] = pd.Categorical(df_loaded["date"].dt.month.map(MONTHS_MAP), categories=MONTH_ORDER, ordered=True)
    df_loaded["month_num"] = df_loaded["date"].dt.month
    df_loaded["day"] = df_loaded["date"].dt.day
    df_loaded["week"] = df_loaded["date"].dt.isocalendar().week.astype(int)
    df_loaded["weekday"] = pd.Categorical(df_loaded["date"].dt.day_name(), categories=WEEKDAY_ORDER, ordered=True)

    # Detect diurnal capability
    has_hour = False
    if "hour" in df_loaded.columns:
        try:
            df_loaded["hour"] = pd.to_numeric(df_loaded["hour"], errors="coerce")
            if df_loaded["hour"].between(0, 23).any():
                has_hour = True
        except Exception:
            has_hour = False

    has_datetime = False
    if "datetime" in df_loaded.columns:
        try:
            dt = pd.to_datetime(df_loaded["datetime"], errors="coerce")
            if dt.notna().any():
                df_loaded["datetime"] = dt
                df_loaded["hour_from_datetime"] = df_loaded["datetime"].dt.hour
                has_datetime = True
        except Exception:
            has_datetime = False

    return df_loaded, " • ".join(msg) if msg else "Loaded", (has_hour or has_datetime), diag


_refresh_key = int(datetime.now().timestamp()) if force_refresh else 0
df, load_msg, has_diurnal, diag = load_data(_refresh_key=_refresh_key)

if isinstance(load_msg, str) and load_msg == "NO_DATA":
    with st.sidebar:
        st.error("No data file found. Upload combined_air_quality.txt to continue.")
        uploaded = st.file_uploader("Upload combined_air_quality.txt", type=["txt"])
    st.markdown("### Data diagnostics")
    st.code(json.dumps(diag, indent=2))
    if uploaded is None:
        st.stop()
    try:
        df = pd.read_csv(uploaded, sep="\t")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["index"] = pd.to_numeric(df["index"], errors="coerce")
        df["pollutant"] = df.get("pollutant", np.nan)
        df["level"] = df.get("level", "Unknown")
        load_msg = "Uploaded combined_air_quality.txt"
        df["year"] = df["date"].dt.year
        df["month"] = pd.Categorical(df["date"].dt.month.map(MONTHS_MAP), categories=MONTH_ORDER, ordered=True)
        df["month_num"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["week"] = df["date"].dt.isocalendar().week.astype(int)
        df["weekday"] = pd.Categorical(df["date"].dt.day_name(), categories=WEEKDAY_ORDER, ordered=True)
        has_diurnal = False
    except Exception as e:
        st.error(f"Upload read failed: {e}")
        st.stop()

if df.empty:
    st.error("No valid rows after parsing. Check your date/index columns.")
    st.code(json.dumps(diag, indent=2))
    st.stop()


# =========================
# SIDEBAR FILTERS
# =========================
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎛️ Filters")

    cities = sorted(df["city"].dropna().unique().tolist())
    default_city = "Delhi" if "Delhi" in cities else (cities[0] if cities else None)

    selected_cities = st.multiselect("Cities", options=cities, default=[default_city] if default_city else [])
    years = sorted(df["year"].dropna().unique().tolist())
    year = st.selectbox("Year", options=years, index=years.index(max(years)) if years else 0)

    month_options = ["All Months"] + MONTH_ORDER
    month_name = st.selectbox("Month", options=month_options, index=0)

    compare_mode = st.toggle("Compare selected cities", value=True)
    top_n = st.slider("Top N cities in rankings", 5, 25, 12)

df_period = df[df["year"] == year].copy()
if month_name != "All Months":
    df_period = df_period[df_period["month"] == month_name].copy()

df_sel = df_period[df_period["city"].isin(selected_cities)].copy() if selected_cities else pd.DataFrame()

period_label = f"{month_name} {year}" if month_name != "All Months" else f"Full Year {year}"


# =========================
# PLOT HELPERS
# =========================
PLOT_CONFIG = {"displayModeBar": True, "responsive": True, "displaylogo": False}

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
        margin=dict(l=45, r=20, t=60 if title else 30, b=45),
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

def add_aqi_bands(fig, yref="y"):
    for name, lo, hi in AQI_BANDS:
        col = CATEGORY_COLORS.get(name, CATEGORY_COLORS["Unknown"])
        fig.add_hrect(
            y0=lo, y1=hi,
            fillcolor=col,
            opacity=0.08,
            line_width=0,
            layer="below"
        )
    return fig


# =========================
# HERO + KPI
# =========================
last_update_txt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

st.markdown(
    f"""
<div class="hero">
  <div class="badge">🌬️ India Air Quality Intelligence</div>
  <h1 class="hero-title">High resolution AQI analytics</h1>
  <div class="hero-sub">
    Period: <b>{period_label}</b> • Source: <b>{load_msg}</b> • Updated: <b>{last_update_txt}</b>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

cities_count = int(df_period["city"].nunique())
days_count = int(df_period["date"].nunique())
avg_aqi = float(df_period["index"].mean()) if df_period["index"].notna().any() else np.nan
p90_aqi = float(df_period["index"].quantile(0.90)) if df_period["index"].notna().any() else np.nan
cat_avg = get_category(avg_aqi) if np.isfinite(avg_aqi) else "Unknown"
cat_color = CATEGORY_COLORS.get(cat_avg, CATEGORY_COLORS["Unknown"])
dominant_poll = safe_mode(df_period["pollutant"], "Other")
poll_color = POLLUTANT_COLORS.get(dominant_poll, POLLUTANT_COLORS["Other"])

st.markdown(
    f"""
<div class="kpi-grid">
  <div class="kpi"><div class="label">Coverage</div><div class="value">{cities_count}</div><div class="hint">Cities</div></div>
  <div class="kpi"><div class="label">Observations</div><div class="value">{days_count}</div><div class="hint">Days in period</div></div>
  <div class="kpi"><div class="label">Mean AQI</div><div class="value" style="color:{cat_color};">{avg_aqi:.1f}</div><div class="hint">{cat_avg}</div></div>
  <div class="kpi"><div class="label">P90 AQI</div><div class="value" style="color:{T["warn"]};">{p90_aqi:.1f}</div><div class="hint">Tail risk</div></div>
</div>
""",
    unsafe_allow_html=True,
)

if show_story:
    st.markdown(
        f"""
<div class="card" style="margin-top:12px;">
  <div class="section-title">Executive insight</div>
  <div class="section-sub">
    Mean AQI is <b style="color:{cat_color};">{avg_aqi:.1f} ({cat_avg})</b>.
    P90 is <b style="color:{T["warn"]};">{p90_aqi:.1f}</b>, showing extreme day risk.
    Dominant pollutant label is <b style="color:{poll_color};">{dominant_poll}</b>.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# =========================
# TABS (DEEP ANALYSIS)
# =========================
tab_rank, tab_calendar, tab_week, tab_trend, tab_dist, tab_episodes, tab_poll, tab_map, tab_export = st.tabs(
    ["Rankings", "Calendar heatmap", "Week patterns", "Trends + anomalies", "Distributions", "Episodes", "Pollutants", "Map", "Export"]
)

# -------------------------
# 1) RANKINGS
# -------------------------
with tab_rank:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">City rankings</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Cleanest and most polluted cities by mean AQI for the selected period.</div>', unsafe_allow_html=True)

    city_avg = df_period.groupby("city")["index"].mean().dropna().sort_values()
    if city_avg.empty:
        st.info("No values available.")
    else:
        n = min(top_n, len(city_avg))
        clean = city_avg.head(n).reset_index(name="AvgAQI")
        dirty = city_avg.tail(n).reset_index(name="AvgAQI")

        clean["Category"] = clean["AvgAQI"].apply(get_category)
        dirty["Category"] = dirty["AvgAQI"].apply(get_category)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            fig = px.bar(
                clean.sort_values("AvgAQI", ascending=False),
                x="AvgAQI", y="city",
                orientation="h",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
                text="AvgAQI",
                labels={"city": "", "AvgAQI": "Mean AQI"},
            )
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
            fig.update_layout(**layout_base("🥇 Cleanest", 340 if compact_mode else 460), showlegend=False)
            fig.update_xaxes(gridcolor=T["border"])
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        with c2:
            fig = px.bar(
                dirty.sort_values("AvgAQI", ascending=True),
                x="AvgAQI", y="city",
                orientation="h",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
                text="AvgAQI",
                labels={"city": "", "AvgAQI": "Mean AQI"},
            )
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
            fig.update_layout(**layout_base("⚠️ Most polluted", 340 if compact_mode else 460), showlegend=False)
            fig.update_xaxes(gridcolor=T["border"])
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# 2) CALENDAR HEATMAP (daily)
# -------------------------
with tab_calendar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Calendar heatmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Daily AQI intensity across weeks and weekdays. Pick one city for a clean calendar view.</div>', unsafe_allow_html=True)

    if not selected_cities:
        st.info("Select at least one city in the sidebar.")
    else:
        city_cal = st.selectbox("City for calendar", selected_cities, index=0, key="cal_city")
        d = df_period[df_period["city"] == city_cal].copy()
        d = d.dropna(subset=["index"])
        if d.empty:
            st.info("No data for this city in this period.")
        else:
            d = d.sort_values("date")
            d["dow"] = d["date"].dt.dayofweek  # Monday=0
            d["wk"] = d["date"].dt.isocalendar().week.astype(int)
            d["year_for_week"] = d["date"].dt.isocalendar().year.astype(int)

            # Stable week index for plotting within a year
            # For All Months, weeks across year are fine. For a single month, still ok.
            pivot = d.pivot_table(index="dow", columns="wk", values="index", aggfunc="mean")
            pivot = pivot.reindex(index=list(range(0, 7)))

            fig = px.imshow(
                pivot,
                aspect="auto",
                labels=dict(x="ISO week", y="Weekday", color="AQI"),
                color_continuous_scale=T["cont_scale"],
            )
            fig.update_layout(**layout_base(f"{city_cal}: calendar heatmap", 360 if compact_mode else 520))
            fig.update_yaxes(
                tickmode="array",
                tickvals=list(range(0, 7)),
                ticktext=WEEKDAY_ORDER,
                title_text=""
            )
            fig.update_xaxes(title_text="")
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

            if show_story:
                worst = d.loc[d["index"].idxmax()]
                best = d.loc[d["index"].idxmin()]
                st.caption(
                    f"Worst day: {worst['date'].date()} (AQI {worst['index']:.0f}). "
                    f"Best day: {best['date'].date()} (AQI {best['index']:.0f})."
                )

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# 3) WEEK PATTERNS (weekday + weekly aggregation)
# -------------------------
with tab_week:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Week patterns</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Weekday distribution and week-by-week evolution.</div>', unsafe_allow_html=True)

    if not selected_cities:
        st.info("Select at least one city.")
    else:
        city_w = st.selectbox("City", selected_cities, index=0, key="week_city")
        d = df_period[df_period["city"] == city_w].dropna(subset=["index"]).copy()
        if d.empty:
            st.info("No data.")
        else:
            c1, c2 = st.columns([1.1, 1.9], gap="large")

            with c1:
                fig = px.violin(
                    d,
                    x="weekday",
                    y="index",
                    box=True,
                    points="outliers",
                    labels={"weekday": "", "index": "AQI"},
                )
                fig.update_layout(**layout_base("Weekday distribution", 320 if compact_mode else 440))
                fig.update_xaxes(title_text=None, gridcolor=T["border"])
                fig.update_yaxes(gridcolor=T["border"])
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

            with c2:
                w = d.groupby(["year", "week"], as_index=False)["index"].mean()
                w["week_id"] = w["year"].astype(str) + "-W" + w["week"].astype(str).str.zfill(2)

                fig = px.bar(
                    w.sort_values(["year", "week"]),
                    x="week_id",
                    y="index",
                    labels={"week_id": "ISO Week", "index": "Weekly mean AQI"},
                )
                fig.update_layout(**layout_base("Weekly mean AQI", 320 if compact_mode else 440))
                fig.update_xaxes(title_text=None, gridcolor=T["border"], tickangle=45)
                fig.update_yaxes(gridcolor=T["border"])
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# 4) TRENDS + ANOMALIES + P10-P90 BAND
# -------------------------
with tab_trend:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Trends, uncertainty band, and anomaly detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Daily trend with rolling mean and anomaly markers based on rolling baseline.</div>', unsafe_allow_html=True)

    if not selected_cities:
        st.info("Select at least one city.")
    else:
        city_t = st.selectbox("City", selected_cities, index=0, key="trend_city")
        d = df_period[df_period["city"] == city_t].dropna(subset=["index"]).sort_values("date").copy()
        if len(d) < 10:
            st.info("Not enough points.")
        else:
            # Rolling stats
            d["roll7"] = d["index"].rolling(7, min_periods=1).mean()
            d["roll30"] = d["index"].rolling(30, min_periods=2).mean()
            d["std30"] = d["index"].rolling(30, min_periods=2).std().fillna(0.0)
            d["thr"] = d["roll30"] + float(anomaly_sigma) * d["std30"]
            d["anom"] = d["index"] > d["thr"]

            # Percentile band per day-of-year (seasonal uncertainty)
            band_df = None
            if show_uncertainty:
                tmp = df[df["city"] == city_t].dropna(subset=["index"]).copy()
                tmp["doy"] = tmp["date"].dt.dayofyear
                band = tmp.groupby("doy")["index"].agg(p10=lambda x: np.nanpercentile(x, 10),
                                                     p50=lambda x: np.nanpercentile(x, 50),
                                                     p90=lambda x: np.nanpercentile(x, 90)).reset_index()
                # Map current period to doy
                d["doy"] = d["date"].dt.dayofyear
                band_df = d.merge(band, on="doy", how="left")

            fig = go.Figure()

            # Bands
            if band_df is not None and band_df[["p10", "p90"]].notna().any().any():
                fig.add_trace(go.Scatter(
                    x=band_df["date"], y=band_df["p90"],
                    mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip"
                ))
                fig.add_trace(go.Scatter(
                    x=band_df["date"], y=band_df["p10"],
                    mode="lines", line=dict(width=0),
                    fill="tonexty",
                    name="Seasonal band (P10–P90)",
                    fillcolor="rgba(0,212,255,0.10)" if theme_name.startswith("Dark") else "rgba(0,91,255,0.10)",
                    hoverinfo="skip"
                ))

            # Daily
            fig.add_trace(go.Scatter(
                x=d["date"], y=d["index"],
                mode="lines+markers",
                name="Daily AQI",
                line=dict(width=1.6, shape="spline" if smooth_lines else "linear", color=T["muted2"]),
                marker=dict(size=4, opacity=0.85, color=T["muted2"]),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>AQI: %{y:.0f}<extra></extra>",
            ))

            # Rolling
            fig.add_trace(go.Scatter(
                x=d["date"], y=d["roll7"],
                mode="lines",
                name="7-day mean",
                line=dict(width=3.0, color=T["accent"]),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>7-day mean: %{y:.1f}<extra></extra>",
            ))

            # Anomalies
            an = d[d["anom"]]
            if not an.empty:
                fig.add_trace(go.Scatter(
                    x=an["date"], y=an["index"],
                    mode="markers",
                    name="Anomaly",
                    marker=dict(size=11, color=T["danger"], line=dict(width=1, color=T["bg"])),
                    hovertemplate="<b>Anomaly</b><br>%{x|%Y-%m-%d}<br>AQI: %{y:.0f}<extra></extra>",
                ))

            fig.update_layout(**layout_base(f"{city_t}: trend + anomalies", 420 if compact_mode else 620))
            fig.update_layout(hovermode="x unified")
            fig.update_xaxes(title_text=None, gridcolor=T["border"])
            fig.update_yaxes(title_text="AQI", gridcolor=T["border"])
            add_aqi_bands(fig)
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

            if show_story:
                if not an.empty:
                    st.warning(f"Detected {len(an)} anomaly day(s) using rolling baseline + {anomaly_sigma:.1f}σ.")
                    with st.expander("Show anomaly table"):
                        st.dataframe(an[["date", "index", "level", "pollutant"]].rename(columns={"index": "AQI"}), use_container_width=True)
                else:
                    st.success("No anomalies detected under current sensitivity.")

            # Optional true diurnal
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown("#### Diurnal analysis (hourly)")
            if not has_diurnal:
                st.info("Your file looks daily only (no hour/datetime). Upload hourly data to unlock diurnal plots.")
            else:
                st.success("Hourly information detected. Diurnal plots are enabled.")
                dd = df_period[df_period["city"] == city_t].copy()
                if "hour_from_datetime" in dd.columns and dd["hour_from_datetime"].notna().any():
                    dd["hour_use"] = dd["hour_from_datetime"]
                else:
                    dd["hour_use"] = dd["hour"]

                dd = dd.dropna(subset=["hour_use", "index"])
                if dd.empty:
                    st.info("No hourly points for current selection.")
                else:
                    hr = dd.groupby("hour_use")["index"].agg(["mean", "median"]).reset_index()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hr["hour_use"], y=hr["mean"], mode="lines+markers", name="Mean",
                        line=dict(width=3.0, color=T["accent"])
                    ))
                    fig.add_trace(go.Scatter(
                        x=hr["hour_use"], y=hr["median"], mode="lines+markers", name="Median",
                        line=dict(width=2.0, dash="dot", color=T["accent2"])
                    ))
                    fig.update_layout(**layout_base("Diurnal profile", 320 if compact_mode else 420))
                    fig.update_xaxes(title_text="Hour of day", dtick=1, gridcolor=T["border"])
                    fig.update_yaxes(title_text="AQI", gridcolor=T["border"])
                    add_aqi_bands(fig)
                    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# 5) DISTRIBUTIONS (hist + category shares + box compare)
# -------------------------
with tab_dist:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Distributions and category structure</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Understand the full shape, not only averages.</div>', unsafe_allow_html=True)

    if df_period["index"].dropna().empty:
        st.info("No data.")
    else:
        c1, c2 = st.columns([1.6, 1.4], gap="large")

        with c1:
            # National distribution
            fig = px.histogram(
                df_period.dropna(subset=["index"]),
                x="index",
                nbins=40,
                labels={"index": "AQI"},
                marginal="box",
            )
            fig.update_layout(**layout_base("AQI distribution (national)", 340 if compact_mode else 460))
            fig.update_xaxes(gridcolor=T["border"])
            fig.update_yaxes(gridcolor=T["border"])
            add_aqi_bands(fig)
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        with c2:
            # Category share
            tmp = df_period.dropna(subset=["index"]).copy()
            tmp["cat"] = tmp["index"].apply(get_category)
            share = tmp["cat"].value_counts(normalize=True).reset_index()
            share.columns = ["Category", "Share"]
            share["Share"] = share["Share"] * 100.0

            fig = px.bar(
                share,
                x="Share",
                y="Category",
                orientation="h",
                text="Share",
                color="Category",
                color_discrete_map=CATEGORY_COLORS,
                labels={"Share": "Share of days (%)", "Category": ""},
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
            fig.update_layout(**layout_base("Category share", 340 if compact_mode else 460), showlegend=False)
            fig.update_xaxes(gridcolor=T["border"])
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        # City compare distribution
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("#### City comparison distribution")
        if not selected_cities:
            st.info("Select cities to compare.")
        else:
            d = df_sel.dropna(subset=["index"]).copy()
            if d.empty:
                st.info("No data for selected cities.")
            else:
                fig = px.box(
                    d,
                    x="city",
                    y="index",
                    points=False,
                    labels={"city": "", "index": "AQI"},
                )
                fig.update_layout(**layout_base("City distribution (box)", 320 if compact_mode else 440))
                fig.update_xaxes(gridcolor=T["border"], tickangle=30)
                fig.update_yaxes(gridcolor=T["border"])
                add_aqi_bands(fig)
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# 6) EPISODES (contiguous high AQI periods + streaks)
# -------------------------
with tab_episodes:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Episodes and streaks</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Contiguous high AQI events and their severity.</div>', unsafe_allow_html=True)

    if not selected_cities:
        st.info("Select a city.")
    else:
        city_e = st.selectbox("City", selected_cities, index=0, key="ep_city")
        d = df_period[df_period["city"] == city_e].dropna(subset=["index"]).sort_values("date").copy()
        thr = float(episode_threshold)

        if d.empty:
            st.info("No data.")
        else:
            d["high"] = d["index"] >= thr
            # Identify runs
            run_id = (d["high"] != d["high"].shift(1)).cumsum()
            runs = d[d["high"]].groupby(run_id).agg(
                start=("date", "min"),
                end=("date", "max"),
                days=("date", "count"),
                peak=("index", "max"),
                mean=("index", "mean"),
            ).reset_index(drop=True)

            c1, c2 = st.columns([1.4, 1.6], gap="large")

            with c1:
                # Streak length distribution
                if runs.empty:
                    st.success(f"No episodes found with AQI >= {int(thr)}.")
                else:
                    fig = px.bar(
                        runs.sort_values("peak", ascending=False).head(15),
                        x="peak",
                        y=runs.sort_values("peak", ascending=False).head(15).index.astype(str),
                        orientation="h",
                        labels={"peak": "Peak AQI", "y": "Episode"},
                    )
                    fig.update_layout(**layout_base(f"Top episodes (peak AQI) | threshold {int(thr)}", 320 if compact_mode else 440))
                    fig.update_xaxes(gridcolor=T["border"])
                    fig.update_yaxes(title_text="")
                    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

            with c2:
                # Timeline view (Gantt-like)
                if runs.empty:
                    st.info("No episode timeline.")
                else:
                    fig = go.Figure()
                    for i, r in runs.sort_values("peak", ascending=False).head(20).iterrows():
                        fig.add_trace(go.Scatter(
                            x=[r["start"], r["end"]],
                            y=[f"Ep {i+1}", f"Ep {i+1}"],
                            mode="lines",
                            line=dict(width=10),
                            name=f"Ep {i+1}",
                            hovertemplate=(
                                f"<b>Episode {i+1}</b><br>"
                                f"Start: {r['start'].date()}<br>"
                                f"End: {r['end'].date()}<br>"
                                f"Days: {int(r['days'])}<br>"
                                f"Mean: {r['mean']:.1f}<br>"
                                f"Peak: {r['peak']:.0f}<extra></extra>"
                            ),
                            showlegend=False
                        ))
                    fig.update_layout(**layout_base("Episode timeline (top 20 by peak)", 320 if compact_mode else 440))
                    fig.update_xaxes(gridcolor=T["border"])
                    fig.update_yaxes(title_text="", gridcolor=T["border"])
                    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

            if not runs.empty:
                with st.expander("Episode table"):
                    out = runs.sort_values(["peak", "days"], ascending=False).copy()
                    out["start"] = out["start"].dt.date
                    out["end"] = out["end"].dt.date
                    st.dataframe(out, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# 7) POLLUTANTS (mix + multi-year share + city matrix)
# -------------------------
with tab_poll:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Pollutant intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Dominant pollutant distribution and its change over years.</div>', unsafe_allow_html=True)

    if df_period.empty:
        st.info("No data.")
    else:
        c1, c2 = st.columns(2, gap="large")

        with c1:
            # National mix
            g = df_period.groupby("pollutant").size().reset_index(name="days").sort_values("days", ascending=False)
            fig = px.pie(
                g,
                values="days",
                names="pollutant",
                hole=0.55,
                color="pollutant",
                color_discrete_map=POLLUTANT_COLORS,
            )
            fig.update_layout(**layout_base("National pollutant mix (selected period)", 340 if compact_mode else 460))
            st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        with c2:
            # City x pollutant matrix (share)
            mat = df_period.groupby(["city", "pollutant"]).size().reset_index(name="n")
            tot = mat.groupby("city")["n"].transform("sum")
            mat["pct"] = (mat["n"] / tot) * 100.0
            # Take top cities by volume for readability
            top_cities = df_period["city"].value_counts().head(25).index.tolist()
            mat = mat[mat["city"].isin(top_cities)].copy()
            piv = mat.pivot_table(index="city", columns="pollutant", values="pct", aggfunc="sum", fill_value=0)

            if piv.empty:
                st.info("Not enough data for matrix.")
            else:
                fig = px.imshow(
                    piv,
                    aspect="auto",
                    labels=dict(x="Pollutant", y="City", color="Share (%)"),
                    color_continuous_scale=T["cont_scale"],
                )
                fig.update_layout(**layout_base("City × pollutant share (top 25 cities)", 340 if compact_mode else 460))
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

        # Multi-year stacked share for selected city
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("#### Multi-year pollutant share (city)")
        if not selected_cities:
            st.info("Select a city.")
        else:
            city_p = st.selectbox("City", selected_cities, index=0, key="poll_city_share")
            dd = df[df["city"] == city_p].copy()
            if dd.empty:
                st.info("No data.")
            else:
                y = dd.groupby(["year", "pollutant"]).size().unstack(fill_value=0)
                pct = y.apply(lambda r: (r / r.sum() * 100.0) if r.sum() > 0 else r, axis=1).fillna(0)
                long = pct.reset_index().melt(id_vars="year", var_name="pollutant", value_name="pct")

                fig = px.bar(
                    long,
                    x="year",
                    y="pct",
                    color="pollutant",
                    barmode="stack",
                    color_discrete_map=POLLUTANT_COLORS,
                    labels={"pct": "Share of days (%)", "year": "", "pollutant": "Pollutant"},
                )
                fig.update_layout(**layout_base(f"{city_p}: pollutant share by year", 320 if compact_mode else 440))
                fig.update_xaxes(type="category", gridcolor=T["border"])
                fig.update_yaxes(ticksuffix="%", gridcolor=T["border"])
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# 8) MAP (lat_long.txt)
# -------------------------
with tab_map:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Hotspots map</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Bubble size is severity. Color is AQI category.</div>', unsafe_allow_html=True)

    coords_path = next((p for p in candidate_paths("lat_long.txt") if p.exists()), None)
    if coords_path is None:
        st.warning("lat_long.txt not found in repo root.")
    else:
        # Aggregate
        grouped = df_period.groupby("city").agg(
            avg_aqi=("index", "mean"),
            dominant_pollutant=("pollutant", lambda x: safe_mode(x, "Other")),
        ).reset_index().dropna(subset=["avg_aqi"])

        # Load coords
        city_coords_data = {}
        try:
            exec(coords_path.read_text(encoding="utf-8", errors="ignore"), {}, city_coords_data)
            city_coords = city_coords_data.get("city_coords", {})
        except Exception as e:
            st.error(f"lat_long.txt read error: {e}")
            city_coords = {}

        if not city_coords:
            st.warning("city_coords dict not found in lat_long.txt")
        else:
            latlong = pd.DataFrame([{"city": k, "lat": v[0], "lon": v[1]} for k, v in city_coords.items()])
            m = pd.merge(grouped, latlong, on="city", how="inner")
            if m.empty:
                st.warning("No match between city names in data and lat_long.txt.")
            else:
                m["AQI Category"] = m["avg_aqi"].apply(get_category)
                m["size"] = np.maximum(m["avg_aqi"] / 10.0, 5.0)

                fig = px.scatter_mapbox(
                    m,
                    lat="lat",
                    lon="lon",
                    size="size",
                    size_max=30,
                    color="AQI Category",
                    color_discrete_map=CATEGORY_COLORS,
                    hover_name="city",
                    custom_data=["avg_aqi", "dominant_pollutant", "AQI Category"],
                    zoom=4.2,
                    center={"lat": 23.5, "lon": 82.0},
                )
                fig.update_traces(
                    hovertemplate=(
                        "<b>%{hovertext}</b><br>"
                        "Mean AQI: %{customdata[0]:.1f} (%{customdata[2]})<br>"
                        "Dominant pollutant: %{customdata[1]}<extra></extra>"
                    )
                )
                fig.update_layout(**layout_base(f"Average AQI hotspots • {period_label}", 520 if compact_mode else 720))
                fig.update_layout(mapbox_style=T["map_style"], margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# 9) EXPORT
# -------------------------
with tab_export:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Export and diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Download filtered data. If anything fails, check diagnostics.</div>', unsafe_allow_html=True)

    if df_sel.empty:
        st.info("Select cities to export.")
    else:
        export_df = df_sel.sort_values(["city", "date"]).copy()
        buf = StringIO()
        export_df.to_csv(buf, index=False)
        st.download_button(
            "Download selected slice (CSV)",
            data=buf.getvalue(),
            file_name=f"aqi_{year}_{month_name.replace(' ','')}_{len(selected_cities)}cities.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("Show data diagnostics"):
        st.code(json.dumps(diag, indent=2))

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown(
    f"""
<div class="card" style="margin-top:12px;">
  <div class="section-title">Notes</div>
  <div class="section-sub">
    Calendar, weekday, episodes, anomalies, distributions are computed from daily AQI.
    Diurnal plots require hourly timestamps (hour or datetime column).
  </div>
  <div class="badge">Theme: <b>{theme_name}</b> • Period: <b>{period_label}</b></div>
</div>
""",
    unsafe_allow_html=True,
)

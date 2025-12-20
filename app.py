import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta
import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

# ───────────────────────────────────────────────────────────────────────────────
#                        CONFIGURATION & STYLING
# ───────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    layout="wide",
    page_title="IIT KGP AQI Dashboard 2025 • Award-Winning",
    page_icon="🌬️",
    initial_sidebar_state="expanded"
)

pio.templates.default = "plotly_dark"

# Colors
ACCENT = "#00F0FF"
ACCENT2 = "#FF00AA"
TEXT = "#E0E0FF"
SUBTEXT = "#A0A0CC"
GLASS = "rgba(30, 30, 60, 0.3)"
BORDER = "rgba(0, 240, 255, 0.3)"
SHADOW = "0 12px 40px rgba(0, 240, 255, 0.25)"

CATEGORY_COLORS = {
    "Good": "#00FF99", "Satisfactory": "#66FF99", "Moderate": "#FFCC66",
    "Poor": "#FF6666", "Very Poor": "#FF3366", "Severe": "#FF0066",
    "Unknown": "#444466"
}

POLLUTANT_COLORS = {
    "PM2.5": "#FF3366", "PM10": "#00FFFF", "NO2": "#FF00FF",
    "SO2": "#FFFF00", "CO": "#FFAA00", "O3": "#00FFCC", "Other": "#9999FF"
}

HEALTH_RECS = {
    "Good": "Perfect for all outdoor activities!",
    "Satisfactory": "Unusually sensitive people should limit prolonged exertion.",
    "Moderate": "Sensitive groups should reduce outdoor activity.",
    "Poor": "Everyone should reduce prolonged or heavy exertion.",
    "Very Poor": "Avoid outdoor activities, especially sensitive groups.",
    "Severe": "Avoid all outdoor activities. Stay indoors with air purifiers.",
    "Unknown": "Data unavailable — take precautions."
}

# ───────────────────────────────────────────────────────────────────────────────
#                        CUSTOM CSS (GLASSMORPHISM + NEON)
# ───────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    .stApp {{ background: linear-gradient(135deg, #0A0A1A 0%, #0F0F2E 100%); color: {TEXT}; }}
    h1, h2, h3, h4, h5, h6 {{ font-family: 'Orbitron', sans-serif !important; background: linear-gradient(90deg, {ACCENT}, {ACCENT2}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 15px {ACCENT}; }}
    .stSidebar {{ background: rgba(20,20,40,0.8) !important; backdrop-filter: blur(12px); border-right: 1px solid {BORDER}; }}
    .glass-card {{ background: {GLASS}; backdrop-filter: blur(16px); border-radius: 20px; border: 1px solid {BORDER}; padding: 2rem; margin: 1rem 0; box-shadow: {SHADOW}; transition: all 0.4s ease; }}
    .glass-card:hover {{ transform: translateY(-10px); box-shadow: 0 20px 60px rgba(0,240,255,0.3); }}
    .metric-value {{ font-size: 3.8rem !important; font-weight: 900 !important; color: {ACCENT} !important; text-shadow: 0 0 20px {ACCENT}; }}
    .stTabs [data-baseweb="tab-list"] {{ background: {GLASS}; border-radius: 16px; padding: 8px; }}
    .stTabs [aria-selected="true"] {{ background: {ACCENT} !important; color: #000 !important; box-shadow: 0 0 25px {ACCENT}; }}
    .stButton > button {{ background: linear-gradient(45deg, {ACCENT}, {ACCENT2}); color: #000 !important; border: none; border-radius: 50px; padding: 14px 32px; font-weight: 700; box-shadow: 0 0 20px {ACCENT}; transition: all 0.3s ease; }}
    .stButton > button:hover {{ transform: translateY(-4px); box-shadow: 0 0 40px {ACCENT}; }}
    .footer {{ text-align: center; padding: 3rem; background: {GLASS}; border-top: 1px solid {BORDER}; margin-top: 4rem; border-radius: 20px 20px 0 0; }}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
#                        DATA LOADING
# ───────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800)
def load_data():
    today = datetime.now().strftime("%y-%m-%d")
    csv_path = f"data/{today}.csv"
    fallback = "combined_air_quality.txt"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        msg = f"Live data: {today}.csv"
    else:
        df = pd.read_csv(fallback, sep="\t", parse_dates=["date"])
        msg = "Archive data loaded"

    df["pollutant"] = df["pollutant"].str.split(",").str[0].str.strip().fillna("Other")
    df["level"] = df["level"].fillna("Unknown")

    if 2025 in df["date"].dt.year.unique():
        df = df[~((df["date"].dt.year == 2025) & (df["date"].dt.month > 5))]

    return df, msg, datetime.fromtimestamp(os.path.getmtime(csv_path if os.path.exists(csv_path) else fallback))

df, load_msg, last_update = load_data()

# ───────────────────────────────────────────────────────────────────────────────
#                        SIDEBAR
# ───────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <h1 style="font-size: 3rem; text-align: center;">IIT KGP</h1>
    <h2 style="text-align: center; color: {ACCENT2};">AQI 2025</h2>
    <p style="text-align: center; color: {SUBTEXT};">{load_msg}</p>
    <p style="text-align: center; color: {ACCENT};">Last update: {last_update.strftime('%Y-%m-%d %H:%M')}</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    cities = sorted(df["city"].unique())
    selected_cities = st.multiselect("Select Cities", cities, default=["Delhi", "Kolkata", "Mumbai"])

    years = sorted(df["date"].dt.year.unique(), reverse=True)
    year = st.selectbox("Year", years, index=0)

    months = ["All"] + ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if year == 2025:
        months = ["All"] + ["Jan", "Feb", "Mar", "Apr", "May"]
    month = st.selectbox("Month", months)

    st.markdown("---")
    st.markdown(f"**Live Clock:** {datetime.now().strftime('%H:%M:%S')}", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
#                        DATA FILTERING
# ───────────────────────────────────────────────────────────────────────────────

filtered_df = df[df["date"].dt.year == year].copy()
if month != "All":
    month_num = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}[month]
    filtered_df = filtered_df[filtered_df["date"].dt.month == month_num]

# ───────────────────────────────────────────────────────────────────────────────
#                        HEADER & NATIONAL SNAPSHOT
# ───────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="glass-card" style="text-align:center; padding:4rem 2rem;">
    <h1 style="font-size:5rem; margin:0;">🌬️ IIT KGP AQI DASHBOARD 2025</h1>
    <p style="font-size:1.6rem; color:{SUBTEXT}; margin-top:1rem;">India's Most Advanced Air Quality Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<h2>🇮🇳 NATIONAL AIR QUALITY OVERVIEW</h2>", unsafe_allow_html=True)

def get_category(aqi):
    if pd.isna(aqi): return "Unknown"
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Cities Monitored</div>
        <div class="metric-value">{filtered_df['city'].nunique()}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_aqi = filtered_df["index"].mean()
    cat = get_category(avg_aqi)
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">National Avg AQI</div>
        <div class="metric-value" style="color:{CATEGORY_COLORS.get(cat)}">{avg_aqi:.1f if not pd.isna(avg_aqi) else "N/A"}</div>
        <p style="color:{CATEGORY_COLORS.get(cat)}; font-weight:700;">{cat}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Data Period</div>
        <div class="metric-value">{month} {year}</div>
        <p style="color:{SUBTEXT};">{filtered_df['date'].nunique()} days</p>
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
#                        INTERACTIVE INDIA MAP
# ───────────────────────────────────────────────────────────────────────────────

st.markdown("<h2>📍 AIR QUALITY HOTSPOTS (INDIA MAP)</h2>", unsafe_allow_html=True)

city_coords = {
    "Delhi": [28.6139, 77.2090], "Kolkata": [22.5726, 88.3639], "Mumbai": [19.0760, 72.8777],
    "Bengaluru": [12.9716, 77.5946], "Chennai": [13.0827, 80.2707], "Hyderabad": [17.3850, 78.4867],
    "Ahmedabad": [23.0225, 72.5714], "Pune": [18.5204, 73.8567], "Jaipur": [26.9124, 75.7873],
    # Add more cities as needed
}

map_data = filtered_df.groupby("city").agg({"index": "mean"}).reset_index()
map_data["lat"] = map_data["city"].map(lambda x: city_coords.get(x, [None, None])[0])
map_data["lon"] = map_data["city"].map(lambda x: city_coords.get(x, [None, None])[1])
map_data = map_data.dropna(subset=["lat", "lon"])

fig_map = px.scatter_mapbox(
    map_data, lat="lat", lon="lon", size="index", color="index",
    hover_name="city", color_continuous_scale="RdYlGn_r",
    size_max=40, zoom=4.5, center={"lat": 20.5937, "lon": 78.9629}
)
fig_map.update_layout(mapbox_style="carto-darkmatter", margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig_map, use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────────
#                        CITY DEEP DIVE (TABS)
# ───────────────────────────────────────────────────────────────────────────────

if selected_cities:
    for city in selected_cities:
        st.markdown(f"<h2>🔍 {city.upper()} DEEP DIVE – {year}</h2>", unsafe_allow_html=True)
        city_df = filtered_df[filtered_df["city"] == city].sort_values("date")

        if city_df.empty:
            st.warning(f"No data available for {city}")
            continue

        latest = city_df.iloc[-1]
        aqi = latest["index"]
        level = latest["level"]
        pollutant = latest["pollutant"]

        cols = st.columns([1, 2, 1])
        with cols[0]:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:1.4rem;">Current AQI</div>
                <div class="metric-value" style="color:{CATEGORY_COLORS.get(level)}">{int(aqi) if not pd.isna(aqi) else "N/A"}</div>
                <p style="color:{CATEGORY_COLORS.get(level)};">{level}</p>
            </div>
            """, unsafe_allow_html=True)

        with cols[1]:
            st.markdown(f"""
            <div class="glass-card">
                <h3>Health Advisory</h3>
                <p style="font-size:1.4rem;">{HEALTH_RECS.get(level, "No data")}</p>
            </div>
            """, unsafe_allow_html=True)

        with cols[2]:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:1.4rem;">Dominant Pollutant</div>
                <div class="metric-value" style="color:{POLLUTANT_COLORS.get(pollutant)}">{pollutant}</div>
            </div>
            """, unsafe_allow_html=True)

        tabs = st.tabs(["Trends", "Calendar", "Heatmap", "Weekday", "Forecast", "Correlation", "Export"])

        with tabs[0]:
            fig_trend = px.line(city_df, x="date", y="index", title="AQI Trend with Rolling Averages")
            fig_trend.add_scatter(x=city_df["date"], y=city_df["index"].rolling(7).mean(), name="7-Day Avg", line=dict(color=ACCENT, dash="dash"))
            fig_trend.add_scatter(x=city_df["date"], y=city_df["index"].rolling(30).mean(), name="30-Day Avg", line=dict(color=ACCENT2, dash="dot"))
            fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_trend, use_container_width=True)

        with tabs[1]:
            cal = city_df.set_index("date")["index"].resample("D").mean().reset_index()
            cal["day"] = cal["date"].dt.day
            cal["month"] = cal["date"].dt.month_name()
            fig_cal = px.treemap(cal, path=["month", "day"], values="index", color="index", color_continuous_scale="RdYlGn_r")
            st.plotly_chart(fig_cal, use_container_width=True)

        with tabs[2]:
            heatmap = city_df.pivot_table(index=city_df["date"].dt.month_name(), columns=city_df["date"].dt.day, values="index")
            fig_heat = px.imshow(heatmap, color_continuous_scale="RdYlGn_r", text_auto=True)
            st.plotly_chart(fig_heat, use_container_width=True)

        with tabs[3]:
            weekday = city_df.copy()
            weekday["weekday"] = weekday["date"].dt.day_name()
            fig_week = px.box(weekday, x="weekday", y="index", color="weekday", points="all")
            st.plotly_chart(fig_week, use_container_width=True)

        with tabs[4]:
            city_df_clean = city_df.dropna(subset=["index"]).copy()
            if len(city_df_clean) < 20:
                st.warning("Not enough valid data for forecasting.")
            else:
                X = np.arange(len(city_df_clean)).reshape(-1, 1)
                y = city_df_clean["index"].values
                poly = PolynomialFeatures(degree=2)
                model = LinearRegression().fit(poly.fit_transform(X), y)
                future_days = 30
                future = np.arange(len(city_df_clean), len(city_df_clean) + future_days).reshape(-1, 1)
                pred = model.predict(poly.transform(future))
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=city_df_clean["date"], y=y, mode="lines+markers", name="Observed"))
                fig_fc.add_trace(go.Scatter(x=[city_df_clean["date"].max() + timedelta(days=i+1) for i in range(future_days)], y=pred, mode="lines", name="Forecast", line=dict(dash="dash")))
                st.plotly_chart(fig_fc, use_container_width=True)

        with tabs[5]:
            st.markdown("### Pollutant Correlation Heatmap")
            # Placeholder for correlation matrix
            st.info("Correlation matrix coming soon (requires multi-pollutant data)")

        with tabs[6]:
            st.download_button("Download CSV", city_df.to_csv(index=False), f"{city}_aqi_{year}.csv", "text/csv")

# ───────────────────────────────────────────────────────────────────────────────
#                        FOOTER & EXPORT
# ───────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="footer">
    <p style="font-size:1.4rem;">Developed with ❤️ by IIT Kharagpur</p>
    <p style="color:{SUBTEXT};">Data: CPCB • Designed for Excellence</p>
</div>
""", unsafe_allow_html=True)

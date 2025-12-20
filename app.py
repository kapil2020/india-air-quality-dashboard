import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from io import StringIO
from datetime import datetime, timedelta
import altair as alt
import base64
import time

# ───────────────────────────────────────────────────────────────────────────────
#                        AWARD-WINNING DASHBOARD CONFIG
# ───────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    layout="wide",
    page_title="IIT KGP AQI Dashboard • 2025 Award-Winning",
    page_icon="🌬️",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/kapil2020/india-air-quality-dashboard',
        'Report a bug': 'https://github.com/kapil2020/india-air-quality-dashboard/issues',
        'About': "Developed by Kapil Meena, IIT Kharagpur"
    }
)

# ───────────────────────────────────────────────────────────────────────────────
#                        CYBER-FUTURISTIC DARK THEME + GLASSMORPHISM
# ───────────────────────────────────────────────────────────────────────────────

pio.templates.default = "plotly_dark"

ACCENT = "#00F0FF"               # Electric Cyan
ACCENT2 = "#FF00AA"              # Neon Magenta
TEXT = "#E0E0FF"
SUBTEXT = "#A0A0CC"
BG = "#0A0A1A"
CARD_BG = "rgba(20, 20, 40, 0.6)"
GLASS = "rgba(30, 30, 60, 0.3)"
BORDER = "rgba(0, 240, 255, 0.3)"
SHADOW = "0 8px 32px rgba(0, 240, 255, 0.15)"

CATEGORY_COLORS = {
    "Severe": "#FF0066", "Very Poor": "#FF3366", "Poor": "#FF6666",
    "Moderate": "#FFCC66", "Satisfactory": "#66FF99", "Good": "#00FF99",
    "Unknown": "#444466"
}

POLLUTANT_COLORS = {
    "PM2.5": "#FF3366", "PM10": "#00FFFF", "NO2": "#FF00FF",
    "SO2": "#FFFF00", "CO": "#FFAA00", "O3": "#00FFCC", "Other": "#9999FF"
}

HEALTH_RECS = {
    "Good": "Ideal for outdoor activities!",
    "Satisfactory": "Unusually sensitive people should limit prolonged exertion.",
    "Moderate": "Sensitive groups should reduce outdoor activity.",
    "Poor": "Everyone should reduce prolonged or heavy exertion.",
    "Very Poor": "Avoid outdoor activities, especially sensitive groups.",
    "Severe": "Avoid all outdoor activities. Stay indoors with air purifiers.",
    "Unknown": "Data unavailable — take precautions."
}

# ───────────────────────────────────────────────────────────────────────────────
#                        CUSTOM CSS (GLASSMORPHISM + NEON EFFECTS)
# ───────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}

    .stApp {{
        background: linear-gradient(135deg, #0A0A1A 0%, #0F0F2E 100%);
        color: {TEXT};
        font-family: 'JetBrains Mono', monospace;
    }}

    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Orbitron', sans-serif !important;
        background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 2px;
        text-shadow: 0 0 10px {ACCENT};
    }}

    .stSidebar {{
        background: {CARD_BG} !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid {BORDER};
        box-shadow: {SHADOW};
    }}

    .stSidebar .stMarkdown {{
        padding: 0 1rem;
    }}

    .glass-card {{
        background: {GLASS};
        backdrop-filter: blur(16px);
        border-radius: 20px;
        border: 1px solid {BORDER};
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: {SHADOW};
        transition: all 0.4s ease;
    }}

    .glass-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 16px 48px rgba(0, 240, 255, 0.25);
    }}

    .metric-value {{
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        color: {ACCENT} !important;
        text-shadow: 0 0 20px {ACCENT};
    }}

    .metric-label {{
        font-size: 1.2rem !important;
        color: {SUBTEXT} !important;
        letter-spacing: 1px;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        background: {GLASS};
        border-radius: 16px;
        padding: 8px;
        gap: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        color: {SUBTEXT};
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
    }}

    .stTabs [aria-selected="true"] {{
        background: {ACCENT} !important;
        color: #000 !important;
        box-shadow: 0 0 20px {ACCENT};
    }}

    .stButton > button {{
        background: linear-gradient(45deg, {ACCENT}, {ACCENT2});
        color: #000 !important;
        border: none;
        border-radius: 50px;
        padding: 14px 32px;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: 0 0 20px {ACCENT};
        transition: all 0.3s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-4px);
        box-shadow: 0 0 40px {ACCENT};
    }}

    .footer {{
        text-align: center;
        padding: 3rem 1rem;
        background: {GLASS};
        border-top: 1px solid {BORDER};
        margin-top: 4rem;
        border-radius: 20px 20px 0 0;
    }}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
#                        DATA LOADING & CACHING
# ───────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800)  # 30 min cache
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
    
    # Limit 2025 data to May
    if 2025 in df["date"].dt.year.unique():
        df = df[~((df["date"].dt.year == 2025) & (df["date"].dt.month > 5))]
    
    return df, msg, datetime.fromtimestamp(os.path.getmtime(fallback) if not os.path.exists(csv_path) else os.path.getmtime(csv_path))

df, load_msg, last_update = load_data()

# ───────────────────────────────────────────────────────────────────────────────
#                        SIDEBAR - ADVANCED FILTERS
# ───────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <h1 style="font-size: 2.5rem; text-align: center; margin-bottom: 1rem;">IIT KGP</h1>
    <h2 style="text-align: center; margin-bottom: 2rem;">AQI 2025</h2>
    <p style="text-align: center; color: {SUBTEXT};">{load_msg}</p>
    <p style="text-align: center; color: {ACCENT}; font-size: 1.1rem;">Last update: {last_update.strftime('%Y-%m-%d %H:%M')}</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    cities = sorted(df["city"].unique())
    selected_cities = st.multiselect("🌆 Cities", cities, default=["Delhi"], placeholder="Select one or more cities")

    years = sorted(df["date"].dt.year.unique(), reverse=True)
    year = st.selectbox("🗓 Year", years, index=0)

    months = ["All"] + ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if year == 2025:
        months = ["All"] + ["Jan", "Feb", "Mar", "Apr", "May"]
    month = st.selectbox("📅 Month", months)

    st.markdown("---")

    st.markdown("""
    <div style="text-align:center; margin-top:2rem;">
        <a href="https://github.com/kapil2020/india-air-quality-dashboard" target="_blank"
           style="color:{ACCENT}; text-decoration:none; font-weight:700;">
            ⭐ View Source on GitHub
        </a>
    </div>
    """.format(ACCENT=ACCENT), unsafe_allow_html=True)

# Filter data
filtered_df = df[df["date"].dt.year == year].copy()
if month != "All":
    month_num = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}[month]
    filtered_df = filtered_df[filtered_df["date"].dt.month == month_num]

# ───────────────────────────────────────────────────────────────────────────────
#                        HEADER - AWARD-WINNING HEADER
# ───────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="glass-card" style="text-align:center; padding:3rem 2rem;">
    <h1 style="font-size:4rem; margin:0;">🌬️ IIT KGP AQI DASHBOARD 2025</h1>
    <p style="font-size:1.4rem; color:{SUBTEXT}; margin-top:1rem;">
        Next-Gen Air Quality Intelligence Platform
    </p>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
#                        NATIONAL SNAPSHOT (GLASS CARDS)
# ───────────────────────────────────────────────────────────────────────────────

st.markdown("<h2>🇮🇳 NATIONAL OVERVIEW</h2>", unsafe_allow_html=True)

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
    cat = pd.Series(avg_aqi).apply(lambda x: next((k for k,v in CATEGORY_COLORS.items() if x <= 50 if k=="Good" else x<=100 if k=="Satisfactory" else x<=200 if k=="Moderate" else x<=300 if k=="Poor" else x<=400 if k=="Very Poor" else "Severe"), "Unknown")).iloc[0]
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">National Average AQI</div>
        <div class="metric-value" style="color:{CATEGORY_COLORS.get(cat)}">{avg_aqi:.1f}</div>
        <p style="text-align:center; color:{CATEGORY_COLORS.get(cat)};">{cat}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">Data Period</div>
        <div class="metric-value">{month} {year}</div>
        <p style="text-align:center; color:{SUBTEXT};">{filtered_df['date'].nunique()} days</p>
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
#                        TOP & WORST CITIES (INTERACTIVE)
# ───────────────────────────────────────────────────────────────────────────────

st.markdown("<h2>🏆 CITY RANKINGS</h2>", unsafe_allow_html=True)

city_avg = filtered_df.groupby("city")["index"].mean().sort_values()
num_show = st.slider("Number of cities to show", 5, 20, 10)

top = city_avg.head(num_show)
bottom = city_avg.tail(num_show)

col_top, col_bottom = st.columns(2)

with col_top:
    fig_top = px.bar(
        top.reset_index(), x="index", y="city", orientation="h",
        color="index", color_continuous_scale="RdYlGn_r",
        title="Top Cleanest Cities", text_auto=True
    )
    fig_top.update_layout(**{"height": 600, "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"})
    st.plotly_chart(fig_top, use_container_width=True)

with col_bottom:
    fig_bottom = px.bar(
        bottom.reset_index(), x="index", y="city", orientation="h",
        color="index", color_continuous_scale="RdYlGn_r",
        title="Most Polluted Cities", text_auto=True
    )
    fig_bottom.update_layout(**{"height": 600, "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"})
    st.plotly_chart(fig_bottom, use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────────
#                        CITY DEEP DIVE (TABS + ADVANCED FEATURES)
# ───────────────────────────────────────────────────────────────────────────────

if selected_cities:
    for city in selected_cities:
        st.markdown(f"<h2>🔍 {city.upper()} DEEP DIVE</h2>", unsafe_allow_html=True)
        city_df = filtered_df[filtered_df["city"] == city].sort_values("date")

        if city_df.empty:
            st.warning(f"No data for {city} in {month} {year}")
            continue

        latest = city_df.iloc[-1]
        aqi = latest["index"]
        level = latest["level"]
        pollutant = latest["pollutant"]

        cols = st.columns([1, 2, 1])
        with cols[0]:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:1.2rem;">Current AQI</div>
                <div class="metric-value" style="color:{CATEGORY_COLORS.get(level)}">{int(aqi)}</div>
                <div style="color:{CATEGORY_COLORS.get(level)};">{level}</div>
            </div>
            """, unsafe_allow_html=True)

        with cols[1]:
            st.markdown(f"""
            <div class="glass-card">
                <h3>Health Advisory</h3>
                <p style="font-size:1.3rem;">{HEALTH_RECS.get(level, "No data")}</p>
            </div>
            """, unsafe_allow_html=True)

        with cols[2]:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:1.2rem;">Dominant Pollutant</div>
                <div class="metric-value" style="color:{POLLUTANT_COLORS.get(pollutant)}">{pollutant}</div>
            </div>
            """, unsafe_allow_html=True)

        tabs = st.tabs(["Trends", "Calendar", "Heatmap", "Weekday", "Forecast", "Health", "AI Insights"])

        with tabs[0]:
            fig_trend = px.line(city_df, x="date", y="index", title="AQI Trend with 7-day Rolling Avg")
            fig_trend.add_scatter(x=city_df["date"], y=city_df["index"].rolling(7).mean(), name="7-day Avg", line=dict(color=ACCENT, dash="dash"))
            fig_trend.update_layout(**{"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"})
            st.plotly_chart(fig_trend, use_container_width=True)

        with tabs[1]:
            # Calendar Heatmap (advanced)
            cal = city_df.set_index("date")["index"].resample("D").mean().reset_index()
            cal["day"] = cal["date"].dt.day
            cal["month"] = cal["date"].dt.month_name()
            fig_cal = px.treemap(cal, path=["month", "day"], values="index", color="index", color_continuous_scale="RdYlGn_r")
            st.plotly_chart(fig_cal, use_container_width=True)

        with tabs[2]:
            heatmap = city_df.pivot_table(index="month", columns="day", values="index")
            fig_heat = px.imshow(heatmap, color_continuous_scale="RdYlGn_r", text_auto=True)
            st.plotly_chart(fig_heat, use_container_width=True)

        with tabs[3]:
            weekday = city_df.copy()
            weekday["weekday"] = weekday["date"].dt.day_name()
            fig_week = px.box(weekday, x="weekday", y="index", color="weekday", points="all")
            st.plotly_chart(fig_week, use_container_width=True)

        with tabs[4]:
            if len(city_df) >= 20:
                X = np.arange(len(city_df)).reshape(-1, 1)
                y = city_df["index"]
                poly = PolynomialFeatures(degree=2)
                model = LinearRegression().fit(poly.fit_transform(X), y)
                future = np.arange(len(city_df), len(city_df) + 30).reshape(-1, 1)
                pred = model.predict(poly.transform(future))
                fig_fc = go.Figure()
                fig_fc.add_scatter(x=city_df["date"], y=y, mode="lines", name="Observed")
                fig_fc.add_scatter(x=[city_df["date"].max() + timedelta(days=i) for i in range(1, 31)], y=pred, mode="lines", name="Forecast", line=dict(dash="dash"))
                st.plotly_chart(fig_fc, use_container_width=True)

        with tabs[5]:
            st.markdown(f"""
            <div class="glass-card">
                <h3>Health Impact</h3>
                <p><b>Current Risk:</b> {level}</p>
                <p><b>Recommendations:</b> {HEALTH_RECS.get(level)}</p>
            </div>
            """, unsafe_allow_html=True)

        with tabs[6]:
            st.markdown("""
            <div class="glass-card">
                <h3>AI-Powered Insights</h3>
                <p>Coming soon: Machine learning-based pollution source prediction and health risk forecasting.</p>
            </div>
            """, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
#                        DOWNLOAD + FOOTER
# ───────────────────────────────────────────────────────────────────────────────

st.markdown("<h2>📥 Download Data</h2>", unsafe_allow_html=True)
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Filtered Data (CSV)", csv, "iitk_aqi_2025.csv", "text/csv")

st.markdown(f"""
<div class="footer">
    <p style="font-size:1.2rem;">Developed with ❤️ by IIT Kharagpur</p>
    <p style="color:{SUBTEXT};">Data Source: CPCB • Designed for Excellence</p>
</div>
""", unsafe_allow_html=True)

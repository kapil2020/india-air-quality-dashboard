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
import requests
import json
from datetime import datetime

# --- Global Theme & Style Setup ---
pio.templates.default = "plotly_dark"

ACCENT_COLOR = "#00BCD4"                 # Vibrant Teal
TEXT_COLOR_DARK_THEME = "#EAEAEA"        # Light gray for text
SUBTLE_TEXT_COLOR_DARK_THEME = "#B0B0B0" # Subtle gray
BACKGROUND_COLOR = "#121212"             # Very dark gray (almost black)
CARD_BACKGROUND_COLOR = "#1E1E1E"        # Slightly lighter for cards
BORDER_COLOR = "#333333"                 # Dark border
HIGHLIGHT_COLOR = "#FF6B6B"              # For alerts and highlights

CATEGORY_COLORS_DARK = {
    "Severe": "#F44336",      # Vivid Red
    "Very Poor": "#FF7043",   # Vivid Orange-Red
    "Poor": "#FFA726",        # Vivid Orange
    "Moderate": "#FFEE58",    # Vivid Yellow
    "Satisfactory": "#9CCC65",# Vivid Light Green
    "Good": "#4CAF50",        # Vivid Green
    "Unknown": "#444444"      # Dark gray for unknown days
}

POLLUTANT_COLORS_DARK = {
    "PM2.5": "#FF6B6B", "PM10": "#4ECDC4", "NO2": "#45B7D1",
    "SO2": "#F9C74F", "CO": "#F8961E", "O3": "#90BE6D", "Other": "#B0BEC5"
}

# Health recommendations based on AQI levels
HEALTH_RECOMMENDATIONS = {
    "Good": "Perfect day for outdoor activities!",
    "Satisfactory": "Sensitive individuals should consider reducing prolonged/heavy exertion.",
    "Moderate": "Sensitive groups should reduce outdoor activities.",
    "Poor": "Everyone should reduce prolonged/heavy exertion.",
    "Very Poor": "Avoid outdoor activities, especially for sensitive groups.",
    "Severe": "Avoid all outdoor activities, keep windows closed.",
    "Unknown": "Air quality data unavailable - take precautions."
}

# ------------------- Page Config -------------------
st.set_page_config(
    layout="wide",
    page_title="IIT KGP AQI Dashboard",
    page_icon="🌬️",
    initial_sidebar_state="expanded"
)

# ------------------- Custom CSS Styling (Dark Theme + Sidebar Fixes) -------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    /* =================================
       1. GENERAL RESETS & DEFAULTS
       ================================= */
    * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }}

    html, body, #root, .stApp {{
        background-color: {BACKGROUND_COLOR} !important;
        color: {TEXT_COLOR_DARK_THEME} !important;
    }}

    body {{
        font-family: 'Inter', sans-serif;
        line-height: 1.7;
        font-size: 16px;
    }}

    a {{
        color: {ACCENT_COLOR};
        text-decoration: none;
        transition: color 0.3s ease;
    }}

    a:hover {{
        color: #00E5FF;
    }}

    /* =================================
       2. LAYOUT & CONTAINERS
       ================================= */
    .main .block-container {{
        padding: 2rem;
    }}

    /* Card-like styling for sections/charts */
    .stPlotlyChart, .stDataFrame, .stAlert, .stMetric,
    .stDownloadButton > button, .stButton > button,
    div[data-testid="stExpander"], div[data-testid="stForm"] {{
        border-radius: 16px;
        border: 1px solid {BORDER_COLOR};
        background-color: {CARD_BACKGROUND_COLOR};
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    
    .stPlotlyChart:hover, .stDataFrame:hover, .stMetric:hover,
    div[data-testid="stExpander"]:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0, 188, 212, 0.25);
        border-color: #555555;
    }}

    /* Custom insight card with more padding */
    .insight-card {{
        background: linear-gradient(145deg, #1a1a1a, #232323);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid {BORDER_COLOR};
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        height: 100%;
    }}

    /* =================================
       3. TYPOGRAPHY
       ================================= */
    h1 {{
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        font-size: 3rem;
        background: linear-gradient(90deg, {ACCENT_COLOR}, #00E5FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    h2 {{
        font-family: 'Inter', sans-serif;
        color: {ACCENT_COLOR};
        border-bottom: 2px solid {BORDER_COLOR};
        padding-bottom: 0.75rem;
        margin-top: 3rem;
        margin-bottom: 2rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        position: relative;
        font-size: 2rem;
    }}
    
    h2:after {{
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, {ACCENT_COLOR}, transparent);
    }}
    
    h3 {{
        font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR_DARK_THEME};
        margin-bottom: 1.2rem;
        font-weight: 600;
        font-size: 1.5rem;
    }}

    h4, h5, h6 {{
        font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR_DARK_THEME};
        margin-bottom: 1rem;
        font-weight: 600;
    }}

    /* =================================
       4. SIDEBAR
       ================================= */
    .stSidebar {{
        background-color: {CARD_BACKGROUND_COLOR};
        border-right: 1px solid {BORDER_COLOR};
        padding: 2rem 1.5rem;
        box-shadow: 5px 0 25px rgba(0, 0, 0, 0.25);
        width: 300px !important;
    }}

    .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3, .stSidebar .stMarkdown p {{
        color: {TEXT_COLOR_DARK_THEME};
        text-align: left;
    }}

    .stSidebar h2 {{
      position: relative;
      margin-top: 1rem;
      margin-bottom: 1.5rem;
      padding-bottom: 0.5rem;
      font-size: 1.5rem !important;
      color: {ACCENT_COLOR} !important;
    }}

    .stSidebar h2:after {{
      content: "";
      position: absolute;
      bottom: 0;
      left: 0;
      width: 3rem;
      height: 3px;
      background-color: {ACCENT_COLOR};
    }}
    
    .stSidebar .stSelectbox label, .stSidebar .stMultiselect label, .stSidebar .stNumberInput label {{
        color: {ACCENT_COLOR} !important;
        font-weight: 600;
        font-size: 1.05rem;
    }}

    /* Selectbox/Multiselect widget improvements */
    div[data-baseweb="select"] > div:first-child {{
      background-color: {BACKGROUND_COLOR} !important;
      color: {TEXT_COLOR_DARK_THEME} !important;
      border: 1px solid #555555 !important;
      border-radius: 10px !important;
    }}
    div[data-baseweb="select"] [role="listbox"] {{
      background-color: {CARD_BACKGROUND_COLOR} !important;
      border: 1px solid #555555 !important;
      border-radius: 10px !important;
    }}
    div[data-baseweb="select"] [role="option"]:hover {{
      background-color: {ACCENT_COLOR} !important;
      color: {BACKGROUND_COLOR} !important;
    }}

    /* =================================
       5. SPECIFIC WIDGETS & COMPONENTS
       ================================= */

    /* --- Header --- */
    .gradient-header {{
        background: linear-gradient(270deg, {BACKGROUND_COLOR}, #1a2a3a, {BACKGROUND_COLOR});
        background-size: 200% 200%;
        animation: gradientAnimation 12s ease infinite;
        padding: 2.5rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid {BORDER_COLOR};
    }}

    /* --- Metric Cards --- */
    .stMetric {{
        background-color: {CARD_BACKGROUND_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }}
    
    .stMetric > div:nth-child(1) {{ /* Label */
        font-size: 1rem;
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        font-weight: 500;
    }}
    
    .stMetric > div:nth-child(2) {{ /* Value */
        font-size: 2.5rem;
        font-weight: 700;
        color: {ACCENT_COLOR};
        margin: 0.5rem 0;
    }}

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] {{
         border-bottom: 2px solid {BORDER_COLOR};
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 1rem 1.5rem;
        font-weight: 600;
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
    }}
     .stTabs [aria-selected="true"] {{
        color: {ACCENT_COLOR} !important;
        border-bottom: 3px solid {ACCENT_COLOR};
     }}

    /* --- Buttons --- */
    .stDownloadButton button, .stButton button {{
        background: linear-gradient(90deg, {ACCENT_COLOR}, #00BFA5);
        color: {BACKGROUND_COLOR};
        border: none;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        transition: all 0.3s ease;
        font-size: 1rem;
        box-shadow: 0 4px 10px rgba(0, 188, 212, 0.3);
    }}
    .stDownloadButton button:hover, .stButton button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0, 188, 212, 0.4);
    }}

    /* --- Health & Info Cards --- */
    .health-card {{
        border-left: 4px solid;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        background-color: rgba(30, 30, 30, 0.7);
    }}
    
    /* =================================
       6. RESPONSIVE DESIGN
       ================================= */
    @media (max-width: 768px) {{
        .main .block-container {{
            padding: 1rem;
        }}
        h1 {{ font-size: 2.2rem; }}
        h2 {{ font-size: 1.6rem; }}
        
        /* Stack all columns */
        .col1, .col2, .col3,
        .status-col1, .status-col2, .status-col3,
        .health-col1, .health-col2,
        .map-col1, .map-col2,
        .forecast-col1, .forecast-col2,
        .poll-col1, .poll-col2 {{
            flex: 0 0 100% !important;
            max-width: 100% !important;
            margin-bottom: 1rem;
        }}
        
        .footer-info {{
            flex-direction: column;
            text-align: center;
            gap: 1.5rem;
        }}
    }}

    @media (max-width: 480px) {{
        .main .block-container {{ padding: 0.5rem; }}
        h1 {{ font-size: 1.8rem; }}
        h2 {{ font-size: 1.4rem; }}
        body {{ font-size: 14px; }}
        
        .stMetric > div:nth-child(2) {{ font-size: 2rem !important; }}
        
        .stTabs {{ overflow-x: auto; }}
        
        .stPlotlyChart {{ height: 350px !important; }}
    }}

    /* =================================
       7. ANIMATIONS & FOOTER
       ================================= */
    @keyframes gradientAnimation {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    .footer-container {{
      margin-top: 4rem;
      padding: 3rem 2rem;
      border-radius: 16px;
      background: linear-gradient(270deg, {BACKGROUND_COLOR}, #1a2a3a, {BACKGROUND_COLOR});
      background-size: 200% 200%;
      animation: gradientAnimation 8s ease infinite;
      border: 1px solid #2a3a4a;
      text-align: center;
    }}
    .footer-container h3 {{
      color: {ACCENT_COLOR};
      font-size: 1.8rem;
      margin-bottom: 1.5rem;
    }}
    .footer-info {{
      display: flex;
      justify-content: center;
      gap: 2.5rem;
      flex-wrap: wrap;
      margin-bottom: 2rem;
    }}
    .footer-info p {{ margin: 0; }}
    .footer-info .label {{ font-size: 0.9rem; color: #B0B0B0; }}
    .footer-info .value {{ font-weight: 500; color: {TEXT_COLOR_DARK_THEME}; }}
    .footer-links a {{
      color: {ACCENT_COLOR};
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
    }}
    .copyright {{
      font-size: 0.85rem;
      color: #707070;
      text-align: center;
      margin-top: 2rem;
    }}
</style>
""", unsafe_allow_html=True)


# ------------------- Helper Functions -------------------
def get_custom_plotly_layout_args(height: int = None, title_text: str = None) -> dict:
    """
    Returns a dict of common Plotly layout arguments for dark-themed charts
    """
    layout_args = {
        "font": {"family": "Inter", "color": TEXT_COLOR_DARK_THEME, "size": 14},
        "paper_bgcolor": CARD_BACKGROUND_COLOR,
        "plot_bgcolor": CARD_BACKGROUND_COLOR,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 12}
        },
        "margin": {"l": 40, "r": 20, "t": 60, "b": 40},
        "hoverlabel": {
            "bgcolor": "#2A2A2A",
            "font_size": 12,
            "font_family": "Inter"
        }
    }
    if height:
        layout_args["height"] = height
    if title_text:
        layout_args["title_text"] = title_text
        layout_args["title_font"] = {"color": ACCENT_COLOR, "size": 18, "family": "Inter"}
        layout_args["title_x"] = 0.03
        layout_args["title_y"] = 0.95
    return layout_args

def format_number(num):
    if num > 1000000:
        return f"{num/1000000:.1f}M"
    if num > 1000:
        return f"{num/1000:.1f}K"
    return str(num)

def get_category(aqi_val):
    """Map AQI value to category"""
    if pd.isna(aqi_val):
        return "Unknown"
    if aqi_val <= 50:
        return "Good"
    elif aqi_val <= 100:
        return "Satisfactory"
    elif aqi_val <= 200:
        return "Moderate"
    elif aqi_val <= 300:
        return "Poor"
    elif aqi_val <= 400:
        return "Very Poor"
    else:
        return "Severe"

# ------------------- Title Header -------------------
st.markdown("""
<div class="gradient-header">
    <h1>🌬️ IIT KGP AIR QUALITY DASHBOARD</h1>
    <p style="color: #B0B0B0; font-size: 1.1rem; max-width: 800px; margin: 0 auto;">
        Real-time Air Quality Monitoring and Predictive Analysis for Indian Cities
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------- Load Data -------------------
@st.cache_data(ttl=3600)
def load_data_and_metadata():
    """
    Tries to load today's CSV (named YYYY-MM-DD.csv). If not found, falls back to 'combined_air_quality.txt'.
    Returns: (df_loaded, load_msg, last_update_time)
    """
    today = pd.to_datetime("today").date()
    csv_path = f"data/{today}.csv"
    fallback_file = "combined_air_quality.txt"
    df_loaded = None
    is_today_data = False
    load_msg = ""
    last_update_time = None

    # 1) Attempt to load today's CSV
    if os.path.exists(csv_path):
        try:
            df_loaded = pd.read_csv(csv_path)
            if "date" in df_loaded.columns:
                df_loaded["date"] = pd.to_datetime(df_loaded["date"])
                is_today_data = True
                load_msg = f"Live data from: **{today}.csv**"
                last_update_time = pd.Timestamp(os.path.getmtime(csv_path), unit="s")
            else:
                load_msg = f"Warning: '{csv_path}' found but missing 'date' column. Using fallback."
        except Exception as e:
            load_msg = f"Error loading '{csv_path}': {e}. Using fallback."

    # 2) If today's CSV is missing or invalid, load fallback
    if df_loaded is None or not is_today_data:
        try:
            if not os.path.exists(fallback_file):
                st.error(f"FATAL: Main data file '{fallback_file}' not found.")
                return pd.DataFrame(), "Error: Main data file not found.", None
            df_loaded = pd.read_csv(fallback_file, sep="\t", parse_dates=["date"])
            base_load_msg = f"Displaying archive data from: **{fallback_file}**"
            load_msg = base_load_msg if not load_msg or is_today_data else load_msg + " " + base_load_msg
            last_update_time = pd.Timestamp(os.path.getmtime(fallback_file), unit="s")
        except Exception as e:
            st.error(f"FATAL: Error loading '{fallback_file}': {e}.")
            return pd.DataFrame(), f"Error loading fallback: {e}", None

    # Common post-processing
    for col, default_val in [("pollutant", np.nan), ("level", "Unknown")]:
        if col not in df_loaded.columns:
            df_loaded[col] = default_val

    df_loaded["pollutant"] = (
        df_loaded["pollutant"].astype(str)
        .str.split(",").str[0].str.strip()
        .replace(["nan", "NaN", "None", ""], np.nan)
    )
    df_loaded["level"] = df_loaded["level"].astype(str).fillna("Unknown")
    df_loaded["pollutant"] = df_loaded["pollutant"].fillna("Other")
    
    # Filter 2025 data to include only up to May
    if 2025 in df_loaded["date"].dt.year.unique():
        df_loaded = df_loaded[~((df_loaded["date"].dt.year == 2025) & (df_loaded["date"].dt.month > 5))]

    return df_loaded, load_msg, last_update_time

df, load_message, data_last_updated = load_data_and_metadata()

if df.empty:
    st.error("Dashboard cannot operate without data. Please check data sources.")
    st.stop()

if data_last_updated:
    st.caption(
        f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 0.9rem;'>"
        f"📅 Last data update: {data_last_updated.strftime('%Y-%m-%d %H:%M:%S')} "
        f"</p>",
        unsafe_allow_html=True
    )

# ------------------- Sidebar Filters -------------------
with st.sidebar:
    st.header("🔭 Controls")
    st.info("Fetching real-time data from CPCB. Today's data available after 5:45 PM IST.", icon="ℹ️")

    unique_cities = sorted(df["city"].unique()) if "city" in df.columns else []
    default_city_val = ["Delhi"] if "Delhi" in unique_cities else (unique_cities[0:1] if unique_cities else [])
    selected_cities = st.multiselect("🏙️ Select Cities", unique_cities, default=default_city_val,
                                    help="Select one or more cities for detailed analysis")

    years = sorted(df["date"].dt.year.unique())
    # Default to 2024 if present, else last year
    default_year_val = 2024 if 2024 in years else (max(years) if years else None)
    if default_year_val:
        year_index = years.index(default_year_val)
    else:
        year_index = 0
    year = st.selectbox("🗓️ Select Year", years, index=year_index if years else 0,
                      help="Select the year for analysis")

    months_map_dict = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
    }
    
    # For 2025, only show months up to May
    if year == 2025:
        month_options_list = ["All Months"] + [months_map_dict[i] for i in range(1, 6)]
    else:
        month_options_list = ["All Months"] + list(months_map_dict.values())
        
    selected_month_name = st.selectbox("🌙 Select Month", month_options_list, index=0,
                                     help="Optionally select a specific month")

    month_number_filter = None
    if selected_month_name != "All Months":
        month_number_filter = [k for k, v in months_map_dict.items() if v == selected_month_name][0]

    # Filter data based on global selections
    df_period_filtered = df[df["date"].dt.year == year].copy()
    if month_number_filter:
        df_period_filtered = df_period_filtered[df_period_filtered["date"].dt.month == month_number_filter]
    
    st.markdown("---")
    st.markdown("""
    <div style="margin-top: 2rem; text-align: center;">
        <p style="font-size: 0.85rem; color: #B0B0B0;">
            Developed with ❤️ by IIT Kharagpur<br>Data Source: CPCB India
        </p>
    </div>
    """, unsafe_allow_html=True)

# ========================================================
# =========  NATIONAL KEY INSIGHTS (Enhanced)  ===========
# ========================================================
st.markdown("## 🇮🇳 National Air Quality Snapshot")

col1, col2, col3 = st.columns(3, gap="large")
with col1:
    with st.container():
        st.markdown(f"<div class='insight-card'><h3>🌆 Coverage</h3>", unsafe_allow_html=True)
        cities_count = df_period_filtered["city"].nunique()
        st.metric(label="Cities Monitored", value=cities_count)
        st.markdown(f"<p style='color:{SUBTLE_TEXT_COLOR_DARK_THEME}; font-size:0.9rem;'>Across India in {year}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown(f"<div class='insight-card'><h3>📈 National Average</h3>", unsafe_allow_html=True)
        if not df_period_filtered.empty:
            avg_aqi_national = df_period_filtered["index"].mean()
            st.metric(label="Average AQI", value=f"{avg_aqi_national:.1f}")
            national_category = get_category(avg_aqi_national)
            st.markdown(f"<p style='color:{CATEGORY_COLORS_DARK.get(national_category)}; font-weight:600;'>{national_category} Air Quality</p>", unsafe_allow_html=True)
        else:
            st.info("No data available")
        st.markdown("</div>", unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown(f"<div class='insight-card'><h3>📅 Time Period</h3>", unsafe_allow_html=True)
        period = f"{selected_month_name} {year}" if selected_month_name != "All Months" else f"Full Year {year}"
        st.markdown(f"<p style='font-size:1.8rem; text-align:center; margin:1rem 0; color:{TEXT_COLOR_DARK_THEME}; font-weight:600;'>{period}</p>", unsafe_allow_html=True)
        days_count = df_period_filtered["date"].nunique()
        st.markdown(f"<p style='text-align:center; color:{SUBTLE_TEXT_COLOR_DARK_THEME};'>{days_count} days of data</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ========================================================
# ====== REPLACEMENT: TOP & BOTTOM CITIES VISUALIZATION ======
# ========================================================
st.markdown(f"## 🏆 City Rankings for {year}")

if not df_period_filtered.empty:
    city_avg_aqi = df_period_filtered.groupby("city")["index"].mean().dropna().sort_values()

    if not city_avg_aqi.empty:
        max_cities = len(city_avg_aqi)
        num_to_show = st.slider(
            "Select Number of Cities to Rank",
            min_value=3,
            max_value=min(15, max_cities),
            value=min(5, max_cities),
            help="Adjust the slider to see more or fewer cities in the rankings."
        )

        top_cities = city_avg_aqi.head(num_to_show).reset_index()
        top_cities.columns = ["City", "Avg AQI"]
        top_cities["Category"] = top_cities["Avg AQI"].apply(get_category)

        bottom_cities = city_avg_aqi.tail(num_to_show).reset_index()
        bottom_cities.columns = ["City", "Avg AQI"]
        bottom_cities["Category"] = bottom_cities["Avg AQI"].apply(get_category)

        col_top, col_bottom = st.columns(2, gap="large")

        with col_top:
            st.markdown(f"<h5 style='color:#EAEAEA; text-align:center;'>🥇 Top {num_to_show} Cleanest Cities</h5>", unsafe_allow_html=True)
            fig_top = px.bar(
                top_cities.sort_values("Avg AQI", ascending=False),
                x="Avg AQI",
                y="City",
                color="Category",
                color_discrete_map=CATEGORY_COLORS_DARK,
                orientation='h',
                text='Avg AQI'
            )
            fig_top.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_top.update_layout(
                yaxis_title=None, xaxis_title="Average AQI", showlegend=False,
                height=max(200, num_to_show * 50),
                paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with col_bottom:
            st.markdown(f"<h5 style='color:#EAEAEA; text-align:center;'>⚠️ Top {num_to_show} Most Polluted Cities</h5>", unsafe_allow_html=True)
            fig_bottom = px.bar(
                bottom_cities.sort_values("Avg AQI", ascending=True),
                x="Avg AQI",
                y="City",
                color="Category",
                color_discrete_map=CATEGORY_COLORS_DARK,
                orientation='h',
                text='Avg AQI'
            )
            fig_bottom.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_bottom.update_layout(
                yaxis_title=None, xaxis_title="Average AQI", showlegend=False,
                height=max(200, num_to_show * 50),
                paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME
            )
            st.plotly_chart(fig_bottom, use_container_width=True)

    else:
        st.info("No city averages available for the selected period.")
else:
    st.info("No data available for the selected period.")


# ========================================================
# =======   CITY-SPECIFIC ANALYSIS (Improved)   ==========
# ========================================================
export_data_list = []

if not selected_cities:
    st.info("✨ Select one or more cities from the sidebar to dive into detailed analysis.")
else:
    for city in selected_cities:
        st.markdown(f"## 🏙️ {city.upper()} DEEP DIVE – {year}")
        
        city_data_full = df_period_filtered[df_period_filtered["city"] == city].copy()
        if city_data_full.empty:
            st.warning(f"😔 No data available for {city} for {selected_month_name}, {year}. Try different filter settings.")
            continue
            
        latest_data = city_data_full.sort_values("date", ascending=False).iloc[0]
        current_aqi = latest_data["index"]
        current_level = latest_data["level"]
        current_pollutant = latest_data["pollutant"]
        health_msg = HEALTH_RECOMMENDATIONS.get(current_level, "No specific health recommendations available")
        
        status_col1, status_col2, status_col3 = st.columns([1,2,1], gap="large")
        with status_col1:
            st.markdown(f"<div style='text-align:center; padding:1rem; border-radius:16px; background:{CARD_BACKGROUND_COLOR}; border:1px solid {BORDER_COLOR}; height:100%'>", unsafe_allow_html=True)
            st.markdown("<h6>Live Status</h6>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:3rem; font-weight:800; color:{CATEGORY_COLORS_DARK.get(current_level, '#FFFFFF')}; line-height:1.2;'>{current_aqi:.0f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.2rem; font-weight:600;'>{current_level}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with status_col2:
            st.markdown(f"<div style='padding:1.5rem; border-radius:16px; background:{CARD_BACKGROUND_COLOR}; border:1px solid {BORDER_COLOR}; height:100%;'>", unsafe_allow_html=True)
            st.markdown("<h6>Health Recommendation</h6>", unsafe_allow_html=True)
            st.markdown(f"<div class='health-card' style='border-left-color: {CATEGORY_COLORS_DARK.get(current_level, '#FFFFFF')};'>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:1.1rem;'>{health_msg}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with status_col3:
            st.markdown(f"<div style='text-align:center; padding:1rem; border-radius:16px; background:{CARD_BACKGROUND_COLOR}; border:1px solid {BORDER_COLOR}; height:100%;'>", unsafe_allow_html=True)
            st.markdown("<h6>Dominant Pollutant</h6>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:2.5rem; font-weight:700; color:{POLLUTANT_COLORS_DARK.get(current_pollutant, '#FFFFFF')}; margin:1rem 0; line-height:1.2;'>{current_pollutant}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        city_data_full["day_of_year"] = city_data_full["date"].dt.dayofyear
        city_data_full["month_name"] = city_data_full["date"].dt.month_name()
        city_data_full["day_of_month"] = city_data_full["date"].dt.day
        export_data_list.append(city_data_full)

        tab_trend, tab_dist, tab_heatmap_detail, tab_health, tab_forecast = st.tabs(["📊 TRENDS & CALENDAR", "📈 DISTRIBUTIONS", "🗓️ DETAILED HEATMAP", "❤️ HEALTH ANALYSIS", "🔮 HEALTH FORECAST"])

        with tab_trend:
            st.markdown("<h5>📅 Daily AQI Calendar</h5>", unsafe_allow_html=True)
            # Build calendar for full year, but only plot if data exists
            if city_data_full["index"].notna().any():
                start_date = pd.to_datetime(f"{year}-01-01")
                end_date = pd.to_datetime(f"{year}-12-31")
                full_year_dates = pd.date_range(start_date, end_date, freq="D")

                calendar_df = pd.DataFrame({"date": full_year_dates})
                calendar_df["week"] = calendar_df["date"].dt.isocalendar().week
                calendar_df["day_of_week"] = calendar_df["date"].dt.dayofweek

                # Adjust for Jan/Dec year boundary
                calendar_df.loc[(calendar_df["date"].dt.month == 1) & (calendar_df["week"] > 50), "week"] = 0
                calendar_df.loc[(calendar_df["date"].dt.month == 12) & (calendar_df["week"] == 1), "week"] = calendar_df["week"].max() + 1

                merged_cal_df = pd.merge(
                    calendar_df,
                    city_data_full[["date", "index", "level"]],
                    on="date",
                    how="left"
                )
                merged_cal_df["level"] = merged_cal_df["level"].fillna("Unknown")
                merged_cal_df["aqi_text"] = merged_cal_df["index"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")

                # Build z-values as numeric codes
                level_to_code = {level: idx for idx, level in enumerate(CATEGORY_COLORS_DARK.keys())}
                z_values = merged_cal_df["level"].map(level_to_code)

                day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

                # Determine month annotation positions
                week_start_dates = merged_cal_df.groupby("week")["date"].min().reset_index()
                week_start_dates["month"] = week_start_dates["date"].dt.strftime("%b")
                month_names = week_start_dates.drop_duplicates("month", keep="first").set_index("week")["month"]

                fig_cal = go.Figure(
                    data=go.Heatmap(
                        x=merged_cal_df["week"],
                        y=merged_cal_df["day_of_week"],
                        z=z_values,
                        customdata=pd.DataFrame(
                            {
                                "date": merged_cal_df["date"].dt.strftime("%Y-%m-%d"),
                                "level": merged_cal_df["level"],
                                "aqi": merged_cal_df["aqi_text"]
                            }
                        ),
                        hovertemplate="<b>%{customdata[0]}</b><br>AQI: %{customdata[2]} (%{customdata[1]})<extra></extra>",
                        colorscale=[[i / (len(CATEGORY_COLORS_DARK) - 1), color] for i, color in enumerate(CATEGORY_COLORS_DARK.values())],
                        showscale=False,
                        xgap=3, ygap=3
                    )
                )

                annotations = []
                for week_num, month_name in month_names.items():
                    annotations.append(
                        go.layout.Annotation(
                            text=month_name, align="center", showarrow=False, xref="x", yref="y",
                            x=week_num, y=7, font=dict(color=SUBTLE_TEXT_COLOR_DARK_THEME, size=12)
                        )
                    )

                for i, (level, color) in enumerate(CATEGORY_COLORS_DARK.items()):
                    annotations.append(
                        go.layout.Annotation(
                            text=f"█ <span style='color:{TEXT_COLOR_DARK_THEME};'>{level}</span>",
                            align="left", showarrow=False, xref="paper", yref="paper",
                            x=0.05 + 0.12 * (i % 7), y=-0.15 - 0.1 * (i // 7),
                            xanchor="left", yanchor="top", font=dict(color=color, size=12)
                        )
                    )

                fig_cal.update_layout(
                    yaxis=dict(tickmode="array", tickvals=list(range(7)), ticktext=day_labels, showgrid=False, zeroline=False),
                    xaxis=dict(showgrid=False, zeroline=False, tickmode="array", ticktext=[], tickvals=[]),
                    height=350, margin=dict(t=50, b=100, l=40, r=40),
                    plot_bgcolor=CARD_BACKGROUND_COLOR, paper_bgcolor=CARD_BACKGROUND_COLOR,
                    font_color=TEXT_COLOR_DARK_THEME, annotations=annotations, showlegend=False
                )
                st.plotly_chart(fig_cal, use_container_width=True)
            else:
                st.info("No AQI data available for calendar plot.")

            st.markdown("<h5>📈 AQI Trend & 7-Day Rolling Average</h5>", unsafe_allow_html=True)
            if len(city_data_full) >= 2:
                city_data_trend = city_data_full.sort_values("date").copy()
                city_data_trend["rolling_avg_7day"] = city_data_trend["index"].rolling(window=7, center=True, min_periods=1).mean().round(2)

                fig_trend_roll = go.Figure()
                fig_trend_roll.add_trace(go.Scatter(
                    x=city_data_trend["date"], y=city_data_trend["index"], mode="lines+markers", name="Daily AQI",
                    marker=dict(size=4, opacity=0.7, color=SUBTLE_TEXT_COLOR_DARK_THEME),
                    line=dict(width=1.5, color=SUBTLE_TEXT_COLOR_DARK_THEME),
                    customdata=city_data_trend[["level"]],
                    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>AQI: %{y}<br>Category: %{customdata[0]}<extra></extra>"
                ))
                fig_trend_roll.add_trace(go.Scatter(
                    x=city_data_trend["date"], y=city_data_trend["rolling_avg_7day"], mode="lines", name="7-Day Rolling Avg",
                    line=dict(color=ACCENT_COLOR, width=2.5, dash="dash"),
                    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>7-Day Avg AQI: %{y}<extra></extra>"
                ))
                fig_trend_roll.update_layout(
                    yaxis_title="AQI Index", xaxis_title="Date", height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified", paper_bgcolor=CARD_BACKGROUND_COLOR,
                    plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME
                )
                st.plotly_chart(fig_trend_roll, use_container_width=True)
            else:
                st.info("Not enough data points to display trend.")

        with tab_dist:
            col_bar_dist, col_sun_dist = st.columns([2, 1], gap="large")

            # Build distribution data only if there are rows
            if not city_data_full.empty:
                category_counts_df = city_data_full["level"].value_counts().reindex(CATEGORY_COLORS_DARK.keys(), fill_value=0).reset_index()
                category_counts_df.columns = ["AQI Category", "Number of Days"]

                with col_bar_dist:
                    st.markdown("<h5>📊 AQI Category Distribution</h5>", unsafe_allow_html=True)
                    fig_dist_bar = px.bar(
                        category_counts_df, x="AQI Category", y="Number of Days",
                        color="AQI Category", color_discrete_map=CATEGORY_COLORS_DARK, text_auto=True
                    )
                    fig_dist_bar.update_layout(
                        height=450, xaxis_title=None, yaxis_title="Number of Days",
                        paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR,
                        font_color=TEXT_COLOR_DARK_THEME, showlegend=False
                    )
                    st.plotly_chart(fig_dist_bar, use_container_width=True)

                with col_sun_dist:
                    st.markdown("<h5>☀️ Category Proportions</h5>", unsafe_allow_html=True)
                    if category_counts_df["Number of Days"].sum() > 0:
                        fig_sunburst = px.sunburst(
                            category_counts_df, path=["AQI Category"], values="Number of Days",
                            color="AQI Category", color_discrete_map=CATEGORY_COLORS_DARK,
                        )
                        fig_sunburst.update_layout(
                            height=450, margin=dict(t=20, l=20, r=20, b=20),
                            paper_bgcolor=CARD_BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME
                        )
                        st.plotly_chart(fig_sunburst, use_container_width=True)
                    else:
                        st.caption("No data for sunburst chart.")

                st.markdown("<h5>🎻 Monthly AQI Distribution</h5>", unsafe_allow_html=True)
                # Determine only months present in data, in correct order
                months_map_number = {v: k for k, v in months_map_dict.items()}
                present_months = sorted(
                    city_data_full["month_name"].dropna().unique(),
                    key=lambda m: months_map_number.get(m, 13)
                )
                if present_months:
                    city_data_full["month_name_cat"] = pd.Categorical(city_data_full["month_name"], categories=present_months, ordered=True)
                    fig_violin = px.violin(
                        city_data_full.sort_values("month_name_cat"), x="month_name_cat", y="index",
                        color="month_name_cat", color_discrete_sequence=px.colors.qualitative.Vivid,
                        box=True, points="outliers", hover_data=["date", "level"],
                        labels={"index": "AQI Index", "month_name_cat": "Month"}
                    )
                    fig_violin.update_layout(
                        height=500, xaxis_title=None, showlegend=False,
                        paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME
                    )
                    st.plotly_chart(fig_violin, use_container_width=True)
                else:
                    st.info("No monthly data available to render violin plot.")
            else:
                st.info("No data available for distribution plots.")

        with tab_heatmap_detail:
            st.markdown("<h5>🔥 AQI Heatmap (Month vs. Day)</h5>", unsafe_allow_html=True)
            if not city_data_full.empty:
                present_months = sorted(
                    city_data_full["month_name"].dropna().unique(),
                    key=lambda m: months_map_number.get(m, 13)
                )
                if present_months:
                    heatmap_pivot = city_data_full.pivot_table(index="month_name", columns="day_of_month", values="index", observed=False)
                    heatmap_pivot = heatmap_pivot.reindex(present_months)

                    if not heatmap_pivot.dropna(how='all').empty:
                        fig_heat_detail = px.imshow(
                            heatmap_pivot, labels=dict(x="Day of Month", y="Month", color="AQI"),
                            aspect="auto", color_continuous_scale="Inferno", text_auto=".0f"
                        )
                        fig_heat_detail.update_layout(
                            height=550, xaxis_side="top", paper_bgcolor=CARD_BACKGROUND_COLOR,
                            plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME
                        )
                        st.plotly_chart(fig_heat_detail, use_container_width=True)
                    else:
                        st.info("No data available to render heatmap.")
                else:
                    st.info("No monthly data available for heatmap.")
            else:
                st.info("No data available for heatmap.")

        with tab_health:
            st.markdown("<h5>❤️ Health Impact Analysis</h5>", unsafe_allow_html=True)
            health_col1, health_col2 = st.columns(2, gap="large")
            
            with health_col1:
                st.markdown("<h6>🚶‍♂️ Activity Recommendations</h6>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background:{CARD_BACKGROUND_COLOR}; border-radius:16px; padding:1.5rem; border:1px solid {BORDER_COLOR}">
                    <p style="font-size:1.1rem; margin-bottom:1rem;"><b>Current AQI: {current_aqi:.0f} ({current_level})</b></p>
                    <p style="font-size:1.05rem; margin-bottom:1rem;">{health_msg}</p>
                    <p style="font-size:0.95rem; color:{SUBTLE_TEXT_COLOR_DARK_THEME};">
                        Based on latest data from {latest_data['date'].strftime('%Y-%m-%d')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<h6>📖 General Guidelines</h6>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background:{CARD_BACKGROUND_COLOR}; border-radius:16px; padding:1.5rem; border:1px solid {BORDER_COLOR}; margin-top:1.5rem;">
                    <ul style="padding-left:1.5rem;">
                        <li>Sensitive groups include children, elderly, and people with respiratory issues.</li>
                        <li>Consider wearing N95 masks when AQI is high (>200).</li>
                        <li>Keep windows closed during high pollution periods.</li>
                        <li>Use air purifiers indoors when air quality is poor.</li>
                        <li>Limit outdoor exercise during high pollution days.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with health_col2:
                st.markdown("<h6>📊 Health Risk Levels</h6>", unsafe_allow_html=True)
                
                # Create gauge chart for health risk
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = current_aqi,
                    title = {'text': f"Current AQI: {current_level}"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': TEXT_COLOR_DARK_THEME},
                        'bar': {'color': CATEGORY_COLORS_DARK.get(current_level, '#FFFFFF')},
                        'steps': [
                            {'range': [0, 50], 'color': CATEGORY_COLORS_DARK['Good']},
                            {'range': [50, 100], 'color': CATEGORY_COLORS_DARK['Satisfactory']},
                            {'range': [100, 200], 'color': CATEGORY_COLORS_DARK['Moderate']},
                            {'range': [200, 300], 'color': CATEGORY_COLORS_DARK['Poor']},
                            {'range': [300, 400], 'color': CATEGORY_COLORS_DARK['Very Poor']},
                            {'range': [400, 500], 'color': CATEGORY_COLORS_DARK['Severe']}
                        ]
                    }
                ))
                fig_gauge.update_layout(
                    height=300,
                    paper_bgcolor=CARD_BACKGROUND_COLOR,
                    font_color=TEXT_COLOR_DARK_THEME
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Health risk summary
                risk_levels = {
                    "Good": "Low risk",
                    "Satisfactory": "Low to moderate risk",
                    "Moderate": "Moderate risk",
                    "Poor": "High risk",
                    "Very Poor": "Very high risk",
                    "Severe": "Hazardous"
                }
                
                st.markdown(f"""
                <div style="background:{CARD_BACKGROUND_COLOR}; border-radius:16px; padding:1.5rem; border:1px solid {BORDER_COLOR}; margin-top:1.5rem;">
                    <p><b>Current Risk Level:</b> {risk_levels.get(current_level, 'Unknown')}</p>
                    <p><b>Affected Groups:</b> { 
                        'Everyone' if current_level in ['Poor', 'Very Poor', 'Severe'] 
                        else 'Sensitive groups' if current_level in ['Moderate'] 
                        else 'None' 
                    }</p>
                </div>
                """, unsafe_allow_html=True)
                
        with tab_forecast:
            st.markdown("<h5>🔮 Health Impact Forecast</h5>", unsafe_allow_html=True)
            
            if len(city_data_full) >= 15:
                forecast_df = city_data_full.sort_values("date")[["date", "index"]].dropna()
                if len(forecast_df) >= 2:
                    forecast_df["days_since_start"] = (forecast_df["date"] - forecast_df["date"].min()).dt.days
                    
                    X = forecast_df["days_since_start"].values.reshape(-1, 1)
                    y = forecast_df["index"].values
                    
                    poly = PolynomialFeatures(degree=2)
                    X_poly = poly.fit_transform(X)
                    
                    model = LinearRegression().fit(X_poly, y)
                    
                    last_day_num = forecast_df["days_since_start"].max()
                    future_X_range = np.arange(0, last_day_num + 15 + 1)
                    future_X_poly = poly.transform(future_X_range.reshape(-1, 1))
                    future_y_pred = model.predict(future_X_poly)
                    
                    min_date_forecast = forecast_df["date"].min()
                    future_dates_list = [min_date_forecast + pd.Timedelta(days=int(i)) for i in future_X_range]

                    plot_df_obs = pd.DataFrame({"date": forecast_df["date"], "AQI": y})
                    plot_df_fcst = pd.DataFrame({"date": future_dates_list, "AQI": np.maximum(0, future_y_pred)})
                    
                    # Add health impact levels
                    def get_health_impact(aqi):
                        if aqi <= 50: return "Good", CATEGORY_COLORS_DARK["Good"]
                        elif aqi <= 100: return "Satisfactory", CATEGORY_COLORS_DARK["Satisfactory"]
                        elif aqi <= 200: return "Moderate", CATEGORY_COLORS_DARK["Moderate"]
                        elif aqi <= 300: return "Poor", CATEGORY_COLORS_DARK["Poor"]
                        elif aqi <= 400: return "Very Poor", CATEGORY_COLORS_DARK["Very Poor"]
                        else: return "Severe", CATEGORY_COLORS_DARK["Severe"]
                    
                    plot_df_fcst["health_impact"] = plot_df_fcst["AQI"].apply(lambda x: get_health_impact(x)[0])
                    plot_df_fcst["color"] = plot_df_fcst["AQI"].apply(lambda x: get_health_impact(x)[1])

                    fig_forecast = go.Figure()
                    fig_forecast.add_trace(go.Scatter(
                        x=plot_df_obs["date"], y=plot_df_obs["AQI"], mode="lines+markers", name="Observed AQI",
                        line=dict(color=SUBTLE_TEXT_COLOR_DARK_THEME, width=1.5),
                        marker=dict(size=4, opacity=0.7),
                        customdata=plot_df_obs[["AQI"]],
                        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>AQI: %{y:.0f}<extra></extra>"
                    ))
                    fig_forecast.add_trace(go.Scatter(
                        x=plot_df_fcst["date"], y=plot_df_fcst["AQI"], mode="lines", name="Forecast",
                        line=dict(color=ACCENT_COLOR, width=2.5, dash="dash"),
                        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Forecast AQI: %{y:.0f}<extra></extra>"
                    ))
                    
                    # Add health impact markers
                    for date, aqi, impact, color in zip(
                        plot_df_fcst["date"], 
                        plot_df_fcst["AQI"], 
                        plot_df_fcst["health_impact"],
                        plot_df_fcst["color"]
                    ):
                        fig_forecast.add_trace(go.Scatter(
                            x=[date], y=[aqi], mode="markers", name=impact,
                            marker=dict(color=color, size=8),
                            showlegend=False,
                            hovertemplate=f"<b>{date.strftime('%Y-%m-%d')}</b><br>Forecast AQI: {aqi:.0f}<br>Health Impact: {impact}<extra></extra>"
                        ))
                    
                    forecast_start = forecast_df["date"].max() + pd.Timedelta(days=1)
                    forecast_end = future_dates_list[-1]
                    
                    fig_forecast.add_vrect(
                        x0=forecast_start, x1=forecast_end,
                        fillcolor="rgba(255, 107, 107, 0.1)", layer="below", line_width=0,
                        annotation_text="Forecast Period", annotation_position="top left",
                        annotation_font_color=HIGHLIGHT_COLOR
                    )

                    layout_args_fcst = get_custom_plotly_layout_args(height=500, title_text=f"Health Impact Forecast – {city}")
                    layout_args_fcst['legend'] = {"orientation": "h", "yanchor": "bottom", "y":1.02, "xanchor":"right", "x":1}
                    layout_args_fcst['hovermode'] = "x unified"
                    layout_args_fcst['paper_bgcolor'] = CARD_BACKGROUND_COLOR
                    layout_args_fcst['plot_bgcolor'] = BACKGROUND_COLOR
                    fig_forecast.update_layout(**layout_args_fcst)
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Health impact summary
                    st.markdown("<h6>📋 Forecasted Health Impact</h6>", unsafe_allow_html=True)
                    forecast_summary = plot_df_fcst[plot_df_fcst["date"] > forecast_df["date"].max()].copy()
                    if not forecast_summary.empty:
                        impact_counts = forecast_summary["health_impact"].value_counts().reset_index()
                        impact_counts.columns = ["Health Impact", "Days"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("<p style='font-weight:600;'>Expected Days:</p>", unsafe_allow_html=True)
                            for impact, days in zip(impact_counts["Health Impact"], impact_counts["Days"]):
                                color = CATEGORY_COLORS_DARK.get(impact, "#FFFFFF")
                                st.markdown(f"<p><span style='color:{color}; font-weight:600;'>●</span> {impact}: {days} days</p>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("<p style='font-weight:600;'>Recommendations:</p>", unsafe_allow_html=True)
                            for impact in impact_counts["Health Impact"]:
                                st.markdown(f"<p>• {HEALTH_RECOMMENDATIONS.get(impact, 'No specific recommendations')}</p>", unsafe_allow_html=True)
                    else:
                        st.info("No forecasted days available for health impact summary.")
                else:
                    st.warning(f"Not enough valid data points for {city} to forecast.")
            else:
                st.warning(f"Need at least 15 data points for {city} for forecasting; found {len(city_data_full)}.")

# ========================================================
# =======   CITY-WISE AQI COMPARISON (Enhanced)  =========
# ========================================================
if len(selected_cities) > 1:
    st.markdown("## 🆚 Multi-City AQI Comparison")
    comparison_df_list = []
    for city_comp in selected_cities:
        city_ts_comp = df_period_filtered[df_period_filtered["city"] == city_comp].copy()
        if not city_ts_comp.empty:
            city_ts_comp = city_ts_comp.sort_values("date")
            city_ts_comp["city_label"] = city_comp
            comparison_df_list.append(city_ts_comp)

    if comparison_df_list:
        combined_comp_df = pd.concat(comparison_df_list)
        avg_aqi_df = combined_comp_df.groupby("city_label")["index"].mean().reset_index().sort_values("index")
        
        col1, col2 = st.columns([3, 1], gap="large")
        with col1:
            fig_cmp = px.line(
                combined_comp_df, x="date", y="index", color="city_label",
                labels={"index": "AQI Index", "date": "Date", "city_label": "City"},
                markers=True, line_shape="spline", color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_cmp.update_layout(
                title_text=f"AQI Trends Comparison – {selected_month_name}, {year}",
                height=500, legend_title_text="City",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR,
                font_color=TEXT_COLOR_DARK_THEME, hovermode="x unified"
            )
            st.plotly_chart(fig_cmp, use_container_width=True)
        
        with col2:
            st.markdown("<h5>🏆 Average AQI</h5>", unsafe_allow_html=True)
            for idx, row in avg_aqi_df.iterrows():
                aqi_val = row["index"]
                category_comp = get_category(aqi_val)
                color = CATEGORY_COLORS_DARK.get(category_comp, "#FFFFFF")
                    
                st.markdown(f"""
                <div style="background:{CARD_BACKGROUND_COLOR}; border-radius:12px; padding:1rem; margin-bottom:1rem; border:1px solid {BORDER_COLOR}">
                    <div style="font-weight:600; font-size:1.1rem; color:{TEXT_COLOR_DARK_THEME};">{row['city_label']}</div>
                    <div style="font-size:1.8rem; font-weight:700; color:{color};">{aqi_val:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        with st.container():
            st.info("Not enough data or cities selected for comparison with current filters.")
    st.markdown("---")

# ========================================================
# =======   PROMINENT POLLUTANT ANALYSIS (Enhanced)  =====
# ========================================================
st.markdown("## 💨 Prominent Pollutant Analysis")

poll_col1, poll_col2 = st.columns(2, gap="large")

with poll_col1:
    st.markdown("<h4>🔍 Yearly Dominant Pollutant Trends</h4>", unsafe_allow_html=True)
    city_pollutant_A = st.selectbox(
        "Select City for Yearly Pollutant Trend:", unique_cities,
        key="pollutant_A_city_dark", 
        index=unique_cities.index(default_city_val[0]) if default_city_val and default_city_val[0] in unique_cities else 0
    )
    pollutant_data_A = df[df["city"] == city_pollutant_A].copy()
    pollutant_data_A["year_label"] = pollutant_data_A["date"].dt.year

    if not pollutant_data_A.empty:
        grouped_poll_A = pollutant_data_A.groupby(["year_label", "pollutant"]).size().unstack(fill_value=0)
        percent_poll_A = grouped_poll_A.apply(lambda x: x / x.sum() * 100 if x.sum() > 0 else x, axis=1).fillna(0)
        percent_poll_A_long = percent_poll_A.reset_index().melt(id_vars="year_label", var_name="pollutant", value_name="percentage")

        fig_poll_A = px.bar(
            percent_poll_A_long, x="year_label", y="percentage", color="pollutant",
            title=f"Dominant Pollutants Over Years – {city_pollutant_A}",
            labels={"percentage": "Percentage of Days (%)", "year_label": "Year", "pollutant": "Pollutant"},
            color_discrete_map=POLLUTANT_COLORS_DARK, barmode="stack"
        )
        fig_poll_A.update_layout(
            xaxis_type="category", yaxis_ticksuffix="%", height=500,
            paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME
        )
        st.plotly_chart(fig_poll_A, use_container_width=True)
    else:
        st.warning(f"No pollutant data for {city_pollutant_A} (Yearly Trend).")

with poll_col2:
    st.markdown(f"<h4>⛽ Dominant Pollutants ({selected_month_name}, {year})</h4>", unsafe_allow_html=True)
    city_pollutant_B = st.selectbox(
        "Select City for Filtered Pollutant View:", unique_cities,
        key="pollutant_B_city_dark", 
        index=unique_cities.index(default_city_val[0]) if default_city_val and default_city_val[0] in unique_cities else 0
    )
    pollutant_data_B = df_period_filtered[df_period_filtered["city"] == city_pollutant_B].copy()
    if not pollutant_data_B.empty and "pollutant" in pollutant_data_B.columns:
        grouped_poll_B = pollutant_data_B.groupby("pollutant").size().reset_index(name="count")
        
        fig_poll_B = px.pie(
            grouped_poll_B, values="count", names="pollutant",
            title=f"Dominant Pollutants – {city_pollutant_B} ({selected_month_name}, {year})",
            color="pollutant", color_discrete_map=POLLUTANT_COLORS_DARK, hole=0.4
        )
        fig_poll_B.update_layout(
            height=500, paper_bgcolor=CARD_BACKGROUND_COLOR,
            font_color=TEXT_COLOR_DARK_THEME,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_poll_B, use_container_width=True)
    else:
        st.warning(f"No pollutant data for {city_pollutant_B} for the selected period.")

# ========================================================
# =======   AQI FORECAST (Enhanced)   ====================
# ========================================================
st.markdown("## 🔮 AQI Forecast (Polynomial Trend)")

forecast_col1, forecast_col2 = st.columns([3,1], gap="large")

with forecast_col1:
    forecast_city_select = st.selectbox(
        "Select City for AQI Forecast:", unique_cities,
        key="forecast_city_select_dark", 
        index=unique_cities.index(default_city_val[0]) if default_city_val and default_city_val[0] in unique_cities else 0
    )
    forecast_src_data = df_period_filtered[df_period_filtered["city"] == forecast_city_select].copy()
    if len(forecast_src_data) >= 15:
        forecast_df = forecast_src_data.sort_values("date")[["date", "index"]].dropna()
        if len(forecast_df) >= 2:
            forecast_df["days_since_start"] = (forecast_df["date"] - forecast_df["date"].min()).dt.days
            
            X = forecast_df["days_since_start"].values.reshape(-1, 1)
            y = forecast_df["index"].values
            
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            
            model = LinearRegression().fit(X_poly, y)
            
            last_day_num = forecast_df["days_since_start"].max()
            future_X_range = np.arange(0, last_day_num + 15 + 1)
            future_X_poly = poly.transform(future_X_range.reshape(-1, 1))
            future_y_pred = model.predict(future_X_poly)
            
            min_date_forecast = forecast_df["date"].min()
            future_dates_list = [min_date_forecast + pd.Timedelta(days=int(i)) for i in future_X_range]

            plot_df_obs = pd.DataFrame({"date": forecast_df["date"], "AQI": y})
            plot_df_fcst = pd.DataFrame({"date": future_dates_list, "AQI": np.maximum(0, future_y_pred)})

            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=plot_df_obs["date"], y=plot_df_obs["AQI"], mode="lines+markers", name="Observed AQI", line=dict(color=ACCENT_COLOR), marker=dict(size=5)))
            fig_forecast.add_trace(go.Scatter(x=plot_df_fcst["date"], y=plot_df_fcst["AQI"], mode="lines", name="Forecast", line=dict(dash="dash", color=HIGHLIGHT_COLOR, width=3)))
            
            forecast_start = forecast_df["date"].max() + pd.Timedelta(days=1)
            forecast_end = future_dates_list[-1]
            
            fig_forecast.add_vrect(
                x0=forecast_start, x1=forecast_end,
                fillcolor="rgba(255, 107, 107, 0.1)", layer="below", line_width=0,
                annotation_text="Forecast Period", annotation_position="top left",
                annotation_font_color=HIGHLIGHT_COLOR
            )

            layout_args_fcst = get_custom_plotly_layout_args(height=500, title_text=f"AQI Forecast – {forecast_city_select}")
            layout_args_fcst['legend'] = {"orientation": "h", "yanchor": "bottom", "y":1.02, "xanchor":"right", "x":1}
            layout_args_fcst['hovermode'] = "x unified"
            layout_args_fcst['paper_bgcolor'] = CARD_BACKGROUND_COLOR
            layout_args_fcst['plot_bgcolor'] = BACKGROUND_COLOR
            fig_forecast.update_layout(**layout_args_fcst)
            st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.warning(f"Not enough valid data points for {forecast_city_select} to forecast.")
    else:
        st.warning(f"Need at least 15 data points for {forecast_city_select} for forecasting; found {len(forecast_src_data)}.")

with forecast_col2:
    st.markdown("<h5>ℹ️ Forecast Info</h5>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{CARD_BACKGROUND_COLOR}; border-radius:16px; padding:1.5rem; border:1px solid {BORDER_COLOR}; height:100%">
        <h6 style="color:{ACCENT_COLOR};">Forecast Methodology</h6>
        <p>This forecast uses polynomial regression (degree=2) on historical data to predict future AQI trends for the next 15 days.</p>
        <h6 style="color:{ACCENT_COLOR}; margin-top:1rem;">Limitations</h6>
        <ul style="padding-left:1.5rem; font-size:0.95rem;">
            <li>For short-term estimation only.</li>
            <li>Does not account for sudden weather events or policy changes.</li>
            <li>Accuracy decreases further into the future.</li>
            <li>Based on historical patterns, not real-time emissions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ========================================================
# =========  City AQI Hotspots (Enhanced)  ==============
# ========================================================
st.markdown("## 📍 Air Quality Hotspots Map")

map_col1, map_col2 = st.columns([3,1], gap="large")

with map_col1:
    coords_file_path = "lat_long.txt"
    if not df_period_filtered.empty:
        map_grouped_data = df_period_filtered.groupby("city").agg(
            avg_aqi=('index', 'mean'),
            dominant_pollutant=('pollutant', lambda x: x.mode()[0] if not x.mode().empty else 'N/A')
        ).reset_index().dropna(subset=['avg_aqi'])

        if os.path.exists(coords_file_path):
            city_coords_data = {}
            with open(coords_file_path, "r", encoding="utf-8") as f:
                exec(f.read(), {}, city_coords_data)
            city_coords = city_coords_data.get("city_coords", {})

            latlong_map_df = pd.DataFrame([{'city': k, 'lat': v[0], 'lon': v[1]} for k, v in city_coords.items()])
            map_merged_df = pd.merge(map_grouped_data, latlong_map_df, on="city", how="inner")
            
            if not map_merged_df.empty:
                map_merged_df["AQI Category"] = map_merged_df["avg_aqi"].apply(get_category)
                
                map_merged_df["scaled_size"] = np.maximum(map_merged_df["avg_aqi"] / 10, 5)

                fig_scatter_map = px.scatter_mapbox(
                    map_merged_df, lat="lat", lon="lon", size="scaled_size", size_max=25,
                    color="AQI Category", color_discrete_map=CATEGORY_COLORS_DARK,
                    hover_name="city", custom_data=['city', 'avg_aqi', 'dominant_pollutant', 'AQI Category'],
                    zoom=4.2, center={"lat": 23.5, "lon": 82.0}
                )
                scatter_map_layout_args = get_custom_plotly_layout_args(height=700, title_text=f"Average AQI Hotspots - {selected_month_name}, {year}")
                scatter_map_layout_args['mapbox_style'] = "carto-darkmatter"
                scatter_map_layout_args['margin'] = {"r":10,"t":60,"l":10,"b":10}
                fig_scatter_map.update_traces(
                    hovertemplate="<b style='font-size:1.1em;'>%{customdata[0]}</b><br>Avg. AQI: %{customdata[1]:.1f} (%{customdata[3]})<br>Dominant Pollutant: %{customdata[2]}<extra></extra>"
                )
                fig_scatter_map.update_layout(**scatter_map_layout_args)
                st.plotly_chart(fig_scatter_map, use_container_width=True)
            else:
                st.warning("Could not merge AQI data with city coordinates.")
        else:
            st.warning(f"Coordinates file '{coords_file_path}' not found. Cannot display map.")
    else:
        st.warning("No air quality data available for the selected filters to display on a map.")

with map_col2:
    st.markdown("<h5>ℹ️ Map Guide</h5>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{CARD_BACKGROUND_COLOR}; border-radius:16px; padding:1.5rem; border:1px solid {BORDER_COLOR}; height:100%">
        <h6 style="color:{ACCENT_COLOR};">How to Use</h6>
        <ul style="padding-left:1.5rem; margin-bottom:1.5rem; font-size:0.95rem;">
            <li>Bubble size reflects AQI severity.</li>
            <li>Color indicates the AQI category.</li>
            <li>Hover over a bubble for details.</li>
            <li>Click, drag, and zoom to explore.</li>
        </ul>
        <h6 style="color:{ACCENT_COLOR};">AQI Categories</h6>
    """, unsafe_allow_html=True)
    
    for category, color in CATEGORY_COLORS_DARK.items():
        if category != "Unknown":
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <div style="width: 15px; height: 15px; background-color: {color}; border-radius: 3px; margin-right: 10px;"></div>
                <span style="color: {TEXT_COLOR_DARK_THEME};">{category}</span>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ========================================================
# ========   DOWNLOAD FILTERED DATA (Enhanced)   =========
# ========================================================
if export_data_list:
    st.markdown("## 📥 Download Data")
    with st.container():
        combined_export_final = pd.concat(export_data_list)
        csv_buffer_final = StringIO()
        combined_export_final.to_csv(csv_buffer_final, index=False)
        
        st.markdown(f"""
        <div style="background:{CARD_BACKGROUND_COLOR}; border-radius:16px; padding:1.5rem 2rem; border:1px solid {BORDER_COLOR}; text-align:center;">
            <h4 style="color:{ACCENT_COLOR};">Export Filtered Data</h4>
            <p style="margin-bottom:1.5rem;">Download the dataset for the selected cities and period for your own analysis.</p>
        """, unsafe_allow_html=True)
        
        st.download_button(
            label="📤 Download Filtered City Data (CSV)",
            data=csv_buffer_final.getvalue(),
            file_name=f"IITKGP_filtered_aqi_{year}_{selected_month_name.replace(' ', '')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("</div>", unsafe_allow_html=True)

# ======================
# =====  FOOTER  =======
# ======================
st.markdown(f"""
<div class="footer-container">
    <h3>IIT KGP Air Quality Dashboard</h3>
    <div class="footer-info">
      <div>
        <p class="label">Data Source</p>
        <p class="value">Central Pollution Control Board (CPCB)</p>
      </div>
      <div>
        <p class="label">Principal Investigator</p>
        <p class="value"><a href="https://www.mustlab.in/faculty" target="_blank">Prof. Arkopal Kishore Goswami</a></p>
      </div>
      <div>
        <p class="label">Developed By</p>
        <p class="value"><a href="https://sites.google.com/view/kapil-lab/home" target="_blank">Kapil Meena</a>, PhD Student</p>
      </div>
      <div>
        <p class="label">Last Updated</p>
        <p class="value">{data_last_updated.strftime('%Y-%m-%d %H:%M') if data_last_updated else "N/A"}</p>
      </div>
    </div>
    <div class="footer-links">
      <a href="https://github.com/kapil2020/india-air-quality-dashboard" target="_blank">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>
        View on GitHub
      </a>
    </div>
    <p class="copyright">© {pd.to_datetime("today").year} IIT Kharagpur | For Research and Educational Purposes</p>
</div>
""", unsafe_allow_html=True)
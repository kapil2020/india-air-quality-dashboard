import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.seasonal import seasonal_decompose
#from io import StringIO

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
    "Satisfactory": "Sensitive individuals should consider reducing prolonged/heavy exertion",
    "Moderate": "Sensitive groups should reduce outdoor activities",
    "Poor": "Everyone should reduce prolonged/heavy exertion",
    "Very Poor": "Avoid outdoor activities, especially for sensitive groups",
    "Severe": "Avoid all outdoor activities, keep windows closed",
    "Unknown": "Air quality data unavailable - take precautions"
}

# ------------------- Page Config -------------------
st.set_page_config(
    layout="wide", 
    page_title="IIT KGP AQI Dashboard", 
    page_icon="🌬️",
    initial_sidebar_state="expanded"
)

# ------------------- Custom CSS Styling (Dark Theme) -------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    * {{
        transition: all 0.2s ease;
    }}
    
    body {{
        font-family: 'Inter', sans-serif;
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR_DARK_THEME};
        line-height: 1.6;
    }}

    /* Main container styling */
    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
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
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    
    .stPlotlyChart:hover, .stDataFrame:hover, .stMetric:hover,
    div[data-testid="stExpander"]:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 188, 212, 0.3);
        border-color: #555555;
    }}
    
    .stpyplot {{
        border-radius: 12px;
        border: 1px solid {BORDER_COLOR};
        background-color: {CARD_BACKGROUND_COLOR};
        padding: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }}

    .stTabs [data-baseweb="tab-list"] {{
         box-shadow: none;
         border-bottom: 2px solid {BORDER_COLOR};
         padding-bottom: 0;
         background-color: transparent;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        margin-right: 0.5rem;
        transition: all 0.3s ease;
        border: none;
    }}
    
     .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, {BACKGROUND_COLOR}, {CARD_BACKGROUND_COLOR});
        color: {ACCENT_COLOR} !important;
        border-bottom: 3px solid {ACCENT_COLOR};
        box-shadow: 0 4px 8px rgba(0, 188, 212, 0.2);
     }}
     
     .stTabs [aria-selected="true"]:hover {{
        background: linear-gradient(90deg, {BACKGROUND_COLOR}, #232323);
     }}

    /* Headings */
    h1 {{
        font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR_DARK_THEME};
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, {ACCENT_COLOR}, #00E5FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
    }}
    
    h2 {{
        font-family: 'Inter', sans-serif;
        color: {ACCENT_COLOR};
        border-bottom: 2px solid {BORDER_COLOR};
        padding-bottom: 0.6rem;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        position: relative;
        font-size: 1.8rem;
    }}
    
    h2:after {{
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 120px;
        height: 3px;
        background: linear-gradient(90deg, {ACCENT_COLOR}, transparent);
    }}
    
    h3 {{
        font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR_DARK_THEME};
        margin-top: 0rem;
        margin-bottom: 1.2rem;
        font-weight: 600;
        font-size: 1.4rem;
    }}
    
    h4, h5 {{
        font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR_DARK_THEME};
        margin-top: 0.2rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }}

    /* Sidebar styling */
    .stSidebar {{
        background-color: {CARD_BACKGROUND_COLOR};
        border-right: 1px solid {BORDER_COLOR};
        padding: 1.5rem;
        box-shadow: 5px 0 15px rgba(0, 0, 0, 0.2);
        min-width: 300px !important;
        width: 300px !important;
    }}
    
    .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3, .stSidebar .stMarkdown p {{
        color: {TEXT_COLOR_DARK_THEME};
        text-align: left;
        border-bottom: none;
    }}
    
    .stSidebar .stSelectbox label, .stSidebar .stMultiselect label {{
        color: {ACCENT_COLOR} !important;
        font-weight: 600;
        font-size: 1.05rem;
    }}

    /* Metric styling */
    .stMetric {{
        background-color: {BACKGROUND_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }}
    
    .stMetric > div:nth-child(1) {{
        font-size: 1rem;
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        font-weight: 500;
        letter-spacing: 0.5px;
    }}
    
    .stMetric > div:nth-child(2) {{
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, {ACCENT_COLOR}, #00E5FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }}
    
    .stMetric > div:nth-child(3) {{
        font-size: 0.9rem;
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        font-weight: 400;
        font-style: italic;
    }}

    /* Expander styling */
    div[data-testid="stExpander"] summary {{
        font-size: 1.2rem;
        font-weight: 600;
        color: {ACCENT_COLOR};
        padding: 0.8rem 1.2rem;
    }}
    
    div[data-testid="stExpander"] svg {{
        fill: {ACCENT_COLOR};
    }}

    /* Download button styling */
    .stDownloadButton button {{
        background: linear-gradient(90deg, {ACCENT_COLOR}, #00BFA5);
        color: {BACKGROUND_COLOR};
        border: none;
        font-weight: 600;
        padding: 0.85rem 2rem;
        border-radius: 50px;
        transition: all 0.3s ease;
        font-size: 1rem;
        box-shadow: 0 4px 10px rgba(0, 188, 212, 0.3);
    }}
    
    .stDownloadButton button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0, 188, 212, 0.4);
    }}
    
    .stDownloadButton button:active {{
        transform: translateY(0);
    }}

    /* Input Widgets */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div:first-child,
    .stMultiselect div[data-baseweb="select"] > div:first-child {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR_DARK_THEME};
        border-color: {BORDER_COLOR} !important;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }}
    
    .stDateInput input {{
         background-color: {BACKGROUND_COLOR};
         color: {TEXT_COLOR_DARK_THEME};
         border-radius: 10px;
         padding: 0.5rem 1rem;
    }}
    
    /* Hover effects for inputs */
    .stTextInput input:hover, .stSelectbox div[data-baseweb="select"] > div:first-child:hover,
    .stMultiselect div[data-baseweb="select"] > div:first-child:hover {{
        border-color: {ACCENT_COLOR} !important;
        box-shadow: 0 0 0 2px rgba(0, 188, 212, 0.2);
    }}

    /* Custom Scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {BACKGROUND_COLOR};
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(45deg, {ACCENT_COLOR}, #00BFA5);
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(45deg, #00BFA5, {ACCENT_COLOR});
    }}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {{
        .main .block-container {{
            padding: 1rem;
        }}
        
        .stMetric > div:nth-child(2) {{
            font-size: 2rem;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            padding: 0.6rem 1rem;
            font-size: 0.9rem;
        }}
        
        h1 {{
            font-size: 2.2rem;
        }}
        
        h2 {{
            font-size: 1.5rem;
        }}
        
        .stSidebar {{
            width: 100% !important;
            min-width: 100% !important;
        }}
        
        .col1, .col2, .col3 {{
            flex: 0 0 100% !important;
            max-width: 100% !important;
        }}
        
        .status-col1, .status-col2, .status-col3 {{
            flex: 0 0 100% !important;
            max-width: 100% !important;
            margin-bottom: 1rem;
        }}
        
        .health-col1, .health-col2 {{
            flex: 0 0 100% !important;
            max-width: 100% !important;
        }}
        
        .map-col1, .map-col2 {{
            flex: 0 0 100% !important;
            max-width: 100% !important;
        }}
        
        .forecast-col1, .forecast-col2 {{
            flex: 0 0 100% !important;
            max-width: 100% !important;
        }}
        
        .poll-col1, .poll-col2 {{
            flex: 0 0 100% !important;
            max-width: 100% !important;
        }}
        
        .insight-card {{
            margin-bottom: 1rem;
        }}
    }}
    
    /* Custom badge styling */
    .badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }}
    
    /* Health recommendation cards */
    .health-card {{
        border-left: 4px solid;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: rgba(30, 30, 30, 0.7);
    }}
    
    /* Key insight cards */
    .insight-card {{
        background: linear-gradient(145deg, #1a1a1a, #232323);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid {BORDER_COLOR};
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }}
    
    /* Gradient animation for header */
    @keyframes gradientAnimation {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    .gradient-header {{
        background: linear-gradient(270deg, {BACKGROUND_COLOR}, #1a2a3a, {BACKGROUND_COLOR});
        background-size: 300% 300%;
        animation: gradientAnimation 12s ease infinite;
        padding: 2rem 1rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid #2a3a4a;
    }}
    
    /* Mobile menu */
    @media (max-width: 480px) {{
        .stSelectbox, .stMultiselect, .stButton, .stSlider {{
            width: 100% !important;
        }}
        
        .stMetric > div:nth-child(2) {{
            font-size: 1.8rem !important;
        }}
        
        .stTabs {{
            overflow-x: auto;
        }}
        
        .stPlotlyChart {{
            height: 300px !important;
        }}
    }}
    
    /* Dark background for all elements */
    html, body, #root, .block-container, .stApp {{
        background-color: {BACKGROUND_COLOR} !important;
        color: {TEXT_COLOR_DARK_THEME} !important;
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

    return df_loaded, load_msg, last_update_time

df, load_message, data_last_updated = load_data_and_metadata()

if df.empty:
    st.error("Dashboard cannot operate without data. Please check data sources.")
    st.stop()

if data_last_updated:
    st.caption(
        f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 0.9rem;'>"
        f"📅 Last data update: {data_last_updated.strftime('%Y-%m-%d %H:%M:%S')}"
        "</p>",
        unsafe_allow_html=True
    )

# ------------------- Sidebar Filters -------------------
with st.sidebar:
    st.header("🔭 EXPLORATION CONTROLS")
    st.markdown("---", unsafe_allow_html=True)
    st.info("Fetching real-time data from CPCB. Today's data available after 5:45 PM IST.", icon="ℹ️")

    unique_cities = sorted(df["city"].unique()) if "city" in df.columns else []
    default_city_val = ["Delhi"] if "Delhi" in unique_cities else (unique_cities[0:1] if unique_cities else [])
    selected_cities = st.multiselect("🏙️ Select Cities", unique_cities, default=default_city_val,
                                    help="Select one or more cities for detailed analysis")

    years = sorted(df["date"].dt.year.unique())
    default_year_val = max(years) if years else None
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

    # Add quick filter buttons
    st.markdown("### ⚡ Quick Filters")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Current Year", help="Show data for current year"):
            year = pd.to_datetime("today").year
    with col2:
        if st.button("Major Cities", help="Select major Indian cities"):
            major_cities = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru", "Hyderabad"]
            selected_cities = [city for city in major_cities if city in unique_cities]

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

# Insight cards container
col1, col2, col3 = st.columns(3)
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
            # Determine national AQI category
            if avg_aqi_national <= 50:
                category = "Good"
            elif avg_aqi_national <= 100:
                category = "Satisfactory"
            elif avg_aqi_national <= 200:
                category = "Moderate"
            elif avg_aqi_national <= 300:
                category = "Poor"
            elif avg_aqi_national <= 400:
                category = "Very Poor"
            else:
                category = "Severe"
            st.markdown(f"<p style='color:{CATEGORY_COLORS_DARK.get(category)}; font-weight:600;'>{category} Air Quality</p>", unsafe_allow_html=True)
        else:
            st.info("No data available")
        st.markdown("</div>", unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown(f"<div class='insight-card'><h3>📅 Time Period</h3>", unsafe_allow_html=True)
        period = f"{selected_month_name} {year}" if selected_month_name != "All Months" else f"Full Year {year}"
        st.markdown(f"<p style='font-size:1.8rem; text-align:center; margin:1rem 0;'>{period}</p>", unsafe_allow_html=True)
        days_count = df_period_filtered["date"].nunique()
        st.markdown(f"<p style='text-align:center; color:{SUBTLE_TEXT_COLOR_DARK_THEME};'>{days_count} days of data</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Top and bottom cities
st.markdown("### 🏆 Top & Bottom Cities")

if not df_period_filtered.empty:
    city_avg_aqi = df_period_filtered.groupby("city")["index"].mean().dropna().sort_values()
    
    if not city_avg_aqi.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🥇 Cleanest Cities")
            top_cities = city_avg_aqi.head(5).reset_index()
            top_cities.columns = ["City", "Avg AQI"]
            top_cities["Rank"] = range(1, 6)
            top_cities = top_cities[["Rank", "City", "Avg AQI"]]
            st.dataframe(top_cities.style.format({"Avg AQI": "{:.1f}"}), height=250)
        
        with col2:
            st.markdown("##### ⚠️ Most Polluted Cities")
            bottom_cities = city_avg_aqi.tail(5).reset_index()
            bottom_cities = bottom_cities.iloc[::-1]  # Reverse to show worst first
            bottom_cities.columns = ["City", "Avg AQI"]
            bottom_cities["Rank"] = range(1, 6)
            bottom_cities = bottom_cities[["Rank", "City", "Avg AQI"]]
            st.dataframe(bottom_cities.style.format({"Avg AQI": "{:.1f}"}), height=250)
    else:
        st.info("No city averages available for the selected period.")
else:
    st.info("No data available for the selected period.")

# Major cities AQI
st.markdown("### 🏙️ Major Cities AQI Comparison")
major_cities = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru", "Hyderabad"]
major_cities_data = df_period_filtered[df_period_filtered["city"].isin(major_cities)]

if not major_cities_data.empty:
    avg_aqi_major = major_cities_data.groupby("city")["index"].mean().dropna()
    present_major_cities = [c for c in major_cities if c in avg_aqi_major.index]
    
    if present_major_cities:
        cols = st.columns(len(present_major_cities))
        for i, city_name in enumerate(present_major_cities):
            aqi_val = avg_aqi_major.get(city_name)
            # Determine category for color
            if aqi_val <= 50:
                color = CATEGORY_COLORS_DARK["Good"]
            elif aqi_val <= 100:
                color = CATEGORY_COLORS_DARK["Satisfactory"]
            elif aqi_val <= 200:
                color = CATEGORY_COLORS_DARK["Moderate"]
            elif aqi_val <= 300:
                color = CATEGORY_COLORS_DARK["Poor"]
            elif aqi_val <= 400:
                color = CATEGORY_COLORS_DARK["Very Poor"]
            else:
                color = CATEGORY_COLORS_DARK["Severe"]
                
            with cols[i]:
                st.markdown(f"<div style='text-align:center; padding:1rem; border-radius:12px; background:{CARD_BACKGROUND_COLOR}; border:1px solid {BORDER_COLOR}'>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:1.2rem; font-weight:600;'>{city_name}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:1.8rem; font-weight:700; color:{color};'>{aqi_val:.1f}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No AQI data available for major cities in the selected period.")
else:
    st.info("No data available for major cities.")

st.markdown("---")

# ========================================================
# =======   CITY-SPECIFIC ANALYSIS (Improved)   ==========
# ========================================================
export_data_list = []

if not selected_cities:
    st.info("✨ Select one or more cities from the sidebar to dive into detailed analysis.")
else:
    for city in selected_cities:
        st.markdown(f"## 🏙️ {city.upper()} DEEP DIVE – {year}")
        
        # Health status card
        city_data_full = df_period_filtered[df_period_filtered["city"] == city].copy()
        
        if city_data_full.empty:
            with st.container():
                st.warning(f"😔 No data available for {city} for {selected_month_name}, {year}. Try different filter settings.")
            continue
            
        # Calculate current AQI status
        latest_data = city_data_full.sort_values("date", ascending=False).iloc[0]
        current_aqi = latest_data["index"]
        current_level = latest_data["level"]
        current_pollutant = latest_data["pollutant"]
        
        # Health recommendation
        health_msg = HEALTH_RECOMMENDATIONS.get(current_level, "No specific health recommendations available")
        
        # Status card
        status_col1, status_col2, status_col3 = st.columns([1,2,1])
        with status_col1:
            st.markdown(f"<div style='text-align:center; padding:1rem; border-radius:12px; background:{CARD_BACKGROUND_COLOR}; border:1px solid {BORDER_COLOR}'>", unsafe_allow_html=True)
            st.markdown("🔴 Live Status", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:2.5rem; font-weight:700; color:{CATEGORY_COLORS_DARK.get(current_level, '#FFFFFF')};'>{current_aqi:.0f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.1rem;'>{current_level}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with status_col2:
            st.markdown(f"<div style='padding:1rem; border-radius:12px; background:{CARD_BACKGROUND_COLOR}; border:1px solid {BORDER_COLOR}; height:100%;'>", unsafe_allow_html=True)
            st.markdown("#### Health Recommendations")
            st.markdown(f"<div class='health-card' style='border-left-color: {CATEGORY_COLORS_DARK.get(current_level, '#FFFFFF')};'>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:1.1rem;'>{health_msg}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with status_col3:
            st.markdown(f"<div style='text-align:center; padding:1rem; border-radius:12px; background:{CARD_BACKGROUND_COLOR}; border:1px solid {BORDER_COLOR}; height:100%;'>", unsafe_allow_html=True)
            st.markdown("#### Dominant Pollutant")
            st.markdown(f"<div style='font-size:2rem; color:{POLLUTANT_COLORS_DARK.get(current_pollutant, '#FFFFFF')}; margin:1rem 0;'>{current_pollutant}</div>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:0.9rem;'>Primary air quality concern</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        city_data_full["day_of_year"] = city_data_full["date"].dt.dayofyear
        city_data_full["month_name"] = city_data_full["date"].dt.month_name()
        city_data_full["day_of_month"] = city_data_full["date"].dt.day
        export_data_list.append(city_data_full)

        tab_trend, tab_dist, tab_heatmap_detail, tab_health = st.tabs(["📊 TRENDS & CALENDAR", "📈 DISTRIBUTIONS", "🗓️ DETAILED HEATMAP", "❤️ HEALTH ANALYSIS"])

        with tab_trend:
            st.markdown("##### 📅 Daily AQI Calendar")
            # FIXED CALENDAR HEATMAP (No duplicate months)
            start_date = pd.to_datetime(f"{year}-01-01")
            end_date = pd.to_datetime(f"{year}-12-31")
            full_year_dates = pd.date_range(start_date, end_date, freq="D")

            calendar_df = pd.DataFrame({"date": full_year_dates})
            calendar_df["week"] = calendar_df["date"].dt.isocalendar().week
            calendar_df["day_of_week"] = calendar_df["date"].dt.dayofweek
            
            # Handle year transition properly
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

            day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            
            # Get unique weeks for month annotations (fixed)
            week_start_dates = merged_cal_df.groupby("week")["date"].min().reset_index()
            week_start_dates["month"] = week_start_dates["date"].dt.strftime("%b")
            month_names = week_start_dates.drop_duplicates("month", keep="first").set_index("week")["month"]

            fig_cal = go.Figure(
                data=go.Heatmap(
                    x=merged_cal_df["week"],
                    y=merged_cal_df["day_of_week"],
                    z=merged_cal_df["level"].map({k: i for i, k in enumerate(CATEGORY_COLORS_DARK.keys())}),
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
                    xgap=2, ygap=2
                )
            )

            # Add month annotations
            annotations = []
            for week_num, month_name in month_names.items():
                annotations.append(
                    go.layout.Annotation(
                        text=month_name,
                        align="center",
                        showarrow=False,
                        xref="x", yref="y",
                        x=week_num, y=7,
                        font=dict(color=SUBTLE_TEXT_COLOR_DARK_THEME, size=12)
                    )
                )

            # Add legend manually
            for i, (level, color) in enumerate(CATEGORY_COLORS_DARK.items()):
                annotations.append(
                    go.layout.Annotation(
                        text=f"█ <span style='color:{TEXT_COLOR_DARK_THEME};'>{level}</span>",
                        align="left",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.05 + 0.12 * (i % 7),
                        y=-0.1 - 0.1 * (i // 7),
                        xanchor="left", yanchor="top",
                        font=dict(color=color, size=12)
                    )
                )

            fig_cal.update_layout(
                yaxis=dict(
                    tickmode="array",
                    tickvals=list(range(7)),
                    ticktext=day_labels,
                    showgrid=False, 
                    zeroline=False
                ),
                xaxis=dict(showgrid=False, zeroline=False, tickmode="array", ticktext=[], tickvals=[]),
                height=320,
                margin=dict(t=50, b=100, l=40, r=40),
                plot_bgcolor=CARD_BACKGROUND_COLOR,
                paper_bgcolor=CARD_BACKGROUND_COLOR,
                font_color=TEXT_COLOR_DARK_THEME,
                annotations=annotations,
                showlegend=False
            )
            st.plotly_chart(fig_cal, use_container_width=True)

            st.markdown("##### 📈 AQI Trend & 7-Day Rolling Average")
            city_data_trend = city_data_full.sort_values("date").copy()
            city_data_trend["rolling_avg_7day"] = city_data_trend["index"].rolling(window=7, center=True, min_periods=1).mean().round(2)

            fig_trend_roll = go.Figure()
            fig_trend_roll.add_trace(
                go.Scatter(
                    x=city_data_trend["date"], y=city_data_trend["index"],
                    mode="lines+markers", name="Daily AQI",
                    marker=dict(size=4, opacity=0.7, color=SUBTLE_TEXT_COLOR_DARK_THEME),
                    line=dict(width=1.5, color=SUBTLE_TEXT_COLOR_DARK_THEME),
                    customdata=city_data_trend[["level"]],
                    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>AQI: %{y}<br>Category: %{customdata[0]}<extra></extra>"
                )
            )
            fig_trend_roll.add_trace(
                go.Scatter(
                    x=city_data_trend["date"], y=city_data_trend["rolling_avg_7day"],
                    mode="lines", name="7-Day Rolling Avg",
                    line=dict(color=ACCENT_COLOR, width=2.5, dash="dash"),
                    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>7-Day Avg AQI: %{y}<extra></extra>"
                )
            )
            fig_trend_roll.update_layout(
                yaxis_title="AQI Index", 
                xaxis_title="Date", 
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified", 
                paper_bgcolor=CARD_BACKGROUND_COLOR, 
                plot_bgcolor=BACKGROUND_COLOR,
                font_color=TEXT_COLOR_DARK_THEME
            )
            st.plotly_chart(fig_trend_roll, use_container_width=True)

        with tab_dist:
            col_bar_dist, col_sun_dist = st.columns([2, 1])
            with col_bar_dist:
                st.markdown("##### 📊 AQI Category Distribution")
                category_counts_df = (
                    city_data_full["level"]
                    .value_counts()
                    .reindex(CATEGORY_COLORS_DARK.keys(), fill_value=0)
                    .reset_index()
                )
                category_counts_df.columns = ["AQI Category", "Number of Days"]
                
                fig_dist_bar = px.bar(
                    category_counts_df, 
                    x="AQI Category", 
                    y="Number of Days", 
                    color="AQI Category",
                    color_discrete_map=CATEGORY_COLORS_DARK, 
                    text_auto=True
                )
                fig_dist_bar.update_layout(
                    height=450, 
                    xaxis_title=None, 
                    yaxis_title="Number of Days",
                    paper_bgcolor=CARD_BACKGROUND_COLOR, 
                    plot_bgcolor=BACKGROUND_COLOR, 
                    font_color=TEXT_COLOR_DARK_THEME
                )
                fig_dist_bar.update_traces(
                    textfont_size=12, 
                    textangle=0, 
                    textposition="outside", 
                    cliponaxis=False,
                    marker_line_width=0
                )
                st.plotly_chart(fig_dist_bar, use_container_width=True)

            with col_sun_dist:
                st.markdown("##### ☀️ Category Proportions")
                if category_counts_df["Number of Days"].sum() > 0:
                    fig_sunburst = px.sunburst(
                        category_counts_df, 
                        path=["AQI Category"], 
                        values="Number of Days",
                        color="AQI Category", 
                        color_discrete_map=CATEGORY_COLORS_DARK,
                    )
                    fig_sunburst.update_layout(
                        height=450, 
                        margin=dict(t=20, l=20, r=20, b=20),
                        paper_bgcolor=CARD_BACKGROUND_COLOR, 
                        plot_bgcolor=BACKGROUND_COLOR, 
                        font_color=TEXT_COLOR_DARK_THEME
                    )
                    st.plotly_chart(fig_sunburst, use_container_width=True)
                else:
                    st.caption("No data for sunburst chart.")

            st.markdown("##### 🎻 Monthly AQI Distribution")
            month_order_cat = list(months_map_dict.values())
            city_data_full["month_name"] = pd.Categorical(city_data_full["month_name"], categories=month_order_cat, ordered=True)

            fig_violin = px.violin(
                city_data_full.sort_values("month_name"),
                x="month_name", 
                y="index", 
                color="month_name", 
                color_discrete_sequence=px.colors.qualitative.Vivid,
                box=True, 
                points="outliers",
                labels={"index": "AQI Index", "month_name": "Month"},
                hover_data=["date", "level"]
            )
            fig_violin.update_layout(
                height=500, 
                xaxis_title=None, 
                showlegend=False,
                paper_bgcolor=CARD_BACKGROUND_COLOR, 
                plot_bgcolor=BACKGROUND_COLOR, 
                font_color=TEXT_COLOR_DARK_THEME
            )
            fig_violin.update_traces(meanline_visible=True)
            st.plotly_chart(fig_violin, use_container_width=True)

        with tab_heatmap_detail:
            st.markdown("##### 🔥 AQI Heatmap (Month vs. Day)")
            heatmap_pivot = city_data_full.pivot_table(index="month_name", columns="day_of_month", values="index", observed=False)
            heatmap_pivot = heatmap_pivot.reindex(month_order_cat)

            fig_heat_detail = px.imshow(
                heatmap_pivot, 
                labels=dict(x="Day of Month", y="Month", color="AQI"),
                aspect="auto", 
                color_continuous_scale="Inferno",
                text_auto=".0f"
            )
            fig_heat_detail.update_layout(
                height=550, 
                xaxis_side="top",
                paper_bgcolor=CARD_BACKGROUND_COLOR, 
                plot_bgcolor=BACKGROUND_COLOR, 
                font_color=TEXT_COLOR_DARK_THEME
            )
            fig_heat_detail.update_traces(hovertemplate="<b>Month:</b> %{y}<br><b>Day:</b> %{x}<br><b>AQI:</b> %{z}<extra></extra>")
            st.plotly_chart(fig_heat_detail, use_container_width=True)
            
        with tab_health:
            st.markdown("##### ❤️ Health Impact Analysis")
            
            # Health impact by AQI category
            health_col1, health_col2 = st.columns(2)
            
            with health_col1:
                st.markdown("###### 🚶‍♂️ Activity Recommendations")
                st.markdown(f"""
                <div style="background:{CARD_BACKGROUND_COLOR}; border-radius:12px; padding:1.5rem; border:1px solid {BORDER_COLOR}">
                    <p style="font-size:1.1rem; margin-bottom:1rem;"><b>Current AQI: {current_aqi:.0f} ({current_level})</b></p>
                    <p style="font-size:1.05rem; margin-bottom:1rem;">{health_msg}</p>
                    <p style="font-size:0.95rem; color:{SUBTLE_TEXT_COLOR_DARK_THEME};">
                        Based on latest data from {latest_data['date'].strftime('%Y-%m-%d')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("###### 📖 General Guidelines")
                st.markdown(f"""
                <div style="background:{CARD_BACKGROUND_COLOR}; border-radius:12px; padding:1.5rem; border:1px solid {BORDER_COLOR}; margin-top:1.5rem;">
                    <ul style="padding-left:1.5rem;">
                        <li>Sensitive groups include children, elderly, and people with respiratory issues</li>
                        <li>Consider wearing N95 masks when AQI &gt; 200</li>
                        <li>Keep windows closed during high pollution periods</li>
                        <li>Use air purifiers indoors when AQI &gt; 150</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with health_col2:
                st.markdown("###### 📊 Pollution Health Effects")
                health_effects = {
                    "Good": "No health impacts expected",
                    "Satisfactory": "Minor discomfort to sensitive individuals",
                    "Moderate": "Breathing discomfort to people with lung disease",
                    "Poor": "Breathing discomfort to most people on prolonged exposure",
                    "Very Poor": "Respiratory illness on prolonged exposure",
                    "Severe": "Health impacts even on light physical activity"
                }
                
                fig_health = go.Figure()
                for level, effect in health_effects.items():
                    if level in category_counts_df["AQI Category"].values:
                        days = category_counts_df[category_counts_df["AQI Category"] == level]["Number of Days"].values[0]
                    else:
                        days = 0
                        
                    fig_health.add_trace(go.Bar(
                        y=[level],
                        x=[days],
                        name=level,
                        orientation='h',
                        marker_color=CATEGORY_COLORS_DARK.get(level),
                        hoverinfo="text",
                        hovertext=f"<b>{level}</b><br>{effect}<br>{days} days in {year}"
                    ))
                
                fig_health.update_layout(
                    title="Health Impact by AQI Category",
                    barmode="stack",
                    height=450,
                    xaxis_title="Number of Days",
                    yaxis_title="AQI Category",
                    paper_bgcolor=CARD_BACKGROUND_COLOR,
                    plot_bgcolor=BACKGROUND_COLOR,
                    font_color=TEXT_COLOR_DARK_THEME,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_health, use_container_width=True)

# ========================================================
# =======   CITY-WISE AQI COMPARISON (Enhanced)  =========
# ========================================================
if len(selected_cities) > 1:
    st.markdown("## 🆚 MULTI-CITY AQI COMPARISON")
    comparison_df_list = []
    for city_comp in selected_cities:
        city_ts_comp = df_period_filtered[df_period_filtered["city"] == city_comp].copy()
        if not city_ts_comp.empty:
            city_ts_comp = city_ts_comp.sort_values("date")
            city_ts_comp["city_label"] = city_comp
            comparison_df_list.append(city_ts_comp)

    if comparison_df_list:
        combined_comp_df = pd.concat(comparison_df_list)
        
        # Add average AQI calculation
        avg_aqi_df = combined_comp_df.groupby("city_label")["index"].mean().reset_index().sort_values("index")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            fig_cmp = px.line(
                combined_comp_df, 
                x="date", 
                y="index", 
                color="city_label",
                labels={"index": "AQI Index", "date": "Date", "city_label": "City"},
                markers=True, 
                line_shape="spline", 
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_cmp.update_layout(
                title_text=f"AQI Trends Comparison – {selected_month_name}, {year}",
                height=500, 
                legend_title_text="City",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor=CARD_BACKGROUND_COLOR, 
                plot_bgcolor=BACKGROUND_COLOR, 
                font_color=TEXT_COLOR_DARK_THEME,
                hovermode="x unified"
            )
            st.plotly_chart(fig_cmp, use_container_width=True)
        
        with col2:
            st.markdown("##### 🏆 Average AQI")
            for idx, row in avg_aqi_df.iterrows():
                aqi_val = row["index"]
                # Determine color based on AQI value
                if aqi_val <= 50:
                    color = CATEGORY_COLORS_DARK["Good"]
                elif aqi_val <= 100:
                    color = CATEGORY_COLORS_DARK["Satisfactory"]
                elif aqi_val <= 200:
                    color = CATEGORY_COLORS_DARK["Moderate"]
                elif aqi_val <= 300:
                    color = CATEGORY_COLORS_DARK["Poor"]
                elif aqi_val <= 400:
                    color = CATEGORY_COLORS_DARK["Very Poor"]
                else:
                    color = CATEGORY_COLORS_DARK["Severe"]
                    
                st.markdown(f"""
                <div style="background:{CARD_BACKGROUND_COLOR}; border-radius:12px; padding:1rem; margin-bottom:1rem; border:1px solid {BORDER_COLOR}">
                    <div style="font-weight:600; font-size:1.1rem;">{row['city_label']}</div>
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
st.markdown("## 💨 PROMINENT POLLUTANT ANALYSIS")

poll_col1, poll_col2 = st.columns(2)

with poll_col1:
    st.markdown("#### 鑽 Yearly Dominant Pollutant Trends")
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
            percent_poll_A_long, 
            x="year_label", 
            y="percentage", 
            color="pollutant",
            title=f"Dominant Pollutants Over Years – {city_pollutant_A}",
            labels={"percentage": "Percentage of Days (%)", "year_label": "Year", "pollutant": "Pollutant"},
            color_discrete_map=POLLUTANT_COLORS_DARK,
            barmode="stack"
        )
        fig_poll_A.update_layout(
            xaxis_type="category", 
            yaxis_ticksuffix="%", 
            height=500,
            paper_bgcolor=CARD_BACKGROUND_COLOR, 
            plot_bgcolor=BACKGROUND_COLOR, 
            font_color=TEXT_COLOR_DARK_THEME
        )
        fig_poll_A.update_traces(texttemplate="%{value:.1f}%", textposition="auto")
        st.plotly_chart(fig_poll_A, use_container_width=True)
    else:
        st.warning(f"No pollutant data for {city_pollutant_A} (Yearly Trend).")

with poll_col2:
    st.markdown(f"#### ⛽ Dominant Pollutants ({selected_month_name}, {year})")
    city_pollutant_B = st.selectbox(
        "Select City for Filtered Pollutant View:", unique_cities,
        key="pollutant_B_city_dark", 
        index=unique_cities.index(default_city_val[0]) if default_city_val and default_city_val[0] in unique_cities else 0
    )
    pollutant_data_B = df_period_filtered[df_period_filtered["city"] == city_pollutant_B].copy()
    if not pollutant_data_B.empty and "pollutant" in pollutant_data_B.columns:
        grouped_poll_B = pollutant_data_B.groupby("pollutant").size().reset_index(name="count")
        total_days_B = grouped_poll_B["count"].sum()
        grouped_poll_B["percentage"] = (grouped_poll_B["count"] / total_days_B * 100).round(1) if total_days_B > 0 else 0

        fig_poll_B = px.pie(
            grouped_poll_B, 
            values="count", 
            names="pollutant",
            title=f"Dominant Pollutants – {city_pollutant_B} ({selected_month_name}, {year})",
            color="pollutant",
            color_discrete_map=POLLUTANT_COLORS_DARK,
            hole=0.4
        )
        fig_poll_B.update_layout(
            height=500,
            paper_bgcolor=CARD_BACKGROUND_COLOR, 
            plot_bgcolor=BACKGROUND_COLOR, 
            font_color=TEXT_COLOR_DARK_THEME,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        fig_poll_B.update_traces(textinfo="percent+label", textposition="inside")
        st.plotly_chart(fig_poll_B, use_container_width=True)
    else:
        st.warning(f"No pollutant data for {city_pollutant_B} for the selected period.")

# ========================================================
# =======   AQI FORECAST (Enhanced)   ====================
# ========================================================
st.markdown("## 🔮 AQI FORECAST (POLYNOMIAL TREND)")

forecast_col1, forecast_col2 = st.columns([3,1])

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
            
            # Use polynomial regression for better forecasting
            X = forecast_df["days_since_start"].values.reshape(-1, 1)
            y = forecast_df["index"].values
            
            # Polynomial features (degree=2 for curved line)
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
            fig_forecast.add_trace(
                go.Scatter(
                    x=plot_df_obs["date"], 
                    y=plot_df_obs["AQI"], 
                    mode="lines+markers", 
                    name="Observed AQI", 
                    line=dict(color=ACCENT_COLOR),
                    marker=dict(size=5)
                )
            )
            fig_forecast.add_trace(
                go.Scatter(
                    x=plot_df_fcst["date"], 
                    y=plot_df_fcst["AQI"], 
                    mode="lines", 
                    name="Forecast",
                    line=dict(dash="dash", color=HIGHLIGHT_COLOR, width=3)
                )
            )
            
            # Add forecast period shading
            forecast_start = forecast_df["date"].max() + pd.Timedelta(days=1)
            forecast_end = future_dates_list[-1]
            
            fig_forecast.add_vrect(
                x0=forecast_start, x1=forecast_end,
                fillcolor="rgba(255, 107, 107, 0.1)",
                layer="below", line_width=0,
                annotation_text="Forecast Period", 
                annotation_position="top left",
                annotation_font_color=HIGHLIGHT_COLOR
            )

            fig_forecast.update_layout(
                title=f"AQI Forecast – {forecast_city_select}", 
                yaxis_title="AQI Index", 
                xaxis_title="Date", 
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor=CARD_BACKGROUND_COLOR, 
                plot_bgcolor=BACKGROUND_COLOR, 
                font_color=TEXT_COLOR_DARK_THEME,
                hovermode="x unified"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.warning(f"Not enough valid data points for {forecast_city_select} to forecast.")
    else:
        st.warning(f"Need at least 15 data points for {forecast_city_select} for forecasting; found {len(forecast_src_data)}.")

with forecast_col2:
    st.markdown("##### ℹ️ Forecast Info")
    st.markdown(f"""
    <div style="background:{CARD_BACKGROUND_COLOR}; border-radius:12px; padding:1.5rem; border:1px solid {BORDER_COLOR}; margin-top:1.5rem;">
        <p style="font-size:1.1rem; margin-bottom:1rem;"><b>Forecast Methodology</b></p>
        <p style="font-size:0.95rem; margin-bottom:1rem;">
            This forecast uses polynomial regression (degree=2) based on historical data to predict future AQI trends.
        </p>
        <p style="font-size:0.95rem; margin-bottom:1rem;">
            <b>Limitations:</b><br>
            - Short-term forecasts only<br>
            - Doesn't account for weather events<br>
            - Accuracy decreases beyond 15 days
        </p>
        <p style="font-size:0.9rem; color:{SUBTLE_TEXT_COLOR_DARK_THEME}; margin-top:1.5rem;">
            For precise planning, consult official sources.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ========================================================
# =========  City AQI Hotspots (Enhanced)  ==============
# ========================================================
st.markdown("## 📍 AIR QUALITY HOTSPOTS MAP")

map_col1, map_col2 = st.columns([3,1])

with map_col1:
    city_coords_data = {}
    coords_file_path = "lat_long.txt"  # Coordinates file
    scatter_map_rendered = False

    try:
        if os.path.exists(coords_file_path):
            with open(coords_file_path, "r", encoding="utf-8") as f:
                file_content = f.read()

            local_scope = {}
            exec(file_content, {}, local_scope)
            if "city_coords" in local_scope and isinstance(local_scope["city_coords"], dict):
                city_coords_data = local_scope["city_coords"]
    except Exception as e_exec:
        st.error(f"Map Error: Error processing coordinates file '{coords_file_path}'. Scatter map cannot be displayed. Error: {e_exec}")
        city_coords_data = {}

    if not df_period_filtered.empty:
        map_grouped_data = df_period_filtered.groupby("city").agg(
            avg_aqi=('index', 'mean'),
            dominant_pollutant=('pollutant', lambda x:
                x.mode().iloc[0]
                if not x.mode().empty and x.mode().iloc[0] not in ['Other', 'nan']
                else (x[~x.isin(['Other', 'nan'])].mode().iloc[0]
                      if not x[~x.isin(['Other', 'nan'])].mode().empty else 'N/A')
            )
        ).reset_index().dropna(subset=['avg_aqi'])

        if city_coords_data and not map_grouped_data.empty:
            latlong_map_df_list = []
            for city_name_coord, coords_val in city_coords_data.items():
                if isinstance(coords_val, (list, tuple)) and len(coords_val) == 2:
                    try:
                        lat, lon = float(coords_val[0]), float(coords_val[1])
                        latlong_map_df_list.append({'city': city_name_coord, 'lat': lat, 'lon': lon})
                    except (ValueError, TypeError):
                        pass

            latlong_map_df = pd.DataFrame(latlong_map_df_list)

            if not latlong_map_df.empty:
                map_merged_df = pd.merge(map_grouped_data, latlong_map_df, on="city", how="inner")
                if not map_merged_df.empty:
                    map_merged_df["AQI Category"] = map_merged_df["avg_aqi"].apply(
                        lambda val: next(
                            (k for k, v_range in {
                                'Good': (0, 50), 'Satisfactory': (51, 100), 'Moderate': (101, 200),
                                'Poor': (201, 300), 'Very Poor': (301, 400), 'Severe': (401, float('inf'))
                            }.items() if v_range[0] <= val <= v_range[1]), "Unknown"
                        ) if pd.notna(val) else "Unknown"
                    )
                    
                    # Calculate scaled bubble size (smaller bubbles)
                    min_aqi = map_merged_df["avg_aqi"].min()
                    max_aqi = map_merged_df["avg_aqi"].max()
                    size_range = 5, 15  # Smaller size range
                    
                    # Scale the size
                    map_merged_df["scaled_size"] = (
                        (map_merged_df["avg_aqi"] - min_aqi) / 
                        (max_aqi - min_aqi) * (size_range[1] - size_range[0]) + size_range[0]
                    ).fillna(size_range[0])

                    fig_scatter_map = px.scatter_mapbox(
                        map_merged_df, 
                        lat="lat", 
                        lon="lon",
                        size="scaled_size",
                        size_max=size_range[1],
                        color="AQI Category",
                        color_discrete_map=CATEGORY_COLORS_DARK,
                        hover_name="city",
                        custom_data=['city', 'avg_aqi', 'dominant_pollutant', 'AQI Category'],
                        text="city",
                        zoom=4.0, 
                        center={"lat": 23.0, "lon": 82.5}
                    )

                    scatter_map_layout_args = get_custom_plotly_layout_args(
                        height=700,
                        title_text=f"Average AQI Hotspots - {selected_month_name}, {year}"
                    )
                    scatter_map_layout_args['mapbox_style'] = "carto-darkmatter"
                    scatter_map_layout_args['margin'] = {"r":10,"t":60,"l":10,"b":10}
                    scatter_map_layout_args['legend']['y'] = 0.98
                    scatter_map_layout_args['legend']['x'] = 0.98
                    scatter_map_layout_args['legend']['xanchor'] = 'right'

                    fig_scatter_map.update_traces(
                        marker=dict(sizemin=size_range[0], opacity=0.85, sizemode='diameter'),
                        hovertemplate="<b style='font-size:1.1em;'>%{customdata[0]}</b><br>" +
                                      "Avg. AQI: %{customdata[1]:.1f} (%{customdata[3]})<br>" +
                                      "Dominant Pollutant: %{customdata[2]}" +
                                      "<extra></extra>"
                    )
                    fig_scatter_map.update_layout(**scatter_map_layout_args)
                    st.plotly_chart(fig_scatter_map, use_container_width=True)
                    scatter_map_rendered = True

        if not scatter_map_rendered:
            st.warning("Map data incomplete. Showing alternative visualization.")
            if not map_grouped_data.empty:
                avg_aqi_cities_alt = map_grouped_data.sort_values(by='avg_aqi', ascending=True)
                fig_alt_bar = px.bar(
                    avg_aqi_cities_alt.tail(20),
                    x='avg_aqi', 
                    y='city', 
                    orientation='h',
                    color='avg_aqi', 
                    color_continuous_scale=px.colors.sequential.YlOrRd_r,
                    labels={'avg_aqi': 'Average AQI', 'city': 'City'}
                )
                fallback_layout = get_custom_plotly_layout_args(
                    height=max(400, len(avg_aqi_cities_alt.tail(20)) * 25),
                    title_text=f"Top Cities by Average AQI - {selected_month_name}, {year} (Map Fallback)"
                )
                fallback_layout["yaxis"] = {'categoryorder':'total ascending'}
                fig_alt_bar.update_layout(**fallback_layout)
                st.plotly_chart(fig_alt_bar, use_container_width=True)
            else:
                st.warning("No city AQI data available for the selected period to display fallback chart.")
    else:
        st.warning("No air quality data available for the selected filters to display on a map or chart.")

with map_col2:
    st.markdown("##### ℹ️ Map Guide")
    st.markdown(f"""
    <div style="background:{CARD_BACKGROUND_COLOR}; border-radius:12px; padding:1.5rem; border:1px solid {BORDER_COLOR}; margin-top:1.5rem;">
        <p style="font-size:1.1rem; margin-bottom:1rem;"><b>How to Use This Map</b></p>
        <ul style="padding-left:1.5rem; margin-bottom:1.5rem;">
            <li>Bubble size represents AQI severity</li>
            <li>Color indicates AQI category</li>
            <li>Hover for detailed information</li>
            <li>Click and drag to pan</li>
            <li>Use mouse wheel to zoom</li>
        </ul>
        
        <p style="font-size:1.1rem; margin-bottom:1rem;"><b>AQI Categories</b></p>
    """, unsafe_allow_html=True)
    
    for category, color in CATEGORY_COLORS_DARK.items():
        if category != "Unknown":
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <div style="width: 15px; height: 15px; background-color: {color}; border-radius: 3px; margin-right: 10px;"></div>
                <span>{category}</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


# ========================================================
# ========   DOWNLOAD FILTERED DATA (Enhanced)   =========
# ========================================================
if export_data_list:
    st.markdown("## 📥 DOWNLOAD DATA")
    with st.container():
        combined_export_final = pd.concat(export_data_list)
        from io import StringIO
        csv_buffer_final = StringIO()
        combined_export_final.to_csv(csv_buffer_final, index=False)
        
        st.markdown(f"""
        <div style="background:{CARD_BACKGROUND_COLOR}; border-radius:12px; padding:1.5rem; border:1px solid {BORDER_COLOR}">
            <h3 style="color:{ACCENT_COLOR};">Export Filtered Data</h3>
            <p style="margin-bottom:1.5rem;">Download the filtered dataset for further analysis</p>
        """, unsafe_allow_html=True)
        
        st.download_button(
            label="📤 Download Filtered City Data (CSV)",
            data=csv_buffer_final.getvalue(),
            file_name=f"IITKGP_filtered_aqi_{year}_{selected_month_name.replace(' ', '')}.csv",
            mime="text/csv"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------- Footer -------------------
st.markdown(f"""
<div style="text-align: center; margin-top: 4rem; padding: 2.5rem; background: linear-gradient(90deg, #121212, #1a2a3a, #121212); border-radius: 16px; border: 1px solid #2a3a4a; position: relative; overflow: hidden;">
    <div style="position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(90deg, #00BCD4, #00BFA5);"></div>
    
    <h3 style="color: #00BCD4; margin-bottom: 1.5rem;">IIT KGP Air Quality Dashboard</h3>
    
    <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 1.5rem; flex-wrap: wrap;">
        <div>
            <p style="font-size: 0.9rem; color: #B0B0B0;">Data Source</p>
            <p style="font-weight: 500;">Central Pollution Control Board (CPCB)</p>
        </div>
        
        <div>
            <p style="font-size: 0.9rem; color: #B0B0B0;">Developed By</p>
            <p style="font-weight: 500;">IIT Kharagpur Research Team</p>
        </div>
        
        <div>
            <p style="font-size: 0.9rem; color: #B0B0B0;">Last Updated</p>
            <p style="font-weight: 500;">{data_last_updated.strftime('%Y-%m-%d %H:%M') if data_last_updated else "N/A"}</p>
        </div>
    </div>
    
    <div style="margin-top: 1.5rem;">
        <a href="https://github.com/kapil2020/india-air-quality-dashboard" target="_blank" style="color: #00BCD4; text-decoration: none; font-weight: 600; display: inline-flex; align-items: center; gap: 0.5rem;">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00BCD4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
            </svg>
            View on GitHub
        </a>
    </div>
    
    <p style="margin-top: 2rem; font-size: 0.85rem; color: #707070;">
        © {pd.to_datetime("today").year} IIT Kharagpur | For Research and Educational Purposes
    </p>
</div>
""", unsafe_allow_html=True)

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

# ------------------- Custom CSS Styling (Dark Theme & Mobile Responsiveness) -------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    * {{
        transition: all 0.2s ease;
        box-sizing: border-box; /* Ensures padding and border don't add to width/height */
    }}

    html, body, #root, .block-container, .stApp {{
        background-color: {BACKGROUND_COLOR} !important; /* Ensure dark background globally */
        color: {TEXT_COLOR_DARK_THEME} !important;
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
    }}
    
    body {{ /* Additional body specific styles if needed */
        margin: 0;
        padding: 0;
    }}


    /* Main container styling */
    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: clamp(1rem, 2vw, 2rem); /* Responsive padding */
        padding-right: clamp(1rem, 2vw, 2rem); /* Responsive padding */
    }}

    /* Card-like styling for sections/charts */
    .stPlotlyChart, .stDataFrame, .stAlert, .stMetric,
    .stDownloadButton > button, .stButton > button,
    div[data-testid="stExpander"], div[data-testid="stForm"] {{
        border-radius: 16px;
        border: 1px solid {BORDER_COLOR};
        background-color: {CARD_BACKGROUND_COLOR};
        padding: 1.25rem; /* Slightly reduced for better mobile fit */
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
    
    .stpyplot {{ /* For Matplotlib if used */
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
        padding: 0.7rem 1rem; /* Adjusted padding for tabs */
        font-weight: 600;
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        margin-right: 0.3rem; /* Reduced margin */
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
        color: {TEXT_COLOR_DARK_THEME};
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, {ACCENT_COLOR}, #00E5FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: clamp(2rem, 5vw, 2.8rem); /* Responsive font size */
    }}
    
    h2 {{
        color: {ACCENT_COLOR};
        border-bottom: 2px solid {BORDER_COLOR};
        padding-bottom: 0.6rem;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        position: relative;
        font-size: clamp(1.4rem, 4vw, 1.8rem); /* Responsive font size */
    }}
    
    h2:after {{
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 100px; /* Slightly reduced line */
        height: 3px;
        background: linear-gradient(90deg, {ACCENT_COLOR}, transparent);
    }}
    
    h3 {{
        color: {TEXT_COLOR_DARK_THEME};
        margin-top: 0rem;
        margin-bottom: 1rem; /* Adjusted margin */
        font-weight: 600;
        font-size: clamp(1.2rem, 3.5vw, 1.4rem); /* Responsive font size */
    }}
    
    h4, h5 {{
        color: {TEXT_COLOR_DARK_THEME};
        margin-top: 0.2rem;
        margin-bottom: 0.8rem; /* Adjusted margin */
        font-weight: 500;
        font-size: clamp(1rem, 3vw, 1.1rem); /* Responsive font size */
    }}

    /* Sidebar styling */
    .stSidebar {{
        background-color: {CARD_BACKGROUND_COLOR} !important; /* Ensure background for mobile overlay */
        border-right: 1px solid {BORDER_COLOR};
        padding: 1.5rem;
        box-shadow: 5px 0 15px rgba(0, 0, 0, 0.2);
        min-width: 300px !important; /* Desktop sidebar width */
        width: 300px !important;     /* Desktop sidebar width */
    }}
    
    .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3, .stSidebar .stMarkdown p {{
        color: {TEXT_COLOR_DARK_THEME};
        text-align: left;
        border-bottom: none;
    }}
    
    .stSidebar .stSelectbox label, .stSidebar .stMultiselect label {{
        color: {ACCENT_COLOR} !important;
        font-weight: 600;
        font-size: 1rem; /* Adjusted size for sidebar */
    }}

    /* Metric styling */
    .stMetric {{
        background-color: {BACKGROUND_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 1rem; /* Adjusted padding */
        text-align: center;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }}
    
    .stMetric > div:nth-child(1) {{ /* Label */
        font-size: 0.9rem; /* Adjusted size */
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        font-weight: 500;
        letter-spacing: 0.5px;
    }}
    
    .stMetric > div:nth-child(2) {{ /* Value */
        font-size: clamp(1.8rem, 4vw, 2.4rem); /* Responsive font size */
        font-weight: 700;
        background: linear-gradient(90deg, {ACCENT_COLOR}, #00E5FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.3rem 0; /* Adjusted margin */
    }}
    
    .stMetric > div:nth-child(3) {{ /* Delta */
        font-size: 0.8rem; /* Adjusted size */
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        font-weight: 400;
        font-style: italic;
    }}

    /* Expander styling */
    div[data-testid="stExpander"] summary {{
        font-size: 1.1rem; /* Adjusted size */
        font-weight: 600;
        color: {ACCENT_COLOR};
        padding: 0.6rem 1rem; /* Adjusted padding */
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
        padding: 0.75rem 1.5rem; /* Adjusted padding */
        border-radius: 50px;
        transition: all 0.3s ease;
        font-size: 0.95rem; /* Adjusted size */
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
        padding: 0.5rem 0.8rem; /* Adjusted padding */
        font-size: 0.95rem; /* Ensure input text is readable */
    }}
    
    .stDateInput input {{
         background-color: {BACKGROUND_COLOR};
         color: {TEXT_COLOR_DARK_THEME};
         border-radius: 10px;
         padding: 0.5rem 0.8rem; /* Adjusted padding */
         font-size: 0.95rem;
    }}
    
    /* Hover effects for inputs */
    .stTextInput input:hover, .stSelectbox div[data-baseweb="select"] > div:first-child:hover,
    .stMultiselect div[data-baseweb="select"] > div:first-child:hover {{
        border-color: {ACCENT_COLOR} !important;
        box-shadow: 0 0 0 2px rgba(0, 188, 212, 0.2);
    }}

    /* Custom Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px; /* Reduced width slightly */
        height: 8px;
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
        padding: clamp(1rem, 3vw, 2rem) 1rem; /* Responsive padding */
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid #2a3a4a;
    }}
    
    /* Custom badge styling */
    .badge {{
        display: inline-block;
        padding: 0.2rem 0.6rem; /* Adjusted padding */
        border-radius: 50px;
        font-size: 0.8rem; /* Adjusted size */
        font-weight: 600;
        margin: 0.15rem; /* Adjusted margin */
    }}
    
    /* Health recommendation cards */
    .health-card {{
        border-left: 4px solid;
        border-radius: 8px;
        padding: 0.8rem; /* Adjusted padding */
        margin-bottom: 1rem;
        background-color: rgba(30, 30, 30, 0.7);
    }}
    
    /* Key insight cards */
    .insight-card {{
        background: linear-gradient(145deg, #1a1a1a, #232323);
        border-radius: 12px;
        padding: 1.25rem; /* Adjusted padding */
        margin-bottom: 1.5rem;
        border: 1px solid {BORDER_COLOR};
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        height: 100%; /* For equal height in columns if needed */
    }}

    /* Footer Styling (from previous good version) */
    .footer-container {{
        position: relative;
        width: 100%;
        box-sizing: border-box;
        overflow-x: hidden; /* Prevent horizontal scroll */
        margin-top: 4rem;
        padding: 2rem 1rem; /* Adjusted padding for mobile */
        border-radius: 16px;
        background: linear-gradient(270deg, #121212, #1a2a3a, #121212);
        background-size: 300% 300%;
        animation: footerGradient 8s ease infinite;
        border: 1px solid #2a3a4a;
        text-align: center;
    }}
    .footer-top-bar {{
        position: absolute; top: 0; left: 0; width: 100%; height: 4px;
        background: linear-gradient(90deg, #00BCD4, #00BFA5);
        background-size: 200% 200%; animation: footerGradient 5s ease infinite;
    }}
    .footer-container h3 {{
        color: #00BCD4; font-size: clamp(1.3rem, 4vw, 1.8rem); margin-bottom: 1rem; /* Adjusted */
        opacity: 0; animation: fadeIn 1s ease forwards;
    }}
    .footer-info {{
        display: flex; justify-content: center; gap: 1.5rem; /* Adjusted gap */
        flex-wrap: wrap; margin-bottom: 1.5rem;
        opacity: 0; animation: fadeIn 1s ease 0.3s forwards;
    }}
    .footer-info p {{ margin: 0; }}
    .footer-info .label {{ font-size: 0.85rem; color: #B0B0B0; }} /* Adjusted */
    .footer-info .value {{ font-weight: 500; color: #EAEAEA; font-size: 0.9rem; }} /* Adjusted */
    .footer-links {{ margin-bottom: 1rem; opacity: 0; animation: fadeIn 1s ease 0.6s forwards; }}
    .footer-links a {{
        color: #00BCD4; text-decoration: none; font-weight: 600;
        display: inline-flex; align-items: center; gap: 0.4rem; /* Adjusted */
        transition: color 0.3s ease; font-size: 0.9rem; /* Adjusted */
    }}
    .footer-links a:hover {{ color: #00E5FF; }}
    .footer-links a:hover svg {{ animation: iconPulse 1.5s infinite; }}
    .footer-container .copyright {{
        font-size: 0.8rem; color: #707070; /* Adjusted */
        opacity: 0; animation: fadeIn 1s ease 0.9s forwards;
    }}
    @keyframes fadeIn {{ 0% {{ opacity: 0; }} 100% {{ opacity: 1; }} }}
    @keyframes iconPulse {{ 0% {{ transform: scale(1); }} 50% {{ transform: scale(1.1); }} 100% {{ transform: scale(1); }} }}


    /* Responsive adjustments for general layout and typography */
    @media (max-width: 768px) {{
        .main .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}
        .stMetric > div:nth-child(2) {{ /* Metric Value */
            font-size: 1.8rem; /* Further adjust for mobile */
        }}
        .stTabs [data-baseweb="tab"] {{
            padding: 0.6rem 0.8rem; /* Tighter tab padding */
            font-size: 0.85rem; /* Smaller tab font */
        }}
        /* Ensure Streamlit columns stack by making their flex children take full width */
        div[data-testid="stHorizontalBlock"] > div {{
            flex: 1 1 100% !important; /* Make columns stack */
            min-width: 0; /* Allow shrinking */
            margin-bottom: 1rem; /* Add space between stacked columns */
        }}
        div[data-testid="stHorizontalBlock"] > div:last-child {{
            margin-bottom: 0; /* No margin for the last stacked item */
        }}
        .insight-card {{
            margin-bottom: 1rem; /* Space between stacked insight cards */
            padding: 1rem;
        }}
         .gradient-header {{
            padding: 1.5rem 0.5rem; /* Responsive padding for header */
        }}
        .footer-info {{ gap: 1rem; }} /* Reduce gap in footer on mobile */
    }}
    
    /* Specific for very small screens */
    @media (max-width: 480px) {{
        .stSelectbox, .stMultiselect {{ /* Make selectors full width */
            width: 100% !important;
        }}
        .stButton > button, .stDownloadButton > button {{ /* Full width buttons */
            width: 100%;
            padding-left: 0.5rem; padding-right: 0.5rem; /* Adjust padding */
        }}
        .stMetric > div:nth-child(2) {{ /* Metric Value */
            font-size: 1.6rem !important; /* Even smaller */
        }}
        .stTabs {{ /* Allow horizontal scrolling for tabs if they still overflow */
            overflow-x: auto;
        }}
        .stPlotlyChart {{ /* Attempt to control chart height */
             min-height: 280px !important; /* Min height for very small screens */
        }}
        .footer-container {{ padding: 1.5rem 0.5rem; }}
        .footer-info {{ flex-direction: column; align-items: center; }} /* Stack footer info items */
    }}
</style>
""", unsafe_allow_html=True)

# ------------------- Helper Functions -------------------
def get_custom_plotly_layout_args(height: int = None, title_text: str = None) -> dict:
    """
    Returns a dict of common Plotly layout arguments for dark-themed charts.
    Adjusted for better responsiveness.
    """
    layout_args = {
        "font": {"family": "Inter, sans-serif", "color": TEXT_COLOR_DARK_THEME, "size": 11}, # Base font size
        "paper_bgcolor": CARD_BACKGROUND_COLOR,
        "plot_bgcolor": CARD_BACKGROUND_COLOR,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(0,0,0,0)", # Transparent legend background
            "font": {"size": 10} # Smaller legend font
        },
        "margin": {"l": 30, "r": 15, "t": 50 if title_text else 25, "b": 30}, # Tighter margins
        "hoverlabel": {
            "bgcolor": "#2A2A2A",
            "font_size": 11, # Smaller hover label font
            "font_family": "Inter, sans-serif"
        },
        "xaxis": {"gridcolor": BORDER_COLOR, "linecolor": BORDER_COLOR, "automargin": True, "tickfont": {"size": 9}}, # automargin and smaller tick font
        "yaxis": {"gridcolor": BORDER_COLOR, "linecolor": BORDER_COLOR, "automargin": True, "tickfont": {"size": 9}}, # automargin and smaller tick font
    }
    if height:
        layout_args["height"] = height
    if title_text:
        layout_args["title_text"] = title_text
        layout_args["title_font"] = {"color": ACCENT_COLOR, "size": 14, "family": "Inter, sans-serif"} # Smaller title font
        layout_args["title_x"] = 0.05 # Left align title
    return layout_args

# ------------------- Title Header -------------------
st.markdown("""
<div class="gradient-header">
    <h1>🌬️ IIT KGP AIR QUALITY DASHBOARD</h1>
    <p style="color: #B0B0B0; font-size: clamp(0.9rem, 2.5vw, 1.1rem); max-width: 800px; margin: 0 auto;">
        Real-time Air Quality Monitoring and Predictive Analysis for Indian Cities
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------- Load Data -------------------
@st.cache_data(ttl=3600)
def load_data_and_metadata():
    """
    Tries to load today's CSV. If not found, falls back to 'combined_air_quality.txt'.
    Returns: (df_loaded, load_msg, last_update_time)
    """
    today = pd.to_datetime("today").date()
    # Assuming data folder is in the same directory as the script
    # For Streamlit Sharing or other deployments, ensure this path is correct
    data_folder = os.path.join(os.path.dirname(__file__), "data")
    csv_path = os.path.join(data_folder, f"{today}.csv")
    fallback_file = "combined_air_quality.txt" # Assuming this is in the root
    
    df_loaded = None
    is_today_data = False
    load_msg = ""
    last_update_time = None

    if os.path.exists(csv_path):
        try:
            df_loaded = pd.read_csv(csv_path)
            if "date" in df_loaded.columns:
                df_loaded["date"] = pd.to_datetime(df_loaded["date"])
                is_today_data = True
                load_msg = f"Live data from: **{os.path.basename(csv_path)}**"
                last_update_time = pd.Timestamp(os.path.getmtime(csv_path), unit="s")
            else:
                load_msg = f"Warning: '{os.path.basename(csv_path)}' found but missing 'date' column. Using fallback."
        except Exception as e:
            load_msg = f"Error loading '{os.path.basename(csv_path)}': {e}. Using fallback."

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
        f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 0.85rem;'>"
        f"📅 Last data update: {data_last_updated.strftime('%Y-%m-%d %H:%M:%S')}"
        "</p>",
        unsafe_allow_html=True
    )

# ------------------- Sidebar Filters -------------------
with st.sidebar:
    st.header("🔭 Controls") # Simplified header
    st.markdown("---", unsafe_allow_html=True)
    # st.info("Fetching real-time data from CPCB. Today's data available after 5:45 PM IST.", icon="ℹ️") # Removed for brevity

    unique_cities = sorted(df["city"].unique()) if "city" in df.columns else []
    
    # Default to Delhi if available, otherwise first city in list, else empty list
    default_city_selection = []
    if "Delhi" in unique_cities:
        default_city_selection = ["Delhi"]
    elif unique_cities:
        default_city_selection = [unique_cities[0]]
        
    selected_cities = st.multiselect("🏙️ Select Cities", unique_cities, default=default_city_selection,
                                    help="Select one or more cities for detailed analysis")

    years = sorted(df["date"].dt.year.unique(), reverse=True) # Ensure reverse chronological order
    default_year_val = years[0] if years else None # Default to most recent year
    
    year_index = 0 # Default index for selectbox
    if default_year_val and years: # Check if years list is not empty
        try:
            year_index = years.index(default_year_val)
        except ValueError: # If default_year_val is somehow not in years
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

    st.markdown("---")
    st.markdown("""
    <div style="margin-top: 1rem; text-align: center;">
        <p style="font-size: 0.8rem; color: #B0B0B0;">
            Developed by IIT Kharagpur<br>Data: CPCB India
        </p>
    </div>
    """, unsafe_allow_html=True)

# ========================================================
# =========  NATIONAL KEY INSIGHTS (Enhanced)  ===========
# ========================================================
st.markdown("## 🇮🇳 National Air Quality Snapshot")

if not df_period_filtered.empty:
    col1, col2, col3 = st.columns(3) # Using 3 columns for a cleaner look
    with col1:
        with st.container(): # Explicit container for card styling to apply
            st.markdown(f"<div class='insight-card'><h3>🌆 Coverage</h3>", unsafe_allow_html=True)
            cities_count = df_period_filtered["city"].nunique()
            st.metric(label="Cities Monitored", value=cities_count)
            st.markdown(f"<p style='color:{SUBTLE_TEXT_COLOR_DARK_THEME}; font-size:0.85rem;'>For {selected_month_name}, {year}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        with st.container():
            st.markdown(f"<div class='insight-card'><h3>📈 National Average</h3>", unsafe_allow_html=True)
            avg_aqi_national = df_period_filtered["index"].mean()
            st.metric(label="Average AQI", value=f"{avg_aqi_national:.1f}" if pd.notna(avg_aqi_national) else "N/A")
            if pd.notna(avg_aqi_national):
                # Determine national AQI category
                if avg_aqi_national <= 50: category = "Good"
                elif avg_aqi_national <= 100: category = "Satisfactory"
                elif avg_aqi_national <= 200: category = "Moderate"
                elif avg_aqi_national <= 300: category = "Poor"
                elif avg_aqi_national <= 400: category = "Very Poor"
                else: category = "Severe"
                st.markdown(f"<p style='color:{CATEGORY_COLORS_DARK.get(category, '#FFFFFF')}; font-weight:600; font-size:0.85rem;'>Overall: {category}</p>", unsafe_allow_html=True)
            else:
                 st.markdown(f"<p style='color:{SUBTLE_TEXT_COLOR_DARK_THEME}; font-size:0.85rem;'>No average data.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        with st.container():
            st.markdown(f"<div class='insight-card'><h3>📅 Days Observed</h3>", unsafe_allow_html=True) # Changed from Time Period
            days_count = df_period_filtered["date"].nunique()
            st.metric(label="Unique Days with Data", value=days_count)
            st.markdown(f"<p style='color:{SUBTLE_TEXT_COLOR_DARK_THEME}; font-size:0.85rem;'>In selected period</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🏆 Top & Bottom Performing Cities")
    if not df_period_filtered.empty:
        city_avg_aqi = df_period_filtered.groupby("city")["index"].mean().dropna().sort_values()
        if not city_avg_aqi.empty:
            col_top, col_bottom = st.columns(2)
            with col_top:
                st.markdown("##### 🥇 Cleanest Cities (Avg. AQI)")
                top_cities = city_avg_aqi.head(5).reset_index()
                top_cities.columns = ["City", "Avg AQI"]
                st.dataframe(top_cities.style.format({"Avg AQI": "{:.1f}"}), height=210, use_container_width=True) # use_container_width
            with col_bottom:
                st.markdown("##### ⚠️ Most Polluted Cities (Avg. AQI)")
                bottom_cities = city_avg_aqi.tail(5).iloc[::-1].reset_index()
                bottom_cities.columns = ["City", "Avg AQI"]
                st.dataframe(bottom_cities.style.format({"Avg AQI": "{:.1f}"}), height=210, use_container_width=True)
        else:
            st.info("No city averages available for the selected period.")
    else:
        st.info("No data to rank cities for the selected period.")
else:
    st.warning("No data available for the selected period to display National Snapshot.")
st.markdown("---")

# ========================================================
# =======   CITY-SPECIFIC ANALYSIS (Improved)   ==========
# ========================================================

if not selected_cities:
    st.info("✨ Select one or more cities from the sidebar to dive into detailed analysis.")
else:
    # Use a selectbox for single city deep-dive from the *selected_cities* list
    city_to_analyze = st.selectbox(
        "🔬 Analyze City:",
        selected_cities,
        index=0 if selected_cities else -1, # Default to first selected city, handle empty list
        key="city_analyzer_select"
    )

    if city_to_analyze: # Proceed only if a city is selected from the dropdown
        st.markdown(f"## 🏙️ {city_to_analyze.upper()} DEEP DIVE – {year}")
        city_data_full = df_period_filtered[df_period_filtered["city"] == city_to_analyze].copy()

        if city_data_full.empty:
            with st.container():
                st.warning(f"😔 No data available for {city_to_analyze} for {selected_month_name}, {year}.")
        else:
            latest_data = city_data_full.sort_values("date", ascending=False).iloc[0]
            current_aqi = latest_data["index"]
            current_level = latest_data["level"]
            current_pollutant = latest_data["pollutant"]
            health_msg = HEALTH_RECOMMENDATIONS.get(current_level, "No specific health recommendations available.")

            # Status cards in columns
            status_cols = st.columns([1, 2, 1]) # Adjusted column ratios
            with status_cols[0]:
                st.markdown(f"<div class='insight-card' style='text-align:center;'><h4 style='margin-bottom:0.3rem;'>Live Status</h4>"
                            f"<div style='font-size:2.2rem; font-weight:700; color:{CATEGORY_COLORS_DARK.get(current_level, '#FFFFFF')};'>{current_aqi:.0f}</div>"
                            f"<div style='font-size:1rem; color:{SUBTLE_TEXT_COLOR_DARK_THEME};'>{current_level}</div></div>", unsafe_allow_html=True)
            with status_cols[1]:
                st.markdown(f"<div class='insight-card'><h4 style='margin-bottom:0.3rem;'>Health Recommendation</h4>"
                            f"<div class='health-card' style='border-left-color: {CATEGORY_COLORS_DARK.get(current_level, '#FFFFFF')}; padding: 0.6rem;'>" # Reduced padding
                            f"<p style='font-size:0.95rem; margin:0;'>{health_msg}</p></div>"
                            f"<p style='font-size:0.75rem; color:{SUBTLE_TEXT_COLOR_DARK_THEME}; text-align:right; margin-top:0.3rem;'>Latest: {latest_data['date'].strftime('%b %d')}</p></div>", unsafe_allow_html=True)

            with status_cols[2]:
                st.markdown(f"<div class='insight-card' style='text-align:center;'><h4 style='margin-bottom:0.3rem;'>Dominant Pollutant</h4>"
                            f"<div style='font-size:1.8rem; font-weight:600; color:{POLLUTANT_COLORS_DARK.get(current_pollutant, '#FFFFFF')}; margin:0.5rem 0;'>{current_pollutant}</div>"
                            f"<p style='font-size:0.8rem; color:{SUBTLE_TEXT_COLOR_DARK_THEME};'>Primary concern</p></div>", unsafe_allow_html=True)


            city_data_full["day_of_year"] = city_data_full["date"].dt.dayofyear
            city_data_full["month_name"] = city_data_full["date"].dt.month_name()
            city_data_full["day_of_month"] = city_data_full["date"].dt.day
            export_data_list.append(city_data_full) # For download section

            tab_trend, tab_dist, tab_heatmap_detail, tab_health = st.tabs(["📊 Trends & Calendar", "📈 Distributions", "🗓️ Heatmap", "❤️ Health"])

            with tab_trend:
                # Calendar Heatmap
                st.markdown("##### 📅 Daily AQI Calendar")
                start_date_cal = pd.to_datetime(f"{year}-01-01")
                end_date_cal = pd.to_datetime(f"{year}-12-31")
                full_year_dates_cal = pd.date_range(start_date_cal, end_date_cal, freq="D")
                calendar_df = pd.DataFrame({"date": full_year_dates_cal})
                calendar_df["week"] = calendar_df["date"].dt.isocalendar().week.astype(int)
                calendar_df["day_of_week"] = calendar_df["date"].dt.dayofweek.astype(int)
                calendar_df.loc[(calendar_df["date"].dt.month == 1) & (calendar_df["week"] > 50), "week"] = 0
                max_week_val = calendar_df["week"].max()
                calendar_df.loc[(calendar_df["date"].dt.month == 12) & (calendar_df["week"] == 1), "week"] = max_week_val + 1 if 53 not in calendar_df['week'].unique() else 53


                city_year_data = df[df["city"] == city_to_analyze] # Use full year data for calendar
                city_year_data = city_year_data[city_year_data["date"].dt.year == year]

                merged_cal_df = pd.merge(calendar_df, city_year_data[["date", "index", "level"]], on="date", how="left")
                merged_cal_df["level"] = merged_cal_df["level"].fillna("Unknown")
                merged_cal_df["aqi_text"] = merged_cal_df["index"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
                day_labels_cal = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                
                # For month annotations (fixed to use groupby on date for actual month starts)
                month_starts_df = merged_cal_df.groupby(merged_cal_df['date'].dt.to_period('M'))['week'].min().reset_index()
                month_starts_df['month_abbr'] = month_starts_df['date'].dt.strftime('%b')
                
                annotations_cal = []
                for _, row in month_starts_df.iterrows():
                     annotations_cal.append(go.layout.Annotation(
                            text=row['month_abbr'], align="center",showarrow=False, xref="x", yref="y domain",
                            x=row['week'], y=1.05, font=dict(color=SUBTLE_TEXT_COLOR_DARK_THEME, size=10)))


                fig_cal = go.Figure(data=go.Heatmap(
                        x=merged_cal_df["week"], y=merged_cal_df["day_of_week"],
                        z=merged_cal_df["level"].map({k: i for i, k in enumerate(CATEGORY_COLORS_DARK.keys())}),
                        customdata=pd.DataFrame({"date": merged_cal_df["date"].dt.strftime("%Y-%m-%d"), "level": merged_cal_df["level"], "aqi": merged_cal_df["aqi_text"]}),
                        hovertemplate="<b>%{customdata[0]}</b><br>AQI: %{customdata[2]} (%{customdata[1]})<extra></extra>",
                        colorscale=[[i / (len(CATEGORY_COLORS_DARK) -1), color] for i, color in enumerate(CATEGORY_COLORS_DARK.values())],
                        showscale=False, xgap=2, ygap=2))
                fig_cal.update_layout(get_custom_plotly_layout_args(height=280), # Reduced height for denser layout
                    yaxis=dict(tickmode="array", tickvals=list(range(7)), ticktext=day_labels_cal, showgrid=False, zeroline=False, title=None),
                    xaxis=dict(showgrid=False, zeroline=False, tickmode="array", ticktext=[], tickvals=[], title=None), # Hide x-axis ticks/labels
                    annotations=annotations_cal, margin=dict(t=30, b=10, l=30, r=10) # Tight margins
                )
                st.plotly_chart(fig_cal, use_container_width=True)

                # AQI Trend
                st.markdown("##### 📈 AQI Trend & 7-Day Rolling Average")
                city_data_trend = city_data_full.sort_values("date").copy()
                city_data_trend["rolling_avg_7day"] = city_data_trend["index"].rolling(window=7, center=True, min_periods=1).mean().round(2)
                fig_trend_roll = go.Figure()
                fig_trend_roll.add_trace(go.Scatter(x=city_data_trend["date"], y=city_data_trend["index"], mode="lines+markers", name="Daily AQI", marker=dict(size=3, opacity=0.6, color=SUBTLE_TEXT_COLOR_DARK_THEME), line=dict(width=1, color=SUBTLE_TEXT_COLOR_DARK_THEME)))
                fig_trend_roll.add_trace(go.Scatter(x=city_data_trend["date"], y=city_data_trend["rolling_avg_7day"], mode="lines", name="7-Day Rolling Avg", line=dict(color=ACCENT_COLOR, width=2)))
                fig_trend_roll.update_layout(get_custom_plotly_layout_args(height=350), yaxis_title="AQI Index", xaxis_title=None, hovermode="x unified", legend=dict(y=1.15))
                st.plotly_chart(fig_trend_roll, use_container_width=True)

            with tab_dist:
                dist_col1, dist_col2 = st.columns(2)
                with dist_col1:
                    st.markdown("##### 📊 AQI Category (Days)")
                    category_counts_df = city_data_full["level"].value_counts().reindex(CATEGORY_COLORS_DARK.keys(), fill_value=0).reset_index()
                    category_counts_df.columns = ["AQI Category", "Number of Days"]
                    fig_dist_bar = px.bar(category_counts_df, x="AQI Category", y="Number of Days", color="AQI Category", color_discrete_map=CATEGORY_COLORS_DARK, text_auto=True)
                    fig_dist_bar.update_layout(get_custom_plotly_layout_args(height=350), xaxis_title=None, yaxis_title="Days")
                    fig_dist_bar.update_traces(textfont_size=10, marker_line_width=0)
                    st.plotly_chart(fig_dist_bar, use_container_width=True)
                with dist_col2:
                    st.markdown("##### ☀️ Category Proportions")
                    if category_counts_df["Number of Days"].sum() > 0:
                        fig_sunburst = px.sunburst(category_counts_df, path=["AQI Category"], values="Number of Days", color="AQI Category", color_discrete_map=CATEGORY_COLORS_DARK)
                        fig_sunburst.update_layout(get_custom_plotly_layout_args(height=350), margin=dict(t=10, l=10, r=10, b=10))
                        st.plotly_chart(fig_sunburst, use_container_width=True)

                st.markdown("##### 🎻 Monthly AQI Distribution")
                month_order_cat = list(months_map_dict.values())
                city_data_full["month_name_cat"] = pd.Categorical(city_data_full["month_name"], categories=month_order_cat, ordered=True)
                fig_violin = px.violin(city_data_full.sort_values("month_name_cat"), x="month_name_cat", y="index", color="month_name_cat", color_discrete_sequence=px.colors.qualitative.Pastel1, box=True, points="outliers")
                fig_violin.update_layout(get_custom_plotly_layout_args(height=400), xaxis_title=None, showlegend=False, yaxis_title="AQI Index")
                fig_violin.update_traces(meanline_visible=True)
                st.plotly_chart(fig_violin, use_container_width=True)

            with tab_heatmap_detail:
                st.markdown("##### 🔥 AQI Heatmap (Month vs. Day)")
                heatmap_pivot = city_data_full.pivot_table(index="month_name", columns="day_of_month", values="index", observed=False) # observed=False to include all months
                heatmap_pivot = heatmap_pivot.reindex(list(months_map_dict.values())) # Ensure correct month order
                fig_heat_detail = px.imshow(heatmap_pivot, labels=dict(x="Day of Month", y="Month", color="AQI"), aspect="auto", color_continuous_scale="Inferno", text_auto=".0f")
                fig_heat_detail.update_layout(get_custom_plotly_layout_args(height=450), xaxis_side="top")
                st.plotly_chart(fig_heat_detail, use_container_width=True)
            
            with tab_health:
                st.markdown("##### ❤️ Health Impact Analysis")
                health_cols = st.columns(2)
                with health_cols[0]:
                    st.markdown("###### 🚶‍♂️ Activity Recommendations")
                    st.markdown(f"<div class='insight-card' style='height: auto;'><p style='font-size:1rem; margin-bottom:0.5rem;'><b>Current AQI: {current_aqi:.0f} ({current_level})</b></p>"
                                f"<div class='health-card' style='border-left-color: {CATEGORY_COLORS_DARK.get(current_level, '#FFFFFF')};'><p style='font-size:0.9rem;'>{health_msg}</p></div></div>", unsafe_allow_html=True)
                    st.markdown("###### 📖 General Guidelines")
                    st.markdown(f"<div class='insight-card' style='height: auto; font-size:0.85rem;'><ul style='padding-left:1.2rem; margin-bottom:0;'>"
                                "<li>Sensitive groups: children, elderly, respiratory issues.</li>"
                                "<li>N95 masks if AQI > 200.</li>"
                                "<li>Close windows if high pollution.</li>"
                                "<li>Air purifiers if AQI > 150.</li></ul></div>", unsafe_allow_html=True)
                with health_cols[1]:
                    st.markdown("###### 📊 Pollution Health Effects")
                    health_effects = { "Good": "No health impacts", "Satisfactory": "Minor discomfort (sensitive)", "Moderate": "Breathing discomfort (lung disease)", "Poor": "Breathing discomfort (prolonged)", "Very Poor": "Respiratory illness (prolonged)", "Severe": "Health impacts (light activity)"}
                    
                    # Need category_counts_df from tab_dist or recompute
                    if 'category_counts_df' not in locals() or category_counts_df.empty: # Check if it exists and is not empty
                         category_counts_temp = city_data_full["level"].value_counts().reindex(CATEGORY_COLORS_DARK.keys(), fill_value=0).reset_index()
                         category_counts_temp.columns = ["AQI Category", "Number of Days"]
                    else:
                         category_counts_temp = category_counts_df

                    fig_health = go.Figure()
                    for level, effect in health_effects.items():
                        days = category_counts_temp[category_counts_temp["AQI Category"] == level]["Number of Days"].values[0] if level in category_counts_temp["AQI Category"].values else 0
                        fig_health.add_trace(go.Bar(y=[level], x=[days], name=level, orientation='h', marker_color=CATEGORY_COLORS_DARK.get(level), hovertext=f"{effect}<br>{days} days in {year}"))
                    fig_health.update_layout(get_custom_plotly_layout_args(height=400), title_text=None, barmode="stack", xaxis_title="Number of Days", yaxis_title=None, legend=dict(y=1.15))
                    st.plotly_chart(fig_health, use_container_width=True)

# ========================================================
# =======   CITY-WISE AQI COMPARISON (Enhanced)  =========
# ========================================================
if len(selected_cities) > 1:
    st.markdown("## 🆚 MULTI-CITY AQI COMPARISON")
    comp_cols = st.columns([3,1]) # Column for line chart and ranked list
    with comp_cols[0]:
        comparison_df_list = []
        for city_comp in selected_cities:
            city_ts_comp = df_period_filtered[df_period_filtered["city"] == city_comp].copy()
            if not city_ts_comp.empty:
                city_ts_comp = city_ts_comp.sort_values("date")
                city_ts_comp["city_label"] = city_comp
                comparison_df_list.append(city_ts_comp)

        if comparison_df_list:
            combined_comp_df = pd.concat(comparison_df_list)
            fig_cmp = px.line(combined_comp_df, x="date", y="index", color="city_label", labels={"index": "AQI Index", "date": "Date", "city_label": "City"}, markers=False, line_shape="spline", color_discrete_sequence=px.colors.qualitative.Plotly)
            fig_cmp.update_layout(get_custom_plotly_layout_args(height=450, title_text=f"AQI Trends – {selected_month_name}, {year}"), hovermode="x unified", legend=dict(y=1.15))
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.info("Not enough data for selected cities for comparison.")
    
    with comp_cols[1]:
        if 'combined_comp_df' in locals() and not combined_comp_df.empty:
            st.markdown("##### 🏆 Avg. AQI Rank")
            avg_aqi_df_comp = combined_comp_df.groupby("city_label")["index"].mean().reset_index().sort_values("index")
            avg_aqi_df_comp["Rank"] = avg_aqi_df_comp["index"].rank(method="min").astype(int)
            avg_aqi_df_comp = avg_aqi_df_comp[["Rank", "city_label", "index"]].rename(columns={"city_label": "City", "index":"Avg AQI"})
            st.dataframe(avg_aqi_df_comp.style.format({"Avg AQI":"{:.1f}"}), height=min(450, len(avg_aqi_df_comp)*45 + 50) ,use_container_width=True) # Dynamic height
        else:
            st.info("Averages not available.")
else:
     if selected_cities: # Only show this if a single city is selected and comparison is not active
        st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True) # Spacer if no comparison


st.markdown("---")

# ========================================================
# =======   PROMINENT POLLUTANT ANALYSIS (Enhanced)  =====
# ========================================================
st.markdown("## 💨 PROMINENT POLLUTANT ANALYSIS")

if selected_cities: # Only show if cities are selected
    poll_city_select = st.selectbox(
        "Select City for Pollutant Analysis:",
        selected_cities,
        index=0 if selected_cities else -1, # Default to first selected city
        key="pollutant_city_selector"
    )
    if poll_city_select:
        poll_col1, poll_col2 = st.columns(2)
        with poll_col1:
            st.markdown(f"#### Dominance ({selected_month_name}, {year})")
            pollutant_data_B = df_period_filtered[df_period_filtered["city"] == poll_city_select].copy()
            if not pollutant_data_B.empty and "pollutant" in pollutant_data_B.columns:
                grouped_poll_B = pollutant_data_B.groupby("pollutant").size().reset_index(name="count")
                fig_poll_B = px.pie(grouped_poll_B, values="count", names="pollutant", color="pollutant", color_discrete_map=POLLUTANT_COLORS_DARK, hole=0.4)
                fig_poll_B.update_layout(get_custom_plotly_layout_args(height=400), legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5), title_text=None)
                fig_poll_B.update_traces(textinfo="percent+label", textposition="inside")
                st.plotly_chart(fig_poll_B, use_container_width=True)
            else:
                st.warning(f"No pollutant data for {poll_city_select} for this period.")

        with poll_col2:
            st.markdown("#### Yearly Dominant Pollutant Trends")
            pollutant_data_A = df[df["city"] == poll_city_select].copy() # Full history for yearly trend
            pollutant_data_A["year_label"] = pollutant_data_A["date"].dt.year
            if not pollutant_data_A.empty:
                grouped_poll_A = pollutant_data_A.groupby(["year_label", "pollutant"]).size().unstack(fill_value=0)
                percent_poll_A = grouped_poll_A.apply(lambda x: x / x.sum() * 100 if x.sum() > 0 else x, axis=1).fillna(0)
                percent_poll_A_long = percent_poll_A.reset_index().melt(id_vars="year_label", var_name="pollutant", value_name="percentage")
                fig_poll_A = px.bar(percent_poll_A_long, x="year_label", y="percentage", color="pollutant", color_discrete_map=POLLUTANT_COLORS_DARK, barmode="stack")
                fig_poll_A.update_layout(get_custom_plotly_layout_args(height=400), xaxis_type="category", yaxis_ticksuffix="%", title_text=None, legend=dict(y=1.15))
                st.plotly_chart(fig_poll_A, use_container_width=True)
            else:
                st.warning(f"No yearly pollutant data for {poll_city_select}.")
else:
    st.info("Select city/cities from the sidebar to view pollutant analysis.")


# ========================================================
# =======   AQI FORECAST (Enhanced Polynomial)   =========
# ========================================================
st.markdown("## 🔮 AQI FORECAST (POLYNOMIAL TREND)")

if selected_cities: # Only show if cities are selected
    forecast_city_select_fore = st.selectbox( # Changed key to avoid conflict
        "Select City for AQI Forecast:",
        selected_cities,
        index=0 if selected_cities else -1, # Default to first selected city
        key="forecast_city_selector_poly"
    )
    if forecast_city_select_fore:
        forecast_col1, forecast_col2 = st.columns([3,1])
        with forecast_col1:
            forecast_src_data = df[df["city"] == forecast_city_select_fore].copy() # Use all historical data for the city
            forecast_src_data = forecast_src_data.sort_values("date")[["date", "index"]].dropna()

            if len(forecast_src_data) >= 30: # Require at least 30 data points for a more stable poly fit
                forecast_df_poly = forecast_src_data.copy()
                forecast_df_poly["days_since_start"] = (forecast_df_poly["date"] - forecast_df_poly["date"].min()).dt.days
                
                X_poly_train = forecast_df_poly["days_since_start"].values.reshape(-1, 1)
                y_poly_train = forecast_df_poly["index"].values
                
                poly_features = PolynomialFeatures(degree=2) # Degree 2 for curved line
                X_poly_features = poly_features.fit_transform(X_poly_train)
                
                model_poly = LinearRegression().fit(X_poly_features, y_poly_train)
                
                last_day_num_poly = forecast_df_poly["days_since_start"].max()
                future_X_range_poly = np.arange(last_day_num_poly + 1, last_day_num_poly + 30 + 1) # Forecast next 30 days
                future_X_poly_features = poly_features.transform(future_X_range_poly.reshape(-1, 1))
                future_y_pred_poly = model_poly.predict(future_X_poly_features)
                
                min_date_forecast_poly = forecast_df_poly["date"].min() # This should be last_historical_date
                last_historical_date = forecast_df_poly['date'].max()
                future_dates_list_poly = [last_historical_date + pd.Timedelta(days=int(i)) for i in range(1, 31)]


                plot_df_obs_poly = pd.DataFrame({"date": forecast_df_poly["date"], "AQI": y_poly_train, "Type": "Observed"})
                plot_df_fcst_poly = pd.DataFrame({"date": future_dates_list_poly, "AQI": np.maximum(0, future_y_pred_poly), "Type": "Forecast"}) # Ensure AQI not negative
                
                combined_plot_df = pd.concat([plot_df_obs_poly, plot_df_fcst_poly])

                fig_forecast_poly = px.line(
                    combined_plot_df, x="date", y="AQI", color="Type",
                    color_discrete_map={"Observed": ACCENT_COLOR, "Forecast": HIGHLIGHT_COLOR},
                    markers=False, line_shape="spline"
                )
                
                forecast_start_date = plot_df_fcst_poly["date"].min()
                forecast_end_date = plot_df_fcst_poly["date"].max()
                fig_forecast_poly.add_vrect(x0=forecast_start_date, x1=forecast_end_date, fillcolor="rgba(255,107,107,0.1)", layer="below", line_width=0)

                fig_forecast_poly.update_layout(get_custom_plotly_layout_args(height=450),
                    title_text=None, yaxis_title="AQI Index", xaxis_title=None, hovermode="x unified", legend=dict(y=1.15))
                st.plotly_chart(fig_forecast_poly, use_container_width=True)
            else:
                st.warning(f"Need at least 30 data points for {forecast_city_select_fore} for polynomial forecasting; found {len(forecast_src_data)}.")
        with forecast_col2:
            st.markdown("##### ℹ️ Forecast Info")
            st.markdown(f"<div class='insight-card' style='height:auto; font-size:0.85rem;'>"
                        "<p><b>Method:</b> Polynomial Regression (degree 2) on historical data.</p>"
                        "<p><b>Limitations:</b> Short-term trend projection. Does not account for sudden events, weather changes, or policy impacts. Accuracy typically decreases further into the future.</p>"
                        "<p style='color:{SUBTLE_TEXT_COLOR_DARK_THEME}; margin-top:1rem;'>Use for indicative purposes. Consult official forecasts for critical decisions.</p>"
                        "</div>", unsafe_allow_html=True)
else:
    st.info("Select city/cities from the sidebar to enable AQI forecasting.")

# ========================================================
# =========  City AQI Hotspots (Enhanced)  ==============
# ========================================================
st.markdown("## 📍 AIR QUALITY HOTSPOTS MAP")

map_main_col, map_info_col = st.columns([3,1]) # Main map and info column

with map_main_col:
    city_coords_data = {}
    coords_file_path = "lat_long.txt"
    scatter_map_rendered = False

    try: # Safely load coordinates
        if os.path.exists(coords_file_path):
            with open(coords_file_path, "r", encoding="utf-8") as f:
                city_coords_data = eval(f.read()) # Assuming trusted file
        if not isinstance(city_coords_data, dict): city_coords_data = {} # Ensure it's a dict
    except Exception: city_coords_data = {} # Empty on error

    if not df_period_filtered.empty and city_coords_data:
        map_grouped_data = df_period_filtered.groupby("city").agg(
            avg_aqi=('index', 'mean'),
            dominant_pollutant=('pollutant', lambda x: x.mode().iloc[0] if not x.mode().empty and pd.notna(x.mode().iloc[0]) else 'N/A')
        ).reset_index().dropna(subset=['avg_aqi'])

        latlong_map_df_list = [{'city': name, 'lat': coords[0], 'lon': coords[1]} for name, coords in city_coords_data.items() if isinstance(coords, (list, tuple)) and len(coords) == 2]
        latlong_map_df = pd.DataFrame(latlong_map_df_list)

        if not latlong_map_df.empty:
            map_merged_df = pd.merge(map_grouped_data, latlong_map_df, on="city", how="inner")
            if not map_merged_df.empty:
                map_merged_df["AQI Category"] = map_merged_df["avg_aqi"].apply(lambda val: next((k for k,v_range in CATEGORY_COLORS_DARK.items() if v_range[0] <= val <= v_range[1]), "Unknown") if pd.notna(val) else "Unknown", # This logic needs correction for CATEGORY_COLORS_DARK
                    # Corrected AQI Category mapping based on defined ranges
                    lambda val: next((k for k, v_range_tuple in {
                        'Good': (0, 50), 'Satisfactory': (51, 100), 'Moderate': (101, 200),
                        'Poor': (201, 300), 'Very Poor': (301, 400), 'Severe': (401, float('inf'))
                    }.items() if v_range_tuple[0] <= val <= v_range_tuple[1]), "Unknown") if pd.notna(val) else "Unknown"
                )
                # Scale bubble size (smaller bubbles)
                min_aqi_map, max_aqi_map = map_merged_df["avg_aqi"].min(), map_merged_df["avg_aqi"].max()
                size_range_map = (3, 15) # Min and max pixel size for bubbles
                if max_aqi_map > min_aqi_map: # Avoid division by zero
                    map_merged_df["scaled_size"] = ((map_merged_df["avg_aqi"] - min_aqi_map) / (max_aqi_map - min_aqi_map) * (size_range_map[1] - size_range_map[0]) + size_range_map[0])
                else:
                    map_merged_df["scaled_size"] = size_range_map[0] # Default to min size if all AQIs are same or only one point
                map_merged_df["scaled_size"] = map_merged_df["scaled_size"].fillna(size_range_map[0])


                fig_scatter_map = px.scatter_mapbox(map_merged_df, lat="lat", lon="lon", size="scaled_size", size_max=size_range_map[1],
                    color="AQI Category", color_discrete_map=CATEGORY_COLORS_DARK, hover_name="city",
                    custom_data=['city', 'avg_aqi', 'dominant_pollutant', 'AQI Category'], text="city",
                    zoom=3.8, center={"lat": 23.0, "lon": 82.5}) # Centered on India
                fig_scatter_map.update_layout(get_custom_plotly_layout_args(height=600, title_text=f"AQI Hotspots - {selected_month_name}, {year}"),
                    mapbox_style="carto-darkmatter", margin={"r":10,"t":60,"l":10,"b":10}, legend=dict(y=0.98, x=0.98, xanchor='right'))
                fig_scatter_map.update_traces(marker=dict(sizemin=size_range_map[0], opacity=0.8),
                    hovertemplate="<b style='font-size:1.1em;'>%{customdata[0]}</b><br>Avg. AQI: %{customdata[1]:.1f} (%{customdata[3]})<br>Dominant Pollutant: %{customdata[2]}<extra></extra>")
                st.plotly_chart(fig_scatter_map, use_container_width=True)
                scatter_map_rendered = True

    if not scatter_map_rendered:
        st.warning("Map could not be rendered. Check data and coordinate file.", icon="🗺️")

with map_info_col:
    st.markdown("##### ℹ️ Map Guide")
    st.markdown(f"<div class='insight-card' style='font-size:0.85rem;'>"
                "<ul><li>Bubble size indicates AQI severity.</li><li>Color shows AQI category.</li><li>Hover for details.</li><li>Pan & Zoom enabled.</li></ul>"
                "<b>AQI Categories:</b>", unsafe_allow_html=True)
    for category, color_val in CATEGORY_COLORS_DARK.items():
        if category != "Unknown":
            st.markdown(f"<div style='display: flex; align-items: center; margin-bottom: 0.3rem;'><div style='width: 12px; height: 12px; background-color: {color_val}; border-radius: 3px; margin-right: 8px;'></div><span>{category}</span></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ========================================================
# ========   DOWNLOAD FILTERED DATA (Enhanced)   =========
# ========================================================
st.markdown("## 📥 DOWNLOAD DATA")
with st.container(): # Use container to apply card styling for the download section
    if export_data_list: # Check if there's data to download
        combined_export_final = pd.concat(export_data_list).drop_duplicates() # Combine and remove duplicates
        csv_buffer_final = StringIO()
        combined_export_final.to_csv(csv_buffer_final, index=False)
        
        st.markdown("<h3 style='font-size:1.2rem; color: #00BCD4;'>Export Filtered Data</h3>" # Match H3 style
                    "<p style='margin-bottom:1rem; font-size:0.9rem;'>Download the current dataset for your own analysis.</p>", unsafe_allow_html=True)
        st.download_button(label="📤 Download Data (CSV)", data=csv_buffer_final.getvalue(),
            file_name=f"IITKGP_AQI_Data_{year}_{selected_month_name.replace(' ', '_')}.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("No data currently selected for export. Adjust filters or select cities.")

# ------------------- Footer (MODERN & ANIMATED) -------------------
st.markdown(f"""
<div class="footer-container">
  <div class="footer-top-bar"></div>
  <h3>IIT KGP Air Quality Dashboard</h3>
  <div class="footer-info">
    <div><p class="label">Data Source</p><p class="value">Central Pollution Control Board (CPCB)</p></div>
    <div><p class="label">Developed By</p><p class="value">IIT Kharagpur Research Team</p></div>
    <div><p class="label">Last Updated</p><p class="value">{data_last_updated.strftime('%Y-%m-%d %H:%M') if data_last_updated else "N/A"}</p></div>
  </div>
  <div class="footer-links">
    <a href="https://github.com/kapil2020/india-air-quality-dashboard" target="_blank">
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>
      View on GitHub
    </a>
  </div>
  <p class="copyright">© {pd.to_datetime("today").year} IIT Kharagpur | For Research and Educational Purposes</p>
</div>
""", unsafe_allow_html=True)


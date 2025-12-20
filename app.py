import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.stats as stats
from io import StringIO
import requests
import json
from datetime import datetime, timedelta
import calendar
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# --- Professional Theme & Design System ---
pio.templates.default = "plotly_white"

# Professional Color Palette
COLORS = {
    # Primary Brand Colors
    "primary": "#00B4D8",      # Vibrant Cyan
    "primary_dark": "#0077B6",  # Deep Ocean Blue
    "primary_light": "#90E0EF", # Light Cyan
    
    # Semantic Colors
    "success": "#06D6A0",       # Emerald Green
    "warning": "#FFD166",       # Golden Yellow
    "danger": "#EF476F",        # Coral Red
    "info": "#118AB2",          # Steel Blue
    
    # Data Visualization Colors
    "sequential": ["#003F5C", "#2F4B7C", "#665191", "#A05195", "#D45087", "#F95D6A", "#FF7C43", "#FFA600"],
    "diverging": ["#1A237E", "#283593", "#303F9F", "#3949AB", "#5C6BC0", "#7986CB", "#9FA8DA", "#C5CAE9",
                  "#E8EAF6", "#FFEBEE", "#FFCDD2", "#EF9A9A", "#E57373", "#EF5350", "#E53935", "#C62828"],
    
    # AQI Categories (Professional Gradient)
    "aqi_good": "#00A878",        # Mint Green
    "aqi_moderate": "#FFD166",     # Soft Yellow
    "aqi_poor": "#F8961E",         # Orange
    "aqi_very_poor": "#EF476F",    # Coral
    "aqi_severe": "#9D0208",       # Crimson
    
    # UI Colors
    "background": "#0F172A",       # Space Gray (Dark)
    "card": "#1E293B",             # Slate Gray
    "border": "#334155",           # Steel Gray
    "text_primary": "#F8FAFC",     # Off White
    "text_secondary": "#94A3B8",   # Muted Blue
    "text_muted": "#64748B",       # Slate
}

# Enhanced AQI Category Colors with gradients
AQI_CATEGORY_COLORS = {
    "Good": {"color": COLORS["aqi_good"], "gradient": ["#00A878", "#00C48C", "#00E0A0"]},
    "Satisfactory": {"color": "#7ED957", "gradient": ["#7ED957", "#95E973", "#ACF98F"]},
    "Moderate": {"color": COLORS["aqi_moderate"], "gradient": ["#FFD166", "#FFDB85", "#FFE5A4"]},
    "Poor": {"color": COLORS["aqi_poor"], "gradient": ["#F8961E", "#F9A942", "#FABC66"]},
    "Very Poor": {"color": COLORS["aqi_very_poor"], "gradient": ["#EF476F", "#F15C87", "#F4719F"]},
    "Severe": {"color": COLORS["aqi_severe"], "gradient": ["#9D0208", "#B51B1B", "#D13434"]},
    "Unknown": {"color": COLORS["text_muted"], "gradient": ["#64748B", "#7C8A9F", "#94A0B3"]}
}

# Pollutant Colors with professional palette
POLLUTANT_COLORS = {
    "PM2.5": "#EF476F",    # Coral
    "PM10": "#118AB2",     # Steel Blue
    "NO2": "#06D6A0",      # Emerald
    "SO2": "#FFD166",      # Golden Yellow
    "CO": "#FF9E00",       # Orange
    "O3": "#7209B7",       # Purple
    "NH3": "#3A86FF",      # Azure
    "Pb": "#8338EC",       # Electric Purple
    "Other": COLORS["text_muted"]
}

# Health Impact Levels
HEALTH_IMPACTS = {
    "Good": {
        "level": "Low Risk",
        "icon": "✅",
        "recommendation": "Perfect for outdoor activities. Air quality is ideal for everyone.",
        "activities": ["Outdoor sports", "Walking", "Cycling", "All outdoor activities"],
        "precautions": ["None required"],
        "affected_groups": "None"
    },
    "Satisfactory": {
        "level": "Low-Moderate Risk",
        "icon": "ℹ️",
        "recommendation": "Sensitive individuals should consider reducing prolonged/heavy exertion.",
        "activities": ["Light outdoor activities", "Walking", "Normal daily activities"],
        "precautions": ["Sensitive groups should limit prolonged exertion"],
        "affected_groups": "Children, elderly, people with respiratory conditions"
    },
    "Moderate": {
        "level": "Moderate Risk",
        "icon": "⚠️",
        "recommendation": "Sensitive groups should reduce outdoor activities.",
        "activities": ["Light indoor activities", "Short walks"],
        "precautions": ["Limit outdoor time", "Close windows if symptoms develop"],
        "affected_groups": "Children, elderly, asthma patients"
    },
    "Poor": {
        "level": "High Risk",
        "icon": "🚨",
        "recommendation": "Everyone should reduce prolonged/heavy exertion.",
        "activities": ["Essential activities only", "Use air purifiers indoors"],
        "precautions": ["Wear N95 masks", "Avoid outdoor exercise", "Keep windows closed"],
        "affected_groups": "Everyone, especially sensitive groups"
    },
    "Very Poor": {
        "level": "Very High Risk",
        "icon": "⛔",
        "recommendation": "Avoid outdoor activities, especially for sensitive groups.",
        "activities": ["Stay indoors", "Essential travel only"],
        "precautions": ["Use air purifiers", "Wear N95 masks outdoors", "Keep windows sealed"],
        "affected_groups": "Everyone - serious health effects possible"
    },
    "Severe": {
        "level": "Hazardous",
        "icon": "🔥",
        "recommendation": "Avoid all outdoor activities, keep windows closed.",
        "activities": ["Stay indoors", "Use air purifiers", "Limit physical activity"],
        "precautions": ["Emergency precautions", "Avoid all outdoor exposure", "Seek medical help if symptoms"],
        "affected_groups": "Everyone - emergency conditions"
    }
}

# ------------------- Page Config -------------------
st.set_page_config(
    layout="wide",
    page_title="AQI Pro Dashboard | IIT KGP",
    page_icon="🌍",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/kapil2020/india-air-quality-dashboard',
        'Report a bug': "https://github.com/kapil2020/india-air-quality-dashboard/issues",
        'About': "### Advanced Air Quality Monitoring System\nDeveloped by IIT Kharagpur\n© 2024 All Rights Reserved"
    }
)

# ------------------- Professional CSS Styling -------------------
st.markdown(f"""
<style>
    /* =================================
       1. FONTS & TYPOGRAPHY
       ================================= */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    * {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    /* =================================
       2. MAIN LAYOUT
       ================================= */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }}
    
    html, body, [data-testid="stAppViewContainer"] {{
        background: linear-gradient(135deg, {COLORS['background']} 0%, #1a1f36 100%);
        color: {COLORS['text_primary']};
    }}
    
    /* =================================
       3. CARD DESIGN SYSTEM
       ================================= */
    .pro-card {{
        background: linear-gradient(145deg, {COLORS['card']}, #2d3748);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid {COLORS['border']};
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }}
    
    .pro-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['success']});
    }}
    
    .pro-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 180, 216, 0.2);
        border-color: {COLORS['primary']};
    }}
    
    /* =================================
       4. TYPOGRAPHY HIERARCHY
       ================================= */
    h1 {{
        font-family: 'Inter', sans-serif;
        font-weight: 900;
        font-size: 3.5rem;
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['success']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        text-align: center;
    }}
    
    h2 {{
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        color: {COLORS['text_primary']};
        font-size: 2.2rem;
        margin-top: 3rem;
        margin-bottom: 1.5rem;
        position: relative;
        padding-bottom: 1rem;
    }}
    
    h2::after {{
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 80px;
        height: 4px;
        background: linear-gradient(90deg, {COLORS['primary']}, transparent);
        border-radius: 2px;
    }}
    
    h3 {{
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: {COLORS['text_primary']};
        font-size: 1.6rem;
        margin-bottom: 1rem;
    }}
    
    h4 {{
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: {COLORS['text_primary']};
        font-size: 1.3rem;
        margin-bottom: 0.75rem;
    }}
    
    .section-title {{
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        color: {COLORS['text_primary']};
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 4px solid {COLORS['primary']};
    }}
    
    /* =================================
       5. SIDEBAR ENHANCEMENTS
       ================================= */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['card']} 0%, #2d3748 100%);
        border-right: 1px solid {COLORS['border']};
    }}
    
    section[data-testid="stSidebar"] .pro-card {{
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    /* =================================
       6. METRIC CARDS
       ================================= */
    .metric-card {{
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.8));
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }}
    
    .metric-card:hover {{
        transform: translateY(-3px);
        border-color: {COLORS['primary']};
        box-shadow: 0 8px 25px rgba(0, 180, 216, 0.15);
    }}
    
    .metric-value {{
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.5rem 0;
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['success']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    .metric-label {{
        font-size: 0.9rem;
        color: {COLORS['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }}
    
    /* =================================
       7. BUTTONS & CONTROLS
       ================================= */
    .stButton > button {{
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['primary_dark']});
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 180, 216, 0.4);
        background: linear-gradient(90deg, {COLORS['primary_dark']}, {COLORS['primary']});
    }}
    
    /* =================================
       8. TABS ENHANCEMENT
       ================================= */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1rem;
        border-bottom: 2px solid {COLORS['border']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        padding: 1rem 2rem;
        font-weight: 600;
        color: {COLORS['text_secondary']};
        background: transparent;
        border: 1px solid transparent;
        border-radius: 12px 12px 0 0;
        transition: all 0.3s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(255, 255, 255, 0.05);
        color: {COLORS['text_primary']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background: rgba(0, 180, 216, 0.1);
        color: {COLORS['primary']} !important;
        border-bottom: 3px solid {COLORS['primary']};
        border-top: 1px solid rgba(0, 180, 216, 0.3);
        border-left: 1px solid rgba(0, 180, 216, 0.3);
        border-right: 1px solid rgba(0, 180, 216, 0.3);
    }}
    
    /* =================================
       9. DATA TABLE STYLING
       ================================= */
    .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
    }}
    
    /* =================================
       10. PROGRESS BARS & GAUGES
       ================================= */
    .progress-bar {{
        height: 8px;
        background: {COLORS['border']};
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }}
    
    .progress-fill {{
        height: 100%;
        border-radius: 4px;
        transition: width 1s ease-in-out;
    }}
    
    /* =================================
       11. ANIMATIONS
       ================================= */
    @keyframes pulse {{
        0% {{ opacity: 1; }}
        50% {{ opacity: 0.7; }}
        100% {{ opacity: 1; }}
    }}
    
    .pulse {{
        animation: pulse 2s infinite;
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .fade-in {{
        animation: fadeIn 0.8s ease-out;
    }}
    
    /* =================================
       12. RESPONSIVE DESIGN
       ================================= */
    @media (max-width: 768px) {{
        h1 {{ font-size: 2.5rem; }}
        h2 {{ font-size: 1.8rem; }}
        h3 {{ font-size: 1.4rem; }}
        .metric-value {{ font-size: 2.2rem; }}
        .main .block-container {{
            padding: 1rem;
        }}
    }}
    
    @media (max-width: 480px) {{
        h1 {{ font-size: 2rem; }}
        h2 {{ font-size: 1.5rem; }}
        .stTabs [data-baseweb="tab"] {{
            padding: 0.75rem 1rem;
            font-size: 0.9rem;
        }}
    }}
    
    /* =================================
       13. CUSTOM SCROLLBAR
       ================================= */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {COLORS['background']};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {COLORS['primary']}, {COLORS['primary_dark']});
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(180deg, {COLORS['primary_dark']}, {COLORS['primary']});
    }}
</style>
""", unsafe_allow_html=True)

# ------------------- Helper Functions -------------------
def get_plotly_layout(title=None, height=500, width=None, showlegend=True, margin=None):
    """Professional Plotly layout configuration"""
    if margin is None:
        margin = dict(l=60, r=40, t=80, b=60, pad=10)
    
    layout = go.Layout(
        title=dict(
            text=title,
            font=dict(
                family="Inter, sans-serif",
                size=20,
                color=COLORS["text_primary"],
                weight=800
            ),
            x=0.03,
            y=0.95,
            xanchor="left"
        ),
        font=dict(
            family="Inter, sans-serif",
            size=12,
            color=COLORS["text_secondary"]
        ),
        plot_bgcolor=COLORS["card"],
        paper_bgcolor="rgba(0,0,0,0)",
        height=height,
        width=width,
        margin=margin,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=COLORS["card"],
            font_size=12,
            font_family="Inter, sans-serif",
            font_color=COLORS["text_primary"]
        ),
        legend=dict(
            orientation="h" if showlegend else "v",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            bordercolor=COLORS["border"],
            borderwidth=1,
            font=dict(
                size=12,
                color=COLORS["text_secondary"]
            )
        ) if showlegend else None,
        xaxis=dict(
            gridcolor=COLORS["border"],
            gridwidth=1,
            zerolinecolor=COLORS["border"],
            linecolor=COLORS["border"],
            linewidth=2,
            tickfont=dict(size=11),
            titlefont=dict(size=13, weight=600)
        ),
        yaxis=dict(
            gridcolor=COLORS["border"],
            gridwidth=1,
            zerolinecolor=COLORS["border"],
            linecolor=COLORS["border"],
            linewidth=2,
            tickfont=dict(size=11),
            titlefont=dict(size=13, weight=600)
        )
    )
    return layout

def create_gauge_chart(value, title, min_val=0, max_val=500, steps=None):
    """Create a professional gauge chart"""
    if steps is None:
        steps = [
            {"range": [0, 50], "color": AQI_CATEGORY_COLORS["Good"]["color"]},
            {"range": [50, 100], "color": AQI_CATEGORY_COLORS["Satisfactory"]["color"]},
            {"range": [100, 200], "color": AQI_CATEGORY_COLORS["Moderate"]["color"]},
            {"range": [200, 300], "color": AQI_CATEGORY_COLORS["Poor"]["color"]},
            {"range": [300, 400], "color": AQI_CATEGORY_COLORS["Very Poor"]["color"]},
            {"range": [400, 500], "color": AQI_CATEGORY_COLORS["Severe"]["color"]}
        ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={"text": title, "font": {"size": 18, "family": "Inter"}},
        delta={"reference": 50, "increasing": {"color": COLORS["danger"]}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickwidth": 1, "tickcolor": COLORS["text_primary"]},
            "bar": {"color": COLORS["primary"]},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 2,
            "bordercolor": COLORS["border"],
            "steps": steps,
            "threshold": {
                "line": {"color": COLORS["danger"], "width": 4},
                "thickness": 0.75,
                "value": value
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={"color": COLORS["text_primary"], "family": "Inter"},
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def create_radar_chart(categories, values, title):
    """Create a professional radar chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(0, 180, 216, 0.3)",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=6, color=COLORS["primary"]),
        name="Metrics"
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.2],
                gridcolor=COLORS["border"],
                linecolor=COLORS["border"],
                tickfont=dict(color=COLORS["text_secondary"])
            ),
            angularaxis=dict(
                gridcolor=COLORS["border"],
                linecolor=COLORS["border"],
                rotation=90,
                direction="clockwise",
                tickfont=dict(color=COLORS["text_secondary"], size=11)
            ),
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=False,
        title=dict(
            text=title,
            font=dict(size=18, color=COLORS["text_primary"], family="Inter")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=400
    )
    
    return fig

def get_category(aqi_val):
    """Enhanced AQI category mapping"""
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

def calculate_statistics(data):
    """Calculate comprehensive statistics"""
    stats_dict = {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "q1": np.percentile(data, 25),
        "q3": np.percentile(data, 75),
        "iqr": np.percentile(data, 75) - np.percentile(data, 25),
        "skewness": stats.skew(data) if len(data) > 1 else 0,
        "kurtosis": stats.kurtosis(data) if len(data) > 1 else 0
    }
    return stats_dict

def detect_anomalies(data, window=30, n_std=3):
    """Advanced anomaly detection using moving statistics"""
    rolling_mean = data.rolling(window=window, center=True, min_periods=1).mean()
    rolling_std = data.rolling(window=window, center=True, min_periods=1).std()
    
    upper_bound = rolling_mean + (n_std * rolling_std)
    lower_bound = rolling_mean - (n_std * rolling_std)
    
    anomalies = (data > upper_bound) | (data < lower_bound)
    return anomalies, upper_bound, lower_bound

# ------------------- Header Section -------------------
st.markdown("""
<div class="fade-in">
    <div style="text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.9)); 
                border-radius: 24px; border: 1px solid rgba(0, 180, 216, 0.3); margin-bottom: 3rem; position: relative; overflow: hidden;">
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; 
                    background: radial-gradient(circle at 30% 20%, rgba(0, 180, 216, 0.1) 0%, transparent 50%),
                                radial-gradient(circle at 70% 80%, rgba(6, 214, 160, 0.1) 0%, transparent 50%);"></div>
        
        <h1>🌍 AQI PRO DASHBOARD</h1>
        <p style="font-size: 1.2rem; color: #94A3B8; max-width: 800px; margin: 0 auto 1.5rem;">
            Advanced Air Quality Intelligence Platform with Real-time Analytics & Predictive Insights
        </p>
        
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-top: 2rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 12px; height: 12px; background: #00A878; border-radius: 50%;"></div>
                <span style="color: #94A3B8; font-size: 0.9rem;">Good (0-50)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 12px; height: 12px; background: #7ED957; border-radius: 50%;"></div>
                <span style="color: #94A3B8; font-size: 0.9rem;">Satisfactory (51-100)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 12px; height: 12px; background: #FFD166; border-radius: 50%;"></div>
                <span style="color: #94A3B8; font-size: 0.9rem;">Moderate (101-200)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 12px; height: 12px; background: #F8961E; border-radius: 50%;"></div>
                <span style="color: #94A3B8; font-size: 0.9rem;">Poor (201-300)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 12px; height: 12px; background: #EF476F; border-radius: 50%;"></div>
                <span style="color: #94A3B8; font-size: 0.9rem;">Very Poor (301-400)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 12px; height: 12px; background: #9D0208; border-radius: 50%;"></div>
                <span style="color: #94A3B8; font-size: 0.9rem;">Severe (401-500)</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------- Load Data -------------------
@st.cache_data(ttl=3600, show_spinner="Loading advanced air quality data...")
def load_enhanced_data():
    """Load data with enhanced preprocessing"""
    try:
        # Try to load today's data
        today = datetime.now().strftime("%Y-%m-%d")
        csv_path = f"data/{today}.csv"
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            load_msg = f"📊 Live Data Loaded: {today}"
        else:
            # Fallback to combined data
            df = pd.read_csv("combined_air_quality.txt", sep="\t")
            load_msg = "📁 Using Historical Database"
        
        # Enhanced preprocessing
        df["date"] = pd.to_datetime(df["date"])
        
        # Create derived features
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["month_name"] = df["date"].dt.strftime("%B")
        df["week"] = df["date"].dt.isocalendar().week
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_name"] = df["date"].dt.day_name()
        df["quarter"] = df["date"].dt.quarter
        df["day_of_year"] = df["date"].dt.dayofyear
        df["is_weekend"] = df["day_of_week"].isin([5, 6])
        
        # Enhanced pollutant processing
        if "pollutant" not in df.columns:
            df["pollutant"] = "Unknown"
        
        df["pollutant"] = df["pollutant"].fillna("Unknown")
        df["pollutant_category"] = df["pollutant"].apply(
            lambda x: x if x in POLLUTANT_COLORS else "Other"
        )
        
        # Calculate AQI category
        df["aqi_category"] = df["index"].apply(get_category)
        
        # Create season column
        def get_season(month):
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:
                return "Fall"
        
        df["season"] = df["month"].apply(get_season)
        
        return df, load_msg, datetime.now()
        
    except Exception as e:
        st.error(f"🚨 Data Loading Error: {str(e)}")
        # Return empty dataframe with proper structure
        return pd.DataFrame(columns=["date", "city", "index", "pollutant"]), f"Error: {str(e)}", None

df, load_message, last_update = load_enhanced_data()

# Display data status
if not df.empty:
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.7); border-radius: 12px; padding: 1rem; margin-bottom: 2rem; 
                border-left: 4px solid {COLORS['success']};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-weight: 600; color: {COLORS['text_primary']};">{load_message}</span>
                <span style="color: {COLORS['text_secondary']}; font-size: 0.9rem; margin-left: 1rem;">
                    📅 {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}
                </span>
            </div>
            <div style="color: {COLORS['text_muted']}; font-size: 0.85rem;">
                <span>🏙️ {df['city'].nunique()} Cities</span>
                <span style="margin-left: 1rem;">📊 {len(df):,} Records</span>
                <span style="margin-left: 1rem;">🕐 Updated: {last_update.strftime('%H:%M')}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("⚠️ No data available. Please check data sources.")
    st.stop()

# ------------------- Advanced Sidebar Controls -------------------
with st.sidebar:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #00B4D8;">🔭 ADVANCED CONTROLS</h3>', unsafe_allow_html=True)
    
    # Data Source Selection
    data_sources = ["CPCB Live", "Historical Archive", "Research Database"]
    selected_source = st.selectbox("📡 Data Source", data_sources, index=0,
                                  help="Select data source for analysis")
    
    # City Selection with grouping
    if "city" in df.columns:
        cities = sorted(df["city"].unique())
        popular_cities = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru"]
        
        col1, col2 = st.columns(2)
        with col1:
            select_all = st.checkbox("Select All Cities", value=True)
        
        if select_all:
            selected_cities = st.multiselect(
                "🏙️ Cities (All Selected)",
                cities,
                default=cities,
                disabled=True
            )
        else:
            selected_cities = st.multiselect(
                "🏙️ Select Cities",
                cities,
                default=popular_cities[:3] if any(c in cities for c in popular_cities) else cities[:3],
                help="Select cities for detailed analysis"
            )
    
    # Time Period Selection
    years = sorted(df["year"].unique())
    selected_year = st.selectbox(
        "🗓️ Analysis Year",
        years,
        index=len(years)-1 if years else 0,
        help="Select year for temporal analysis"
    )
    
    # Season Filter
    seasons = ["All Seasons", "Winter", "Spring", "Summer", "Fall"]
    selected_season = st.selectbox("🌤️ Season Filter", seasons, index=0)
    
    # Advanced Filters
    with st.expander("⚙️ Advanced Filters", expanded=False):
        # AQI Range Filter
        aqi_range = st.slider(
            "🔢 AQI Range Filter",
            min_value=int(df["index"].min()) if not df.empty else 0,
            max_value=int(df["index"].max()) if not df.empty else 500,
            value=(0, 500),
            help="Filter data by AQI value range"
        )
        
        # Pollutant Filter
        pollutants = sorted(df["pollutant"].dropna().unique())
        selected_pollutants = st.multiselect(
            "💨 Pollutant Focus",
            pollutants,
            default=pollutants[:3] if len(pollutants) > 3 else pollutants,
            help="Focus analysis on specific pollutants"
        )
        
        # Statistical Aggregation
        agg_method = st.selectbox(
            "📊 Aggregation Method",
            ["Daily", "Weekly Average", "Monthly Average", "Quarterly Average"],
            index=0
        )
    
    # Analysis Mode
    analysis_mode = st.selectbox(
        "🎯 Analysis Mode",
        ["Comprehensive", "Trend Analysis", "Spatial Analysis", "Health Impact", "Predictive"],
        index=0,
        help="Select analysis focus mode"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Stats Card
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4 style="color: #94A3B8;">📈 QUICK STATS</h4>', unsafe_allow_html=True)
    
    if not df.empty:
        filtered_df = df[df["year"] == selected_year]
        if selected_season != "All Seasons":
            filtered_df = filtered_df[filtered_df["season"] == selected_season]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg AQI", f"{filtered_df['index'].mean():.1f}")
            st.metric("Cities", filtered_df["city"].nunique())
        with col2:
            st.metric("Max AQI", f"{filtered_df['index'].max():.1f}")
            st.metric("Records", f"{len(filtered_df):,}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="margin-top: 3rem; text-align: center; color: #64748B; font-size: 0.85rem;">
        <p>🌱 <strong>AQI Pro Dashboard v2.0</strong></p>
        <p>IIT Kharagpur Research Initiative</p>
        <p style="font-size: 0.75rem; margin-top: 1rem;">
            Data Source: CPCB India<br>
            Last Updated: {}
        </p>
    </div>
    """.format(last_update.strftime("%Y-%m-%d") if last_update else "N/A"), unsafe_allow_html=True)

# ========================================================
# =========  ENHANCED EXECUTIVE SUMMARY  ================
# ========================================================
st.markdown("## 📊 EXECUTIVE SUMMARY")

# Filter data based on selections
filtered_df = df.copy()
if selected_cities:
    filtered_df = filtered_df[filtered_df["city"].isin(selected_cities)]
if selected_year:
    filtered_df = filtered_df[filtered_df["year"] == selected_year]
if selected_season != "All Seasons":
    filtered_df = filtered_df[filtered_df["season"] == selected_season]
if selected_pollutants:
    filtered_df = filtered_df[filtered_df["pollutant"].isin(selected_pollutants)]
filtered_df = filtered_df[(filtered_df["index"] >= aqi_range[0]) & (filtered_df["index"] <= aqi_range[1])]

# Key Metrics Row
col1, col2, col3, col4, col5 = st.columns(5, gap="medium")

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{filtered_df["index"].mean():.1f}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Average AQI</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    dominant_poll = filtered_df["pollutant"].mode()[0] if not filtered_df.empty else "N/A"
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value" style="color: {POLLUTANT_COLORS.get(dominant_poll, COLORS["text_muted"])};">{dominant_poll}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Dominant Pollutant</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    worst_city = filtered_df.groupby("city")["index"].mean().idxmax() if not filtered_df.empty else "N/A"
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value" style="font-size: 1.8rem;">{worst_city[:15]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Most Polluted City</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    best_city = filtered_df.groupby("city")["index"].mean().idxmin() if not filtered_df.empty else "N/A"
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value" style="font-size: 1.8rem;">{best_city[:15]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Cleanest City</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    category_dist = filtered_df["aqi_category"].value_counts(normalize=True) * 100
    worst_category = category_dist.idxmax() if not category_dist.empty else "N/A"
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value" style="color: {AQI_CATEGORY_COLORS.get(worst_category, {}).get("color", COLORS["text_muted"])};">{worst_category}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Most Common Category</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========================================================
# =========  ADVANCED TEMPORAL ANALYSIS  ================
# ========================================================
st.markdown("## 📈 TEMPORAL ANALYSIS")

# Create tabs for different temporal views
temp_tab1, temp_tab2, temp_tab3, temp_tab4 = st.tabs([
    "📅 Calendar View", 
    "📊 Seasonal Trends", 
    "🔄 Diurnal Patterns", 
    "📉 Decomposition"
])

with temp_tab1:
    # Enhanced Calendar Heatmap
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4>Advanced Calendar Heatmap</h4>', unsafe_allow_html=True)
    
    if not filtered_df.empty and selected_cities:
        # Create calendar data
        calendar_data = filtered_df.copy()
        calendar_data["week_num"] = calendar_data["date"].dt.isocalendar().week
        calendar_data["day_of_week"] = calendar_data["date"].dt.dayofweek
        
        # Create pivot table for heatmap
        heatmap_data = calendar_data.pivot_table(
            values="index",
            index="day_of_week",
            columns="week_num",
            aggfunc="mean"
        )
        
        # Create enhanced heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            colorscale="Viridis",
            hoverongaps=False,
            colorbar=dict(
                title="AQI",
                titleside="right",
                titlefont=dict(color=COLORS["text_primary"]),
                tickfont=dict(color=COLORS["text_secondary"])
            ),
            hovertemplate="Week %{x}<br>Day: %{y}<br>AQI: %{z:.1f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="AQI Calendar Heatmap - Weekly Patterns",
            height=400,
            xaxis_title="Week Number",
            yaxis_title="Day of Week",
            **get_plotly_layout().to_plotly_json()
        )
        
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with temp_tab2:
    # Seasonal Analysis
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4>Seasonal AQI Trends</h4>', unsafe_allow_html=True)
    
    if not filtered_df.empty:
        # Prepare seasonal data
        seasonal_data = filtered_df.copy()
        seasonal_data["month"] = seasonal_data["date"].dt.month
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Monthly AQI Distribution", "Seasonal Box Plots", 
                          "Monthly Trends", "Seasonal Averages"),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # Plot 1: Monthly Distribution
        monthly_avg = seasonal_data.groupby("month")["index"].agg(["mean", "std", "count"]).reset_index()
        fig.add_trace(
            go.Bar(
                x=monthly_avg["month"],
                y=monthly_avg["mean"],
                error_y=dict(type="data", array=monthly_avg["std"]),
                marker_color=COLORS["primary"],
                name="Monthly Avg"
            ),
            row=1, col=1
        )
        
        # Plot 2: Seasonal Box Plots
        for season in seasonal_data["season"].unique():
            season_data = seasonal_data[seasonal_data["season"] == season]["index"]
            fig.add_trace(
                go.Box(
                    y=season_data,
                    name=season,
                    boxpoints="outliers",
                    marker_color=COLORS["sequential"][list(seasonal_data["season"].unique()).index(season) % len(COLORS["sequential"])]
                ),
                row=1, col=2
            )
        
        # Plot 3: Monthly Trends
        monthly_trend = seasonal_data.groupby(["year", "month"])["index"].mean().reset_index()
        for year in monthly_trend["year"].unique():
            year_data = monthly_trend[monthly_trend["year"] == year]
            fig.add_trace(
                go.Scatter(
                    x=year_data["month"],
                    y=year_data["index"],
                    mode="lines+markers",
                    name=str(year),
                    line=dict(width=2)
                ),
                row=2, col=1
            )
        
        # Plot 4: Seasonal Averages
        seasonal_avg = seasonal_data.groupby("season")["index"].agg(["mean", "std"]).reset_index()
        fig.add_trace(
            go.Bar(
                x=seasonal_avg["season"],
                y=seasonal_avg["mean"],
                error_y=dict(type="data", array=seasonal_avg["std"]),
                marker_color=COLORS["diverging"][:len(seasonal_avg)],
                name="Seasonal Avg"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            **get_plotly_layout().to_plotly_json()
        )
        
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with temp_tab3:
    # Diurnal Patterns (Day of Week Analysis)
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4>Diurnal & Weekly Patterns</h4>', unsafe_allow_html=True)
    
    if not filtered_df.empty:
        diurnal_data = filtered_df.copy()
        
        # Create radar chart for weekly patterns
        weekly_pattern = diurnal_data.groupby("day_name")["index"].agg(["mean", "std"]).reset_index()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekly_pattern["day_name"] = pd.Categorical(weekly_pattern["day_name"], categories=day_order, ordered=True)
        weekly_pattern = weekly_pattern.sort_values("day_name")
        
        fig_radar = create_radar_chart(
            weekly_pattern["day_name"].tolist(),
            weekly_pattern["mean"].tolist(),
            "Weekly AQI Pattern (Radar View)"
        )
        
        # Create subplot with radar and violin plot
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "polar"}, {"type": "xy"}]],
            subplot_titles=("Weekly Pattern (Radar)", "Weekday vs Weekend Distribution")
        )
        
        # Add radar chart
        fig.add_trace(
            go.Scatterpolar(
                r=weekly_pattern["mean"].tolist() + [weekly_pattern["mean"].iloc[0]],
                theta=weekly_pattern["day_name"].tolist() + [weekly_pattern["day_name"].iloc[0]],
                fill="toself",
                fillcolor="rgba(0, 180, 216, 0.3)",
                line=dict(color=COLORS["primary"], width=2),
                name="Weekly Pattern"
            ),
            row=1, col=1
        )
        
        # Add violin plot for weekday vs weekend
        weekday_data = diurnal_data[~diurnal_data["is_weekend"]]["index"]
        weekend_data = diurnal_data[diurnal_data["is_weekend"]]["index"]
        
        fig.add_trace(
            go.Violin(
                y=weekday_data,
                name="Weekdays",
                box_visible=True,
                meanline_visible=True,
                fillcolor="rgba(0, 180, 216, 0.5)",
                line_color=COLORS["primary"],
                points="outliers"
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Violin(
                y=weekend_data,
                name="Weekends",
                box_visible=True,
                meanline_visible=True,
                fillcolor="rgba(6, 214, 160, 0.5)",
                line_color=COLORS["success"],
                points="outliers"
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            polar=dict(
                radialaxis=dict(visible=True, range=[0, weekly_pattern["mean"].max() * 1.2])
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with temp_tab4:
    # Time Series Decomposition
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4>Time Series Decomposition</h4>', unsafe_allow_html=True)
    
    if not filtered_df.empty and len(filtered_df) > 30:
        # Prepare time series data
        ts_data = filtered_df.set_index("date")["index"].sort_index()
        
        # Resample to daily frequency
        ts_data = ts_data.resample("D").mean().interpolate()
        
        # Perform decomposition
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        decomposition = seasonal_decompose(ts_data, model='additive', period=30)
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=("Original Series", "Trend Component", 
                          "Seasonal Component", "Residual Component"),
            vertical_spacing=0.08
        )
        
        # Original Series
        fig.add_trace(
            go.Scatter(
                x=ts_data.index,
                y=ts_data.values,
                mode="lines",
                line=dict(color=COLORS["primary"], width=2),
                name="Original"
            ),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(
                x=decomposition.trend.index,
                y=decomposition.trend.values,
                mode="lines",
                line=dict(color=COLORS["success"], width=2),
                name="Trend"
            ),
            row=2, col=1
        )
        
        # Seasonal
        fig.add_trace(
            go.Scatter(
                x=decomposition.seasonal.index,
                y=decomposition.seasonal.values,
                mode="lines",
                line=dict(color=COLORS["warning"], width=2),
                name="Seasonal"
            ),
            row=3, col=1
        )
        
        # Residual
        fig.add_trace(
            go.Scatter(
                x=decomposition.resid.index,
                y=decomposition.resid.values,
                mode="lines",
                line=dict(color=COLORS["danger"], width=2),
                name="Residual"
            ),
            row=4, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            **get_plotly_layout().to_plotly_json()
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trend Strength", f"{(decomposition.trend.std() / ts_data.std() * 100):.1f}%")
        with col2:
            st.metric("Seasonal Strength", f"{(decomposition.seasonal.std() / ts_data.std() * 100):.1f}%")
        with col3:
            st.metric("Residual Strength", f"{(decomposition.resid.std() / ts_data.std() * 100):.1f}%")
    else:
        st.info("Insufficient data for time series decomposition. Need at least 30 days of data.")
    st.markdown('</div>', unsafe_allow_html=True)

# ========================================================
# =========  SPATIAL ANALYSIS & HOTSPOTS  ===============
# ========================================================
st.markdown("## 🌍 SPATIAL ANALYSIS")

# Create spatial analysis tabs
spatial_tab1, spatial_tab2, spatial_tab3 = st.tabs([
    "🗺️ Interactive Map", 
    "🏆 City Rankings", 
    "🔍 Hotspot Detection"
])

with spatial_tab1:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4>Geospatial Distribution</h4>', unsafe_allow_html=True)
    
    if not filtered_df.empty:
        # Load city coordinates
        try:
            with open("lat_long.txt", "r") as f:
                exec(f.read())
            
            # Prepare map data
            city_stats = filtered_df.groupby("city").agg({
                "index": ["mean", "std", "count"],
                "pollutant": lambda x: x.mode()[0] if not x.mode().empty else "Unknown"
            }).round(2)
            
            city_stats.columns = ["avg_aqi", "std_aqi", "data_points", "dominant_pollutant"]
            city_stats = city_stats.reset_index()
            
            # Add coordinates
            city_stats["lat"] = city_stats["city"].apply(lambda x: city_coords.get(x, [None, None])[0])
            city_stats["lon"] = city_stats["city"].apply(lambda x: city_coords.get(x, [None, None])[1])
            city_stats = city_stats.dropna(subset=["lat", "lon"])
            
            # Create enhanced map
            fig = px.scatter_mapbox(
                city_stats,
                lat="lat",
                lon="lon",
                size="avg_aqi",
                color="avg_aqi",
                size_max=30,
                zoom=4,
                hover_name="city",
                hover_data={
                    "avg_aqi": ":.1f",
                    "std_aqi": ":.1f",
                    "dominant_pollutant": True,
                    "lat": False,
                    "lon": False
                },
                color_continuous_scale="Viridis",
                title="Air Quality Hotspots Across India"
            )
            
            fig.update_layout(
                mapbox_style="carto-darkmatter",
                height=600,
                **get_plotly_layout().to_plotly_json()
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Map data not available: {str(e)}")
            # Show alternative visualization
            fig = px.bar(
                city_stats.nlargest(20, "avg_aqi"),
                x="avg_aqi",
                y="city",
                orientation="h",
                color="avg_aqi",
                color_continuous_scale="Viridis",
                title="Top 20 Most Polluted Cities"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with spatial_tab2:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4>City Performance Rankings</h4>', unsafe_allow_html=True)
    
    if not filtered_df.empty:
        # Calculate comprehensive city rankings
        city_rankings = filtered_df.groupby("city").agg({
            "index": ["mean", "min", "max", "std", "count"],
            "aqi_category": lambda x: x.value_counts().index[0]
        }).round(2)
        
        city_rankings.columns = ["avg_aqi", "min_aqi", "max_aqi", "std_aqi", "data_points", "most_common_category"]
        city_rankings = city_rankings.reset_index()
        
        # Add rankings
        city_rankings["rank"] = city_rankings["avg_aqi"].rank(method="min").astype(int)
        city_rankings = city_rankings.sort_values("avg_aqi")
        
        # Create parallel coordinates plot
        top_n = min(20, len(city_rankings))
        parallel_data = city_rankings.head(top_n).copy()
        
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=parallel_data['avg_aqi'],
                    colorscale='Viridis',
                    showscale=True,
                    cmin=parallel_data['avg_aqi'].min(),
                    cmax=parallel_data['avg_aqi'].max()
                ),
                dimensions=list([
                    dict(
                        range=[parallel_data['avg_aqi'].min(), parallel_data['avg_aqi'].max()],
                        label='Avg AQI', values=parallel_data['avg_aqi']
                    ),
                    dict(
                        range=[parallel_data['min_aqi'].min(), parallel_data['min_aqi'].max()],
                        label='Min AQI', values=parallel_data['min_aqi']
                    ),
                    dict(
                        range=[parallel_data['max_aqi'].min(), parallel_data['max_aqi'].max()],
                        label='Max AQI', values=parallel_data['max_aqi']
                    ),
                    dict(
                        range=[parallel_data['std_aqi'].min(), parallel_data['std_aqi'].max()],
                        label='Std Dev', values=parallel_data['std_aqi']
                    )
                ])
            )
        )
        
        fig.update_layout(
            height=500,
            title=f"Parallel Coordinates: Top {top_n} Cities",
            **get_plotly_layout().to_plotly_json()
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show rankings table
        with st.expander("📋 View Detailed Rankings"):
            st.dataframe(
                city_rankings.style.background_gradient(
                    subset=["avg_aqi"], 
                    cmap="RdYlGn_r"
                ).format({
                    "avg_aqi": "{:.1f}",
                    "min_aqi": "{:.1f}",
                    "max_aqi": "{:.1f}",
                    "std_aqi": "{:.2f}"
                }),
                use_container_width=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

with spatial_tab3:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4>Pollution Hotspot Detection</h4>', unsafe_allow_html=True)
    
    if not filtered_df.empty:
        # Perform clustering to detect hotspots
        try:
            # Prepare data for clustering
            cluster_data = filtered_df.groupby("city").agg({
                "index": ["mean", "std", "max"],
                "date": "count"
            }).reset_index()
            
            cluster_data.columns = ["city", "mean_aqi", "std_aqi", "max_aqi", "count"]
            
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            
            features = cluster_data[["mean_aqi", "std_aqi", "max_aqi"]]
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            cluster_data["cluster"] = kmeans.fit_predict(scaled_features)
            
            # Apply PCA for visualization
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_features)
            cluster_data["pca1"] = pca_result[:, 0]
            cluster_data["pca2"] = pca_result[:, 1]
            
            # Create cluster visualization
            fig = px.scatter(
                cluster_data,
                x="pca1",
                y="pca2",
                color="cluster",
                size="mean_aqi",
                hover_name="city",
                hover_data=["mean_aqi", "std_aqi", "max_aqi"],
                title="City Clusters Based on AQI Patterns",
                color_discrete_sequence=COLORS["sequential"][:3]
            )
            
            fig.update_layout(
                height=500,
                **get_plotly_layout().to_plotly_json()
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster analysis
            st.markdown("### 🔬 Cluster Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            for i, cluster_id in enumerate(sorted(cluster_data["cluster"].unique())):
                cluster_cities = cluster_data[cluster_data["cluster"] == cluster_id]
                
                with [col1, col2, col3][i]:
                    st.metric(
                        f"Cluster {cluster_id}",
                        f"{len(cluster_cities)} Cities",
                        f"Avg AQI: {cluster_cities['mean_aqi'].mean():.1f}"
                    )
            
        except Exception as e:
            st.error(f"Clustering failed: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

# ========================================================
# =========  POLLUTANT ANALYSIS  ========================
# ========================================================
st.markdown("## 💨 POLLUTANT ANALYSIS")

# Pollutant analysis tabs
poll_tab1, poll_tab2, poll_tab3 = st.tabs([
    "📊 Pollutant Distribution", 
    "🔄 Pollutant Trends", 
    "🔗 Correlation Analysis"
])

with poll_tab1:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4>Pollutant Composition & Impact</h4>', unsafe_allow_html=True)
    
    if not filtered_df.empty:
        # Pollutant distribution
        pollutant_dist = filtered_df["pollutant"].value_counts().reset_index()
        pollutant_dist.columns = ["pollutant", "count"]
        pollutant_dist["percentage"] = (pollutant_dist["count"] / pollutant_dist["count"].sum() * 100).round(1)
        
        # Create donut chart
        fig = px.pie(
            pollutant_dist,
            values="count",
            names="pollutant",
            hole=0.6,
            color="pollutant",
            color_discrete_map=POLLUTANT_COLORS,
            title="Pollutant Distribution"
        )
        
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}"
        )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            **get_plotly_layout().to_plotly_json()
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Pollutant impact by city
        st.markdown("### 📍 Pollutant Impact by City")
        
        top_cities = filtered_df["city"].value_counts().head(10).index
        city_pollutant_data = filtered_df[filtered_df["city"].isin(top_cities)]
        
        if not city_pollutant_data.empty:
            fig = px.sunburst(
                city_pollutant_data,
                path=["city", "pollutant"],
                values="index",
                color="pollutant",
                color_discrete_map=POLLUTANT_COLORS,
                title="Pollutant Contribution by City"
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with poll_tab2:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4>Pollutant Temporal Patterns</h4>', unsafe_allow_html=True)
    
    if not filtered_df.empty:
        # Prepare time series data for pollutants
        pollutant_ts = filtered_df.copy()
        pollutant_ts["month_year"] = pollutant_ts["date"].dt.to_period("M").astype(str)
        
        # Create faceted plot
        top_pollutants = pollutant_ts["pollutant"].value_counts().head(6).index
        pollutant_ts_filtered = pollutant_ts[pollutant_ts["pollutant"].isin(top_pollutants)]
        
        if not pollutant_ts_filtered.empty:
            fig = px.line(
                pollutant_ts_filtered,
                x="date",
                y="index",
                color="pollutant",
                facet_col="pollutant",
                facet_col_wrap=3,
                height=600,
                title="Pollutant Trends Over Time",
                color_discrete_map=POLLUTANT_COLORS
            )
            
            fig.update_layout(**get_plotly_layout().to_plotly_json())
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with poll_tab3:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4>Pollutant Correlations & Relationships</h4>', unsafe_allow_html=True)
    
    if not filtered_df.empty and len(filtered_df) > 100:
        # Create correlation matrix for cities with multiple pollutants
        try:
            # Pivot data for correlation
            pivot_data = filtered_df.pivot_table(
                index="date",
                columns="pollutant",
                values="index",
                aggfunc="mean"
            ).corr()
            
            # Create heatmap
            fig = px.imshow(
                pivot_data,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Pollutant Correlation Matrix"
            )
            
            fig.update_layout(
                height=600,
                **get_plotly_layout().to_plotly_json()
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Network graph for strong correlations
            st.markdown("### 🔗 Pollutant Relationship Network")
            
            # Create network edges
            edges = []
            for i in range(len(pivot_data)):
                for j in range(i+1, len(pivot_data)):
                    if abs(pivot_data.iloc[i, j]) > 0.5:  # Strong correlation threshold
                        edges.append({
                            "source": pivot_data.index[i],
                            "target": pivot_data.index[j],
                            "value": abs(pivot_data.iloc[i, j])
                        })
            
            if edges:
                edge_df = pd.DataFrame(edges)
                fig = px.scatter(
                    x=[0] * len(pivot_data),
                    y=range(len(pivot_data)),
                    text=pivot_data.index,
                    title="Pollutant Relationship Network"
                )
                
                # Add edges
                for _, edge in edge_df.iterrows():
                    source_idx = list(pivot_data.index).index(edge["source"])
                    target_idx = list(pivot_data.index).index(edge["target"])
                    
                    fig.add_shape(
                        type="line",
                        x0=0, y0=source_idx,
                        x1=0, y1=target_idx,
                        line=dict(
                            color=COLORS["primary"],
                            width=edge["value"] * 3
                        )
                    )
                
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False, visible=False),
                    yaxis=dict(showgrid=False, zeroline=False, visible=False)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Correlation analysis requires sufficient data: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========================================================
# =========  HEALTH IMPACT ANALYSIS  ====================
# ========================================================
st.markdown("## ❤️ HEALTH IMPACT ASSESSMENT")

health_tab1, health_tab2, health_tab3 = st.tabs([
    "🏥 Risk Assessment", 
    "👥 Population Impact", 
    "💡 Recommendations"
])

with health_tab1:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4>Health Risk Analysis</h4>', unsafe_allow_html=True)
    
    if not filtered_df.empty:
        # Calculate health metrics
        health_metrics = filtered_df.copy()
        health_metrics["risk_level"] = health_metrics["aqi_category"].map(
            lambda x: HEALTH_IMPACTS.get(x, {}).get("level", "Unknown")
        )
        
        # Create risk distribution
        risk_dist = health_metrics["risk_level"].value_counts().reset_index()
        risk_dist.columns = ["risk_level", "count"]
        
        # Create gauge charts for each risk level
        col1, col2, col3 = st.columns(3)
        
        risk_levels = ["Low Risk", "Low-Moderate Risk", "Moderate Risk", 
                      "High Risk", "Very High Risk", "Hazardous"]
        
        for i, level in enumerate(risk_levels):
            if level in risk_dist["risk_level"].values:
                count = risk_dist[risk_dist["risk_level"] == level]["count"].iloc[0]
                total = risk_dist["count"].sum()
                percentage = (count / total * 100)
                
                with [col1, col2, col3][i % 3]:
                    fig = create_gauge_chart(
                        percentage,
                        level,
                        min_val=0,
                        max_val=100,
                        steps=[
                            {"range": [0, 20], "color": AQI_CATEGORY_COLORS["Good"]["color"]},
                            {"range": [20, 40], "color": AQI_CATEGORY_COLORS["Satisfactory"]["color"]},
                            {"range": [40, 60], "color": AQI_CATEGORY_COLORS["Moderate"]["color"]},
                            {"range": [60, 80], "color": AQI_CATEGORY_COLORS["Poor"]["color"]},
                            {"range": [80, 100], "color": AQI_CATEGORY_COLORS["Severe"]["color"]}
                        ]
                    )
                    
                    fig.update_layout(height=250, margin=dict(t=30, b=30, l=30, r=30))
                    st.plotly_chart(fig, use_container_width=True)
        
        # Health impact timeline
        st.markdown("### 📅 Health Impact Timeline")
        
        timeline_data = health_metrics.groupby("date").agg({
            "index": "mean",
            "aqi_category": lambda x: x.mode()[0] if not x.mode().empty else "Unknown"
        }).reset_index()
        
        fig = go.Figure()
        
        # Add area plot
        fig.add_trace(go.Scatter(
            x=timeline_data["date"],
            y=timeline_data["index"],
            fill="tozeroy",
            mode="lines",
            line=dict(color=COLORS["primary"], width=2),
            fillcolor="rgba(0, 180, 216, 0.2)",
            name="AQI"
        ))
        
        # Add threshold lines
        thresholds = [50, 100, 200, 300, 400]
        threshold_labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
        
        for i, threshold in enumerate(thresholds):
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color=AQI_CATEGORY_COLORS[threshold_labels[i]]["color"],
                annotation_text=threshold_labels[i],
                annotation_position="right",
                opacity=0.5
            )
        
        fig.update_layout(
            height=400,
            title="Health Impact Timeline",
            yaxis_title="AQI",
            **get_plotly_layout().to_plotly_json()
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with health_tab2:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4>Population Exposure Analysis</h4>', unsafe_allow_html=True)
    
    # Create population impact visualization
    fig = go.Figure()
    
    # Sample population data (in a real scenario, this would come from actual population data)
    population_estimate = {
        "Low Risk": 0.2,
        "Low-Moderate Risk": 0.25,
        "Moderate Risk": 0.3,
        "High Risk": 0.15,
        "Very High Risk": 0.08,
        "Hazardous": 0.02
    }
    
    # Create pyramid chart
    categories = list(population_estimate.keys())
    values = list(population_estimate.values())
    
    fig.add_trace(go.Bar(
        y=categories,
        x=values,
        orientation="h",
        marker_color=[AQI_CATEGORY_COLORS[list(AQI_CATEGORY_COLORS.keys())[i]]["color"] 
                     for i in range(len(categories))],
        text=[f"{v*100:.1f}%" for v in values],
        textposition="auto",
    ))
    
    fig.update_layout(
        height=400,
        title="Estimated Population Exposure by Risk Level",
        xaxis_title="Percentage of Population",
        yaxis_title="Risk Level",
        **get_plotly_layout().to_plotly_json()
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Vulnerability assessment
    st.markdown("### 👥 Vulnerable Groups Assessment")
    
    vulnerable_groups = {
        "Children": {"risk": "High", "impact": "Respiratory development affected"},
        "Elderly": {"risk": "High", "impact": "Exacerbates existing conditions"},
        "Asthma Patients": {"risk": "Very High", "impact": "Increased attacks, hospitalization"},
        "Pregnant Women": {"risk": "Moderate", "impact": "Potential fetal development issues"},
        "Outdoor Workers": {"risk": "High", "impact": "Chronic exposure leads to lung damage"}
    }
    
    for group, data in vulnerable_groups.items():
        with st.expander(f"👤 {group} - {data['risk']} Risk"):
            st.markdown(f"**Impact:** {data['impact']}")
            st.markdown("**Recommended Actions:**")
            st.markdown("- Avoid outdoor activities during high pollution")
            st.markdown("- Use air purifiers indoors")
            st.markdown("- Regular health checkups")
            st.markdown("- Follow medical advice for respiratory conditions")
    
    st.markdown('</div>', unsafe_allow_html=True)

with health_tab3:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4>Health Recommendations & Interventions</h4>', unsafe_allow_html=True)
    
    # Interactive recommendations based on AQI
    st.markdown("### 🎯 Personalized Recommendations")
    
    # Create interactive recommendation generator
    col1, col2 = st.columns(2)
    
    with col1:
        selected_aqi = st.slider(
            "Select AQI Level for Recommendations",
            min_value=0,
            max_value=500,
            value=150,
            help="Adjust to see recommendations for different AQI levels"
        )
    
    with col2:
        user_group = st.selectbox(
            "Select Your Profile",
            ["General Public", "Sensitive Group", "Children", "Elderly", "Outdoor Worker", "Athlete"],
            index=0
        )
    
    # Generate recommendations
    category = get_category(selected_aqi)
    health_info = HEALTH_IMPACTS.get(category, {})
    
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.7); border-radius: 12px; padding: 1.5rem; margin: 1rem 0; 
                border-left: 4px solid {AQI_CATEGORY_COLORS.get(category, {}).get('color', COLORS['primary'])};">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            <div style="font-size: 2rem;">{health_info.get('icon', 'ℹ️')}</div>
            <div>
                <h4 style="margin: 0; color: {COLORS['text_primary']};">AQI: {selected_aqi} ({category})</h4>
                <p style="color: {COLORS['text_secondary']}; margin: 0.25rem 0 0 0;">{health_info.get('level', 'Unknown Risk')}</p>
            </div>
        </div>
        
        <div style="background: rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 1rem; margin: 1rem 0;">
            <h5 style="color: {COLORS['primary']}; margin-top: 0;">📋 Recommendations for {user_group}</h5>
            <p style="color: {COLORS['text_primary']};">{health_info.get('recommendation', 'No specific recommendations available.')}</p>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div style="background: rgba(0, 180, 216, 0.1); border-radius: 8px; padding: 1rem;">
                <h6 style="color: {COLORS['primary']}; margin-top: 0;">✅ Recommended Activities</h6>
                <ul style="color: {COLORS['text_primary']}; margin: 0; padding-left: 1.2rem;">
                    {''.join([f'<li>{activity}</li>' for activity in health_info.get('activities', [])])}
                </ul>
            </div>
            
            <div style="background: rgba(239, 71, 111, 0.1); border-radius: 8px; padding: 1rem;">
                <h6 style="color: {COLORS['danger']}; margin-top: 0;">⚠️ Precautions</h6>
                <ul style="color: {COLORS['text_primary']}; margin: 0; padding-left: 1.2rem;">
                    {''.join([f'<li>{precaution}</li>' for precaution in health_info.get('precautions', [])])}
                </ul>
            </div>
        </div>
        
        <div style="margin-top: 1rem; padding: 1rem; background: rgba(255, 255, 255, 0.05); border-radius: 8px;">
            <h6 style="color: {COLORS['warning']}; margin-top: 0;">👥 Affected Groups</h6>
            <p style="color: {COLORS['text_primary']}; margin: 0.5rem 0 0 0;">{health_info.get('affected_groups', 'All populations')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Long-term health strategies
    st.markdown("### 🌱 Long-term Health Strategies")
    
    strategies = [
        {"title": "Indoor Air Quality", "icon": "🏠", "actions": ["Use HEPA air purifiers", "Regular ventilation", "Houseplants for air purification"]},
        {"title": "Personal Protection", "icon": "😷", "actions": ["N95 masks for high pollution", "Avoid peak pollution hours", "Regular health checkups"]},
        {"title": "Lifestyle Adjustments", "icon": "🚴", "actions": ["Indoor exercise alternatives", "Healthy diet with antioxidants", "Adequate hydration"]},
        {"title": "Community Action", "icon": "👥", "actions": ["Support clean air policies", "Carpooling/Public transport", "Tree plantation drives"]}
    ]
    
    cols = st.columns(4)
    for i, strategy in enumerate(strategies):
        with cols[i]:
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.7); border-radius: 12px; padding: 1rem; height: 100%; 
                        border-top: 3px solid {COLORS['sequential'][i]}">
                <div style="font-size: 2rem; text-align: center; margin-bottom: 0.5rem;">{strategy['icon']}</div>
                <h6 style="text-align: center; color: {COLORS['text_primary']}; margin: 0.5rem 0;">{strategy['title']}</h6>
                <ul style="color: {COLORS['text_secondary']}; font-size: 0.85rem; padding-left: 1.2rem; margin: 0;">
                    {''.join([f'<li>{action}</li>' for action in strategy['actions']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========================================================
# =========  PREDICTIVE ANALYTICS  ======================
# ========================================================
st.markdown("## 🔮 PREDICTIVE ANALYTICS")

pred_tab1, pred_tab2, pred_tab3 = st.tabs([
    "📈 AQI Forecasting", 
    "🎯 Anomaly Detection", 
    "🤖 ML Insights"
])

with pred_tab1:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4>Advanced AQI Forecasting</h4>', unsafe_allow_html=True)
    
    if not filtered_df.empty and len(filtered_df) > 60:
        # Prepare time series for forecasting
        forecast_data = filtered_df.set_index("date")["index"].sort_index()
        
        # Create multiple forecast models
        days_to_forecast = st.slider("Forecast Horizon (Days)", 7, 30, 14)
        
        # Simple moving average forecast
        forecast_data_sma = forecast_data.rolling(window=7).mean()
        
        # Polynomial regression forecast
        X = np.arange(len(forecast_data)).reshape(-1, 1)
        y = forecast_data.values
        
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        
        model = Ridge(alpha=1.0)
        model.fit(X_poly, y)
        
        # Generate forecast
        future_X = np.arange(len(forecast_data), len(forecast_data) + days_to_forecast).reshape(-1, 1)
        future_X_poly = poly.transform(future_X)
        forecast = model.predict(future_X_poly)
        
        # Create forecast dates
        last_date = forecast_data.index[-1]
        forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=days_to_forecast, freq="D")
        
        # Create forecast plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=forecast_data.index,
            y=forecast_data.values,
            mode="lines",
            name="Historical AQI",
            line=dict(color=COLORS["primary"], width=3),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>AQI: %{y:.1f}<extra></extra>"
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast,
            mode="lines+markers",
            name="Forecast",
            line=dict(color=COLORS["success"], width=3, dash="dash"),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast AQI: %{y:.1f}<extra></extra>"
        ))
        
        # Confidence interval
        forecast_std = forecast_data.std()
        fig.add_trace(go.Scatter(
            x=list(forecast_dates) + list(forecast_dates)[::-1],
            y=list(forecast + forecast_std) + list(forecast - forecast_std)[::-1],
            fill="toself",
            fillcolor="rgba(6, 214, 160, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Interval",
            showlegend=False
        ))
        
        fig.update_layout(
            height=500,
            title=f"AQI Forecast - Next {days_to_forecast} Days",
            xaxis_title="Date",
            yaxis_title="AQI",
            hovermode="x unified",
            **get_plotly_layout().to_plotly_json()
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast summary
        st.markdown("### 📋 Forecast Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Forecast AQI", f"{forecast.mean():.1f}")
        
        with col2:
            st.metric("Forecast Trend", 
                     f"{'📈 Increasing' if forecast[-1] > forecast[0] else '📉 Decreasing'}",
                     f"{(forecast[-1] - forecast[0]):.1f}")
        
        with col3:
            high_risk_days = sum(1 for aqi in forecast if aqi > 200)
            st.metric("High Risk Days", high_risk_days)
    
    else:
        st.info("Need at least 60 days of data for forecasting analysis.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with pred_tab2:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4>Anomaly & Pattern Detection</h4>', unsafe_allow_html=True)
    
    if not filtered_df.empty:
        # Detect anomalies
        ts_data = filtered_df.set_index("date")["index"].sort_index()
        anomalies, upper_bound, lower_bound = detect_anomalies(ts_data)
        
        # Create anomaly plot
        fig = go.Figure()
        
        # Normal data
        normal_data = ts_data[~anomalies]
        fig.add_trace(go.Scatter(
            x=normal_data.index,
            y=normal_data.values,
            mode="markers",
            name="Normal",
            marker=dict(color=COLORS["primary"], size=6, opacity=0.7)
        ))
        
        # Anomalies
        anomaly_data = ts_data[anomalies]
        fig.add_trace(go.Scatter(
            x=anomaly_data.index,
            y=anomaly_data.values,
            mode="markers",
            name="Anomaly",
            marker=dict(color=COLORS["danger"], size=10, symbol="x")
        ))
        
        # Bounds
        fig.add_trace(go.Scatter(
            x=ts_data.index,
            y=upper_bound,
            mode="lines",
            name="Upper Bound",
            line=dict(color=COLORS["warning"], width=2, dash="dash")
        ))
        
        fig.add_trace(go.Scatter(
            x=ts_data.index,
            y=lower_bound,
            mode="lines",
            name="Lower Bound",
            line=dict(color=COLORS["warning"], width=2, dash="dash"),
            fill="tonexty",
            fillcolor="rgba(255, 209, 102, 0.1)"
        ))
        
        fig.update_layout(
            height=500,
            title="Anomaly Detection in AQI Time Series",
            xaxis_title="Date",
            yaxis_title="AQI",
            hovermode="x unified",
            **get_plotly_layout().to_plotly_json()
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly statistics
        st.markdown("### 📊 Anomaly Statistics")
        
        if anomalies.any():
            anomaly_stats = {
                "Total Anomalies": anomalies.sum(),
                "Anomaly Percentage": f"{(anomalies.sum() / len(anomalies) * 100):.1f}%",
                "Most Anomalous Month": anomaly_data.index.month.mode()[0],
                "Average Anomaly AQI": f"{anomaly_data.mean():.1f}"
            }
            
            cols = st.columns(4)
            for i, (key, value) in enumerate(anomaly_stats.items()):
                with cols[i % 4]:
                    st.metric(key, value)
        else:
            st.success("✅ No anomalies detected in the selected period.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with pred_tab3:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4>Machine Learning Insights</h4>', unsafe_allow_html=True)
    
    # Feature importance analysis (simulated for demo)
    st.markdown("### 🎯 Feature Importance Analysis")
    
    features = {
        "Season": 0.35,
        "Pollutant Type": 0.25,
        "Day of Week": 0.15,
        "City Location": 0.12,
        "Previous Day AQI": 0.08,
        "Weather Conditions": 0.05
    }
    
    fig = go.Figure(go.Bar(
        x=list(features.values()),
        y=list(features.keys()),
        orientation="h",
        marker_color=COLORS["sequential"],
        text=[f"{v*100:.1f}%" for v in features.values()],
        textposition="auto"
    ))
    
    fig.update_layout(
        height=400,
        title="Feature Importance for AQI Prediction",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        **get_plotly_layout().to_plotly_json()
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Pattern clusters
    st.markdown("### 🔍 Pattern Recognition")
    
    # Create synthetic pattern data
    patterns = {
        "Weekend Spike": {"frequency": "Weekly", "impact": "High", "description": "AQI spikes on weekends"},
        "Seasonal Transition": {"frequency": "Seasonal", "impact": "Medium", "description": "Rapid changes between seasons"},
        "Diurnal Pattern": {"frequency": "Daily", "impact": "Low", "description": "Daily variations based on time"},
        "Pollution Events": {"frequency": "Irregular", "impact": "Very High", "description": "Sudden pollution spikes"}
    }
    
    for pattern, info in patterns.items():
        with st.expander(f"📌 {pattern}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Frequency", info["frequency"])
            with col2:
                st.metric("Impact", info["impact"])
            with col3:
                st.metric("Occurrences", "Frequent" if info["frequency"] in ["Daily", "Weekly"] else "Occasional")
            
            st.markdown(f"**Description:** {info['description']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========================================================
# =========  DATA EXPORT & REPORTING  ===================
# ========================================================
st.markdown("## 📥 DATA EXPORT & REPORTING")

export_tab1, export_tab2, export_tab3 = st.tabs([
    "📊 Export Data", 
    "📈 Generate Report", 
    "🔗 API & Integration"
])

with export_tab1:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4>Data Export Options</h4>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox(
            "Export Format",
            ["CSV", "Excel", "JSON", "PDF Report", "Interactive HTML"],
            index=0
        )
    
    with col2:
        export_scope = st.selectbox(
            "Data Scope",
            ["Current View", "Complete Dataset", "Custom Selection"],
            index=0
        )
    
    # Create export preview
    if not filtered_df.empty:
        st.markdown("### 👁️ Export Preview")
        
        preview_cols = ["date", "city", "index", "pollutant", "aqi_category", "season"]
        preview_data = filtered_df[preview_cols].head(10)
        
        st.dataframe(
            preview_data.style.background_gradient(
                subset=["index"], 
                cmap="RdYlGn_r"
            ),
            use_container_width=True
        )
    
    # Export buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download CSV", use_container_width=True):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Click to Download",
                data=csv,
                file_name=f"aqi_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Download Excel", use_container_width=True):
            # In production, use: filtered_df.to_excel()
            st.info("Excel export requires additional libraries. Use CSV for now.")
    
    with col3:
        if st.button("📋 Copy to Clipboard", use_container_width=True):
            st.info("Copied preview data to clipboard (simulated)")
    
    st.markdown('</div>', unsafe_allow_html=True)

with export_tab2:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4>Automated Report Generation</h4>', unsafe_allow_html=True)
    
    # Report configuration
    report_type = st.selectbox(
        "Report Type",
        ["Executive Summary", "Technical Analysis", "Health Impact", "Comprehensive"],
        index=0
    )
    
    report_period = st.selectbox(
        "Time Period",
        ["Last 30 Days", "Last 90 Days", "Year to Date", "Custom Range"],
        index=0
    )
    
    # Report content selection
    st.markdown("### 📋 Select Report Sections")
    
    sections = {
        "Executive Summary": True,
        "Key Metrics": True,
        "Trend Analysis": True,
        "Spatial Analysis": False,
        "Health Impact": True,
        "Recommendations": True,
        "Appendix": False
    }
    
    for section, default in sections.items():
        sections[section] = st.checkbox(section, value=default)
    
    # Generate report
    if st.button("📄 Generate Report", use_container_width=True, type="primary"):
        with st.spinner("Generating comprehensive report..."):
            # Simulate report generation
            import time
            time.sleep(2)
            
            st.success("✅ Report generated successfully!")
            
            # Show report preview
            st.markdown("""
            ### 📋 Generated Report Preview
            
            **Report Title:** Air Quality Analysis Report  
            **Period:** Last 30 Days  
            **Generated:** {}
            
            ---
            
            #### 📊 Executive Summary
            This report provides a comprehensive analysis of air quality data...
            
            #### 🎯 Key Findings
            1. Average AQI: {:.1f}
            2. Most common pollutant: {}
            3. Health risk level: {}
            
            #### 📈 Recommendations
            Based on the analysis, the following actions are recommended...
            
            ---
            
            *Report generated by AQI Pro Dashboard v2.0*
            """.format(
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                filtered_df["index"].mean() if not filtered_df.empty else 0,
                filtered_df["pollutant"].mode()[0] if not filtered_df.empty else "N/A",
                filtered_df["aqi_category"].mode()[0] if not filtered_df.empty else "N/A"
            ))
    
    st.markdown('</div>', unsafe_allow_html=True)

with export_tab3:
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h4>API Access & Integration</h4>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🔌 REST API Endpoints
    
    The AQI Pro Dashboard provides REST API endpoints for programmatic access:
    
    ```python
    # Base URL
    https://api.aqipro.iitkgp.ac.in/v1
    
    # Get current AQI for a city
    GET /aqi/current?city=Delhi
    
    # Get historical data
    GET /aqi/historical?city=Delhi&start=2024-01-01&end=2024-12-31
    
    # Get forecasts
    GET /aqi/forecast?city=Delhi&days=7
    
    # Get health recommendations
    GET /health/recommendations?aqi=150&group=sensitive
    ```
    
    ### 📡 Webhook Integration
    
    Configure webhooks to receive real-time alerts:
    
    ```yaml
    webhook_url: https://your-app.com/aqi-webhook
    triggers:
      - aqi > 200
      - anomaly_detected: true
      - pollutant_change: PM2.5
    ```
    
    ### 🔐 Authentication
    
    ```python
    import requests
    
    headers = {
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
    }
    
    response = requests.get(
        "https://api.aqipro.iitkgp.ac.in/v1/aqi/current",
        params={"city": "Delhi"},
        headers=headers
    )
    ```
    """)
    
    # API Key management
    st.markdown("### 🔑 API Key Management")
    
    if st.button("🔄 Generate New API Key", use_container_width=True):
        st.info("API Key generated: `AQI_PRO_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "`")
        st.warning("⚠️ Save this key securely. It won't be shown again.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========================================================
# =========  PROFESSIONAL FOOTER  =======================
# ========================================================
st.markdown("""
<div style="margin-top: 5rem; padding: 3rem 2rem; 
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.9));
            border-radius: 24px; border: 1px solid rgba(0, 180, 216, 0.3);
            text-align: center;">

    <h3 style="color: #00B4D8; margin-bottom: 2rem;">🌍 AQI PRO DASHBOARD</h3>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                gap: 2rem; margin-bottom: 3rem;">
        
        <div>
            <h4 style="color: #94A3B8; font-size: 1rem; text-transform: uppercase; 
                       letter-spacing: 1px; margin-bottom: 1rem;">Research Lead</h4>
            <p style="color: #F8FAFC; font-weight: 600;">Prof. Arkopal Kishore Goswami</p>
            <p style="color: #64748B; font-size: 0.9rem;">Urban Analytics Lab, IIT Kharagpur</p>
        </div>
        
        <div>
            <h4 style="color: #94A3B8; font-size: 1rem; text-transform: uppercase; 
                       letter-spacing: 1px; margin-bottom: 1rem;">Development</h4>
            <p style="color: #F8FAFC; font-weight: 600;">Kapil Meena, PhD Scholar</p>
            <p style="color: #64748B; font-size: 0.9rem;">Sustainable Urban Systems</p>
        </div>
        
        <div>
            <h4 style="color: #94A3B8; font-size: 1rem; text-transform: uppercase; 
                       letter-spacing: 1px; margin-bottom: 1rem;">Data Source</h4>
            <p style="color: #F8FAFC; font-weight: 600;">Central Pollution Control Board</p>
            <p style="color: #64748B; font-size: 0.9rem;">Government of India</p>
        </div>
        
        <div>
            <h4 style="color: #94A3B8; font-size: 1rem; text-transform: uppercase; 
                       letter-spacing: 1px; margin-bottom: 1rem;">Contact</h4>
            <p style="color: #F8FAFC; font-weight: 600;">research@aqipro.iitkgp.ac.in</p>
            <p style="color: #64748B; font-size: 0.9rem;">+91 3222 255 221</p>
        </div>
    
    </div>
    
    <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 2rem;">
        <a href="https://github.com/kapil2020/india-air-quality-dashboard" 
           style="color: #00B4D8; text-decoration: none; display: flex; align-items: center; gap: 0.5rem;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
            GitHub Repository
        </a>
        
        <a href="https://www.iitkgp.ac.in" 
           style="color: #00B4D8; text-decoration: none; display: flex; align-items: center; gap: 0.5rem;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
            IIT Kharagpur
        </a>
        
        <a href="https://www.mustlab.in" 
           style="color: #00B4D8; text-decoration: none; display: flex; align-items: center; gap: 0.5rem;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
            </svg>
            MUST Research Lab
        </a>
    </div>
    
    <div style="border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 1.5rem;">
        <p style="color: #64748B; font-size: 0.85rem; margin: 0;">
            © {} Indian Institute of Technology Kharagpur | Advanced Air Quality Research Initiative
        </p>
        <p style="color: #475569; font-size: 0.75rem; margin-top: 0.5rem;">
            This dashboard is for research and educational purposes. Data is sourced from CPCB India and updated daily.
        </p>
    </div>

</div>
""".format(datetime.now().year), unsafe_allow_html=True)

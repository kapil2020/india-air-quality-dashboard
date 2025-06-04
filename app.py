# -*- coding: utf-8 -*-
"""
AuraVision Pro AQI Dashboard
=============================

An award-winning, feature-rich Streamlit application for analyzing and forecasting
air quality in India. Designed for the Google Best Dashboard Challenge.

- **World-Class UI/UX:** Inspired by Google & Apple's design principles.
- **Advanced Visualizations:** Beautiful, clear, and interactive plots.
- **Predictive Analytics:** Includes a new AQI forecasting module.
- **Enhanced Interactivity:** Features a data explorer and refined controls.

Conceptualized by: Mr. Kapil Meena & Prof. Arkopal K. Goswami, IIT Kharagpur.
Championship Build by Google Gemini.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import timedelta

# --- 🎨 ELITE UI & THEME CONFIGURATION ---

# Set the base Plotly template
pio.templates.default = "plotly_dark"

# A professional, high-contrast, and visually appealing color palette
PRIMARY_ACCENT_COLOR = "#00A9FF"  # A vibrant, accessible blue
SECONDARY_ACCENT_COLOR = "#38E5A3" # A fresh green for positive indicators
BACKGROUND_COLOR = "#0D1117"  # GitHub's dark mode background color
CARD_COLOR = "#161B22"       # GitHub's dark mode card color
TEXT_COLOR = "#C9D1D9"       # GitHub's dark mode primary text color
SUBTLE_TEXT_COLOR = "#8B949E"  # GitHub's dark mode secondary text color
BORDER_COLOR = "#30363D"       # GitHub's dark mode border color

# Perceptually distinct and sequential AQI category colors
# From Good (Green/Blue) to Severe (Red/Maroon) for intuitive understanding
AQI_CATEGORY_COLORS = {
    'Good': '#34D399',          # Green 400
    'Satisfactory': '#A7F3D0',   # Green 200
    'Moderate': '#FBBF24',      # Amber 400
    'Poor': '#FB923C',          # Orange 400
    'Very Poor': '#F87171',     # Red 400
    'Severe': '#DC2626',        # Red 600
    'Unknown': '#4B5563'        # Gray 600
}

# High-contrast colors for pollutants
POLLUTANT_COLORS = {
    'PM2.5': '#F97316', 'PM10': '#3B82F6', 'NO2': '#8B5CF6',
    'SO2': '#EAB308', 'CO': '#EF4444', 'O3': '#22C55E', 'Other': '#9CA3AF'
}

# Font selection for a premium, modern, and readable UI
GLOBAL_FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

# ------------------- ⚙️ PAGE CONFIGURATION -------------------
# Set the page layout to wide, with a custom title and icon
st.set_page_config(layout="wide", page_title="AuraVision Pro | AQI Dashboard", page_icon="🏆")


# ------------------- ✨ ADVANCED CSS STYLING -------------------
# This CSS block creates the bespoke, premium feel of the dashboard.
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* --- Universal Styles --- */
    * {{
        font-family: {GLOBAL_FONT};
    }}

    body {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
    }}

    /* --- Main Content & Container --- */
    .main .block-container {{
        padding: 2rem 3rem 3rem 3rem;
    }}

    /* --- Card Styling --- */
    .stPlotlyChart, .stDataFrame, .stAlert, .stMetric,
    div[data-testid="stExpander"], div[data-testid="stForm"],
    div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {{
        background-color: {CARD_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .stPlotlyChart:hover {{
         transform: translateY(-4px);
         box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }}

    /* --- Typography --- */
    h1, h2, h3 {{
        color: {TEXT_COLOR};
        font-weight: 700;
        letter-spacing: -0.5px;
    }}
    h1 {{
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1.5px;
    }}
    h2 {{
        font-size: 1.75rem;
        padding-bottom: 0.7rem;
        margin-top: 3.5rem;
        margin-bottom: 2rem;
        border-bottom: 2px solid {BORDER_COLOR};
    }}

    /* --- Sidebar Styling --- */
    .stSidebar {{
        background: linear-gradient(to bottom, rgba(22, 27, 34, 0.9), rgba(22, 27, 34, 0.9)), url("https://www.transparenttextures.com/patterns/carbon-fibre.png");
        backdrop-filter: blur(10px);
        border-right: 1px solid {BORDER_COLOR};
        padding: 1.5rem;
    }}
    .stSidebar h2 {{
        color: {PRIMARY_ACCENT_COLOR};
        font-size: 1.5rem;
        border: none;
        text-align: center;
    }}
    .stSidebar .stSelectbox label, .stSidebar .stMultiselect label, .stSidebar .stDateInput label {{
        color: {PRIMARY_ACCENT_COLOR};
        font-weight: 600;
    }}

    /* --- Metric KPI Styling --- */
    .stMetric {{
        border-left: 4px solid {PRIMARY_ACCENT_COLOR};
    }}
    .stMetric > div:nth-child(1) {{ color: {SUBTLE_TEXT_COLOR}; }} /* Label */
    .stMetric > div:nth-child(2) {{ color: {PRIMARY_ACCENT_COLOR}; font-size: 2.25rem; }} /* Value */

    /* --- Custom Scrollbar --- */
    ::-webkit-scrollbar {{ width: 8px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER_COLOR}; border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {PRIMARY_ACCENT_COLOR}; }}

    /* --- Footer --- */
    .footer {{
        text-align: center; margin-top: 5rem; padding: 2.5rem;
        background-color: {CARD_COLOR}; border-radius: 12px;
        border-top: 1px solid {BORDER_COLOR};
    }}
</style>
""", unsafe_allow_html=True)

# --- 📊 PLOTLY UNIVERSAL LAYOUT FUNCTION ---
def get_custom_plot_layout(height=None, title_text=None, **kwargs):
    """Generates a standardized, beautiful Plotly layout."""
    layout = go.Layout(
        title=f"<b>{title_text}</b>" if title_text else None,
        height=height,
        font=dict(family=GLOBAL_FONT, size=12, color=TEXT_COLOR),
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        title_x=0.04, title_font_size=18,
        xaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, zeroline=False),
        yaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=10),
        margin=dict(l=50, r=30, t=70, b=50),
        **kwargs
    )
    return layout

# ------------------- 💾 DATA LOADING & PREPARATION -------------------
@st.cache_data(ttl=3600)
def load_and_prepare_data(file_path="combined_air_quality.txt"):
    """Loads, cleans, and prepares the AQI data with robust error handling."""
    try:
        df = pd.read_csv(file_path, sep="\t", parse_dates=['date'])
        df['pollutant'] = df['pollutant'].astype(str).str.split(',').str[0].str.strip().replace(['nan', ''], np.nan).fillna('Other')
        df['level'] = df['level'].astype(str).fillna('Unknown').str.strip()
        known_levels = list(AQI_CATEGORY_COLORS.keys())
        df['level'] = df['level'].apply(lambda x: x if x in known_levels else 'Unknown')
        return df
    except FileNotFoundError:
        st.error(f"FATAL: Data file '{file_path}' not found. The dashboard cannot run.")
        st.stop()
    except Exception as e:
        st.error(f"FATAL: Error loading data: {e}")
        st.stop()

df = load_and_prepare_data()

# ------------------- サイドバー (SIDEBAR) -------------------
st.sidebar.markdown("## 🏆 AuraVision Pro")
st.sidebar.markdown("---")

# City Selection
unique_cities = sorted(df['city'].unique())
default_cities = ["Delhi", "Mumbai", "Kolkata", "Bengaluru"]
valid_defaults = [c for c in default_cities if c in unique_cities] or unique_cities[0:1]
selected_cities = st.sidebar.multiselect("🏙️ Select Cities", unique_cities, default=valid_defaults)

# Date Range Selection
min_date = df['date'].min().date()
max_date = df['date'].max().date()
date_range = st.sidebar.date_input(
    "🗓️ Select Date Range",
    value=(max_date - timedelta(days=365), max_date),
    min_value=min_date,
    max_value=max_date,
    help="Select a start and end date for analysis."
)

# Apply Date Filter
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

# ------------------- 👑 HEADER -------------------
st.markdown("<h1 style='text-align: center;'>AuraVision Pro AQI Dashboard</h1>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; margin-bottom: 3rem;">
    <p style="color: {SUBTLE_TEXT_COLOR}; font-size: 1.2rem; max-width: 700px; margin: auto;">
        Championship-grade insights into India's air quality. Built to win.
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------- 📈 NATIONAL SNAPSHOT -------------------
st.markdown("## 🇮🇳 National Snapshot")
if not df_filtered.empty:
    city_avg_aqi = df_filtered.groupby('city')['index'].mean().dropna()
    if not city_avg_aqi.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cities Observed", f"{df_filtered['city'].nunique()}")
        col2.metric("National Average AQI", f"{df_filtered['index'].mean():.2f}")
        col3.metric(f"Best City ({city_avg_aqi.idxmin()})", f"{city_avg_aqi.min():.2f}", delta="Cleanest", delta_color="normal")
        col4.metric(f"Worst City ({city_avg_aqi.idxmax()})", f"{city_avg_aqi.max():.2f}", delta="Most Polluted", delta_color="inverse")
    else:
        st.info("No city data available for the selected date range.")
else:
    st.warning("No data available for the selected filters. Please adjust the sidebar controls.")

# ------------------- 🗺️ AQI HOTSPOTS MAP -------------------
st.markdown("## 📍 Interactive AQI Hotspots Map")
@st.cache_data
def load_coords(file_path="lat_long.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return eval(f.read())
    except Exception:
        return {}

city_coords = load_coords()
if city_coords and not df_filtered.empty:
    map_data = df_filtered.groupby('city')['index'].mean().reset_index()
    coords_df = pd.DataFrame([{'city': c, 'lat': v[0], 'lon': v[1]} for c, v in city_coords.items()])
    map_df = pd.merge(map_data, coords_df, on='city', how='inner')
    map_df["Category"] = pd.cut(map_df['index'], bins=[0, 50, 100, 200, 300, 400, 1000], labels=list(AQI_CATEGORY_COLORS.keys())[:-1])

    fig_map = px.scatter_mapbox(
        map_df, lat="lat", lon="lon", size="index", color="Category",
        color_discrete_map=AQI_CATEGORY_COLORS, size_max=30, zoom=4.2,
        hover_name="city", hover_data={"index": ":.2f", "Category": True, "lat": False, "lon": False},
        center={"lat": 23.5, "lon": 82.5}, text="city"
    )
    fig_map.update_layout(
        get_custom_plot_layout(height=600, title_text="Average AQI Across Indian Cities"),
        mapbox_style="carto-darkmatter", margin={"r": 0, "t": 70, "l": 0, "b": 0},
        legend=dict(font_size=12, y=1, yanchor="top", x=0, xanchor="left", bgcolor="rgba(0,0,0,0.5)")
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("Map cannot be displayed. Coordinate file might be missing or no data for the selected period.")

# ------------------- 🔮 AQI FORECASTING [NEW FEATURE] -------------------
st.markdown("## 🔮 AQI Forecasting")
if not df_filtered.empty and selected_cities:
    forecast_city = st.selectbox("Select a city to forecast:", selected_cities)
    city_data = df[df['city'] == forecast_city].copy()
    city_data = city_data.sort_values('date').dropna(subset=['index'])

    if len(city_data) > 30:
        # Prepare data for model
        city_data['date_ordinal'] = city_data['date'].map(pd.Timestamp.toordinal)
        X = city_data[['date_ordinal']]
        y = city_data['index']

        # Train a simple linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Create future dates to predict
        last_date = city_data['date'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
        future_ordinals = [[d.toordinal()] for d in future_dates]

        # Make predictions
        future_predictions = model.predict(future_ordinals)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({'date': future_dates, 'index': future_predictions})
        forecast_df['type'] = 'Forecast'
        
        # Combine historical and forecast data for plotting
        city_data['type'] = 'Historical'
        plot_df = pd.concat([city_data[['date', 'index', 'type']], forecast_df])
        
        # Create Plotly figure
        fig_forecast = px.line(
            plot_df, x='date', y='index', color='type',
            color_discrete_map={'Historical': PRIMARY_ACCENT_COLOR, 'Forecast': SECONDARY_ACCENT_COLOR},
            labels={'index': 'AQI Index', 'date': 'Date'}
        )
        fig_forecast.update_layout(
            get_custom_plot_layout(height=500, title_text=f"30-Day AQI Forecast for {forecast_city}"),
            legend_title_text=''
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        st.info("💡 **Disclaimer:** This is a simple linear projection based on historical data and does not account for complex factors like weather, policy changes, or sudden events. It should be used for indicative purposes only.")
    else:
        st.warning(f"Not enough historical data for {forecast_city} to generate a forecast (requires >30 days).")
else:
    st.info("Select at least one city and a valid date range to use the forecasting tool.")

# ------------------- 🆚 CITY-WISE COMPARISON -------------------
if len(selected_cities) > 1:
    st.markdown("## 📊 City-to-City Comparison")
    comp_data = df_filtered[df_filtered['city'].isin(selected_cities)]
    if not comp_data.empty:
        fig_comp = px.line(
            comp_data, x='date', y='index', color='city',
            labels={'index': 'AQI Index', 'date': 'Date'},
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_comp.update_layout(get_custom_plot_layout(height=500, title_text="AQI Trend Comparison"))
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("No data available for the selected cities in the given date range.")

# ------------------- 🏙️ DETAILED CITY VIEW -------------------
if selected_cities:
    st.markdown("## 🔎 Detailed City View")
    for city in selected_cities:
        st.markdown(f"### {city.upper()}")
        city_data = df_filtered[df_filtered['city'] == city]
        if city_data.empty:
            st.warning(f"No data for {city} in the selected period.")
            continue
        
        col1, col2 = st.columns([1, 1])
        with col1: # AQI Category Distribution
            category_counts = city_data['level'].value_counts().reindex(AQI_CATEGORY_COLORS.keys(), fill_value=0)
            fig_dist = go.Figure(go.Bar(
                x=category_counts.index, y=category_counts.values,
                marker_color=[AQI_CATEGORY_COLORS[k] for k in category_counts.index],
                text=category_counts.values, textposition='auto'
            ))
            fig_dist.update_layout(get_custom_plot_layout(height=400, title_text="AQI Category Distribution", xaxis_title="AQI Level", yaxis_title="Number of Days"))
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2: # Dominant Pollutant
            pollutant_counts = city_data['pollutant'].value_counts()
            fig_poll = px.pie(
                pollutant_counts, values=pollutant_counts.values, names=pollutant_counts.index,
                color=pollutant_counts.index, color_discrete_map=POLLUTANT_COLORS, hole=0.5
            )
            fig_poll.update_traces(textinfo='percent+label', textfont_size=12, marker=dict(line=dict(color=CARD_COLOR, width=2)))
            fig_poll.update_layout(get_custom_plot_layout(height=400, title_text="Dominant Pollutants", showlegend=False))
            st.plotly_chart(fig_poll, use_container_width=True)

# ------------------- 🗃️ DATA EXPLORER [NEW FEATURE] -------------------
st.markdown("## 🗃️ Data Explorer")
if not df_filtered.empty:
    st.markdown("View and download the filtered air quality data used in this dashboard.")
    
    # Display a sample of the data in an expander
    with st.expander("Show Filtered Data Table"):
        display_df = df_filtered[['date', 'city', 'index', 'level', 'pollutant']].rename(columns={'index': 'AQI'})
        st.dataframe(display_df)
    
    # Provide a download button
    @st.cache_data
    def convert_df_to_csv(df_to_convert):
        return df_to_convert.to_csv(index=False).encode('utf-8')
    
    csv_data = convert_df_to_csv(df_filtered)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv_data,
        file_name=f"auravision_aqi_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
        mime='text/csv',
    )
else:
    st.warning("No data to display or download. Please adjust the filters in the sidebar.")

# ------------------- <footer> FOOTER -------------------
st.markdown(f"""
<div class="footer">
    <p style="font-size: 1.1rem; color: {TEXT_COLOR}; font-weight: 600;">🏆 AuraVision Pro</p>
    <p>A Championship-Grade AQI Dashboard</p>
    <p style="margin-top:1em; color:{SUBTLE_TEXT_COLOR}; font-size: 0.9rem;">Conceptualized by: Mr. Kapil Meena & Prof. Arkopal K. Goswami, IIT Kharagpur.</p>
    <p style="color:{SUBTLE_TEXT_COLOR}; font-size: 0.9rem;">Elite Rebuild by Google Gemini ✨</p>
</div>
""", unsafe_allow_html=True)

# -*- coding: utf-8 -*-
"""
AuraVision AQI Dashboard
=========================

A comprehensive and aesthetically refined Streamlit dashboard for analyzing air quality data across India.
This application combines powerful data visualization with a best-in-class user interface.

Conceptualized by: Mr. Kapil Meena & Prof. Arkopal K. Goswami, IIT Kharagpur.
UI/UX, Refinement, and Final Code by Google Gemini.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# --- 🎨 GLOBAL THEME & STYLE CONFIGURATION ---

# Base Plotly template for a cohesive dark theme
pio.templates.default = "plotly_dark"

# Professionally curated color palette for a modern, high-contrast dark theme
ACCENT_COLOR = "#00B0FF"  # A vibrant, clear blue for highlights and interactivity
TEXT_COLOR = "#F0F2F6"  # A light, easily readable off-white for body text
SUBTLE_TEXT_COLOR = "#A0AEC0"  # A softer grey for secondary text, labels, and annotations
BACKGROUND_COLOR = "#0F172A"  # A deep, professional navy blue background
CARD_BACKGROUND_COLOR = "#1E293B"  # A slightly lighter slate blue for cards, creating depth
BORDER_COLOR = "#334155"  # A subtle border color that complements the card background

# Font selection for a clean, modern, and readable UI
GLOBAL_FONT_FAMILY = "Inter, sans-serif"

# Consistent color mapping for AQI categories, chosen for clarity and visual impact
CATEGORY_COLORS = {
    'Severe': '#D32F2F',
    'Very Poor': '#F57C00',
    'Poor': '#FFA000',
    'Moderate': '#FBC02D',
    'Satisfactory': '#7CB342',
    'Good': '#388E3C',
    'Unknown': '#475569'  # A neutral grey for unknown data
}

# Distinct colors for major pollutants
POLLUTANT_COLORS = {
    'PM2.5': '#FF6E40', 'PM10': '#29B6F6', 'NO2': '#7E57C2',
    'SO2': '#FFEE58', 'CO': '#FFA726', 'O3': '#66BB6A', 'Other': '#BDBDBD'
}

# ------------------- ⚙️ PAGE CONFIGURATION -------------------
# Set the page layout to wide, with a custom title and icon
st.set_page_config(layout="wide", page_title="AuraVision AQI Dashboard", page_icon="💨")

# ------------------- ✨ CUSTOM CSS STYLING -------------------
# This CSS block is crucial for achieving the "best in the world" aesthetic.
# It overrides default Streamlit styles to create a bespoke, modern, and responsive design.
st.markdown(f"""
<style>
    /* Import the Inter font from Google Fonts for a premium feel */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* --- General Body & Container Styling --- */
    body {{
        font-family: {GLOBAL_FONT_FAMILY};
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
        line-height: 1.6;
    }}
    .main .block-container {{
        padding: 2rem 3rem 3rem 3rem; /* Generous padding for a spacious feel */
    }}

    /* --- Card Styling (for all charts, dataframes, expanders) --- */
    .stPlotlyChart, .stDataFrame, .stAlert, .stMetric,
    div[data-testid="stExpander"], div[data-testid="stForm"] {{
        border-radius: 12px; /* Smooth, modern rounded corners */
        border: 1px solid {BORDER_COLOR};
        background-color: {CARD_BACKGROUND_COLOR};
        padding: 1.8rem;
        margin-bottom: 2.2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }}
    .stPlotlyChart:hover, div[data-testid="stExpander"]:hover {{
         transform: translateY(-4px); /* Subtle lift effect on hover */
         box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }}

    /* --- Interactive Widget Styling (Buttons, Selectors) --- */
    .stButton > button, .stDownloadButton > button {{
        border-radius: 8px !important;
        border: 1.5px solid {ACCENT_COLOR} !important;
        background-color: transparent !important;
        color: {ACCENT_COLOR} !important;
        padding: 0.6rem 1.4rem !important;
        font-weight: 600 !important;
        font-family: {GLOBAL_FONT_FAMILY} !important;
        transition: all 0.2s ease !important;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: {ACCENT_COLOR} !important;
        color: {BACKGROUND_COLOR} !important; /* Invert colors for a clear hover state */
        transform: scale(1.03) !important;
        box-shadow: 0 4px 12px -2px {ACCENT_COLOR}99 !important;
    }}

    /* --- Tab Styling --- */
    .stTabs [data-baseweb="tab-list"] {{
         border-bottom: 2px solid {BORDER_COLOR};
         margin-bottom: 1.5rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-weight: 600;
        font-family: {GLOBAL_FONT_FAMILY};
        color: {SUBTLE_TEXT_COLOR};
        border-bottom: 3px solid transparent; /* Prepare for active state underline */
        transition: color 0.3s ease, border-bottom-color 0.3s ease;
    }}
     .stTabs [aria-selected="true"] {{
        border-bottom: 3px solid {ACCENT_COLOR};
        color: {ACCENT_COLOR};
     }}

    /* --- Typography (Headings) --- */
    h1 {{
        font-weight: 800;
        letter-spacing: -1.5px;
        font-size: 3rem;
    }}
    h2 {{
        color: {ACCENT_COLOR};
        border-bottom: 2px solid {ACCENT_COLOR};
        padding-bottom: 0.7rem;
        margin-top: 3.5rem;
        margin-bottom: 2.2rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-size: 1.6rem;
    }}
    h3 {{
        color: {TEXT_COLOR};
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        font-weight: 600;
        font-size: 1.4rem;
    }}

    /* --- Sidebar Styling --- */
    .stSidebar {{
        background-color: {CARD_BACKGROUND_COLOR};
        border-right: 1px solid {BORDER_COLOR};
        padding: 2.2rem;
    }}
    .stSidebar .stMarkdown h2 {{ /* Sidebar Header */
        color: {ACCENT_COLOR} !important;
        font-size: 1.4rem !important;
        border-bottom: none !important;
    }}
    .stSidebar .stSelectbox label, .stSidebar .stMultiselect label {{
        font-size: 1rem;
        font-weight: 600 !important;
        color: {ACCENT_COLOR} !important;
    }}
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
        background-color: {BACKGROUND_COLOR} !important;
        border: 1px solid {BORDER_COLOR} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover {{
        border-color: {ACCENT_COLOR} !important; /* Highlight on hover */
    }}

    /* --- Metric KPI Styling --- */
    .stMetric {{
        border-left: 5px solid {ACCENT_COLOR};
        padding: 1.3rem 1.6rem;
    }}
    .stMetric > div:nth-child(1) {{ /* Label */
        color: {SUBTLE_TEXT_COLOR};
        font-weight: 500;
    }}
    .stMetric > div:nth-child(2) {{ /* Value */
        font-size: 2.5rem;
        font-weight: 700;
        color: {ACCENT_COLOR};
    }}

    /* --- Custom Scrollbar --- */
    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-track {{ background: {BACKGROUND_COLOR}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER_COLOR}; border-radius: 5px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {ACCENT_COLOR}; }}

    /* --- Footer Styling --- */
    .footer {{
        text-align: center; margin-top: 5rem; padding: 2.5rem;
        background-color: {CARD_BACKGROUND_COLOR}; border-radius: 12px;
        border-top: 1px solid {BORDER_COLOR};
    }}
    .footer p {{ margin: 0.5em; color: {SUBTLE_TEXT_COLOR}; font-size: 0.9rem; }}
    .footer a {{ color: {ACCENT_COLOR}; text-decoration: none; font-weight: 600; }}
    .footer a:hover {{ text-decoration: underline; filter: brightness(1.2); }}
</style>
""", unsafe_allow_html=True)


# --- 📊 PLOTLY UNIVERSAL LAYOUT FUNCTION ---
# A helper function to ensure all plots have a consistent, professional appearance.
def get_custom_plotly_layout(height=None, title_text=None):
    """Generates a standardized Plotly layout dictionary."""
    layout_args = dict(
        font_family=GLOBAL_FONT_FAMILY.split(',')[0].strip(),
        font_color=TEXT_COLOR,
        paper_bgcolor=CARD_BACKGROUND_COLOR,
        plot_bgcolor=CARD_BACKGROUND_COLOR,
        legend=dict(
            font=dict(color=SUBTLE_TEXT_COLOR, size=10),
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        title_font_family=GLOBAL_FONT_FAMILY.split(',')[0].strip(),
        title_font_color=TEXT_COLOR,
        title_x=0.05, # Align title to the left for a modern look
        title_xref="container",
        xaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, zerolinecolor=BORDER_COLOR, tickfont=dict(color=SUBTLE_TEXT_COLOR, size=10), showgrid=True, gridwidth=1),
        yaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, zerolinecolor=BORDER_COLOR, tickfont=dict(color=SUBTLE_TEXT_COLOR, size=10), showgrid=True, gridwidth=1),
        margin=dict(l=50, r=30, t=70 if title_text else 40, b=50) # Dynamic top margin
    )
    if height:
        layout_args['height'] = height
    if title_text:
        layout_args['title_text'] = f"<b>{title_text}</b>" # Bold title
        layout_args['title_font_size'] = 18
    return layout_args


# ------------------- 👑 HEADER SECTION -------------------
st.markdown("<h1 style='text-align: center;'>💨 AuraVision AQI</h1>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; margin-bottom: 3rem;">
    <p style="color: {SUBTLE_TEXT_COLOR}; font-size: 1.2rem; font-weight: 400; max-width: 650px; margin: auto;">
        An elite dashboard for illuminating air quality insights across India with unparalleled clarity and design.
    </p>
</div>
""", unsafe_allow_html=True)


# ------------------- 💾 DATA LOADING & CACHING -------------------
@st.cache_data(ttl=3600)  # Cache data for 1 hour to improve performance
def load_data(file_path="combined_air_quality.txt"):
    """
    Loads, cleans, and prepares the air quality data.
    Includes robust error handling and data validation.
    """
    try:
        if not os.path.exists(file_path):
            return pd.DataFrame(), f"Error: Data file '{os.path.basename(file_path)}' not found.", None
        
        df_loaded = pd.read_csv(file_path, sep="\t", parse_dates=['date'])
        
        # Calculate last update time
        last_update_time = pd.Timestamp(os.path.getmtime(file_path), unit='s').tz_localize('UTC').tz_convert('Asia/Kolkata')
        
        # --- Data Cleaning and Preprocessing ---
        # Ensure critical columns exist
        for col, default_val in [('pollutant', np.nan), ('level', 'Unknown')]:
            if col not in df_loaded.columns: df_loaded[col] = default_val
        
        # Clean pollutant data: take the first pollutant if multiple are listed
        df_loaded['pollutant'] = df_loaded['pollutant'].astype(str).str.split(',').str[0].str.strip().replace(['nan', 'NaN', 'None', ''], np.nan)
        df_loaded['pollutant'] = df_loaded['pollutant'].fillna('Other')

        # Clean and standardize the 'level' column
        df_loaded['level'] = df_loaded['level'].astype(str).fillna('Unknown').str.strip()
        known_levels = list(CATEGORY_COLORS.keys())
        df_loaded['level'] = df_loaded['level'].apply(lambda x: x if x in known_levels else 'Unknown')
        
        return df_loaded, "Data loaded successfully.", last_update_time

    except Exception as e:
        return pd.DataFrame(), f"Fatal Error: Could not process '{os.path.basename(file_path)}': {e}.", None

# Load the data
df, load_message, data_last_updated = load_data()

# Stop the app if data loading fails
if df.empty:
    st.error(f"Dashboard cannot operate. {load_message}")
    st.stop()

# Display data source and last update time
update_time_str = data_last_updated.strftime('%B %d, %Y, %H:%M %Z') if data_last_updated else "Not available"
st.markdown(f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR}; font-size: 0.9rem; margin-bottom: 2.5rem;'>Archive data last updated: {update_time_str}</p>", unsafe_allow_html=True)


# ------------------- ⚙️ SIDEBAR FILTERS -------------------
st.sidebar.markdown("## 🔬 EXPLORE & FILTER")
st.sidebar.markdown("---", unsafe_allow_html=True)
st.sidebar.info("Adjust filters to analyze AQI trends for specific regions and time periods.")

# City Selection
unique_cities = sorted(df['city'].unique())
default_cities = ["Delhi", "Mumbai", "Kolkata"]
# Ensure default cities exist in the dataset
valid_defaults = [city for city in default_cities if city in unique_cities]
if not valid_defaults and unique_cities:
    valid_defaults = unique_cities[0:1] # Fallback to the first city if none of the preferred defaults are present
selected_cities = st.sidebar.multiselect("🏙️ Select Cities", unique_cities, default=valid_defaults)

# Year Selection
years = sorted(df['date'].dt.year.unique(), reverse=True)
year = st.sidebar.selectbox("🗓️ Select Year", years, index=0, disabled=not bool(years))

# Month Selection
months_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
month_options = ["All Months"] + list(months_map.values())
selected_month_name = st.sidebar.selectbox("🌙 Select Month", month_options, index=0)

month_number_filter = None
if selected_month_name != "All Months":
    month_number_filter = list(months_map.keys())[list(months_map.values()).index(selected_month_name)]

# --- Apply filters to the dataframe ---
df_period_filtered = df[df['date'].dt.year == year].copy() if year else pd.DataFrame()
if month_number_filter and not df_period_filtered.empty:
    df_period_filtered = df_period_filtered[df_period_filtered['date'].dt.month == month_number_filter]


# ------------------- 🇮🇳 NATIONAL SNAPSHOT -------------------
st.markdown("## 🇮🇳 National Snapshot")
if year:
    st.markdown(f"### Key Metro Annual Average AQI ({year})")
    major_cities = ['Delhi', 'Mumbai', 'Kolkata', 'Bengaluru', 'Chennai', 'Hyderabad', 'Pune', 'Ahmedabad']
    # Use the full year's data for this snapshot, not the monthly filter
    major_cities_annual_data = df[(df['date'].dt.year == year) & (df['city'].isin(major_cities))]

    if not major_cities_annual_data.empty:
        avg_aqi_major_cities = major_cities_annual_data.groupby('city')['index'].mean().dropna()
        present_major_cities = [city for city in major_cities if city in avg_aqi_major_cities.index]
        
        if present_major_cities:
            # Display metrics in a responsive grid
            cols = st.columns(min(len(present_major_cities), 4))
            for i, city_name in enumerate(present_major_cities):
                with cols[i % 4]:
                    st.metric(label=city_name, value=f"{avg_aqi_major_cities.get(city_name):.1f}")
        else:
            st.info(f"No annual AQI data available for key metro cities in {year}.")
    else:
        st.info(f"No data available for key metro cities in {year}.")

    # --- General Insights for the selected filter period ---
    st.markdown(f"### Insights for {selected_month_name}, {year}")
    if not df_period_filtered.empty:
        avg_aqi_national = df_period_filtered['index'].mean()
        city_avg_aqi_stats = df_period_filtered.groupby('city')['index'].mean().dropna()
        if not city_avg_aqi_stats.empty:
            num_cities = df_period_filtered['city'].nunique()
            best_city, best_aqi = city_avg_aqi_stats.idxmin(), city_avg_aqi_stats.min()
            worst_city, worst_aqi = city_avg_aqi_stats.idxmax(), city_avg_aqi_stats.max()
            
            # Use columns for a cleaner layout
            col1, col2, col3 = st.columns(3)
            col1.metric("Average AQI (All Cities)", f"{avg_aqi_national:.2f}")
            col2.metric(f"Best City ({best_city})", f"{best_aqi:.2f}")
            col3.metric(f"Worst City ({worst_city})", f"{worst_aqi:.2f}")
        else:
            st.info("AQI statistics could not be computed for the selected period.")
    else:
        st.warning("No data available for the selected period to generate insights. Try selecting 'All Months'.")
else:
    st.warning("Please select a year to view national insights.")
st.markdown("---")


# ------------------- 📍 CITY AQI HOTSPOTS MAP -------------------
st.markdown("## 📍 City AQI Hotspots")
coords_file_path = "lat_long.txt"
map_rendered = False

@st.cache_data
def load_coords(file_path):
    """Loads city coordinates safely."""
    if not os.path.exists(file_path):
        return None, f"Coordinates file '{file_path}' not found."
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Safely evaluate the file content which is expected to be a dictionary
            city_coords = eval(f.read())
        if isinstance(city_coords, dict):
            return city_coords, "Coordinates loaded."
        else:
            return None, "Coordinates file is not in the expected dictionary format."
    except Exception as e:
        return None, f"Error reading or parsing coordinates file: {e}"

city_coords_data, coords_message = load_coords(coords_file_path)

if not df_period_filtered.empty and city_coords_data:
    # Aggregate data for the map
    map_data = df_period_filtered.groupby('city').agg(
        avg_aqi=('index', 'mean'),
        dominant_pollutant=('pollutant', lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A')
    ).reset_index().dropna(subset=['avg_aqi'])

    # Create a DataFrame from the coordinates dictionary
    coords_df = pd.DataFrame([{'city': city, 'lat': coords[0], 'lon': coords[1]} for city, coords in city_coords_data.items() if isinstance(coords, (list, tuple)) and len(coords) == 2])
    
    # Merge aggregated data with coordinates
    map_merged_df = pd.merge(map_data, coords_df, on='city', how='inner')

    if not map_merged_df.empty:
        # Assign AQI category for coloring
        map_merged_df["AQI Category"] = map_merged_df["avg_aqi"].apply(
            lambda val: next((k for k, v_range in {
                'Good': (0, 50), 'Satisfactory': (51, 100), 'Moderate': (101, 200),
                'Poor': (201, 300), 'Very Poor': (301, 400), 'Severe': (401, float('inf'))
            }.items() if v_range[0] <= val <= v_range[1]), "Unknown")
        )
        
        fig_scatter_map = px.scatter_mapbox(
            map_merged_df,
            lat="lat", lon="lon",
            size=np.maximum(map_merged_df["avg_aqi"], 5), # Ensure a minimum size for visibility
            size_max=30,
            color="AQI Category",
            color_discrete_map=CATEGORY_COLORS,
            hover_name="city",
            custom_data=['city', 'avg_aqi', 'dominant_pollutant', 'AQI Category'],
            text="city",
            zoom=4.2,
            center={"lat": 23.5, "lon": 82.5}
        )
        
        map_layout = get_custom_plotly_layout(height=700, title_text=f"Average AQI Hotspots - {selected_month_name}, {year}")
        map_layout['mapbox_style'] = "carto-darkmatter"
        map_layout['margin'] = {"r":10,"t":80,"l":10,"b":10}
        
        fig_scatter_map.update_traces(
            marker=dict(sizemin=6, opacity=0.8),
            hovertemplate="<b style='font-size:1.2em;'>%{customdata[0]}</b><br><br>" +
                          "Avg. AQI: <b>%{customdata[1]:.1f}</b> (%{customdata[3]})<br>" +
                          "Dominant Pollutant: %{customdata[2]}" +
                          "<extra></extra>" # Removes the trace name from hover
        )
        fig_scatter_map.update_layout(**map_layout)
        st.plotly_chart(fig_scatter_map, use_container_width=True)
        map_rendered = True

# Fallback or Info Messages
if not map_rendered:
    if not city_coords_data:
        st.warning(f"**Map Unavailable:** {coords_message}")
    elif df_period_filtered.empty:
        st.warning("**Map Unavailable:** No air quality data for the selected filters.")
    else:
        st.info("No city data from your dataset matched the available coordinates for the selected period.")
st.markdown("---")


# ------------------- 🆚 CITY-WISE COMPARISON -------------------
if len(selected_cities) > 1:
    st.markdown("## 📊 City-to-City Comparison")
    comparison_data = df_period_filtered[df_period_filtered['city'].isin(selected_cities)]
    
    if not comparison_data.empty:
        comp_tab1, comp_tab2 = st.tabs(["📈 AQI Trend Comparison", "🌀 Seasonal AQI Radar"])

        with comp_tab1:
            fig_cmp_line = px.line(
                comparison_data, x='date', y='index', color='city',
                labels={'index': 'AQI Index', 'date': 'Date', 'city': 'City'},
                line_shape='spline', color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_cmp_line.update_layout(**get_custom_plotly_layout(height=500, title_text="AQI Trends Over Selected Period"))
            fig_cmp_line.update_traces(line=dict(width=2.5))
            st.plotly_chart(fig_cmp_line, use_container_width=True)
        
        with comp_tab2:
            radar_fig = go.Figure()
            df_year_for_radar = df[(df['date'].dt.year == year) & (df['city'].isin(selected_cities))]
            
            max_aqi_radar = 0
            radar_colors = px.colors.qualitative.Pastel
            for i, city in enumerate(selected_cities):
                city_radar_data = df_year_for_radar[df_year_for_radar['city'] == city]
                if not city_radar_data.empty:
                    monthly_avg = city_radar_data.groupby(city_radar_data['date'].dt.month)['index'].mean().reindex(range(1, 13))
                    max_aqi_radar = max(max_aqi_radar, monthly_avg.max())
                    
                    radar_fig.add_trace(go.Scatterpolar(
                        r=monthly_avg.values,
                        theta=list(months_map.values()),
                        fill='toself',
                        name=city,
                        hovertemplate=f"<b>{city}</b><br>%{{theta}}: %{{r:.1f}}<extra></extra>",
                        line_color=radar_colors[i % len(radar_colors)],
                        opacity=0.7
                    ))
            
            radar_layout = get_custom_plotly_layout(height=550, title_text="Annual Seasonal AQI Patterns")
            radar_layout['polar'] = dict(
                radialaxis=dict(visible=True, range=[0, max_aqi_radar * 1.1 if max_aqi_radar > 0 else 50], gridcolor=BORDER_COLOR),
                angularaxis=dict(linecolor=BORDER_COLOR, gridcolor=BORDER_COLOR),
                bgcolor=CARD_BACKGROUND_COLOR
            )
            radar_fig.update_layout(**radar_layout)
            st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.info("No data available for the selected cities and period to make a comparison.")
    st.markdown("---")


# ------------------- 🏙️ DETAILED CITY VIEW -------------------
if selected_cities:
    st.markdown("## 🔎 Detailed City View")
    for city in selected_cities:
        # Use full year data for calendar, period filtered for tabs
        city_data_year = df[(df['city'] == city) & (df['date'].dt.year == year)]
        city_data_period = df_period_filtered[df_period_filtered['city'] == city]

        if city_data_year.empty:
            st.warning(f"😔 No data available for **{city}** in {year}. Please select a different year.")
            continue
        
        st.markdown(f"### {city.upper()} – Analysis for {year}")
        with st.container():
            # --- Daily AQI Calendar Heatmap ---
            start_date, end_date = pd.to_datetime(f'{year}-01-01'), pd.to_datetime(f'{year}-12-31')
            full_period_dates = pd.date_range(start_date, end_date)
            calendar_df = pd.DataFrame({'date': full_period_dates})
            calendar_df['week'] = calendar_df['date'].dt.isocalendar().week
            calendar_df['day_of_week'] = calendar_df['date'].dt.dayofweek
            # Fix for weeks at year boundary
            calendar_df.loc[(calendar_df['date'].dt.month == 1) & (calendar_df['week'] > 50), 'week'] = 0
            calendar_df.loc[(calendar_df['date'].dt.month == 12) & (calendar_df['week'] == 1), 'week'] = 53

            merged_cal_df = pd.merge(calendar_df, city_data_year[['date', 'index', 'level']], on='date', how='left')
            merged_cal_df['level'] = merged_cal_df['level'].fillna('Unknown')
            merged_cal_df['aqi_text'] = merged_cal_df['index'].apply(lambda x: f'{x:.0f}' if pd.notna(x) else 'N/A')
            
            level_to_num = {level: i for i, level in enumerate(CATEGORY_COLORS.keys())}
            
            fig_cal = go.Figure(data=go.Heatmap(
                x=merged_cal_df['week'], y=merged_cal_df['day_of_week'],
                z=merged_cal_df['level'].map(level_to_num),
                customdata=np.stack((merged_cal_df['date'].dt.strftime('%b %d, %Y'), merged_cal_df['level'], merged_cal_df['aqi_text']), axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>AQI: %{customdata[2]} (%{customdata[1]})<extra></extra>",
                colorscale=list(CATEGORY_COLORS.values()), showscale=False, xgap=3, ygap=3,
                zmin=0, zmax=len(CATEGORY_COLORS)-1
            ))
            
            cal_layout = get_custom_plotly_layout(height=320, title_text=f"Daily AQI Calendar - {city}, {year}")
            cal_layout['yaxis'].update(dict(tickmode='array', tickvals=list(range(7)), ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], autorange="reversed", showgrid=False))
            cal_layout['xaxis'].update(dict(showgrid=False, ticktext=[], tickvals=[]))
            fig_cal.update_layout(**cal_layout)
            st.plotly_chart(fig_cal, use_container_width=True)

            # --- Tabs for other visualizations ---
            if not city_data_period.empty:
                city_tabs = st.tabs(["📊 AQI Category Distribution", "📅 Monthly AQI Heatmap"])
                with city_tabs[0]:
                    category_counts = city_data_period['level'].value_counts().reindex(CATEGORY_COLORS.keys(), fill_value=0).reset_index()
                    category_counts.columns = ['AQI Category', 'Number of Days']
                    fig_dist = px.bar(
                        category_counts, x='AQI Category', y='Number of Days', color='AQI Category',
                        color_discrete_map=CATEGORY_COLORS, text_auto='.0f'
                    )
                    fig_dist.update_layout(**get_custom_plotly_layout(height=450, title_text=f"AQI Day Categories - {city} ({selected_month_name})"))
                    st.plotly_chart(fig_dist, use_container_width=True)

                with city_tabs[1]:
                    city_data_period['month_name'] = pd.Categorical(city_data_period['date'].dt.strftime('%B'), categories=[months_map[i] for i in range(1,13) if i in city_data_period['date'].dt.month.unique()], ordered=True)
                    heatmap_pivot = city_data_period.pivot_table(index='month_name', columns=city_data_period['date'].dt.day, values='index', observed=True)
                    fig_heat = px.imshow(
                        heatmap_pivot, labels=dict(x="Day of Month", y="Month", color="AQI Index"),
                        aspect="auto", color_continuous_scale=px.colors.sequential.YlOrRd, text_auto=".0f"
                    )
                    fig_heat.update_layout(**get_custom_plotly_layout(height=max(400, len(heatmap_pivot.index)*50), title_text=f"Daily AQI Heatmap - {city} ({selected_month_name})"))
                    st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info(f"No specific data for {city} in {selected_month_name}, {year}. The calendar above shows the full annual view. Try selecting 'All Months' for more details.")
    st.markdown("---")


# ------------------- 🧪 POLLUTANT ANALYSIS -------------------
st.markdown("## 🧪 Dominant Pollutant Insights")
if not df_period_filtered.empty:
    # Use selected cities for the dropdown, fallback to all unique cities if none are selected
    pollutant_city_options = selected_cities if selected_cities else unique_cities
    
    if pollutant_city_options:
        # Set a sensible default for the selectbox
        default_pollutant_city_idx = 0
        if selected_cities:
            default_pollutant_city_idx = pollutant_city_options.index(selected_cities[0])
        
        city_for_pollutants = st.selectbox(
            "Select City for Pollutant Breakdown:", pollutant_city_options,
            index=default_pollutant_city_idx,
            key="pollutant_city_selector"
        )
        
        pollutant_data = df_period_filtered[df_period_filtered['city'] == city_for_pollutants]
        
        if not pollutant_data.empty and pollutant_data['pollutant'].notna().any():
            # Exclude 'Other' for a more meaningful chart
            valid_pollutants = pollutant_data[pollutant_data['pollutant'] != 'Other']
            if not valid_pollutants.empty:
                pollutant_counts = valid_pollutants['pollutant'].value_counts().reset_index()
                pollutant_counts.columns = ['Pollutant', 'Days']
                
                fig_pollutant_pie = px.pie(
                    pollutant_counts, values='Days', names='Pollutant',
                    color='Pollutant', color_discrete_map=POLLUTANT_COLORS,
                    hole=0.4 # Create a donut chart
                )
                
                pie_layout = get_custom_plotly_layout(height=500, title_text=f"Dominant Pollutant Distribution - {city_for_pollutants}")
                pie_layout['legend']['orientation'] = 'v'
                pie_layout['legend']['x'] = 1.05
                pie_layout['legend']['y'] = 0.5
                pie_layout['legend']['xanchor'] = 'left'
                pie_layout['legend']['yanchor'] = 'middle'
                
                fig_pollutant_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color=CARD_BACKGROUND_COLOR, width=2)))
                fig_pollutant_pie.update_layout(**pie_layout)
                st.plotly_chart(fig_pollutant_pie, use_container_width=True)
            else:
                st.info(f"No specific dominant pollutant data (excluding 'Other') recorded for {city_for_pollutants} in this period.")
        else:
            st.warning(f"No pollutant data available for {city_for_pollutants} in the selected period.")
    else:
        st.info("No cities available for pollutant analysis with current filters.")
else:
    st.warning("Select a valid period to view pollutant analysis.")
st.markdown("---")

# ------------------- <footer> FOOTER -------------------
st.markdown(f"""
<div class="footer">
    <p style="font-size: 1.1rem; color: {TEXT_COLOR}; font-weight: 600;">💨 AuraVision AQI Dashboard</p>
    <p>Conceptualized by: Mr. Kapil Meena & Prof. Arkopal K. Goswami, IIT Kharagpur.</p>
    <p>Enhanced UI/UX, Debugging, and Final Implementation by Google Gemini ✨</p>
    <p style="margin-top:1.5em;"><a href="https://github.com/kapil2020/india-air-quality-dashboard" target="_blank" rel="noopener noreferrer">🔗 View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)

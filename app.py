# -*- coding: utf-8 -*-
"""
AuraVision Pro AQI Dashboard - Championship Edition v4
======================================================

An award-winning, feature-rich Streamlit application for analyzing and forecasting
air quality in India. Designed for the Google Best Dashboard Challenge.

Key Enhancements in v4:
- Default city to "Delhi".
- Increased sidebar width for better filter visibility.
- Corrected and enhanced footer design.
- Upgraded AQI forecast to Polynomial Regression with confidence intervals.
- Ensured consistent dark theme on mobile and improved responsiveness.
- Refined map bubble sizes.

Conceptualized by: Mr. Kapil Meena & Prof. Arkopal K. Goswami, IIT Kharagpur.
Championship Build & Enhancements by Google Gemini.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import timedelta

# --- 🏆 ELITE UI & THEME CONFIGURATION ---

# Set the base Plotly template for a cohesive, professional look
pio.templates.default = "plotly_dark"

# A professional, high-contrast, and visually appealing color palette inspired by modern design systems
PRIMARY_ACCENT_COLOR = "#00A9FF"  # A vibrant, accessible blue for key actions and highlights
SECONDARY_ACCENT_COLOR = "#38E5A3" # A fresh green for positive indicators or secondary highlights
BACKGROUND_COLOR = "#0D1117"  # GitHub's refined dark mode background
CARD_COLOR = "#161B22"       # GitHub's card color provides subtle depth
TEXT_COLOR = "#C9D1D9"       # Primary text color for readability
SUBTLE_TEXT_COLOR = "#8B949E"  # Secondary text color for labels, annotations, and captions
BORDER_COLOR = "#30363D"       # A soft border that complements the card background
ERROR_COLOR = "#F87171"       # For error messages or severe AQI

# Perceptually distinct and sequential AQI category colors for intuitive understanding
AQI_CATEGORY_COLORS = {
    'Good': '#34D399', 'Satisfactory': '#A7F3D0', 'Moderate': '#FBBF24',
    'Poor': '#FB923C', 'Very Poor': ERROR_COLOR, 'Severe': '#DC2626', 'Unknown': '#4B5563'
}

# High-contrast colors for pollutants, ensuring clarity in charts
POLLUTANT_COLORS = {
    'PM2.5': '#F97316', 'PM10': '#3B82F6', 'NO2': '#8B5CF6',
    'SO2': '#EAB308', 'CO': '#EF4444', 'O3': '#22C55E', 'Other': '#9CA3AF'
}

# Font selection for a premium, modern, and highly readable UI
GLOBAL_FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

# ------------------- ⚙️ PAGE CONFIGURATION -------------------
st.set_page_config(
    layout="wide",
    page_title="AuraVision Pro | AQI Dashboard",
    page_icon="🏆",
    initial_sidebar_state="expanded" # Keep sidebar open by default
)


# ------------------- ✨ ADVANCED CSS STYLING -------------------
# This master CSS block defines the dashboard's bespoke, premium aesthetic.
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* --- Universal Styles --- */
    * {{
        font-family: {GLOBAL_FONT} !important; /* Ensure font is applied everywhere */
        box-sizing: border-box;
    }}
    html, body {{
        height: 100%;
        margin: 0;
        padding: 0;
    }}
    body {{
        background-color: {BACKGROUND_COLOR} !important; /* Ensure dark background */
        color: {TEXT_COLOR};
    }}

    /* --- Main Layout & Container --- */
    .main .block-container {{
        padding: 1.5rem 2rem 2rem 2rem; /* Optimized padding */
    }}

    /* --- Universal Card Styling --- */
    .stPlotlyChart, .stDataFrame, .stAlert, .stMetric,
    div[data-testid="stExpander"], div[data-testid="stForm"] {{
        background-color: {CARD_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px; /* Consistent rounding */
        padding: 1.25rem; /* Standardized padding */
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .stPlotlyChart:hover {{
         transform: translateY(-3px);
         box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }}
    /* Remove excessive padding/border from internal Streamlit containers to let card style dominate */
    div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {{
        padding: 0 !important;
        border: none !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }}

    /* --- Typography --- */
    h1, h2, h3 {{
        color: {TEXT_COLOR};
        font-weight: 700;
        letter-spacing: -0.5px;
    }}
    h1 {{
        font-size: 2.5rem; /* Adjusted for responsiveness */
        font-weight: 800;
        letter-spacing: -1.2px;
        text-align: center;
        margin-bottom: 0.5rem; /* Reduced bottom margin */
    }}
    h2 {{
        font-size: 1.6rem; /* Adjusted */
        padding-bottom: 0.6rem;
        margin-top: 2.5rem; /* Standardized top margin */
        margin-bottom: 1.5rem; /* Standardized bottom margin */
        border-bottom: 1.5px solid {BORDER_COLOR}; /* Subtler border */
    }}
    h3 {{
        font-size: 1.25rem; /* Adjusted */
        margin-bottom: 0.75rem;
    }}

    /* --- Sidebar Styling --- */
    .stSidebar {{
        background-color: rgba(22, 27, 34, 0.9) !important; /* Darker, semi-transparent */
        backdrop-filter: blur(8px); /* Frosted glass effect */
        border-right: 1px solid {BORDER_COLOR};
        width: 20rem !important; /* Increased sidebar width */
        padding: 1.5rem;
    }}
    .stSidebar .stSelectbox label,
    .stSidebar .stMultiselect label,
    .stSidebar .stDateInput label {{
        color: {PRIMARY_ACCENT_COLOR} !important;
        font-weight: 600 !important;
        font-size: 0.95rem; /* Slightly smaller for more content */
    }}
     .stSidebar h1, .stSidebar h2, .stSidebar h3 {{ /* Sidebar specific headings */
        text-align: left;
        margin-bottom: 0.5rem;
    }}
    .stSidebar .stButton>button {{
        width: 100%;
        margin-top: 0.5rem;
    }}

    /* --- Metric KPI Styling --- */
    .stMetric {{
        border-left: 4px solid {PRIMARY_ACCENT_COLOR};
        padding: 1rem;
    }}
    .stMetric > div:nth-child(1) {{ /* Label */
        color: {SUBTLE_TEXT_COLOR};
        font-size: 0.9rem;
        font-weight: 500;
    }}
    .stMetric > div:nth-child(2) {{ /* Value */
        color: {PRIMARY_ACCENT_COLOR};
        font-size: 2rem; /* Adjusted for space */
        font-weight: 700;
    }}
     .stMetric > div:nth-child(3) > div {{ /* Delta (to style the arrow and text) */
        font-size: 0.85rem;
    }}


    /* --- Tabs Styling --- */
    .stTabs [data-baseweb="tab-list"] {{
        border-bottom-color: {BORDER_COLOR} !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent !important;
        color: {SUBTLE_TEXT_COLOR} !important;
        padding-top: 0.75rem; padding-bottom: 0.75rem;
    }}
    .stTabs [aria-selected="true"] {{
        color: {PRIMARY_ACCENT_COLOR} !important;
        border-bottom-color: {PRIMARY_ACCENT_COLOR} !important;
    }}


    /* --- Custom Scrollbar --- */
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER_COLOR}; border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {PRIMARY_ACCENT_COLOR}; }}

    /* Footer Styling */
    .footer {{
        text-align: center;
        padding: 2rem 1rem;
        margin-top: 3rem;
        background-color: {CARD_COLOR};
        border-top: 1px solid {BORDER_COLOR};
        border-radius: 12px 12px 0 0; /* Rounded top corners for footer card */
    }}
    .footer p {{
        margin: 0.3rem 0;
        color: {SUBTLE_TEXT_COLOR};
        font-size: 0.85rem;
    }}
    .footer a {{
        color: {PRIMARY_ACCENT_COLOR};
        text-decoration: none;
        font-weight: 500;
    }}
    .footer a:hover {{
        text-decoration: underline;
    }}

    /* Responsive adjustments for mobile */
    @media (max-width: 768px) {{
        .main .block-container {{
            padding: 1rem;
        }}
        h1 {{ font-size: 1.8rem; }}
        h2 {{ font-size: 1.4rem; margin-top: 2rem; margin-bottom: 1rem;}}
        h3 {{ font-size: 1.1rem; }}
        .stMetric > div:nth-child(2) {{ font-size: 1.8rem; }}
        .stSidebar {{ width: 100% !important; height: auto; position: relative; }} /* Adjust sidebar for mobile */
        .stTabs [data-baseweb="tab"] {{
            font-size: 0.85rem; padding-left: 0.5rem; padding-right: 0.5rem;
        }}
    }}
</style>
""", unsafe_allow_html=True)


# ------------------- 📊 PLOTLY UNIVERSAL LAYOUT FUNCTION -------------------
def get_plot_layout(height=None, title_text=None, **kwargs):
    """Generates a standardized, beautiful Plotly layout for all charts."""
    layout = go.Layout(
        title=f"<b>{title_text}</b>" if title_text else None,
        height=height,
        font=dict(family=GLOBAL_FONT, size=11, color=TEXT_COLOR), # Smaller base font for plots
        paper_bgcolor=CARD_COLOR, plot_bgcolor=CARD_COLOR,
        title_x=0.05, title_font_size=16, # Adjusted title font size
        xaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, zeroline=False, tickfont_size=10),
        yaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, zeroline=False, tickfont_size=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=10),
        margin=dict(l=40, r=20, t=60 if title_text else 30, b=40), # Optimized margins
        **kwargs
    )
    return layout


# ------------------- 💾 DATA LOADING & PREPARATION -------------------
@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_data(file_path="combined_air_quality.txt"):
    """Loads, cleans, and prepares the AQI data with robust error handling."""
    try:
        df_loaded = pd.read_csv(file_path, sep="\t", parse_dates=['date'])
        # Clean pollutant data: take the first pollutant if multiple, handle NaN representations
        df_loaded['pollutant'] = (
            df_loaded['pollutant'].astype(str)
            .str.split(',').str[0].str.strip()
            .replace(['nan', 'NaN', 'None', ''], np.nan) # More comprehensive NaN cleaning
            .fillna('Other')
        )
        # Clean and standardize the 'level' column
        df_loaded['level'] = df_loaded['level'].astype(str).fillna('Unknown').str.strip()
        known_levels = list(AQI_CATEGORY_COLORS.keys())
        df_loaded['level'] = df_loaded['level'].apply(lambda x: x if x in known_levels else 'Unknown')
        return df_loaded
    except FileNotFoundError:
        st.error(f"FATAL: Data file '{file_path}' not found. The dashboard cannot run.")
        st.stop() # Stop execution if data is not found
    except Exception as e:
        st.error(f"FATAL: Error loading or processing data: {e}")
        st.stop() # Stop execution on other data errors

df = load_data()


# -------------------  SIDEBAR FILTERS -------------------
st.sidebar.title("🏆 AuraVision Pro")
st.sidebar.markdown("---")
st.sidebar.header("Global Controls")

# City Selection
unique_cities = sorted(df['city'].unique())
# Ensure "Delhi" is the default if it exists, otherwise use the first available city or an empty list
default_city = []
if "Delhi" in unique_cities:
    default_city = ["Delhi"]
elif unique_cities:
    default_city = [unique_cities[0]]

selected_cities = st.sidebar.multiselect(
    "🏙️ Select Cities",
    unique_cities,
    default=default_city, # Set default to Delhi or first city
    help="Select one or more cities for analysis."
)

# Year and Month Selection
years = sorted(df['date'].dt.year.unique(), reverse=True)
year_default_index = 0 # Default to the most recent year
year = st.sidebar.selectbox("🗓️ Select Year", years, index=year_default_index, help="Select the year for analysis.")

months_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
month_options = ["All Year"] + list(months_map.values())
selected_month_str = st.sidebar.selectbox("🌙 Select Month", month_options, index=0, help="Select 'All Year' or a specific month.")

# Apply filters to the dataframe
df_filtered = df[df['date'].dt.year == year].copy() # Start with year filter
if selected_month_str != "All Year":
    month_num = list(months_map.keys())[list(months_map.values()).index(selected_month_str)]
    df_filtered = df_filtered[df_filtered['date'].dt.month == month_num]


# ------------------- 👑 HEADER -------------------
st.markdown("<h1>AuraVision Pro AQI Dashboard</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR}; font-size: 1.1rem; max-width: 650px; margin: auto; margin-bottom: 2rem;'>An elite analytical tool for air quality in India, combining predictive insights with a world-class user interface.</p>", unsafe_allow_html=True)


# ------------------- 📈 NATIONAL SNAPSHOT -------------------
st.markdown("## 🇮🇳 National Snapshot")
if not df_filtered.empty:
    city_avg_aqi = df_filtered.groupby('city')['index'].mean().dropna()
    if not city_avg_aqi.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cities Observed", f"{df_filtered['city'].nunique()}")
        national_avg_val = df_filtered['index'].mean()
        col2.metric("National Average AQI", f"{national_avg_val:.2f}" if pd.notna(national_avg_val) else "N/A")

        best_city_name = city_avg_aqi.idxmin()
        best_aqi_val = city_avg_aqi.min()
        col3.metric(f"Best City ({best_city_name})", f"{best_aqi_val:.2f}" if pd.notna(best_aqi_val) else "N/A", delta="Cleanest", delta_color="normal")

        worst_city_name = city_avg_aqi.idxmax()
        worst_aqi_val = city_avg_aqi.max()
        col4.metric(f"Worst City ({worst_city_name})", f"{worst_aqi_val:.2f}" if pd.notna(worst_aqi_val) else "N/A", delta="Most Polluted", delta_color="inverse")
    else:
        st.info("No city-level average AQI data available for the selected time period.")
else:
    st.warning("No data available for the selected filters. Please adjust the sidebar controls.")


# ------------------- 🗺️ AQI HOTSPOTS MAP -------------------
st.markdown("## 📍 Interactive AQI Hotspots Map")
@st.cache_data # Cache coordinate loading
def load_coords(file_path="lat_long.txt"):
    """Loads city coordinates safely."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            city_coords_eval = eval(f.read()) # Use eval carefully, assuming trusted file format
        if isinstance(city_coords_eval, dict):
            return city_coords_eval
        return None # Return None if not a dict
    except Exception:
        return None # Return None on any error

city_coords_data = load_coords()

if city_coords_data and not df_filtered.empty:
    map_data_agg = df_filtered.groupby('city')['index'].mean().reset_index()
    coords_df = pd.DataFrame([
        {'city': city_name, 'lat': coords[0], 'lon': coords[1]}
        for city_name, coords in city_coords_data.items()
        if isinstance(coords, (list, tuple)) and len(coords) == 2 # Basic validation
    ])

    if not coords_df.empty:
        map_df_merged = pd.merge(map_data_agg, coords_df, on='city', how='inner')
        if not map_df_merged.empty:
            map_df_merged["Category"] = pd.cut(
                map_df_merged['index'],
                bins=[0, 50, 100, 200, 300, 400, np.inf], # Use np.inf for the upper bound
                labels=list(AQI_CATEGORY_COLORS.keys())[:-1], # Exclude 'Unknown' for binning
                right=True # Ensure 50 is in 'Good', 100 in 'Satisfactory' etc.
            ).astype(str).fillna('Unknown') # Ensure consistent type and handle NaNs from cut

            fig_map = px.scatter_mapbox(
                map_df_merged, lat="lat", lon="lon", size="index", color="Category",
                color_discrete_map=AQI_CATEGORY_COLORS,
                size_max=20, # Reduced max bubble size
                hover_name="city",
                hover_data={"index": ":.2f", "Category": True, "lat": False, "lon": False},
                center={"lat": 23.5, "lon": 82.5}, # Centered on India
                zoom=3.8, # Adjusted zoom
                text="city" # Show city names on map if space permits
            )
            fig_map.update_traces(marker=dict(sizemin=4)) # Ensure minimum bubble size
            fig_map.update_layout(
                get_plot_layout(height=550, title_text=f"Average AQI Hotspots - {selected_month_str}, {year}"),
                mapbox_style="carto-darkmatter",
                margin={"r": 0, "t": 70, "l": 0, "b": 0}
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No city data matched with available coordinates for the selected period.")
    else:
        st.info("Coordinate data could not be processed into a valid format for the map.")
elif df_filtered.empty:
    st.warning("Map cannot be displayed: No air quality data for the selected filters.")
else: # city_coords_data is None or empty
    st.warning("Map cannot be displayed: City coordinates file ('lat_long.txt') not found or is invalid.")


# ------------------- 🆚 CITY-WISE COMPARISON -------------------
if len(selected_cities) > 1:
    st.markdown("## 📊 City-to-City Comparison")
    comp_data = df_filtered[df_filtered['city'].isin(selected_cities)]
    if not comp_data.empty:
        fig_comp = px.line(comp_data, x='date', y='index', color='city', labels={'index': 'AQI Index', 'date': 'Date'},
                           color_discrete_sequence=px.colors.qualitative.Set2) # Using a distinct color sequence
        fig_comp.update_layout(get_plot_layout(height=450, title_text=f"AQI Trend Comparison - {selected_month_str}, {year}"),
                               hovermode="x unified") # Improved hover experience
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("No data available for the selected cities in the given time period to make a comparison.")


# ------------------- 🔎 DETAILED CITY ANALYSIS (RE-ARCHITECTED) -------------------
if selected_cities: # Only show this section if cities are selected
    st.markdown("## 🔎 Detailed City Analysis")
    # Use a selectbox for single city deep-dive, defaults to the first selected city
    city_to_view = st.selectbox(
        "Select a city for detailed analysis:",
        selected_cities,
        index=0, # Default to the first city in the selected_cities list
        key="city_deep_dive_selector",
        help="Choose one of your selected cities to analyze in detail below."
    )

    if city_to_view: # Proceed if a city is chosen from the selectbox
        city_data_selected = df_filtered[df_filtered['city'] == city_to_view].copy()

        if city_data_selected.empty:
            st.warning(f"No data available for **{city_to_view}** for the selected period. Please adjust the filters.")
        else:
            # Create tabs for different views within the city analysis
            tab_titles = ["**📈 Trends & Calendar**", "**📊 Distributions**", "**🗓️ Heatmap**"]
            tab1, tab2, tab3 = st.tabs(tab_titles)

            with tab1: # Trends & Calendar
                st.markdown("##### AQI Trend & 7-Day Rolling Average")
                city_data_selected['rolling_avg_7day'] = city_data_selected['index'].rolling(window=7, center=True, min_periods=1).mean()
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=city_data_selected['date'], y=city_data_selected['index'], mode='lines', name='Daily AQI',
                                               line=dict(color=SUBTLE_TEXT_COLOR, width=1.5)))
                fig_trend.add_trace(go.Scatter(x=city_data_selected['date'], y=city_data_selected['rolling_avg_7day'], mode='lines', name='7-Day Rolling Avg',
                                               line=dict(color=PRIMARY_ACCENT_COLOR, width=2.5)))
                fig_trend.update_layout(get_plot_layout(height=350), hovermode="x unified") # Reduced height
                st.plotly_chart(fig_trend, use_container_width=True)

                st.markdown("##### Daily AQI Calendar")
                # Full year data for the calendar view of the selected city and year
                city_year_data_calendar = df[(df['city'] == city_to_view) & (df['date'].dt.year == year)].copy()
                if not city_year_data_calendar.empty:
                    # Simplified Calendar Heatmap (can be expanded if needed)
                    # This requires more complex date manipulation for a full GitHub-style calendar
                    # For now, a simpler heatmap representation:
                    city_year_data_calendar['day_of_week_num'] = city_year_data_calendar['date'].dt.dayofweek
                    city_year_data_calendar['week_of_year_num'] = city_year_data_calendar['date'].dt.isocalendar().week.astype(int) # Ensure int for proper sorting

                    fig_cal = go.Figure(data=go.Heatmap(
                        x=city_year_data_calendar['week_of_year_num'],
                        y=city_year_data_calendar['day_of_week_num'],
                        z=city_year_data_calendar['index'],
                        colorscale='Plasma', # A perceptually uniform colorscale
                        showscale=False, # Typically legend is separate or context implies scale
                        hovertemplate="<b>Date:</b> %{customdata|%Y-%m-%d}<br><b>AQI:</b> %{z:.2f}<extra></extra>",
                        customdata=city_year_data_calendar['date']
                    ))
                    fig_cal.update_layout(
                        get_plot_layout(height=220, title_text=None), # No title for cleaner integration
                        yaxis=dict(autorange='reversed', tickmode='array', tickvals=list(range(7)), ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], title=None),
                        xaxis=dict(title='Week of Year', tickmode='auto', nticks=10) # Show fewer week ticks
                    )
                    st.plotly_chart(fig_cal, use_container_width=True)
                else:
                    st.info(f"No annual data available for {city_to_view} in {year} to display calendar.")


            with tab2: # Distributions
                col_bar, col_pie = st.columns(2)
                with col_bar:
                    st.markdown("##### AQI Category (Days)")
                    category_counts = city_data_selected['level'].value_counts().reindex(AQI_CATEGORY_COLORS.keys(), fill_value=0)
                    fig_bar = go.Figure(go.Bar(x=category_counts.index, y=category_counts.values,
                                                marker_color=[AQI_CATEGORY_COLORS.get(k, SUBTLE_TEXT_COLOR) for k in category_counts.index]))
                    fig_bar.update_layout(get_plot_layout(height=350, yaxis_title="Number of Days"), xaxis_title=None)
                    st.plotly_chart(fig_bar, use_container_width=True)
                with col_pie:
                    st.markdown("##### AQI Category (Proportions)")
                    fig_pie = px.pie(category_counts, values=category_counts.values, names=category_counts.index,
                                     color=category_counts.index, color_discrete_map=AQI_CATEGORY_COLORS, hole=0.5) # Donut chart
                    fig_pie.update_layout(get_plot_layout(height=350, showlegend=False),
                                          annotations=[dict(text='Days', x=0.5, y=0.5, font_size=16, showarrow=False)]) # Center text
                    st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("##### Monthly AQI Distribution")
                if not city_data_selected.empty and 'date' in city_data_selected.columns:
                    month_order = list(months_map.values())
                    city_data_selected['month_name_cat'] = pd.Categorical(city_data_selected['date'].dt.strftime('%b'), categories=month_order, ordered=True)
                    fig_violin = px.violin(city_data_selected.sort_values('month_name_cat'), x='month_name_cat', y='index', box=True,
                                           color_discrete_sequence=[PRIMARY_ACCENT_COLOR]) # Single color for violin
                    fig_violin.update_layout(get_plot_layout(height=350, xaxis_title="Month", yaxis_title="AQI Index"))
                    st.plotly_chart(fig_violin, use_container_width=True)
                else:
                    st.info("Not enough data for monthly distribution.")


            with tab3: # Heatmap
                st.markdown("##### Daily AQI Heatmap")
                if not city_data_selected.empty and 'date' in city_data_selected.columns:
                    city_data_selected['day_of_month_num'] = city_data_selected['date'].dt.day
                    # Ensure month_name_cat is created if not already (e.g., if only this tab is viewed)
                    if 'month_name_cat' not in city_data_selected.columns:
                         city_data_selected['month_name_cat'] = pd.Categorical(city_data_selected['date'].dt.strftime('%b'), categories=list(months_map.values()), ordered=True)

                    heatmap_pivot_df = city_data_selected.pivot_table(index='month_name_cat', columns='day_of_month_num', values='index', observed=False)
                    heatmap_pivot_df = heatmap_pivot_df.reindex(list(months_map.values())) # Ensure correct month order

                    fig_heat = px.imshow(heatmap_pivot_df, labels=dict(x="Day of Month", y="Month", color="AQI Index"),
                                         aspect="auto", color_continuous_scale="Plasma", text_auto=".0f") # Show AQI values on cells
                    fig_heat.update_layout(get_plot_layout(height=400, xaxis_side="top"), # Move Day axis to top
                                           xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("Not enough data for daily heatmap.")


# ------------------- 🔮 AQI FORECAST (POLYNOMIAL REGRESSION) -------------------
st.markdown("## 🔮 AQI Forecast")
if selected_cities: # Allow forecast only if cities are selected
    forecast_city_selected = st.selectbox(
        "Select a city for AQI forecast:",
        selected_cities,
        key="forecast_city_selector",
        help="Forecasting uses Polynomial Regression (degree 2) based on historical data."
    )
    if forecast_city_selected:
        # Use all available data for the selected city for forecasting, not just df_filtered
        city_data_for_forecast = df[df['city'] == forecast_city_selected].copy().sort_values('date').dropna(subset=['index'])

        if len(city_data_for_forecast) > 30: # Need sufficient data points
            city_data_for_forecast['date_ordinal'] = city_data_for_forecast['date'].map(pd.Timestamp.toordinal)
            
            # Create a polynomial regression model
            degree = 2 # Degree of the polynomial
            poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            poly_model.fit(city_data_for_forecast[['date_ordinal']], city_data_for_forecast['index'])

            # Predict on historical data to get residuals for confidence interval
            historical_predictions = poly_model.predict(city_data_for_forecast[['date_ordinal']])
            residuals = city_data_for_forecast['index'] - historical_predictions
            residual_std = residuals.std() # Standard deviation of residuals

            # Generate future dates for forecasting
            last_historical_date = city_data_for_forecast['date'].max()
            future_dates_list = [last_historical_date + timedelta(days=i) for i in range(1, 31)] # Forecast 30 days
            future_ordinals_array = np.array([d.toordinal() for d in future_dates_list]).reshape(-1, 1)

            # Make future predictions
            future_aqi_predictions = poly_model.predict(future_ordinals_array)
            future_aqi_predictions = np.maximum(0, future_aqi_predictions) # AQI cannot be negative

            # Calculate confidence interval (approximate)
            # For polynomial regression, CI is more complex, this is a simplified approach
            confidence_level = 1.96 * residual_std # 95% confidence
            upper_bound = future_aqi_predictions + confidence_level
            lower_bound = np.maximum(0, future_aqi_predictions - confidence_level) # Ensure lower bound is not negative

            # Create forecast DataFrame
            forecast_plot_df = pd.DataFrame({
                'date': future_dates_list,
                'Forecast': future_aqi_predictions,
                'Upper_CI': upper_bound,
                'Lower_CI': lower_bound
            })

            # Create Plotly figure
            fig_poly_forecast = go.Figure()
            # Historical Data
            fig_poly_forecast.add_trace(go.Scatter(
                x=city_data_for_forecast['date'], y=city_data_for_forecast['index'],
                mode='lines', name='Historical AQI', line=dict(color=SUBTLE_TEXT_COLOR)
            ))
            # Forecast Line
            fig_poly_forecast.add_trace(go.Scatter(
                x=forecast_plot_df['date'], y=forecast_plot_df['Forecast'],
                mode='lines', name='Forecasted AQI', line=dict(color=PRIMARY_ACCENT_COLOR, width=2.5)
            ))
            # Confidence Interval Area
            fig_poly_forecast.add_trace(go.Scatter(
                x=forecast_plot_df['date'], y=forecast_plot_df['Upper_CI'],
                mode='lines', name='Upper 95% CI', line=dict(width=0), showlegend=False
            ))
            fig_poly_forecast.add_trace(go.Scatter(
                x=forecast_plot_df['date'], y=forecast_plot_df['Lower_CI'],
                mode='lines', name='Lower 95% CI', line=dict(width=0), fillcolor='rgba(0,169,255,0.2)', # Accent color with opacity
                fill='tonexty', showlegend=False
            ))

            fig_poly_forecast.update_layout(
                get_plot_layout(height=400, title_text=f"30-Day AQI Forecast for {forecast_city_selected} (Polynomial)"),
                legend_title_text='', hovermode="x unified"
            )
            st.plotly_chart(fig_poly_forecast, use_container_width=True)
            st.caption("💡 **Note:** This forecast uses Polynomial Regression (degree 2) and includes an approximate 95% confidence interval. It's an estimation and actual AQI can vary due to numerous unmodelled factors.")
        else:
            st.warning(f"Not enough historical data for {forecast_city_selected} to generate a robust forecast (requires more than 30 data points).")
else:
    st.info("Select at least one city in the sidebar to enable AQI forecasting.")


# ------------------- 💨 POLLUTANT ANALYSIS -------------------
st.markdown("## 💨 Prominent Pollutant Analysis")
if selected_cities: # Allow pollutant analysis only if cities are selected
    pollutant_city_selected = st.selectbox(
        "Select city for pollutant analysis:",
        selected_cities,
        key="pollutant_analysis_selector",
        help="Analyze dominant pollutants for the selected city."
    )
    if pollutant_city_selected:
        # Use two columns for side-by-side pollutant charts
        col_poll_period, col_poll_yearly = st.columns(2)

        with col_poll_period:
            st.markdown(f"##### Dominance in {selected_month_str}, {year}")
            poll_data_period_selected = df_filtered[df_filtered['city'] == pollutant_city_selected]
            if not poll_data_period_selected.empty and 'pollutant' in poll_data_period_selected.columns:
                pollutant_counts_period = poll_data_period_selected['pollutant'].value_counts()
                fig_poll_bar = px.bar(
                    pollutant_counts_period,
                    x=pollutant_counts_period.index,
                    y=pollutant_counts_period.values,
                    color=pollutant_counts_period.index,
                    color_discrete_map=POLLUTANT_COLORS,
                    labels={'index': 'Pollutant', 'y': 'Number of Days'}
                )
                fig_poll_bar.update_layout(get_plot_layout(height=350, title_text=None), showlegend=False, xaxis_title=None) # No title if inside column
                st.plotly_chart(fig_poll_bar, use_container_width=True)
            else:
                st.info(f"No pollutant data for {pollutant_city_selected} in the selected period.")

        with col_poll_yearly:
            st.markdown("##### Trend Over All Years")
            poll_data_all_years = df[df['city'] == pollutant_city_selected].copy() # Use original df for all years
            if not poll_data_all_years.empty and 'pollutant' in poll_data_all_years.columns:
                poll_data_all_years['year_label'] = poll_data_all_years['date'].dt.year
                pollutant_counts_yearly = poll_data_all_years.groupby(['year_label', 'pollutant']).size().unstack(fill_value=0)
                fig_poll_stack = px.bar(
                    pollutant_counts_yearly,
                    x=pollutant_counts_yearly.index,
                    y=pollutant_counts_yearly.columns,
                    color_discrete_map=POLLUTANT_COLORS,
                    labels={'x': 'Year', 'value': 'Number of Days', 'variable': 'Pollutant'}
                )
                fig_poll_stack.update_layout(get_plot_layout(height=350, title_text=None), barmode='stack', xaxis_title=None) # No title
                st.plotly_chart(fig_poll_stack, use_container_width=True)
            else:
                st.info(f"No yearly pollutant data available for {pollutant_city_selected}.")
else:
    st.info("Select at least one city in the sidebar to analyze pollutant data.")


# ------------------- 📥 DATA DOWNLOAD -------------------
st.markdown("## 📥 Download Data")
st.info("Download the data currently being displayed on the dashboard based on your filter selections in the sidebar.", icon="ℹ️")
@st.cache_data # Cache the conversion to CSV
def convert_df_to_csv(df_to_convert):
    """Converts a DataFrame to a CSV string for download."""
    return df_to_convert.to_csv(index=False).encode('utf-8')

if not df_filtered.empty:
    csv_data_to_download = convert_df_to_csv(df_filtered)
    st.download_button(
        label="📤 Download Filtered Data as CSV",
        data=csv_data_to_download,
        file_name=f"AuraVisionPro_filtered_data_{year}_{selected_month_str.replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True # Make button full width
    )
else:
    st.warning("No data to download based on current filters.")

# ------------------- <footer> FOOTER -------------------
# Corrected footer implementation
st.markdown(f"""
<div class="footer" style="background-color: {CARD_COLOR}; border-top: 1px solid {BORDER_COLOR}; border-radius: 12px 12px 0 0; text-align: center; padding: 2rem 1rem; margin-top: 3rem;">
    <p style="font-size: 1.1rem; color: {TEXT_COLOR}; font-weight: 600; margin-bottom: 0.5rem;">🏆 AuraVision Pro</p>
    <p style="color:{SUBTLE_TEXT_COLOR}; margin-bottom: 1rem;">A Championship-Grade AQI Dashboard</p>
    <p style="color:{SUBTLE_TEXT_COLOR}; font-size: 0.8rem;">Conceptualized by: Mr. Kapil Meena & Prof. Arkopal K. Goswami, IIT Kharagpur.</p>
    <p style="color:{SUBTLE_TEXT_COLOR}; font-size: 0.8rem;">Enhanced & Rebuilt by Google Gemini ✨</p>
    <p style="margin-top: 1rem;"><a href="https://github.com/kapil2020/india-air-quality-dashboard" target="_blank" style="color:{PRIMARY_ACCENT_COLOR}; text-decoration:none; font-weight:500;">View Source on GitHub</a></p>
</div>
""", unsafe_allow_html=True)

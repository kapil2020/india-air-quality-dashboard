import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.linear_model import LinearRegression
import os
from datetime import date, timedelta
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# --- Global Theme & Style Setup ---
# Use Plotly's dark template as a base for charts
pio.templates.default = "plotly_dark"

# Define our primary accent color (a vibrant teal for good contrast on dark)
ACCENT_COLOR = "#00BCD4" # Vibrant Teal
TEXT_COLOR_DARK_THEME = "#EAEAEA"
SUBTLE_TEXT_COLOR_DARK_THEME = "#B0B0B0"
BACKGROUND_COLOR = "#121212" # Very dark grey (almost black)
CARD_BACKGROUND_COLOR = "#1E1E1E" # Slightly lighter for cards
BORDER_COLOR = "#333333"

# Updated AQI Category Colors for Dark Theme (brighter, more distinct)
CATEGORY_COLORS_DARK = {
    'Severe': '#F44336',      # Vivid Red
    'Very Poor': '#FF7043',   # Vivid Orange-Red
    'Poor': '#FFA726',        # Vivid Orange
    'Moderate': '#FFEE58',    # Vivid Yellow
    'Satisfactory': '#9CCC65',# Vivid Light Green
    'Good': '#4CAF50',        # Vivid Green
    'Unknown': '#444444'      # Dark Grey for empty days
}

POLLUTANT_COLORS_DARK = { # Ensure these are bright enough for dark theme
    'PM2.5': '#FF6B6B', 'PM10': '#4ECDC4', 'NO2': '#45B7D1',
    'SO2': '#F9C74F', 'CO': '#F8961E', 'O3': '#90BE6D', 'Other': '#B0BEC5'
}


# ------------------- Page Config -------------------
st.set_page_config(layout="wide", page_title="AuraVision AQI Dashboard", page_icon="🌬️")

# ------------------- Custom CSS Styling (Dark Theme Focus) -------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    body {{
        font-family: 'Inter', sans-serif;
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR_DARK_THEME};
    }}

    /* Main container styling */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }}

    /* Card-like styling for sections/charts */
    .stPlotlyChart, .stDataFrame, .stAlert, .stMetric,
    .stDownloadButton > button, .stButton > button,
    div[data-testid="stExpander"], div[data-testid="stForm"] {{
        border-radius: 12px;
        border: 1px solid {BORDER_COLOR};
        background-color: {CARD_BACKGROUND_COLOR};
        padding: 1.5rem;
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
        transition: background-color 0.3s ease, color 0.3s ease;
    }}
     .stTabs [aria-selected="true"] {{
        border-bottom: 3px solid {ACCENT_COLOR};
        color: {ACCENT_COLOR};
        background-color: {BACKGROUND_COLOR};
     }}

    /* Headings */
    h1 {{
        font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR_DARK_THEME};
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        letter-spacing: -1px;
    }}
    h2 {{ /* Main section titles */
        font-family: 'Inter', sans-serif;
        color: {ACCENT_COLOR};
        border-bottom: 2px solid {ACCENT_COLOR};
        padding-bottom: 0.6rem;
        margin-top: 3rem;
        margin-bottom: 1.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    h3, h4, h5 {{ /* Sub-section titles */
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
    }}
    .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3, .stSidebar .stMarkdown p {{
        color: {TEXT_COLOR_DARK_THEME};
        text-align: left;
        border-bottom: none;
    }}
    .stSidebar .stSelectbox label, .stSidebar .stMultiselect label {{
        color: {ACCENT_COLOR} !important;
        font-weight: 600;
    }}

    /* Metric styling */
    .stMetric {{
        background-color: {BACKGROUND_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 8px;
        padding: 1rem;
    }}
    .stMetric > div:nth-child(1) {{ /* Label */
        font-size: 1rem;
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        font-weight: 500;
    }}
    .stMetric > div:nth-child(2) {{ /* Value */
        font-size: 2.2rem;
        font-weight: 700;
        color: {ACCENT_COLOR};
    }}

    /* Custom Scrollbar for Webkit browsers */
    ::-webkit-scrollbar {{ width: 8px; }}
    ::-webkit-scrollbar-track {{ background: {BACKGROUND_COLOR}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER_COLOR}; border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {ACCENT_COLOR}; }}

</style>
""", unsafe_allow_html=True)

# ------------------- Title -------------------
st.markdown("<h1 style='text-align: center; margin-bottom:0.5rem;'>🌬️ AuraVision AQI</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 1.1rem; margin-bottom: 2.5rem;'>Illuminating Air Quality Insights Across India</p>", unsafe_allow_html=True)


# ------------------- Load Data -------------------
@st.cache_data(ttl=3600)
def load_data_and_metadata():
    with st.spinner(f"🌬️ Initializing AuraVision Engine... Please wait."):
        # This function simulates loading data, including a fallback mechanism.
        # In a real scenario, this would connect to a database or a live API.
        fallback_file = "combined_air_quality.txt" # Using a sample static file
        df_loaded = None
        load_msg = ""
        last_update_time = None
        try:
            if not os.path.exists(fallback_file):
                st.error(f"FATAL: Main data file '{fallback_file}' not found.")
                return pd.DataFrame(), "Error: Main data file not found.", None
            df_loaded = pd.read_csv(fallback_file, sep="\t", parse_dates=['date'])
            load_msg = f"Displaying archive data from: **{fallback_file}**"
            last_update_time = pd.Timestamp(os.path.getmtime(fallback_file), unit='s')
        except Exception as e:
            st.error(f"FATAL: Error loading '{fallback_file}': {e}.")
            return pd.DataFrame(), f"Error loading fallback: {e}", None

        # Common processing for data cleaning
        for col, default_val in [('pollutant', np.nan), ('level', 'Unknown')]:
            if col not in df_loaded.columns: df_loaded[col] = default_val

        df_loaded['pollutant'] = df_loaded['pollutant'].astype(str).str.split(',').str[0].str.strip().replace(['nan', 'NaN', 'None', ''], np.nan)
        df_loaded['level'] = df_loaded['level'].astype(str).fillna('Unknown')
        df_loaded['pollutant'] = df_loaded['pollutant'].fillna('Other')

        return df_loaded, load_msg, last_update_time

df, load_message, data_last_updated = load_data_and_metadata()

if df.empty:
    st.error("Dashboard cannot operate without data. Please check data sources.")
    st.stop()

# ------------------- Sidebar Filters -------------------
st.sidebar.header("🔭 EXPLORATION CONTROLS")
st.sidebar.markdown("---")
st.sidebar.info("Fetching real-time data from CPCB, today's data is available after 5:45pm.")

unique_cities = sorted(df['city'].unique()) if 'city' in df.columns else []
default_city_val = ["Delhi"] if "Delhi" in unique_cities else (unique_cities[0:1] if unique_cities else [])
selected_cities = st.sidebar.multiselect("🏙️ Select Cities", unique_cities, default=default_city_val)

years = sorted(df['date'].dt.year.unique())
default_year_val = max(years) if years else None
year = st.sidebar.selectbox("🗓️ Select Year", years, index=years.index(default_year_val) if default_year_val in years else 0)

months_map_dict = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
month_options_list = ["All Months"] + list(months_map_dict.values())
selected_month_name = st.sidebar.selectbox("🌙 Select Month (Optional)", month_options_list, index=0)

month_number_filter = None
if selected_month_name != "All Months":
    month_number_filter = list(months_map_dict.keys())[list(months_map_dict.values()).index(selected_month_name)]

# --- Filter data based on global selections ---
df_period_filtered = df[df['date'].dt.year == year].copy()
if month_number_filter:
    df_period_filtered = df_period_filtered[df_period_filtered['date'].dt.month == month_number_filter]


# ------------------- 💡 NATIONAL KEY INSIGHTS -------------------
st.markdown("## 🇮🇳 NATIONAL KEY INSIGHTS")
with st.container():
    st.markdown(f"##### Key Metro Annual Average AQI ({year})")
    major_cities = ['Delhi', 'Mumbai', 'Kolkata', 'Bengaluru', 'Chennai']
    major_cities_data = df[(df['date'].dt.year == year) & (df['city'].isin(major_cities))]

    if not major_cities_data.empty:
        avg_aqi_major_cities = major_cities_data.groupby('city')['index'].mean()
        cols = st.columns(len(major_cities))
        for i, city_name in enumerate(major_cities):
            with cols[i]:
                aqi_val = avg_aqi_major_cities.get(city_name, None)
                display_val = f"{aqi_val:.2f}" if aqi_val is not None else "N/A"
                st.metric(label=city_name, value=display_val)
    else:
        st.info(f"No data available for key metro cities in {year}.")

    st.markdown(f"##### General Insights for Selected Period ({selected_month_name}, {year})")
    if not df_period_filtered.empty:
        avg_aqi_national = df_period_filtered['index'].mean()
        city_avg_aqi_stats = df_period_filtered.groupby('city')['index'].mean()
        if not city_avg_aqi_stats.empty:
            st.markdown(
                f"Across **{df_period_filtered['city'].nunique()}** observed cities, the average AQI is **{avg_aqi_national:.2f}**. "
                f"The best performing city was **{city_avg_aqi_stats.idxmin()}** ({city_avg_aqi_stats.min():.2f}), "
                f"while the worst was **{city_avg_aqi_stats.idxmax()}** ({city_avg_aqi_stats.max():.2f})."
            )
    else:
        st.info("No data available for the selected period to generate general insights.")

# ------------------- 🆚 CITY-WISE AQI COMPARISON -------------------
if len(selected_cities) > 1:
    st.markdown("## 🆚 CITY-WISE AQI COMPARISON")
    
    comp_tab1, comp_tab2 = st.tabs(["📈 Trend Comparison", "🌀 Seasonal Pattern Radar"])

    comparison_df_list = []
    for city_comp in selected_cities:
        city_ts_comp = df_period_filtered[df_period_filtered['city'] == city_comp].copy()
        if not city_ts_comp.empty:
            comparison_df_list.append(city_ts_comp)

    if comparison_df_list:
        with comp_tab1:
            st.markdown("##### AQI Trends Comparison")
            combined_comp_df = pd.concat(comparison_df_list)
            fig_cmp = px.line(
                combined_comp_df, x='date', y='index', color='city',
                labels={'index': 'AQI Index', 'date': 'Date', 'city': 'City'},
                line_shape='spline', color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_cmp.update_layout(height=500, paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR)
            st.plotly_chart(fig_cmp, use_container_width=True)
        
        with comp_tab2:
            st.markdown("##### Seasonal AQI Radar")
            radar_fig = go.Figure()
            # Use the full year's data for the radar chart for a complete seasonal picture
            df_year_filtered = df[df['date'].dt.year == year]

            for city_name in selected_cities:
                city_radar_data = df_year_filtered[df_year_filtered['city'] == city_name].copy()
                if not city_radar_data.empty:
                    city_radar_data['month'] = city_radar_data['date'].dt.month
                    monthly_avg_aqi = city_radar_data.groupby('month')['index'].mean().reindex(range(1, 13))
                    
                    radar_fig.add_trace(go.Scatterpolar(
                        r=monthly_avg_aqi.values,
                        theta=[m[:3] for m in months_map_dict.values()], # Use 3-letter month abbreviations
                        fill='toself',
                        name=city_name,
                        hovertemplate=f"<b>{city_name}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>"
                    ))
            
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, df_year_filtered['index'].max()]),
                    bgcolor=BACKGROUND_COLOR
                ),
                height=500,
                paper_bgcolor=CARD_BACKGROUND_COLOR,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.info("Not enough data or cities selected for comparison with current filters.")


# ------------------- 🏙️ CITY-SPECIFIC DEEP DIVE -------------------
if selected_cities:
    st.markdown("## 🏙️ CITY-SPECIFIC DEEP DIVE")
    for city in selected_cities:
        st.markdown(f"### {city.upper()} – {year}")
        city_data_full = df_period_filtered[df_period_filtered['city'] == city].copy()

        if city_data_full.empty:
            st.warning(f"😔 No data available for {city} for the selected period. Try different filter settings.")
            continue
            
        # Create a single container for the city's analysis
        with st.container():
            st.markdown("##### 📅 Daily AQI Calendar")
            start_date, end_date = pd.to_datetime(f'{year}-01-01'), pd.to_datetime(f'{year}-12-31')
            full_year_dates = pd.date_range(start_date, end_date)
            
            calendar_df = pd.DataFrame({'date': full_year_dates})
            calendar_df['week'] = calendar_df['date'].dt.isocalendar().week
            calendar_df['day_of_week'] = calendar_df['date'].dt.dayofweek
            calendar_df.loc[(calendar_df['date'].dt.month == 1) & (calendar_df['week'] > 50), 'week'] = 0
            calendar_df.loc[(calendar_df['date'].dt.month == 12) & (calendar_df['week'] == 1), 'week'] = 53

            # Correctly get first week of each month for labels
            month_label_df = calendar_df.copy()
            month_label_df['month'] = month_label_df['date'].dt.month
            first_weeks = month_label_df.groupby('month')['week'].min()
            month_names_map = {month_num: name[:3] for month_num, name in months_map_dict.items()}

            merged_cal_df = pd.merge(calendar_df, city_data_full[['date', 'index', 'level']], on='date', how='left')
            merged_cal_df['level'] = merged_cal_df['level'].fillna('Unknown')
            merged_cal_df['aqi_text'] = merged_cal_df['index'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')
            
            day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            fig_cal = go.Figure(data=go.Heatmap(
                x=merged_cal_df['week'], y=merged_cal_df['day_of_week'],
                z=merged_cal_df['level'].map({k: i for i, k in enumerate(CATEGORY_COLORS_DARK.keys())}),
                customdata=pd.DataFrame({'date': merged_cal_df['date'].dt.strftime('%Y-%m-%d'), 'level': merged_cal_df['level'], 'aqi': merged_cal_df['aqi_text']}),
                hovertemplate="<b>%{customdata[0]}</b><br>AQI: %{customdata[2]} (%{customdata[1]})<extra></extra>",
                colorscale=list(CATEGORY_COLORS_DARK.values()), showscale=False, xgap=2, ygap=2
            ))

            annotations = [go.layout.Annotation(
                text=month_names_map[month_num], align='center', showarrow=False,
                xref='x', yref='y', x=week_num, y=7.5, font=dict(color=SUBTLE_TEXT_COLOR_DARK_THEME, size=12)
            ) for month_num, week_num in first_weeks.items()]
            
            fig_cal.update_layout(
                yaxis=dict(tickmode='array', tickvals=list(range(7)), ticktext=day_labels, showgrid=False, zeroline=False),
                xaxis=dict(showgrid=False, zeroline=False, tickmode='array', ticktext=[], tickvals=[]),
                height=350, margin=dict(t=50, b=20, l=40, r=40),
                plot_bgcolor=CARD_BACKGROUND_COLOR, paper_bgcolor=CARD_BACKGROUND_COLOR,
                annotations=annotations
            )
            st.plotly_chart(fig_cal, use_container_width=True)

            # Other charts for the city
            city_tabs = st.tabs(["📈 AQI Distribution", "🔥 Monthly Heatmap"])
            with city_tabs[0]:
                category_counts_df = city_data_full['level'].value_counts().reindex(CATEGORY_COLORS_DARK.keys(), fill_value=0).reset_index()
                category_counts_df.columns = ['AQI Category', 'Number of Days']
                fig_dist_bar = px.bar(
                    category_counts_df, x='AQI Category', y='Number of Days', color='AQI Category',
                    color_discrete_map=CATEGORY_COLORS_DARK, text_auto=True
                )
                fig_dist_bar.update_layout(height=400, xaxis_title=None, yaxis_title="Number of Days", plot_bgcolor=BACKGROUND_COLOR)
                st.plotly_chart(fig_dist_bar, use_container_width=True)

            with city_tabs[1]:
                city_data_full['month_name'] = pd.Categorical(city_data_full['date'].dt.month_name(), categories=list(months_map_dict.values()), ordered=True)
                heatmap_pivot = city_data_full.pivot_table(index='month_name', columns=city_data_full['date'].dt.day, values='index', observed=True)
                fig_heat_detail = px.imshow(
                    heatmap_pivot, labels=dict(x="Day of Month", y="Month", color="AQI"),
                    aspect="auto", color_continuous_scale="Inferno", text_auto=".0f"
                )
                fig_heat_detail.update_layout(height=500, xaxis_side="top", plot_bgcolor=BACKGROUND_COLOR)
                st.plotly_chart(fig_heat_detail, use_container_width=True)

# ------------------- 💨 POLLUTANT ANALYSIS -------------------
st.markdown("## 💨 PROMINENT POLLUTANT ANALYSIS")
with st.container():
    st.markdown(f"#### ⛽ Dominant Pollutants for Selected Period ({selected_month_name}, {year})")
    city_pollutant_B = st.selectbox(
        "Select City for Filtered Pollutant View:", unique_cities,
        key="pollutant_B_city_dark", index=unique_cities.index(default_city_val[0]) if default_city_val and default_city_val[0] in unique_cities else 0
    )
    pollutant_data_B = df_period_filtered[df_period_filtered['city'] == city_pollutant_B].copy()
    if not pollutant_data_B.empty and 'pollutant' in pollutant_data_B.columns and pollutant_data_B['pollutant'].notna().any():
        grouped_poll_B = pollutant_data_B.groupby('pollutant').size().reset_index(name='count')
        total_days_B = grouped_poll_B['count'].sum()
        if total_days_B > 0:
            grouped_poll_B['percentage'] = (grouped_poll_B['count'] / total_days_B * 100).round(1)
            fig_poll_B = px.bar(
                grouped_poll_B, x='pollutant', y='percentage', color='pollutant',
                labels={'percentage': 'Percentage of Days (%)', 'pollutant': 'Pollutant'},
                color_discrete_map=POLLUTANT_COLORS_DARK, text_auto=True
            )
            fig_poll_B.update_layout(yaxis_ticksuffix="%", height=450, plot_bgcolor=BACKGROUND_COLOR)
            st.plotly_chart(fig_poll_B, use_container_width=True)
    else:
        st.warning(f"No pollutant data for {city_pollutant_B} for the selected period.")

# ------------------- 🗺️ INTERACTIVE AIR QUALITY MAP -------------------
st.markdown("## 🗺️ INTERACTIVE AIR QUALITY MAP")
with st.container():
    city_coords_data = {}
    try:
        # In a real app, load this from a file or config
        with open("lat_long.txt", "r") as f:
             city_coords_data = eval(f.read())
    except Exception as e:
        st.error(f"Map Error: Could not load city coordinates. {e}")

    if city_coords_data and not df_period_filtered.empty:
        latlong_map_df = pd.DataFrame([{'city': k, 'lat': v[0], 'lon': v[1]} for k, v in city_coords_data.items()])
        map_grouped_data = df_period_filtered.groupby('city').agg(
            avg_aqi=('index', 'mean'),
            dominant_pollutant=('pollutant', lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A')
        ).reset_index()
        map_merged_df = pd.merge(map_grouped_data, latlong_map_df, on='city', how='inner')

        def classify_aqi_map(val):
            if pd.isna(val): return "Unknown"
            if val <= 50: return "Good"
            if val <= 100: return "Satisfactory"
            if val <= 200: return "Moderate"
            if val <= 300: return "Poor"
            if val <= 400: return "Very Poor"
            return "Severe"
        map_merged_df["AQI Category"] = map_merged_df["avg_aqi"].apply(classify_aqi_map)
        
        if not map_merged_df.empty:
            fig_map_final = px.scatter_mapbox(
                map_merged_df, lat="lat", lon="lon",
                # --- CORRECTED: Reduced max size of bubbles ---
                size="avg_aqi", size_max=22,
                color="AQI Category", color_discrete_map=CATEGORY_COLORS_DARK,
                hover_name="city", text="city",
                hover_data={"avg_aqi": ":.2f", "dominant_pollutant": True, "city": False},
                zoom=4.5, center={"lat": 22.8, "lon": 82.5}, height=700
            )
            fig_map_final.update_layout(
                mapbox_style="carto-darkmatter",
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
                paper_bgcolor=CARD_BACKGROUND_COLOR,
                legend=dict(font_color=TEXT_COLOR_DARK_THEME, orientation="h", yanchor="bottom", y=-0.1)
            )
            # --- CORRECTED: Adjusted min size for better dynamic range ---
            fig_map_final.update_traces(marker=dict(sizemin=4, opacity=0.7))
            st.plotly_chart(fig_map_final, use_container_width=True)
        else: st.info("No cities match the map filter for the selected period.")
    else:
        st.warning("Map cannot be displayed due to missing data for the selected period.")

# ------------------- Footer -------------------
st.markdown(f"""
<div style="text-align: center; margin-top: 4rem; padding: 1.5rem; background-color: {CARD_BACKGROUND_COLOR}; border-radius: 12px; border: 1px solid {BORDER_COLOR};">
    <p style="margin:0.3em; color: {TEXT_COLOR_DARK_THEME}; font-size:0.9rem;">🌬️ AuraVision AQI Dashboard</p>
    <p style="margin:0.5em 0; color: {TEXT_COLOR_DARK_THEME}; font-size:0.85rem;">Conceptualized by: Mr. Kapil Meena & Prof. Arkopal K. Goswami, IIT Kharagpur.</p>
    <p style="margin-top:0.8em;"><a href="https://github.com/kapil2020/india-air-quality-dashboard" target="_blank" style="color:{ACCENT_COLOR}; text-decoration:none; font-weight:600;">🔗 View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)

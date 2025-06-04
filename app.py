import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.linear_model import LinearRegression # Note: This import is present but not used.
import os
from datetime import date, timedelta
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
# No need for separate json or ast imports if using exec for the specific file format

# --- Global Theme & Style Setup ---
# Use Plotly's dark template as a base for charts
pio.templates.default = "plotly_dark"

# Define our primary accent color (a vibrant teal for good contrast on dark)
ACCENT_COLOR = "#00BCD4" # Vibrant Teal
TEXT_COLOR_DARK_THEME = "#EAEAEA"
SUBTLE_TEXT_COLOR_DARK_THEME = "#B0B0B0"
BACKGROUND_COLOR = "#0F0F0F" # Even darker, almost black for depth
CARD_BACKGROUND_COLOR = "#1A1A1A" # Slightly lighter for cards
BORDER_COLOR = "#2D2D2D" # Softer border color

# Updated AQI Category Colors for Dark Theme (brighter, more distinct)
CATEGORY_COLORS_DARK = {
    'Severe': '#D32F2F',      # Vivid Red (slightly toned down from F44336)
    'Very Poor': '#F57C00',   # Vivid Orange-Red (from FF7043)
    'Poor': '#FFA000',        # Vivid Orange (from FFA726)
    'Moderate': '#FBC02D',    # Vivid Yellow (from FFEE58)
    'Satisfactory': '#7CB342',# Vivid Light Green (from 9CCC65)
    'Good': '#388E3C',        # Vivid Green (from 4CAF50)
    'Unknown': '#424242'      # Dark Grey for empty days
}

POLLUTANT_COLORS_DARK = { # Ensure these are bright enough for dark theme
    'PM2.5': '#E57373', 'PM10': '#4DB6AC', 'NO2': '#4FC3F7',
    'SO2': '#FFD54F', 'CO': '#FFB74D', 'O3': '#AED581', 'Other': '#90A4AE' # Slightly desaturated
}
GLOBAL_FONT_FAMILY = "Inter, sans-serif"

# ------------------- Page Config -------------------
st.set_page_config(layout="wide", page_title="AuraVision AQI Dashboard", page_icon="🌬️")

# ------------------- Custom CSS Styling (Dark Theme Focus) -------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    body {{
        font-family: {GLOBAL_FONT_FAMILY};
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR_DARK_THEME};
        line-height: 1.6;
    }}

    /* Main container styling */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 3rem; /* Increased padding */
        padding-right: 3rem; /* Increased padding */
    }}

    /* Card-like styling for sections/charts - subtle enhancements */
    .stPlotlyChart, .stDataFrame, .stAlert,
    div[data-testid="stExpander"], div[data-testid="stForm"] {{
        border-radius: 16px;
        border: 1px solid {BORDER_COLOR};
        background-color: {CARD_BACKGROUND_COLOR};
        padding: 2rem; /* Increased padding for cards */
        margin-bottom: 2.5rem; /* Increased margin */
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3); /* Refined shadow */
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }}
    .stPlotlyChart:hover, div[data-testid="stExpander"]:hover {{
         transform: translateY(-2px);
         box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }}

    /* Button Styling */
    .stButton > button, .stDownloadButton > button {{
        border-radius: 10px !important;
        border: 1.5px solid {ACCENT_COLOR} !important;
        background-color: transparent !important;
        color: {ACCENT_COLOR} !important;
        padding: 0.7rem 1.5rem !important; /* Larger buttons */
        font-weight: 600 !important;
        font-family: {GLOBAL_FONT_FAMILY} !important;
        transition: background-color 0.2s ease, color 0.2s ease, transform 0.2s ease !important;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: {ACCENT_COLOR} !important;
        color: {BACKGROUND_COLOR} !important;
        transform: scale(1.03) !important;
    }}
    .stButton > button:active, .stDownloadButton > button:active {{
        background-color: {ACCENT_COLOR} !important;
        filter: brightness(0.9);
        transform: scale(0.98) !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
         box-shadow: none;
         border-bottom: 2px solid {BORDER_COLOR};
         padding-bottom: 0;
         background-color: transparent;
         margin-bottom: 1.5rem; /* Space below tab headers */
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 1rem 1.8rem; /* Larger tabs */
        font-weight: 600;
        font-size: 1rem;
        font-family: {GLOBAL_FONT_FAMILY};
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        background-color: transparent;
        border-radius: 10px 10px 0 0;
        margin-right: 0.5rem;
        transition: background-color 0.3s ease, color 0.3s ease, border-bottom-color 0.3s ease;
        border-bottom: 3px solid transparent;
    }}
     .stTabs [aria-selected="true"] {{
        border-bottom: 3px solid {ACCENT_COLOR};
        color: {ACCENT_COLOR};
        background-color: {BACKGROUND_COLOR};
     }}

    /* Headings */
    h1 {{
        font-family: {GLOBAL_FONT_FAMILY};
        color: {TEXT_COLOR_DARK_THEME};
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        font-size: 3rem; /* Prominent H1 */
    }}
    h2 {{ /* Main section titles */
        font-family: {GLOBAL_FONT_FAMILY};
        color: {ACCENT_COLOR};
        border-bottom: 3px solid {ACCENT_COLOR};
        padding-bottom: 0.8rem;
        margin-top: 4rem; /* More separation */
        margin-bottom: 2.5rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-size: 1.6rem;
    }}
    h3 {{ /* Sub-section titles for cities, etc. */
        font-family: {GLOBAL_FONT_FAMILY};
        color: {TEXT_COLOR_DARK_THEME};
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        font-weight: 600;
        font-size: 1.35rem;
    }}
    h4, h5 {{ /* Plot titles, etc. */
        font-family: {GLOBAL_FONT_FAMILY};
        color: {TEXT_COLOR_DARK_THEME};
        margin-top: 0.5rem; /* Usually part of a card, so less top margin */
        margin-bottom: 1.2rem;
        font-weight: 600; /* Bolder for plot titles */
        font-size: 1.15rem;
    }}

    /* Sidebar styling */
    .stSidebar {{
        background-color: {CARD_BACKGROUND_COLOR};
        border-right: 1px solid {BORDER_COLOR};
        padding: 2.5rem; /* Generous padding */
    }}
    .stSidebar .stMarkdown h2 {{ /* Sidebar Header "EXPLORATION CONTROLS" */
        color: {ACCENT_COLOR} !important;
        text-align: left;
        border-bottom: none !important;
        font-size: 1.4rem !important;
        font-family: {GLOBAL_FONT_FAMILY} !important;
        font-weight: 700 !important;
        margin-top: 0 !important;
        margin-bottom: 2rem !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }}
     .stSidebar .stMarkdown h3, .stSidebar .stMarkdown p {{
        color: {TEXT_COLOR_DARK_THEME};
        text-align: left;
        border-bottom: none;
        font-family: {GLOBAL_FONT_FAMILY};
    }}
    .stSidebar .stSelectbox label, .stSidebar .stMultiselect label, .stSidebar .stDateInput label {{
        color: {ACCENT_COLOR} !important;
        font-weight: 600;
        font-size: 1rem; /* Clearer labels */
        font-family: {GLOBAL_FONT_FAMILY} !important;
        margin-bottom: 0.5rem !important;
    }}
    /* Styling for Streamlit input widgets for better dark theme integration */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{ /* Targetting the actual input background */
        background-color: {BACKGROUND_COLOR} !important;
        border-color: {BORDER_COLOR} !important;
        border-radius: 8px !important;
        font-family: {GLOBAL_FONT_FAMILY} !important;
    }}
     div[data-baseweb="select"] input, .stTextInput input {{ /* Input text color */
        color: {TEXT_COLOR_DARK_THEME} !important;
        font-family: {GLOBAL_FONT_FAMILY} !important;
    }}
    div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover {{
        border-color: {ACCENT_COLOR} !important;
    }}


    /* Metric styling - making it a standout element */
    .stMetric {{
        background-color: {CARD_BACKGROUND_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-left: 6px solid {ACCENT_COLOR}; /* Prominent accent border */
        border-radius: 12px;
        padding: 1.5rem 1.8rem; /* Adjusted padding */
        margin-bottom: 1rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.2);
    }}
    .stMetric > div:nth-child(1) {{ /* Label */
        font-size: 1rem;
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        font-weight: 500;
        margin-bottom: 0.4rem;
        font-family: {GLOBAL_FONT_FAMILY};
    }}
    .stMetric > div:nth-child(2) {{ /* Value */
        font-size: 2.8rem;
        font-weight: 700;
        color: {ACCENT_COLOR};
        line-height: 1.2;
        font-family: {GLOBAL_FONT_FAMILY};
    }}
    .stMetric > div:nth-child(3) {{ /* Delta - if present */
        font-size: 0.95rem;
        font-weight: 500;
        font-family: {GLOBAL_FONT_FAMILY};
    }}

    /* Custom Scrollbar for Webkit browsers */
    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-track {{ background: {BACKGROUND_COLOR}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER_COLOR}; border-radius: 5px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {ACCENT_COLOR}; }}

    /* Footer styling */
    .footer {{
        text-align: center;
        margin-top: 5rem; /* More space before footer */
        padding: 2.5rem;
        background-color: {CARD_BACKGROUND_COLOR};
        border-radius: 16px;
        border-top: 1px solid {BORDER_COLOR}; /* Top border for footer */
        box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.1); /* Shadow for footer */
    }}
    .footer p {{
        margin: 0.5em;
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        font-size: 0.95rem;
        font-family: {GLOBAL_FONT_FAMILY};
    }}
    .footer a {{
        color: {ACCENT_COLOR};
        text-decoration: none;
        font-weight: 600;
    }}
    .footer a:hover {{
        text-decoration: underline;
    }}
</style>
""", unsafe_allow_html=True)

# --- Plotly Universal Layout Options ---
def get_custom_plotly_layout_args():
    return dict(
        font_family=GLOBAL_FONT_FAMILY.split(',')[0].strip(), # Get "Inter"
        font_color=TEXT_COLOR_DARK_THEME,
        paper_bgcolor=CARD_BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR, # Plot area slightly darker than card
        legend=dict(
            font=dict(color=SUBTLE_TEXT_COLOR_DARK_THEME, family=GLOBAL_FONT_FAMILY.split(',')[0].strip()),
            bgcolor=CARD_BACKGROUND_COLOR, # Legend bg
            bordercolor=BORDER_COLOR,
            borderwidth=1
        ),
        title_font_family=GLOBAL_FONT_FAMILY.split(',')[0].strip(),
        title_font_color=TEXT_COLOR_DARK_THEME,
        xaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, zerolinecolor=BORDER_COLOR, tickfont=dict(family=GLOBAL_FONT_FAMILY.split(',')[0].strip(), color=SUBTLE_TEXT_COLOR_DARK_THEME)),
        yaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, zerolinecolor=BORDER_COLOR, tickfont=dict(family=GLOBAL_FONT_FAMILY.split(',')[0].strip(), color=SUBTLE_TEXT_COLOR_DARK_THEME)),
    )

# ------------------- Title -------------------
st.markdown("<h1 style='text-align: center; margin-bottom:0.2rem;'>🌬️ AuraVision AQI</h1>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 1.2rem; font-weight: 400; letter-spacing: 0.5px;">
        Illuminating Air Quality Insights Across India
    </p>
</div>
""", unsafe_allow_html=True)


# ------------------- Load Data -------------------
@st.cache_data(ttl=3600)
def load_data_and_metadata():
    with st.spinner(f"🌬️ Initializing AuraVision Engine... Please wait."):
        fallback_file = "combined_air_quality.txt"
        df_loaded = None
        load_msg = ""
        last_update_time = None
        try:
            if not os.path.exists(fallback_file):
                return pd.DataFrame(), f"Error: Main data file '{fallback_file}' not found. Dashboard may not function correctly.", None
            df_loaded = pd.read_csv(fallback_file, sep="\t", parse_dates=['date'])
            load_msg = f"Displaying archive data from **{fallback_file}**"
            try:
                last_update_time = pd.Timestamp(os.path.getmtime(fallback_file), unit='s').tz_localize('UTC').tz_convert('Asia/Kolkata')
            except Exception: # Fallback if tz_convert fails (e.g. on some OS)
                last_update_time = pd.Timestamp(os.path.getmtime(fallback_file), unit='s')

        except Exception as e:
            return pd.DataFrame(), f"FATAL: Error loading '{fallback_file}': {e}. Dashboard cannot operate.", None

        if df_loaded is not None:
            for col, default_val in [('pollutant', np.nan), ('level', 'Unknown')]:
                if col not in df_loaded.columns: df_loaded[col] = default_val

            df_loaded['pollutant'] = df_loaded['pollutant'].astype(str).str.split(',').str[0].str.strip().replace(['nan', 'NaN', 'None', ''], np.nan)
            df_loaded['level'] = df_loaded['level'].astype(str).fillna('Unknown')
            df_loaded['pollutant'] = df_loaded['pollutant'].fillna('Other')
        else: # Should not happen if logic above is correct, but as a safeguard
            df_loaded = pd.DataFrame()
            load_msg = "Error: Dataframe is None after loading attempt."


        return df_loaded, load_msg, last_update_time

df, load_message, data_last_updated = load_data_and_metadata()

if df.empty:
    st.error(f"Dashboard cannot operate without data. {load_message}")
    st.stop()

# Display data update status below subtitle
if data_last_updated:
    st.markdown(f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 0.95rem; margin-bottom: 2.5rem;'>Archive data last updated: {data_last_updated.strftime('%B %d, %Y, %H:%M %Z') if hasattr(data_last_updated, 'tzinfo') and data_last_updated.tzinfo is not None else data_last_updated.strftime('%B %d, %Y, %H:%M UTC')}</p>", unsafe_allow_html=True)
else:
    st.markdown(f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 0.95rem; margin-bottom: 2.5rem;'>{load_message}</p>", unsafe_allow_html=True)


# ------------------- Sidebar Filters -------------------
st.sidebar.markdown("## 🔭 Controls") # Simplified sidebar title
st.sidebar.markdown("---", unsafe_allow_html=True) # Thematic break
st.sidebar.info("Data presented is based on available archives. For latest CPCB data, refer to official sources.")

unique_cities = sorted(df['city'].unique()) if 'city' in df.columns else []
default_city_val = ["Delhi"] if "Delhi" in unique_cities else (unique_cities[0:1] if unique_cities else [])
selected_cities = st.sidebar.multiselect("🏙️ Select Cities", unique_cities, default=default_city_val)

years = sorted(df['date'].dt.year.unique(), reverse=True) # Show recent years first
default_year_val = years[0] if years else None
year_selection_disabled = not bool(years)

year = st.sidebar.selectbox(
    "🗓️ Select Year",
    years,
    index=0 if default_year_val else 0, # Default to first item (most recent)
    disabled=year_selection_disabled
)


months_map_dict = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
month_options_list = ["All Months"] + list(months_map_dict.values())
selected_month_name = st.sidebar.selectbox("🌙 Select Month (Optional)", month_options_list, index=0, disabled=year_selection_disabled)

month_number_filter = None
if selected_month_name != "All Months":
    month_number_filter = list(months_map_dict.keys())[list(months_map_dict.values()).index(selected_month_name)]

# --- Filter data based on global selections ---
if year:
    df_period_filtered = df[df['date'].dt.year == year].copy()
    if month_number_filter:
        df_period_filtered = df_period_filtered[df_period_filtered['date'].dt.month == month_number_filter]
else:
    df_period_filtered = pd.DataFrame()


# ------------------- 💡 NATIONAL KEY INSIGHTS -------------------
st.markdown("## 🇮🇳 NATIONAL KEY INSIGHTS")
with st.container():
    custom_layout = get_custom_plotly_layout_args()
    if year:
        st.markdown(f"##### Key Metro Annual Average AQI ({year})")
        major_cities = ['Delhi', 'Mumbai', 'Kolkata', 'Bengaluru', 'Chennai', 'Hyderabad', 'Pune'] # Added more
        major_cities_data_for_year = df[df['date'].dt.year == year]
        major_cities_data = major_cities_data_for_year[major_cities_data_for_year['city'].isin(major_cities)]

        if not major_cities_data.empty:
            avg_aqi_major_cities = major_cities_data.groupby('city')['index'].mean()
            present_major_cities = [city for city in major_cities if city in avg_aqi_major_cities.index]
            if present_major_cities:
                # Dynamically adjust columns based on number of cities with data
                cols = st.columns(len(present_major_cities))
                for i, city_name in enumerate(present_major_cities):
                    with cols[i]:
                        aqi_val = avg_aqi_major_cities.get(city_name, None)
                        display_val = f"{aqi_val:.1f}" if aqi_val is not None else "N/A" # .1f for cleaner look
                        st.metric(label=city_name, value=display_val)
            else:
                st.info(f"No data available for the selected key metro cities in {year}.")
        else:
            st.info(f"No data available for key metro cities in {year}.")

        st.markdown(f"##### General Insights for Selected Period ({selected_month_name}, {year})")
        if not df_period_filtered.empty:
            avg_aqi_national = df_period_filtered['index'].mean()
            city_avg_aqi_stats = df_period_filtered.groupby('city')['index'].mean()
            if not city_avg_aqi_stats.empty:
                num_cities_observed = df_period_filtered['city'].nunique()
                best_city_name = city_avg_aqi_stats.idxmin()
                best_city_aqi = city_avg_aqi_stats.min()
                worst_city_name = city_avg_aqi_stats.idxmax()
                worst_city_aqi = city_avg_aqi_stats.max()

                insight_html = f"""
                <div style="font-size: 1.05rem; line-height: 1.7; color: {TEXT_COLOR_DARK_THEME};">
                    Across <b>{num_cities_observed}</b> observed cities, the average AQI is <b style="color:{ACCENT_COLOR};">{avg_aqi_national:.2f}</b>.
                    The city with the best average air quality was <b style="color:{CATEGORY_COLORS_DARK['Good']};">{best_city_name}</b> ({best_city_aqi:.2f}),
                    while the most challenged was <b style="color:{CATEGORY_COLORS_DARK['Severe']};">{worst_city_name}</b> ({worst_city_aqi:.2f}).
                </div>
                """
                st.markdown(insight_html, unsafe_allow_html=True)
        else:
            st.info("No data available for the selected period to generate general insights.")
    else:
        st.warning("Please select a year to view national insights.")

# ------------------- 🆚 CITY-WISE AQI COMPARISON -------------------
if len(selected_cities) > 1 and not df_period_filtered.empty:
    st.markdown("## 🆚 CITY-WISE AQI COMPARISON")
    custom_layout_comp = get_custom_plotly_layout_args()

    comp_tab1, comp_tab2 = st.tabs(["📈 AQI Trend Comparison", "🌀 Seasonal AQI Radar"])

    comparison_df_list = []
    for city_comp in selected_cities:
        city_ts_comp = df_period_filtered[df_period_filtered['city'] == city_comp].copy()
        if not city_ts_comp.empty:
            comparison_df_list.append(city_ts_comp)

    if comparison_df_list:
        with comp_tab1:
            st.markdown("##### AQI Trends Over Selected Period")
            combined_comp_df = pd.concat(comparison_df_list)
            fig_cmp = px.line(
                combined_comp_df, x='date', y='index', color='city',
                labels={'index': 'AQI Index', 'date': 'Date', 'city': 'City'},
                line_shape='spline', color_discrete_sequence=px.colors.qualitative.Pastel1 # Softer colors for lines
            )
            fig_cmp.update_layout(**custom_layout_comp, height=500)
            fig_cmp.update_traces(line=dict(width=2.5))
            st.plotly_chart(fig_cmp, use_container_width=True)
        
        with comp_tab2:
            st.markdown("##### Annual Seasonal AQI Patterns")
            radar_fig = go.Figure()
            df_year_filtered_for_radar = df[df['date'].dt.year == year]

            cities_for_radar = [city_name for city_name in selected_cities if not df_year_filtered_for_radar[df_year_filtered_for_radar['city'] == city_name].empty]

            if cities_for_radar:
                max_aqi_for_radar = 0
                radar_colors = px.colors.qualitative.Set2 # Distinct colors for radar
                for i, city_name in enumerate(cities_for_radar):
                    city_radar_data = df_year_filtered_for_radar[df_year_filtered_for_radar['city'] == city_name].copy()
                    city_radar_data['month'] = city_radar_data['date'].dt.month
                    monthly_avg_aqi = city_radar_data.groupby('month')['index'].mean().reindex(range(1, 13))
                    
                    current_max = monthly_avg_aqi.max()
                    if pd.notna(current_max) and current_max > max_aqi_for_radar:
                        max_aqi_for_radar = current_max

                    radar_fig.add_trace(go.Scatterpolar(
                        r=monthly_avg_aqi.values,
                        theta=[m[:3].upper() for m in months_map_dict.values()], 
                        fill='toself',
                        name=city_name,
                        hovertemplate=f"<b>{city_name}</b><br>%{{theta}}: %{{r:.1f}}<extra></extra>",
                        line_color=radar_colors[i % len(radar_colors)],
                        opacity=0.7
                    ))
                
                radar_fig.update_layout(**custom_layout_comp,
                    height=550, # Slightly taller
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, max_aqi_for_radar * 1.1 if max_aqi_for_radar > 0 else 50], gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR),
                        angularaxis=dict(linecolor=BORDER_COLOR, gridcolor=BORDER_COLOR),
                        bgcolor=BACKGROUND_COLOR # Ensure radar plot area bg matches
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5) # Adjusted legend position
                )
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.info("Not enough data for selected cities in the chosen year to display seasonal radar patterns.")
    elif len(selected_cities) > 1:
        st.info("Not enough data for the selected cities and period to make a comparison. Try different filter settings.")

# ------------------- 🏙️ CITY-SPECIFIC DEEP DIVE -------------------
if selected_cities and (not df_period_filtered.empty or year): # Allow if year is selected even if period_filtered empty (for full year calendar)
    st.markdown("## 🏙️ CITY-SPECIFIC DEEP DIVE")
    custom_layout_city = get_custom_plotly_layout_args()

    for city in selected_cities:
        # For calendar, use full year if "All Months" is selected. Otherwise, use df_period_filtered for month-specific data.
        city_data_for_section = df_period_filtered[df_period_filtered['city'] == city].copy() if month_number_filter else df[df['city'] == city][df['date'].dt.year == year].copy()
        
        if city_data_for_section.empty:
            st.markdown(f"### {city.upper()} – {selected_month_name if month_number_filter else 'Full Year'}, {year}")
            st.warning(f"😔 No data available for {city} for the selected period. Try different filter settings.")
            continue
        
        st.markdown(f"### {city.upper()} – {selected_month_name if month_number_filter else 'Full Year'}, {year}")
            
        with st.container(): # This will get the card styling from CSS
            st.markdown("##### 📅 Daily AQI Calendar View")
            
            start_date_cal, end_date_cal = (pd.to_datetime(f'{year}-{month_number_filter:02d}-01'), (pd.to_datetime(f'{year}-{month_number_filter:02d}-01') + pd.offsets.MonthEnd(0))) if month_number_filter else (pd.to_datetime(f'{year}-01-01'), pd.to_datetime(f'{year}-12-31'))
            full_period_dates_cal = pd.date_range(start_date_cal, end_date_cal)
            
            calendar_df = pd.DataFrame({'date': full_period_dates_cal})
            calendar_df['week'] = calendar_df['date'].dt.isocalendar().week
            calendar_df['day_of_week'] = calendar_df['date'].dt.dayofweek # Monday=0, Sunday=6
            
            if not month_number_filter: # Adjust week for Jan/Dec for full year view
                calendar_df.loc[(calendar_df['date'].dt.month == 1) & (calendar_df['week'] > 50), 'week'] = 0
                max_week_num = calendar_df['week'].max()
                if max_week_num < 52: max_week_num = 52 # Ensure at least 52 weeks for year view
                calendar_df.loc[(calendar_df['date'].dt.month == 12) & (calendar_df['week'] == 1), 'week'] = max_week_num + 1


            month_label_df = calendar_df.copy()
            month_label_df['month_num'] = month_label_df['date'].dt.month
            first_weeks = month_label_df.groupby('month_num')['week'].min()
            month_names_map_cal = {month_num: name[:3].upper() for month_num, name in months_map_dict.items()}

            merged_cal_df = pd.merge(calendar_df, city_data_for_section[['date', 'index', 'level']], on='date', how='left')
            merged_cal_df['level'] = merged_cal_df['level'].fillna('Unknown')
            merged_cal_df['aqi_text'] = merged_cal_df['index'].apply(lambda x: f'{x:.0f}' if pd.notna(x) else 'N/A')
            
            day_labels = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
            
            # Ensure colorscale aligns with CATEGORY_COLORS_DARK keys by sorting
            sorted_levels = list(CATEGORY_COLORS_DARK.keys())
            colorscale_cal = [CATEGORY_COLORS_DARK[level] for level in sorted_levels]
            level_to_num = {level: i for i, level in enumerate(sorted_levels)}

            fig_cal = go.Figure(data=go.Heatmap(
                x=merged_cal_df['week'], y=merged_cal_df['day_of_week'],
                z=merged_cal_df['level'].map(level_to_num),
                customdata=pd.DataFrame({'date': merged_cal_df['date'].dt.strftime('%b %d, %Y'), 'level': merged_cal_df['level'], 'aqi': merged_cal_df['aqi_text']}),
                hovertemplate="<b>%{customdata[0]}</b><br>AQI: %{customdata[2]} (%{customdata[1]})<extra></extra>",
                colorscale=colorscale_cal, showscale=False, xgap=3, ygap=3, # Increased gap
                zmin=0, zmax=len(sorted_levels)-1 # Explicitly set z range for colors
            ))

            annotations_cal = []
            if len(calendar_df['week'].unique()) > 3 or not month_number_filter: # Show month names if substantial view
                 annotations_cal = [go.layout.Annotation(
                    text=month_names_map_cal[month_num], align='center', showarrow=False,
                    xref='x domain', yref='paper', 
                    x=(first_weeks.get(month_num, 0) - calendar_df['week'].min() + 0.5) / (calendar_df['week'].max() - calendar_df['week'].min() + 1) if (calendar_df['week'].max() - calendar_df['week'].min() + 1) > 0 else 0.5, # Position relative to weeks shown
                    y=1.06, # Position above plot
                    font=dict(color=SUBTLE_TEXT_COLOR_DARK_THEME, size=11, family=GLOBAL_FONT_FAMILY.split(',')[0].strip())) 
                    for month_num in sorted(first_weeks.index)
                ]
            
            fig_cal.update_layout(**custom_layout_city,
                yaxis=dict(tickmode='array', tickvals=list(range(7)), ticktext=day_labels, showgrid=False, zeroline=False, autorange="reversed", linecolor=BACKGROUND_COLOR, gridcolor=BACKGROUND_COLOR), # Reversed to show Mon at top
                xaxis=dict(showgrid=False, zeroline=False, tickmode='array', ticktext=[], tickvals=[], linecolor=BACKGROUND_COLOR, gridcolor=BACKGROUND_COLOR),
                height= (180 + 20*len(day_labels)) if month_number_filter else (200 + 20*len(day_labels)), # Dynamic height based on days
                margin=dict(t=60 if annotations_cal else 30, b=20, l=50, r=20), # Adjusted margin
                plot_bgcolor=CARD_BACKGROUND_COLOR, # Calendar plot area same as card
                annotations=annotations_cal
            )
            st.plotly_chart(fig_cal, use_container_width=True)

            # City-specific data for tabs should be based on df_period_filtered
            city_data_for_tabs = df_period_filtered[df_period_filtered['city'] == city].copy()
            if not city_data_for_tabs.empty:
                city_tabs = st.tabs(["📊 AQI Category Distribution", "📅 Monthly AQI Heatmap"])
                with city_tabs[0]:
                    category_counts_df = city_data_for_tabs['level'].value_counts().reindex(CATEGORY_COLORS_DARK.keys(), fill_value=0).reset_index()
                    category_counts_df.columns = ['AQI Category', 'Number of Days']
                    fig_dist_bar = px.bar(
                        category_counts_df, x='AQI Category', y='Number of Days', color='AQI Category',
                        color_discrete_map=CATEGORY_COLORS_DARK, text_auto='.0f' # Integer for days
                    )
                    fig_dist_bar.update_layout(**custom_layout_city, height=450, xaxis_title=None, yaxis_title="Number of Days")
                    fig_dist_bar.update_traces(marker_line_width=0) # Remove bar outlines
                    st.plotly_chart(fig_dist_bar, use_container_width=True)

                with city_tabs[1]:
                    city_data_for_tabs['month_name'] = pd.Categorical(city_data_for_tabs['date'].dt.strftime('%B'), categories=[months_map_dict[i] for i in sorted(city_data_for_tabs['date'].dt.month.unique())], ordered=True)
                    heatmap_pivot = city_data_for_tabs.pivot_table(index='month_name', columns=city_data_for_tabs['date'].dt.day, values='index', observed=True)
                    
                    fig_heat_detail = px.imshow(
                        heatmap_pivot, labels=dict(x="Day of Month", y="Month", color="AQI Index"),
                        aspect="auto", color_continuous_scale=px.colors.sequential.Plasma_r, # Different colorscale
                        text_auto=".0f"
                    )
                    fig_heat_detail.update_layout(**custom_layout_city, height=max(350, len(heatmap_pivot.index)*60), xaxis_side="top", yaxis_title=None)
                    fig_heat_detail.update_xaxes(showgrid=False, zeroline=False)
                    fig_heat_detail.update_yaxes(showgrid=False, zeroline=False)
                    st.plotly_chart(fig_heat_detail, use_container_width=True)
            else:
                 st.info(f"Detailed distribution and heatmap not available for {city} for the specific month selected. Showing full year calendar above.")


elif selected_cities and df_period_filtered.empty and year:
    st.markdown("## 🏙️ CITY-SPECIFIC DEEP DIVE")
    st.warning("No data available for the selected month. Please select 'All Months' to see annual data or adjust filters.")


# ------------------- 💨 POLLUTANT ANALYSIS -------------------
st.markdown("## 💨 PROMINENT POLLUTANT ANALYSIS")
with st.container():
    custom_layout_pollutant = get_custom_plotly_layout_args()
    if not df_period_filtered.empty:
        st.markdown(f"#### ⛽ Dominant Pollutants ({selected_month_name}, {year})")
        
        city_options_pollutant = selected_cities if selected_cities else unique_cities
        default_pollutant_city_index = 0
        if default_city_val and default_city_val[0] in city_options_pollutant:
             default_pollutant_city_index = city_options_pollutant.index(default_city_val[0])
        
        if city_options_pollutant:
            city_pollutant_B = st.selectbox(
                "Select City for Pollutant View:", city_options_pollutant,
                key="pollutant_B_city_dark_v2", 
                index=default_pollutant_city_index
            )
            pollutant_data_B = df_period_filtered[df_period_filtered['city'] == city_pollutant_B].copy()
            if not pollutant_data_B.empty and 'pollutant' in pollutant_data_B.columns and pollutant_data_B['pollutant'].notna().any():
                valid_pollutants = pollutant_data_B[(pollutant_data_B['pollutant'] != 'Other') & (pollutant_data_B['pollutant'].notna())].copy() # Exclude 'Other' and NaN explicitly
                if not valid_pollutants.empty:
                    grouped_poll_B = valid_pollutants.groupby('pollutant').size().reset_index(name='count')
                    total_valid_pollutant_days = grouped_poll_B['count'].sum()
                    if total_valid_pollutant_days > 0:
                        grouped_poll_B['percentage'] = (grouped_poll_B['count'] / total_valid_pollutant_days * 100)
                        fig_poll_B = px.bar(
                            grouped_poll_B.sort_values('percentage', ascending=False), 
                            x='pollutant', y='percentage', color='pollutant',
                            labels={'percentage': 'Dominance (% of Days)', 'pollutant': 'Pollutant'},
                            color_discrete_map=POLLUTANT_COLORS_DARK, text_auto='.1f'
                        )
                        fig_poll_B.update_layout(**custom_layout_pollutant, yaxis_ticksuffix="%", height=500, xaxis_title=None)
                        fig_poll_B.update_traces(marker_line_width=0)
                        st.plotly_chart(fig_poll_B, use_container_width=True)
                    else:
                         st.info(f"No specific dominant pollutant data (excluding 'Other' and missing) for {city_pollutant_B} in the selected period.")
                else:
                    st.info(f"No specific dominant pollutant data (excluding 'Other' and missing) for {city_pollutant_B} in the selected period.")
            else:
                st.warning(f"No pollutant data found for {city_pollutant_B} for the selected period.")
        else:
            st.info("No cities available for pollutant analysis with current filters.")
    else:
        st.info("Select a valid period to see pollutant analysis.")


# ------------------- 🗺️ INTERACTIVE AIR QUALITY MAP -------------------
st.markdown("## 🗺️ INTERACTIVE AIR QUALITY MAP") # This is a card now
with st.container():
    custom_layout_map = get_custom_plotly_layout_args()
    city_coords_data = {}
    coords_file_path = "lat_long.txt"
    try:
        if os.path.exists(coords_file_path):
            with open(coords_file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
            
            local_scope = {}
            try:
                exec(file_content, {}, local_scope) 
                if 'city_coords' in local_scope and isinstance(local_scope['city_coords'], dict):
                    city_coords_data = local_scope['city_coords']
                else:
                    st.error(f"Map Error: 'city_coords' dictionary not found or not a valid dictionary in '{coords_file_path}'. Check file structure.")
                    city_coords_data = {}
            except SyntaxError as e_syntax:
                st.error(f"Map Error: Syntax error in '{coords_file_path}'. Please ensure it's a valid Python file defining 'city_coords = {{...}}'. Error: {e_syntax}")
                city_coords_data = {}
            except Exception as e_exec: # Catch other errors during exec
                st.error(f"Map Error: Error executing coordinate file '{coords_file_path}'. Error: {e_exec}")
                city_coords_data = {}
        else:
            st.warning(f"Map Warning: Coordinates file '{coords_file_path}' not found. Interactive map cannot be displayed.")
            city_coords_data = {} # Ensure it's empty if file not found
            
    except Exception as e_file:
        st.error(f"Map Error: An unexpected error occurred while reading '{coords_file_path}': {e_file}")
        city_coords_data = {}

    if not df_period_filtered.empty:
        # Aggregate data first
        map_grouped_data = df_period_filtered.groupby('city').agg(
            avg_aqi=('index', 'mean'),
            dominant_pollutant=('pollutant', lambda x: x.mode().iloc[0] if not x.mode().empty and x.mode().iloc[0] != 'Other' else (x[x != 'Other'].mode().iloc[0] if not x[x != 'Other'].mode().empty else 'N/A'))
        ).reset_index()

        if city_coords_data and not map_grouped_data.empty:
            latlong_map_df_list = []
            for city_name, coords in city_coords_data.items():
                if isinstance(coords, (list, tuple)) and len(coords) == 2:
                    try: # Ensure coordinates are valid numbers
                        lat = float(coords[0])
                        lon = float(coords[1])
                        latlong_map_df_list.append({'city': city_name, 'lat': lat, 'lon': lon})
                    except (ValueError, TypeError):
                        # Silently skip invalid coordinate entries for a specific city, or log if needed
                        pass 
            latlong_map_df = pd.DataFrame(latlong_map_df_list)

            if not latlong_map_df.empty:
                map_merged_df = pd.merge(map_grouped_data, latlong_map_df, on='city', how='inner')

                def classify_aqi_map(val):
                    if pd.isna(val): return "Unknown"
                    if val <= 50: return "Good";
                    if val <= 100: return "Satisfactory"
                    if val <= 200: return "Moderate"
                    if val <= 300: return "Poor"
                    if val <= 400: return "Very Poor"
                    return "Severe"
                map_merged_df["AQI Category"] = map_merged_df["avg_aqi"].apply(classify_aqi_map)
                
                if not map_merged_df.empty:
                    fig_map_final = px.scatter_mapbox(
                        map_merged_df, lat="lat", lon="lon",
                        size=np.maximum(map_merged_df["avg_aqi"], 0.1), # Ensure size is positive
                        size_max=25, # Slightly larger max bubble
                        color="AQI Category", color_discrete_map=CATEGORY_COLORS_DARK,
                        hover_name="city", 
                        custom_data=['city', 'avg_aqi', 'dominant_pollutant', 'AQI Category'], # For hovertemplate
                        text="city", # Show city names on map if space permits (controlled by mapbox)
                        zoom=4.2, center={"lat": 23.5, "lon": 82.0}, height=750 # Adjusted zoom/center
                    )
                    fig_map_final.update_traces(
                        textposition='top right',
                        marker=dict(sizemin=5, opacity=0.75), # Increased sizemin
                        hovertemplate="<b style='font-size:1.1em;'>%{customdata[0]}</b><br>" +
                                      "Avg. AQI: %{customdata[1]:.1f} (%{customdata[3]})<br>" +
                                      "Dominant Pollutant: %{customdata[2]}" +
                                      "<extra></extra>" # More styled hover
                    )
                    fig_map_final.update_layout(**custom_layout_map,
                        mapbox_style="carto-darkmatter",
                        margin={"r": 0, "t": 10, "l": 0, "b": 0}, # Reduced top margin for map
                        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5, title_text='AQI Category')
                    )
                    st.plotly_chart(fig_map_final, use_container_width=True)
                else: 
                    st.info("No city data matched with available coordinates for the selected period to display on map.")
                    # Fallback if merge is empty (display bar chart)
                    if not map_grouped_data.empty:
                        st.markdown("##### City AQI Overview (Map Data Issues)")
                        st.info("Showing a summary of average AQI by city as map data could not be fully processed.")
                        avg_aqi_cities_map_alt = map_grouped_data.sort_values(by='avg_aqi', ascending=False)
                        fig_alt_map = px.bar(avg_aqi_cities_map_alt, 
                                            x='avg_aqi', y='city', orientation='h', color='avg_aqi',
                                            color_continuous_scale=px.colors.sequential.Plasma_r, # Match other heatmap
                                            labels={'avg_aqi': 'Average AQI', 'city': 'City'},
                                            height=max(400, len(avg_aqi_cities_map_alt['city']) * 35))
                        fig_alt_map.update_layout(**custom_layout_map, yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_alt_map, use_container_width=True)
            else: # latlong_map_df is empty (e.g. city_coords_data was empty or malformed)
                 st.warning("Map Error: City coordinates data could not be processed into a usable format. Cannot display scatter map.")
                 # Fallback to bar chart
                 if not map_grouped_data.empty:
                    st.markdown("##### City AQI Overview (Map Coordinates Issue)")
                    avg_aqi_cities_map_alt = map_grouped_data.sort_values(by='avg_aqi', ascending=False)
                    fig_alt_map = px.bar(avg_aqi_cities_map_alt, 
                                        x='avg_aqi', y='city', orientation='h', color='avg_aqi',
                                        color_continuous_scale=px.colors.sequential.Plasma_r,
                                        labels={'avg_aqi': 'Average AQI', 'city': 'City'},
                                        height=max(400, len(avg_aqi_cities_map_alt['city']) * 35))
                    fig_alt_map.update_layout(**custom_layout_map, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_alt_map, use_container_width=True)

        elif not map_grouped_data.empty : # city_coords_data is empty, but we have other data
            st.markdown("##### City AQI Overview (Map Coordinates Unavailable)")
            st.info("Coordinate data for the interactive map is unavailable or could not be loaded. Showing a summary of average AQI by city instead.")
            avg_aqi_cities_map_alt = map_grouped_data.sort_values(by='avg_aqi', ascending=False)
            
            if not avg_aqi_cities_map_alt.empty:
                fig_alt_map = px.bar(avg_aqi_cities_map_alt, 
                                     x='avg_aqi', y='city', orientation='h', color='avg_aqi',
                                     color_continuous_scale=px.colors.sequential.Plasma_r,
                                     labels={'avg_aqi': 'Average AQI', 'city': 'City'},
                                     height=max(400, len(avg_aqi_cities_map_alt['city']) * 40)) # Taller bars
                fig_alt_map.update_layout(**custom_layout_map, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_alt_map, use_container_width=True)
            else:
                st.warning("No data available to display in the city AQI overview for the selected period.")
        else: # df_period_filtered was empty
             st.warning("Map cannot be displayed due to missing air quality data for the selected period.")
    else: # df_period_filtered was empty from the start
        st.warning("Map cannot be displayed: No air quality data available for the selected filters.")


# ------------------- Footer -------------------
st.markdown(f"""
<div class="footer">
    <p style="font-size: 1.05rem; color: {TEXT_COLOR_DARK_THEME}; font-weight: 600;">🌬️ AuraVision AQI Dashboard</p>
    <p>Conceptualized by: Mr. Kapil Meena & Prof. Arkopal K. Goswami, IIT Kharagpur.</p>
    <p>Enhanced with AI by Gemini ✨</p>
    <p style="margin-top:1.2em;"><a href="https://github.com/kapil2020/india-air-quality-dashboard" target="_blank" rel="noopener noreferrer">🔗 View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)

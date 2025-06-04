import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.linear_model import LinearRegression # Note: This import is present but not used in the provided snippet.
import os
from datetime import date, timedelta
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json # Added for safer parsing of coordinates file
import ast  # Added for safer parsing of coordinates file

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
        background-color: {BACKGROUND_COLOR}; /* Changed for consistency with card elements if preferred, or keep CARD_BACKGROUND_COLOR */
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
st.markdown(f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 1.1rem; margin-bottom: 0.5rem;'>Illuminating Air Quality Insights Across India</p>", unsafe_allow_html=True)


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
                st.error(f"FATAL: Main data file '{fallback_file}' not found.")
                return pd.DataFrame(), "Error: Main data file not found.", None
            df_loaded = pd.read_csv(fallback_file, sep="\t", parse_dates=['date'])
            load_msg = f"Displaying archive data from: **{fallback_file}**"
            # Ensure last_update_time is timezone-aware if possible, or at least consistent UTC
            last_update_time = pd.Timestamp(os.path.getmtime(fallback_file), unit='s').tz_localize('UTC').tz_convert('Asia/Kolkata') # Example: Convert to IST
        except Exception as e:
            st.error(f"FATAL: Error loading '{fallback_file}': {e}.")
            return pd.DataFrame(), f"Error loading fallback: {e}", None

        for col, default_val in [('pollutant', np.nan), ('level', 'Unknown')]:
            if col not in df_loaded.columns: df_loaded[col] = default_val

        df_loaded['pollutant'] = df_loaded['pollutant'].astype(str).str.split(',').str[0].str.strip().replace(['nan', 'NaN', 'None', ''], np.nan)
        df_loaded['level'] = df_loaded['level'].astype(str).fillna('Unknown')
        df_loaded['pollutant'] = df_loaded['pollutant'].fillna('Other')

        return df_loaded, load_msg, last_update_time

df, load_message, data_last_updated = load_data_and_metadata()

# Display data update status below subtitle
if data_last_updated:
    st.markdown(f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 0.9rem; margin-bottom: 2rem;'>Archive data last updated: {data_last_updated.strftime('%B %d, %Y, %H:%M:%S %Z')}</p>", unsafe_allow_html=True)
else:
    st.markdown(f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 0.9rem; margin-bottom: 2rem;'>{load_message}</p>", unsafe_allow_html=True)


if df.empty:
    st.error("Dashboard cannot operate without data. Please check data sources.")
    st.stop()

# ------------------- Sidebar Filters -------------------
st.sidebar.header("🔭 EXPLORATION CONTROLS")
st.sidebar.markdown("---")
st.sidebar.info("Data presented is based on available archives. For latest CPCB data, refer to official sources.") # Adjusted message

unique_cities = sorted(df['city'].unique()) if 'city' in df.columns else []
default_city_val = ["Delhi"] if "Delhi" in unique_cities else (unique_cities[0:1] if unique_cities else [])
selected_cities = st.sidebar.multiselect("🏙️ Select Cities", unique_cities, default=default_city_val)

years = sorted(df['date'].dt.year.unique())
default_year_val = max(years) if years else None # Ensure years list is not empty
year_selection_disabled = not bool(years) # Disable if no years found

year = st.sidebar.selectbox(
    "🗓️ Select Year",
    years,
    index=years.index(default_year_val) if default_year_val in years else 0,
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
if year: # Ensure year is selected
    df_period_filtered = df[df['date'].dt.year == year].copy()
    if month_number_filter:
        df_period_filtered = df_period_filtered[df_period_filtered['date'].dt.month == month_number_filter]
else: # If no year can be selected (e.g. df is empty or date column is problematic)
    df_period_filtered = pd.DataFrame() # Empty dataframe to prevent errors downstream


# ------------------- 💡 NATIONAL KEY INSIGHTS -------------------
st.markdown("## 🇮🇳 NATIONAL KEY INSIGHTS")
with st.container(): # Ensure this container gets the card styling if desired, or style elements within
    if year: # Check if a year is selected for insights
        st.markdown(f"##### Key Metro Annual Average AQI ({year})")
        major_cities = ['Delhi', 'Mumbai', 'Kolkata', 'Bengaluru', 'Chennai']
        # Filter df by year first, then by major cities
        major_cities_data = df[df['date'].dt.year == year]
        major_cities_data = major_cities_data[major_cities_data['city'].isin(major_cities)]


        if not major_cities_data.empty:
            avg_aqi_major_cities = major_cities_data.groupby('city')['index'].mean()
            # Ensure we only create columns for cities present in avg_aqi_major_cities to avoid empty columns
            present_major_cities = [city for city in major_cities if city in avg_aqi_major_cities.index]
            if present_major_cities:
                cols = st.columns(len(present_major_cities))
                for i, city_name in enumerate(present_major_cities):
                    with cols[i]:
                        aqi_val = avg_aqi_major_cities.get(city_name, None)
                        display_val = f"{aqi_val:.2f}" if aqi_val is not None else "N/A"
                        st.metric(label=city_name, value=display_val)
            else:
                st.info(f"No data available for key metro cities in {year}.")

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
    else:
        st.warning("Please select a year to view national insights.")

# ------------------- 🆚 CITY-WISE AQI COMPARISON -------------------
if len(selected_cities) > 1 and not df_period_filtered.empty:
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
            df_year_filtered_for_radar = df[df['date'].dt.year == year] # Use full year for radar

            cities_for_radar = [city_name for city_name in selected_cities if not df_year_filtered_for_radar[df_year_filtered_for_radar['city'] == city_name].empty]

            if cities_for_radar:
                max_aqi_for_radar = 0 # To set radar axis range dynamically
                for city_name in cities_for_radar:
                    city_radar_data = df_year_filtered_for_radar[df_year_filtered_for_radar['city'] == city_name].copy()
                    city_radar_data['month'] = city_radar_data['date'].dt.month
                    monthly_avg_aqi = city_radar_data.groupby('month')['index'].mean().reindex(range(1, 13))
                    
                    current_max = monthly_avg_aqi.max()
                    if pd.notna(current_max) and current_max > max_aqi_for_radar:
                        max_aqi_for_radar = current_max

                    radar_fig.add_trace(go.Scatterpolar(
                        r=monthly_avg_aqi.values,
                        theta=[m[:3] for m in months_map_dict.values()], 
                        fill='toself',
                        name=city_name,
                        hovertemplate=f"<b>{city_name}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>"
                    ))
                
                radar_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, max_aqi_for_radar * 1.1 if max_aqi_for_radar > 0 else 50]), # Dynamic range
                        bgcolor=BACKGROUND_COLOR
                    ),
                    height=500,
                    paper_bgcolor=CARD_BACKGROUND_COLOR,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.info("Not enough data for selected cities in the chosen year to display seasonal radar patterns.")
    elif len(selected_cities) > 1: # comparison_df_list is empty, but multiple cities selected
        st.info("Not enough data for the selected cities and period to make a comparison. Try different filter settings.")

# ------------------- 🏙️ CITY-SPECIFIC DEEP DIVE -------------------
if selected_cities and not df_period_filtered.empty:
    st.markdown("## 🏙️ CITY-SPECIFIC DEEP DIVE")
    for city in selected_cities:
        st.markdown(f"### {city.upper()} – {selected_month_name if month_number_filter else 'Full Year'}, {year}")
        # Use df_period_filtered for city-specific deep dive, as it already incorporates month/year filters
        city_data_deep_dive = df_period_filtered[df_period_filtered['city'] == city].copy()

        if city_data_deep_dive.empty:
            st.warning(f"😔 No data available for {city} for the selected period. Try different filter settings.")
            continue
            
        with st.container():
            st.markdown("##### 📅 Daily AQI Calendar")
            
            # Determine start and end dates for the calendar view based on selected period
            if month_number_filter:
                # Calendar for the specific month
                if year and 1 <= month_number_filter <= 12:
                    start_date_cal = pd.to_datetime(f'{year}-{month_number_filter:02d}-01')
                    end_date_cal = (start_date_cal + pd.offsets.MonthEnd(0))
                else: # Should not happen if filters are working
                    st.error("Invalid year/month for calendar.")
                    continue
            else: # Calendar for the full year
                start_date_cal, end_date_cal = pd.to_datetime(f'{year}-01-01'), pd.to_datetime(f'{year}-12-31')

            full_period_dates_cal = pd.date_range(start_date_cal, end_date_cal)
            
            calendar_df = pd.DataFrame({'date': full_period_dates_cal})
            calendar_df['week'] = calendar_df['date'].dt.isocalendar().week
            calendar_df['day_of_week'] = calendar_df['date'].dt.dayofweek
            
            # Adjust week numbering for Jan/Dec display continuity if showing full year
            if not month_number_filter:
                calendar_df.loc[(calendar_df['date'].dt.month == 1) & (calendar_df['week'] > 50), 'week'] = 0
                calendar_df.loc[(calendar_df['date'].dt.month == 12) & (calendar_df['week'] == 1), 'week'] = calendar_df['week'].max() + 1 if 53 not in calendar_df['week'].unique() else 53


            month_label_df = calendar_df.copy()
            month_label_df['month'] = month_label_df['date'].dt.month
            first_weeks = month_label_df.groupby('month')['week'].min()
            month_names_map_cal = {month_num: name[:3] for month_num, name in months_map_dict.items()}

            merged_cal_df = pd.merge(calendar_df, city_data_deep_dive[['date', 'index', 'level']], on='date', how='left')
            merged_cal_df['level'] = merged_cal_df['level'].fillna('Unknown')
            merged_cal_df['aqi_text'] = merged_cal_df['index'].apply(lambda x: f'{x:.0f}' if pd.notna(x) else 'N/A') # Changed to .0f for cleaner look
            
            day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            fig_cal = go.Figure(data=go.Heatmap(
                x=merged_cal_df['week'], y=merged_cal_df['day_of_week'],
                z=merged_cal_df['level'].map({k: i for i, k in enumerate(CATEGORY_COLORS_DARK.keys())}), # Map levels to numeric for colorscale
                customdata=pd.DataFrame({'date': merged_cal_df['date'].dt.strftime('%Y-%m-%d'), 'level': merged_cal_df['level'], 'aqi': merged_cal_df['aqi_text']}),
                hovertemplate="<b>%{customdata[0]}</b><br>AQI: %{customdata[2]} (%{customdata[1]})<extra></extra>",
                colorscale=[CATEGORY_COLORS_DARK[level] for level in CATEGORY_COLORS_DARK], # Use direct mapping of colors
                showscale=False, xgap=2, ygap=2
            ))

            annotations = []
            # Add month annotations only if a reasonable number of weeks are present (e.g., more than just one month's fragment)
            if len(calendar_df['week'].unique()) > 4 or not month_number_filter:
                 annotations = [go.layout.Annotation(
                    text=month_names_map_cal[month_num], align='center', showarrow=False,
                    xref='x', yref='paper', x=week_num, y=1.05, # y slightly above plot
                     font=dict(color=SUBTLE_TEXT_COLOR_DARK_THEME, size=12)
                ) for month_num, week_num in first_weeks.items()]
            
            fig_cal.update_layout(
                yaxis=dict(tickmode='array', tickvals=list(range(7)), ticktext=day_labels, showgrid=False, zeroline=False, autorange="reversed"), # Reversed to show Mon at top
                xaxis=dict(showgrid=False, zeroline=False, tickmode='array', ticktext=[], tickvals=[]), # Hide x-axis ticks
                height= 250 if month_number_filter else 350, # Smaller for single month
                margin=dict(t=50 if annotations else 20, b=20, l=40, r=40),
                plot_bgcolor=CARD_BACKGROUND_COLOR, paper_bgcolor=CARD_BACKGROUND_COLOR,
                annotations=annotations
            )
            st.plotly_chart(fig_cal, use_container_width=True)

            city_tabs = st.tabs(["📈 AQI Distribution", "🔥 Monthly/Daily Heatmap"])
            with city_tabs[0]:
                category_counts_df = city_data_deep_dive['level'].value_counts().reindex(CATEGORY_COLORS_DARK.keys(), fill_value=0).reset_index()
                category_counts_df.columns = ['AQI Category', 'Number of Days']
                fig_dist_bar = px.bar(
                    category_counts_df, x='AQI Category', y='Number of Days', color='AQI Category',
                    color_discrete_map=CATEGORY_COLORS_DARK, text_auto=True
                )
                fig_dist_bar.update_layout(height=400, xaxis_title=None, yaxis_title="Number of Days", plot_bgcolor=BACKGROUND_COLOR)
                st.plotly_chart(fig_dist_bar, use_container_width=True)

            with city_tabs[1]:
                # Use df_year_filtered for this heatmap to show full year context if "All Months"
                # Or df_period_filtered if a specific month is selected.
                data_for_heatmap = city_data_deep_dive # Already filtered by month if applicable
                
                if not data_for_heatmap.empty:
                    data_for_heatmap['month_name'] = pd.Categorical(data_for_heatmap['date'].dt.strftime('%B'), categories=[months_map_dict[i] for i in range(1,13) if i in data_for_heatmap['date'].dt.month.unique()], ordered=True)
                    heatmap_pivot = data_for_heatmap.pivot_table(index='month_name', columns=data_for_heatmap['date'].dt.day, values='index', observed=False) # observed=False to keep all months
                    
                    fig_heat_detail = px.imshow(
                        heatmap_pivot, labels=dict(x="Day of Month", y="Month", color="AQI"),
                        aspect="auto", color_continuous_scale=px.colors.sequential.Inferno_r, # Reversed Inferno
                        text_auto=".0f"
                    )
                    fig_heat_detail.update_layout(height=max(300, len(heatmap_pivot.index)*50), xaxis_side="top", plot_bgcolor=BACKGROUND_COLOR) # Dynamic height
                    st.plotly_chart(fig_heat_detail, use_container_width=True)
                else:
                    st.info("Not enough data for the selected period to display detailed heatmap.")

elif selected_cities and df_period_filtered.empty :
    st.markdown("## 🏙️ CITY-SPECIFIC DEEP DIVE")
    st.warning("No data available for the selected year/month. Please adjust filters.")


# ------------------- 💨 POLLUTANT ANALYSIS -------------------
st.markdown("## 💨 PROMINENT POLLUTANT ANALYSIS")
with st.container():
    if not df_period_filtered.empty:
        st.markdown(f"#### ⛽ Dominant Pollutants for Selected Period ({selected_month_name}, {year})")
        
        # Use selected_cities for the dropdown, or all unique_cities if none are selected in main filter
        city_options_pollutant = selected_cities if selected_cities else unique_cities
        default_pollutant_city = city_options_pollutant[0] if city_options_pollutant else None

        if default_pollutant_city: # Ensure there's a city to select
            city_pollutant_B = st.selectbox(
                "Select City for Pollutant View:", city_options_pollutant,
                key="pollutant_B_city_dark", 
                index=city_options_pollutant.index(default_pollutant_city) # Find index of the default
            )
            pollutant_data_B = df_period_filtered[df_period_filtered['city'] == city_pollutant_B].copy()
            if not pollutant_data_B.empty and 'pollutant' in pollutant_data_B.columns and pollutant_data_B['pollutant'].notna().any():
                # Filter out 'Other' or NaN before grouping if they are not meaningful for dominance
                valid_pollutants = pollutant_data_B[pollutant_data_B['pollutant'] != 'Other'].dropna(subset=['pollutant'])
                if not valid_pollutants.empty:
                    grouped_poll_B = valid_pollutants.groupby('pollutant').size().reset_index(name='count')
                    total_days_B = grouped_poll_B['count'].sum() # Sum of days with valid pollutant data
                    if total_days_B > 0:
                        grouped_poll_B['percentage'] = (grouped_poll_B['count'] / total_days_B * 100).round(1)
                        fig_poll_B = px.bar(
                            grouped_poll_B.sort_values('percentage', ascending=False), 
                            x='pollutant', y='percentage', color='pollutant',
                            labels={'percentage': 'Percentage of Days (%)', 'pollutant': 'Pollutant'},
                            color_discrete_map=POLLUTANT_COLORS_DARK, text_auto='.1f' # Show one decimal place
                        )
                        fig_poll_B.update_layout(yaxis_ticksuffix="%", height=450, plot_bgcolor=BACKGROUND_COLOR)
                        st.plotly_chart(fig_poll_B, use_container_width=True)
                    else:
                         st.info(f"No specific dominant pollutant data (excluding 'Other') for {city_pollutant_B} in the selected period.")
                else:
                    st.info(f"No specific dominant pollutant data (excluding 'Other') for {city_pollutant_B} in the selected period.")
            else:
                st.warning(f"No pollutant data found for {city_pollutant_B} for the selected period.")
        else:
            st.info("No cities available for pollutant analysis with current filters.")
    else:
        st.info("Select a valid period to see pollutant analysis.")


# ------------------- 🗺️ INTERACTIVE AIR QUALITY MAP -------------------
st.markdown("## 🗺️ INTERACTIVE AIR QUALITY MAP")
with st.container():
    city_coords_data = {}
    coords_file_path = "lat_long.txt"
    try:
        if os.path.exists(coords_file_path):
            with open(coords_file_path, "r") as f:
                content = f.read()
                try:
                    city_coords_data = json.loads(content)
                except json.JSONDecodeError:
                    try:
                        city_coords_data = ast.literal_eval(content)
                    except (SyntaxError, ValueError) as e_ast:
                        st.error(f"Map Error: Could not parse city coordinates from '{coords_file_path}'. Invalid format. Please ensure it's a valid JSON or Python dictionary string. Error: {e_ast}")
                        city_coords_data = {} 
        else:
            st.warning(f"Map Warning: Coordinates file '{coords_file_path}' not found. Map features requiring coordinates will be limited.")
            city_coords_data = {}
    except Exception as e:
        st.error(f"Map Error: An unexpected error occurred while loading city coordinates: {e}")
        city_coords_data = {}

    if not df_period_filtered.empty:
        map_grouped_data = df_period_filtered.groupby('city').agg(
            avg_aqi=('index', 'mean'),
            dominant_pollutant=('pollutant', lambda x: x.mode().iloc[0] if not x.mode().empty and x.mode().iloc[0] != 'Other' else (x[x != 'Other'].mode().iloc[0] if not x[x != 'Other'].mode().empty else 'N/A'))
        ).reset_index()

        if city_coords_data and not map_grouped_data.empty:
            latlong_map_df = pd.DataFrame([{'city': k, 'lat': v[0], 'lon': v[1]} for k, v in city_coords_data.items() if isinstance(v, (list, tuple)) and len(v) == 2])
            
            if not latlong_map_df.empty:
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
                        size="avg_aqi", size_max=22, # Reduced max size
                        color="AQI Category", color_discrete_map=CATEGORY_COLORS_DARK,
                        hover_name="city", text="city",
                        hover_data={"avg_aqi": ":.2f", "dominant_pollutant": True, "city": False, "lat":False, "lon":False}, # Cleaner hover
                        zoom=4.5, center={"lat": 22.8, "lon": 82.5}, height=700
                    )
                    fig_map_final.update_layout(
                        mapbox_style="carto-darkmatter",
                        margin={"r": 0, "t": 40, "l": 0, "b": 0},
                        paper_bgcolor=CARD_BACKGROUND_COLOR,
                        legend=dict(font_color=TEXT_COLOR_DARK_THEME, orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
                    )
                    fig_map_final.update_traces(marker=dict(sizemin=4, opacity=0.75)) # Adjusted sizemin and opacity
                    st.plotly_chart(fig_map_final, use_container_width=True)
                else: 
                    st.info("No city data matched with available coordinates for the selected period.")
                    # Fallback if merge results in empty dataframe but coords were loaded
                    if not map_grouped_data.empty:
                        st.markdown("#### City AQI Overview (Map Data Incomplete)")
                        st.info("Showing a summary of average AQI by city as some coordinate data might be missing or not matching.")
                        avg_aqi_cities_map_alt = map_grouped_data.sort_values(by='avg_aqi', ascending=False)
                        fig_alt_map = px.bar(avg_aqi_cities_map_alt, 
                                            x='avg_aqi', y='city', orientation='h', color='avg_aqi',
                                            color_continuous_scale=px.colors.sequential.Inferno_r,
                                            labels={'avg_aqi': 'Average AQI', 'city': 'City'},
                                            height=max(400, len(avg_aqi_cities_map_alt['city']) * 30))
                        fig_alt_map.update_layout(paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_alt_map, use_container_width=True)

            else: # latlong_map_df is empty
                 st.warning("Map Error: City coordinates data loaded but in an unexpected format or empty. Cannot display scatter map.")
                 # Fallback to bar chart if city_coords_data was not usable
                 st.markdown("#### City AQI Overview (Map Coordinates Issue)")
                 avg_aqi_cities_map_alt = map_grouped_data.sort_values(by='avg_aqi', ascending=False)
                 if not avg_aqi_cities_map_alt.empty:
                    fig_alt_map = px.bar(avg_aqi_cities_map_alt, 
                                        x='avg_aqi', y='city', orientation='h', color='avg_aqi',
                                        color_continuous_scale=px.colors.sequential.Inferno_r,
                                        labels={'avg_aqi': 'Average AQI', 'city': 'City'},
                                        height=max(400, len(avg_aqi_cities_map_alt['city']) * 30))
                    fig_alt_map.update_layout(paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_alt_map, use_container_width=True)


        elif not map_grouped_data.empty : # city_coords_data is empty, but we have other data for the map
            st.markdown("#### City AQI Overview (Map Coordinates Unavailable)")
            st.info("Coordinate data for the interactive map is unavailable or could not be loaded. Showing a summary of average AQI by city instead.")
            avg_aqi_cities_map_alt = map_grouped_data.sort_values(by='avg_aqi', ascending=False)
            
            if not avg_aqi_cities_map_alt.empty:
                fig_alt_map = px.bar(avg_aqi_cities_map_alt, 
                                     x='avg_aqi', 
                                     y='city', 
                                     orientation='h',
                                     color='avg_aqi',
                                     color_continuous_scale=px.colors.sequential.Inferno_r,
                                     labels={'avg_aqi': 'Average AQI', 'city': 'City'},
                                     height=max(400, len(avg_aqi_cities_map_alt['city']) * 35) # Dynamic height
                                    )
                fig_alt_map.update_layout(
                    paper_bgcolor=CARD_BACKGROUND_COLOR,
                    plot_bgcolor=BACKGROUND_COLOR,
                    yaxis={'categoryorder':'total ascending'} 
                )
                st.plotly_chart(fig_alt_map, use_container_width=True)
            else:
                st.warning("No data available to display in the city AQI overview for the selected period.")
        else: # df_period_filtered was empty
             st.warning("Map cannot be displayed due to missing air quality data for the selected period.")
    else: # df_period_filtered was empty from the start
        st.warning("Map cannot be displayed: No air quality data available for the selected filters.")


# ------------------- Footer -------------------
st.markdown(f"""
<div style="text-align: center; margin-top: 4rem; padding: 1.5rem; background-color: {CARD_BACKGROUND_COLOR}; border-radius: 12px; border: 1px solid {BORDER_COLOR};">
    <p style="margin:0.3em; color: {TEXT_COLOR_DARK_THEME}; font-size:0.9rem;">🌬️ AuraVision AQI Dashboard</p>
    <p style="margin:0.5em 0; color: {TEXT_COLOR_DARK_THEME}; font-size:0.85rem;">Conceptualized by: Mr. Kapil Meena & Prof. Arkopal K. Goswami, IIT Kharagpur.</p>
    <p style="margin-top:0.8em;"><a href="https://github.com/kapil2020/india-air-quality-dashboard" target="_blank" style="color:{ACCENT_COLOR}; text-decoration:none; font-weight:600;">🔗 View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.linear_model import LinearRegression # For forecast

# --- Global Theme & Style Setup ---
pio.templates.default = "plotly_dark"

ACCENT_COLOR = "#00BCD4"                 # Vibrant Teal
TEXT_COLOR_DARK_THEME = "#EAEAEA"        # Light gray for text
SUBTLE_TEXT_COLOR_DARK_THEME = "#B0B0B0" # Subtle gray
BACKGROUND_COLOR = "#121212"             # Very dark gray (almost black)
CARD_BACKGROUND_COLOR = "#1E1E1E"        # Slightly lighter for cards
BORDER_COLOR = "#333333"                 # Dark border

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


# ------------------- Page Config -------------------
st.set_page_config(layout="wide", page_title="IIT KGP AQI Dashboard", page_icon="🌬️")


# ------------------- Custom CSS Styling (Dark Theme) -------------------
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
    div[data-testid="stExpander"], div[data-testid="stForm"], .stpyplot {{
        border-radius: 12px;
        border: 1px solid {BORDER_COLOR};
        background-color: {CARD_BACKGROUND_COLOR};
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); /* Softer shadow */
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
        background-color: {BACKGROUND_COLOR}; /* Or CARD_BACKGROUND_COLOR for a slightly different effect */
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
    h2 {{
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
    h3 {{ /* General sub-section titles */
        font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR_DARK_THEME};
        margin-top: 0rem; /* Reset for cards */
        margin-bottom: 1.2rem;
        font-weight: 600;
    }}
     h4, h5 {{ /* Titles within cards or smaller sections */
        font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR_DARK_THEME}; /* Primary text color for emphasis */
        margin-top: 0.2rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }}
    h5 {{ /* Slightly smaller sub-titles */
        color: {SUBTLE_TEXT_COLOR_DARK_THEME}; /* More subtle for plot titles inside cards */
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
        border-bottom: none; /* Remove borders for sidebar headings */
    }}
    .stSidebar .stSelectbox label, .stSidebar .stMultiselect label {{
        color: {ACCENT_COLOR} !important;
        font-weight: 600;
    }}

    /* Metric styling */
    .stMetric {{
        background-color: {BACKGROUND_COLOR}; /* Darker background for contrast */
        border: 1px solid {BORDER_COLOR};
        border-radius: 8px;
        padding: 1rem 1.2rem; /* Adjusted padding */
    }}
    .stMetric > div:nth-child(1) {{ /* Label */
        font-size: 0.95rem; /* Slightly smaller label */
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        font-weight: 500;
        margin-bottom: 0.25rem;
    }}
    .stMetric > div:nth-child(2) {{ /* Value */
        font-size: 2.0rem; /* Slightly smaller value for compactness */
        font-weight: 700;
        color: {ACCENT_COLOR};
    }}
    .stMetric > div:nth-child(3) > div {{ /* Delta (if exists) - not used here but good to have */
        font-size: 0.8rem;
        font-weight: 500;
    }}


    /* Expander styling */
    div[data-testid="stExpander"] summary {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {ACCENT_COLOR};
    }}
    div[data-testid="stExpander"] svg {{
        fill: {ACCENT_COLOR};
    }}

    /* Download button styling */
    .stDownloadButton button {{
        background-color: {ACCENT_COLOR};
        color: {BACKGROUND_COLOR}; /* High contrast text */
        border: none;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: background-color 0.3s ease, opacity 0.3s ease;
    }}
    .stDownloadButton button:hover {{
        background-color: {ACCENT_COLOR};
        opacity: 0.85;
    }}

    /* Input Widgets */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div:first-child,
    .stMultiselect div[data-baseweb="select"] > div:first-child {{
        background-color: {BACKGROUND_COLOR}; /* Darker input background */
        color: {TEXT_COLOR_DARK_THEME};
        border-color: {BORDER_COLOR} !important;
    }}
    .stDateInput input {{
         background-color: {BACKGROUND_COLOR};
         color: {TEXT_COLOR_DARK_THEME};
    }}

    /* Custom Scrollbar */
    ::-webkit-scrollbar {{
        width: 10px; /* Slightly wider scrollbar */
        height: 10px;
    }}
    ::-webkit-scrollbar-track {{
        background: {BACKGROUND_COLOR};
        border-radius: 5px;
    }}
    ::-webkit-scrollbar-thumb {{
        background: {BORDER_COLOR};
        border-radius: 5px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {ACCENT_COLOR};
    }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------
#  Helper: Common Plotly layout arguments
# --------------------------------------------
def get_custom_plotly_layout_args(
    height: int = None,
    title_text: str = None,
    legend_orientation: str = "h",
    legend_y: float = 1.02,
    legend_x: float = 1.0,
    margin_t: int = 40, # Default top margin
    margin_b: int = 40, # Default bottom margin
    margin_l: int = 40, # Default left margin
    margin_r: int = 40  # Default right margin
) -> dict:
    """
    Returns a dict of common Plotly layout arguments for dark-themed charts.
    """
    layout_args = {
        "font": {"family": "Inter, sans-serif", "color": TEXT_COLOR_DARK_THEME, "size": 12},
        "paper_bgcolor": CARD_BACKGROUND_COLOR,
        "plot_bgcolor": CARD_BACKGROUND_COLOR, # Often looks cleaner if same as paper for embedded charts
        "legend": {
            "orientation": legend_orientation,
            "yanchor": "bottom", "y": legend_y,
            "xanchor": "right", "x": legend_x,
            "bgcolor": "rgba(0,0,0,0)", # Transparent legend background
            "font": {"size": 11}
        },
        "margin": dict(t=margin_t, b=margin_b, l=margin_l, r=margin_r),
        "hoverlabel": {
            "bgcolor": BACKGROUND_COLOR,
            "font_size": 13,
            "font_family": "Inter, sans-serif",
            "bordercolor": BORDER_COLOR
        }
    }
    if height:
        layout_args["height"] = height
    if title_text:
        layout_args["title_text"] = title_text
        layout_args["title_font"] = {"color": ACCENT_COLOR, "size": 16, "family": "Inter, sans-serif"}
        layout_args["title_x"] = 0.05 # Align title to the left
        layout_args["title_xanchor"] = 'left'
    return layout_args


# ------------------- Title -------------------
st.markdown("<h1 style='text-align: center; margin-bottom:0.5rem;'>🌬️ IIT KGP AQI Dashboard</h1>", unsafe_allow_html=True)
st.markdown(f"""
<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 1.1rem; margin-bottom: 2.5rem;'>
    Illuminating Air Quality Insights Across India
</p>
""", unsafe_allow_html=True)


# ------------------- Load Data -------------------
@st.cache_data(ttl=3600)
def load_data_and_metadata():
    """
    Tries to load today's CSV (named YYYY-MM-DD.csv). If not found, falls back to 'combined_air_quality.txt'.
    Returns: (df_loaded, load_msg, last_update_time)
    """
    today = pd.to_datetime("today").date()
    # Ensure data directory exists or adjust path as needed
    data_dir = "data" 
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True) # Create if it doesn't exist
    csv_path = os.path.join(data_dir, f"{today}.csv")
    fallback_file = "combined_air_quality.txt" # Assume it's in the root or adjust path
    
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
                # load_msg = f"Live data from: **{today}.csv**" # Message handled by timestamp now
                last_update_time = pd.Timestamp(os.path.getmtime(csv_path), unit="s")
            else:
                load_msg = f"Warning: '{csv_path}' found but missing 'date' column. Using fallback."
        except Exception as e:
            load_msg = f"Error loading '{csv_path}': {e}. Using fallback."

    if df_loaded is None or not is_today_data:
        try:
            if not os.path.exists(fallback_file):
                st.error(f"FATAL: Main data file '{fallback_file}' not found at expected location.")
                return pd.DataFrame(), "Error: Main data file not found.", None
            df_loaded = pd.read_csv(fallback_file, sep="\t", parse_dates=["date"])
            # base_load_msg = f"Displaying archive data from: **{fallback_file}**"
            # load_msg = base_load_msg if not load_msg or is_today_data else load_msg + " " + base_load_msg # Message handled by timestamp
            last_update_time = pd.Timestamp(os.path.getmtime(fallback_file), unit="s")
        except Exception as e:
            st.error(f"FATAL: Error loading '{fallback_file}': {e}.")
            return pd.DataFrame(), f"Error loading fallback: {e}", None

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
    
    # Ensure 'index' (AQI) column is numeric, coercing errors
    if 'index' in df_loaded.columns:
        df_loaded['index'] = pd.to_numeric(df_loaded['index'], errors='coerce')

    return df_loaded, load_msg, last_update_time


df, load_message, data_last_updated = load_data_and_metadata()

if df.empty:
    st.error("Dashboard cannot operate without data. Please check data sources and file paths.")
    st.stop()

if load_message: # Display any specific loading messages/warnings
    st.caption(f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 0.85rem;'>{load_message}</p>", unsafe_allow_html=True)

if data_last_updated:
    st.caption(
        f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 0.85rem; margin-bottom:1.5rem;'>"
        f"Last data update: {data_last_updated.strftime('%Y-%m-%d %H:%M:%S')}"
        "</p>",
        unsafe_allow_html=True
    )


# ------------------- Sidebar Filters -------------------
st.sidebar.markdown("<h2 style='margin-top:0; padding-top:0;'>🔭 EXPLORATION CONTROLS</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---", unsafe_allow_html=True)
st.sidebar.info("Data typically updates daily. Today's data for CPCB stations is usually available after 5:45 PM IST.")

unique_cities = sorted(df["city"].unique()) if "city" in df.columns else []
default_city_val = ["Delhi"] if "Delhi" in unique_cities else (unique_cities[0:1] if unique_cities else [])
selected_cities = st.sidebar.multiselect("🏙️ Select Cities", unique_cities, default=default_city_val)

years = sorted(df["date"].dt.year.unique()) if "date" in df.columns and not df.empty else []
default_year_val = max(years) if years else None
year_index = years.index(default_year_val) if default_year_val and years else 0
year = st.sidebar.selectbox("🗓️ Select Year", years, index=year_index, disabled=not years)

months_map_dict = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
} # Using short month names for brevity in UI
month_options_list = ["All Months"] + list(months_map_dict.values())
selected_month_name_short = st.sidebar.selectbox("🌙 Select Month (Optional)", month_options_list, index=0)

month_number_filter = None
if selected_month_name_short != "All Months":
    month_number_filter = [k for k, v in months_map_dict.items() if v == selected_month_name_short][0]

df_period_filtered = pd.DataFrame() # Initialize as empty
if year:
    df_period_filtered = df[df["date"].dt.year == year].copy()
    if month_number_filter:
        df_period_filtered = df_period_filtered[df_period_filtered["date"].dt.month == month_number_filter]
else:
    st.warning("Please select a year to load data.")


# ========================================================
# =========  NATIONAL KEY INSIGHTS  =======
# ========================================================
st.markdown("## 🇮🇳 National Snapshot")
with st.container(border=False): # Use st.container(border=False) if you don't want the default card style for this section wrapper
    if year and not df_period_filtered.empty:
        st.markdown(f"##### Key Metro Annual Average AQI ({year})")
        major_cities = ["Delhi", "Mumbai", "Kolkata", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Ahmedabad"]
        major_cities_annual_data = df[df["date"].dt.year == year] # Use full year data for annual metro stats
        major_cities_annual_data = major_cities_annual_data[major_cities_annual_data["city"].isin(major_cities)]

        if not major_cities_annual_data.empty:
            avg_aqi_major_cities = major_cities_annual_data.groupby("city")["index"].mean().dropna()
            present_major_cities = [c for c in major_cities if c in avg_aqi_major_cities.index]

            if present_major_cities:
                cols = st.columns(min(len(present_major_cities), 4))
                col_idx = 0
                for city_name_metro in present_major_cities:
                    with cols[col_idx % len(cols)]:
                        aqi_val = avg_aqi_major_cities.get(city_name_metro)
                        st.metric(label=city_name_metro, value=f"{aqi_val:.1f}" if pd.notna(aqi_val) else "N/A")
                    col_idx += 1
            else:
                st.info(f"No annual AQI data available for the selected key metro cities in {year}.")
        else:
            st.info(f"No data available for key metro cities in {year}.")

        st.markdown(f"##### General Insights for Selected Period ({selected_month_name_short}, {year})")
        if not df_period_filtered.empty:
            avg_aqi_national = df_period_filtered["index"].mean()
            city_avg_aqi_stats = df_period_filtered.groupby("city")["index"].mean().dropna()

            if not city_avg_aqi_stats.empty:
                num_cities_observed = df_period_filtered["city"].nunique()
                best_city_name, best_city_aqi = city_avg_aqi_stats.idxmin(), city_avg_aqi_stats.min()
                worst_city_name, worst_city_aqi = city_avg_aqi_stats.idxmax(), city_avg_aqi_stats.max()

                st.markdown(
                    f"""<div style="font-size: 1.05rem; line-height: 1.7; background-color: {CARD_BACKGROUND_COLOR}; padding: 1rem; border-radius: 8px; border: 1px solid {BORDER_COLOR};">
                    Across <b>{num_cities_observed}</b> observed cities during {selected_month_name_short}, {year}, the average AQI is
                    <b style="color:{ACCENT_COLOR}; font-size:1.15em;">{avg_aqi_national:.2f}</b>.
                    City with best average air quality:
                    <b style="color:{CATEGORY_COLORS_DARK.get('Good', '#FFFFFF')};">{best_city_name}</b> ({best_city_aqi:.2f}).
                    Most challenged city:
                    <b style="color:{CATEGORY_COLORS_DARK.get('Severe', '#FFFFFF')};">{worst_city_name}</b> ({worst_city_aqi:.2f}).
                    </div>""",
                    unsafe_allow_html=True
                )
            else:
                st.info("AQI statistics could not be computed for the selected period (no valid city averages).")
        else:
            st.info("No data available for the selected period to generate general insights.")
    elif not year:
        st.warning("Please select a year to view national insights.")
    else: # df_period_filtered is empty but year is selected
         st.info(f"No data available for {selected_month_name_short}, {year} to generate national insights.")

st.markdown("<hr style='border-color: {BORDER_COLOR}; margin-top: 2rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)


# ========================================================
# =======   CITY-SPECIFIC ANALYSIS   =========
# ========================================================
export_data_list = []

if not selected_cities:
    st.info("✨ Select one or more cities from the sidebar to dive into detailed analysis.")
else:
    for city in selected_cities:
        st.markdown(f"## 🏙️ {city.upper()} DEEP DIVE – {year}")
        # Use a container for each city's analysis to apply card styling if desired, or remove for flatter look
        # with st.container(border=True): # This would make the whole city section a card

        city_data_full = df_period_filtered[df_period_filtered["city"] == city].copy()
        current_filter_period_city = f"{selected_month_name_short}, {year}"

        if city_data_full.empty:
            st.warning(f"😔 No data available for {city} for {current_filter_period_city}. Try different filter settings.")
            continue

        city_data_full["day_of_year"] = city_data_full["date"].dt.dayofyear
        city_data_full["month_name"] = city_data_full["date"].dt.strftime('%b') # Short month names
        city_data_full["day_of_month"] = city_data_full["date"].dt.day
        export_data_list.append(city_data_full)

        # --- City Key Metrics ---
        avg_aqi_city_val = city_data_full["index"].mean()
        min_aqi_city_row = city_data_full.loc[city_data_full["index"].idxmin()] if pd.notna(city_data_full["index"].min()) else None
        max_aqi_city_row = city_data_full.loc[city_data_full["index"].idxmax()] if pd.notna(city_data_full["index"].max()) else None
        days_with_data = city_data_full["index"].count()

        st.markdown("##### 🔑 Key Metrics for Selected Period")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric(label="Avg. AQI", value=f"{avg_aqi_city_val:.1f}" if pd.notna(avg_aqi_city_val) else "N/A")
        with m_col2:
            st.metric(label="Days w/ Data", value=str(days_with_data)) # Shortened label
        with m_col3:
            if min_aqi_city_row is not None and pd.notna(min_aqi_city_row['index']):
                st.metric(label="Min AQI", value=f"{min_aqi_city_row['index']:.0f}", help=f"On {min_aqi_city_row['date'].strftime('%b %d')}, Level: {min_aqi_city_row['level']}")
            else:
                st.metric(label="Min AQI", value="N/A")
        with m_col4:
            if max_aqi_city_row is not None and pd.notna(max_aqi_city_row['index']):
                st.metric(label="Max AQI", value=f"{max_aqi_city_row['index']:.0f}", help=f"On {max_aqi_city_row['date'].strftime('%b %d')}, Level: {max_aqi_city_row['level']}")
            else:
                st.metric(label="Max AQI", value="N/A")
        # st.markdown("<hr style='margin-top:0.1rem; margin-bottom:1.0rem; border-color: {BORDER_COLOR};'>", unsafe_allow_html=True)


        tab_trend, tab_dist, tab_heatmap_detail = st.tabs(["📊 TRENDS & CALENDAR", "📈 DISTRIBUTIONS", "🗓️ DETAILED HEATMAP"])

        with tab_trend:
            st.markdown("##### 📅 Daily AQI Calendar")
            start_date = pd.to_datetime(f"{year}-01-01")
            end_date = pd.to_datetime(f"{year}-12-31")
            full_year_dates = pd.date_range(start_date, end_date)

            calendar_df = pd.DataFrame({"date": full_year_dates})
            calendar_df["week"] = calendar_df["date"].dt.isocalendar().week
            calendar_df["day_of_week"] = calendar_df["date"].dt.dayofweek # Monday=0, Sunday=6

            calendar_df.loc[(calendar_df["date"].dt.month == 1) & (calendar_df["week"] > 50), "week"] = 0
            calendar_df.loc[(calendar_df["date"].dt.month == 12) & (calendar_df["week"] == 1), "week"] = 53

            merged_cal_df = pd.merge(
                calendar_df,
                city_data_full[["date", "index", "level"]],
                on="date",
                how="left"
            )
            merged_cal_df["level"] = merged_cal_df["level"].fillna("Unknown")
            merged_cal_df["aqi_text"] = merged_cal_df["index"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A") # Changed to .0f for brevity

            day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            
            fig_cal = go.Figure(
                data=go.Heatmap(
                    x=merged_cal_df["week"],
                    y=merged_cal_df["day_of_week"],
                    z=merged_cal_df["level"].map({k: i for i, k in enumerate(CATEGORY_COLORS_DARK.keys())}),
                    customdata=pd.DataFrame({
                        "date_str": merged_cal_df["date"].dt.strftime("%Y-%m-%d (%a)"), # Added day abbr
                        "level": merged_cal_df["level"],
                        "aqi": merged_cal_df["aqi_text"]
                    }),
                    hovertemplate="<b>%{customdata.date_str}</b><br>AQI: %{customdata.aqi} (%{customdata.level})<extra></extra>",
                    colorscale=[[i / (len(CATEGORY_COLORS_DARK) - 1), color] for i, color in enumerate(CATEGORY_COLORS_DARK.values())],
                    showscale=False,
                    xgap=2, ygap=2
                )
            )

            fig_cal_annotations = []
            month_label_weeks_added = set()
            month_starts_info = []
            for m_idx in range(1, 13):
                first_day_of_month_m = pd.Timestamp(f"{year}-{m_idx}-01")
                days_in_month_m_from_merged = merged_cal_df[merged_cal_df['date'].dt.month == m_idx]
                if not days_in_month_m_from_merged.empty:
                    first_data_day_for_month = days_in_month_m_from_merged.sort_values('date').iloc[0]
                    week_val = first_data_day_for_month['week']
                    month_name_val = first_data_day_for_month['date'].strftime("%b")
                    month_starts_info.append({'name': month_name_val, 'week': week_val, 'date': first_data_day_for_month['date']})
            
            month_starts_info_sorted = sorted(month_starts_info, key=lambda item: item['date'])

            for month_info_item in month_starts_info_sorted:
                week_num_cal = month_info_item['week']
                month_name_cal = month_info_item['name']
                if week_num_cal not in month_label_weeks_added:
                    fig_cal_annotations.append(
                        go.layout.Annotation(
                            text=month_name_cal, align="center", showarrow=False,
                            xref="x", yref="y domain", x=week_num_cal, y=1.06, # Position above cells
                            font=dict(color=SUBTLE_TEXT_COLOR_DARK_THEME, size=11), xanchor='center'
                        ))
                    month_label_weeks_added.add(week_num_cal)
            
            legend_y_start_cal, legend_y_step_cal = -0.12, -0.07
            legend_x_start_cal, legend_x_step_cal = 0.01, 0.14 
            for i_legend, (level_legend, color_legend) in enumerate(CATEGORY_COLORS_DARK.items()):
                fig_cal_annotations.append(
                    go.layout.Annotation(
                        text=f"█ <span style='color:{TEXT_COLOR_DARK_THEME};'>{level_legend}</span>",
                        align="left", showarrow=False, xref="paper", yref="paper",
                        x=legend_x_start_cal + legend_x_step_cal * (i_legend % 7),
                        y=legend_y_start_cal + legend_y_step_cal * (i_legend // 7),
                        xanchor="left", yanchor="top", font=dict(color=color_legend, size=11) # Smaller font
                    ))

            fig_cal.update_layout(
                yaxis=dict(tickmode="array", tickvals=list(range(7)), ticktext=day_labels, showgrid=False, zeroline=False, autorange="reversed"), # Reversed y-axis
                xaxis=dict(showgrid=False, zeroline=False, tickmode="array", ticktext=[], tickvals=[]), # Hide x-axis ticks/labels
                height=320, # Adjusted height
                margin=dict(t=60, b=80, l=40, r=20), # t for month names, b for legend
                plot_bgcolor=CARD_BACKGROUND_COLOR, paper_bgcolor=CARD_BACKGROUND_COLOR,
                font_color=TEXT_COLOR_DARK_THEME, annotations=fig_cal_annotations, showlegend=False
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
                    marker=dict(size=5, opacity=0.7, color=SUBTLE_TEXT_COLOR_DARK_THEME), # Slightly larger markers
                    line=dict(width=1.5, color=SUBTLE_TEXT_COLOR_DARK_THEME),
                    customdata=city_data_trend[["level"]],
                    hovertemplate="<b>%{x|%Y-%m-%d (%a)}</b><br>AQI: %{y:.1f}<br>Category: %{customdata[0]}<extra></extra>"
                )
            )
            fig_trend_roll.add_trace(
                go.Scatter(
                    x=city_data_trend["date"], y=city_data_trend["rolling_avg_7day"],
                    mode="lines", name="7-Day Rolling Avg",
                    line=dict(color=ACCENT_COLOR, width=2.5, dash="dash"),
                    hovertemplate="<b>%{x|%Y-%m-%d (%a)}</b><br>7-Day Avg AQI: %{y:.1f}<extra></extra>"
                )
            )
            fig_trend_roll.update_layout(
                **get_custom_plotly_layout_args(height=400, margin_b=50, margin_t=60),
                yaxis_title="AQI Index", xaxis_title=None, # "Date" is clear from context
                hovermode="x unified"
            )
            st.plotly_chart(fig_trend_roll, use_container_width=True)

        with tab_dist:
            col_bar_dist, col_sun_dist = st.columns(2)
            with col_bar_dist:
                st.markdown("##### 📊 AQI Category (Days)")
                category_counts_df = (
                    city_data_full["level"]
                    .value_counts()
                    .reindex(CATEGORY_COLORS_DARK.keys(), fill_value=0)
                    .reset_index()
                )
                category_counts_df.columns = ["AQI Category", "Number of Days"]
                fig_dist_bar = px.bar(
                    category_counts_df, x="AQI Category", y="Number of Days", color="AQI Category",
                    color_discrete_map=CATEGORY_COLORS_DARK, text_auto=True
                )
                fig_dist_bar.update_layout(
                    **get_custom_plotly_layout_args(height=400, margin_t=30, margin_b=30),
                    xaxis_title=None, yaxis_title="Number of Days", showlegend=False
                )
                fig_dist_bar.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
                st.plotly_chart(fig_dist_bar, use_container_width=True)

            with col_sun_dist:
                st.markdown("##### ☀️ AQI Category (Proportions)")
                if category_counts_df["Number of Days"].sum() > 0:
                    fig_sunburst = px.sunburst(
                        category_counts_df, path=["AQI Category"], values="Number of Days",
                        color="AQI Category", color_discrete_map=CATEGORY_COLORS_DARK,
                    )
                    fig_sunburst.update_layout(
                         **get_custom_plotly_layout_args(height=400, margin_t=30, margin_l=20, margin_r=20, margin_b=30)
                    )
                    st.plotly_chart(fig_sunburst, use_container_width=True)
                else:
                    st.caption("No data for sunburst chart.")

            st.markdown("##### 🎻 Monthly AQI Distribution")
            # Use full month names for categories for better sorting, but display short names from 'month_name' column
            month_order_cat_full = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            temp_month_full_name = city_data_full['date'].dt.month_name() # For ordering
            city_data_full_sorted_month = city_data_full.assign(month_order_col=pd.Categorical(temp_month_full_name, categories=month_order_cat_full, ordered=True))
            city_data_full_sorted_month = city_data_full_sorted_month.sort_values("month_order_col")


            fig_violin = px.violin(
                city_data_full_sorted_month,
                x="month_name", y="index", color="month_name", # 'month_name' is already short
                color_discrete_sequence=px.colors.qualitative.Vivid, # Using a Plotly sequence
                box=True, points="outliers",
                labels={"index": "AQI Index", "month_name": "Month"},
                hover_data={"date": "|%b %d, %Y", "level": True, "index": ":.1f"}
            )
            fig_violin.update_layout(
                **get_custom_plotly_layout_args(height=450, margin_t=30, margin_b=50),
                xaxis_title=None, showlegend=False,
                xaxis={'categoryorder': 'array', 'categoryarray': [months_map_dict[i] for i in range(1,13) if months_map_dict[i] in city_data_full_sorted_month['month_name'].unique()]} # Ensure correct month order
            )
            fig_violin.update_traces(meanline_visible=True)
            st.plotly_chart(fig_violin, use_container_width=True)

        with tab_heatmap_detail:
            st.markdown("##### 🔥 AQI Heatmap (Month vs. Day)")
            heatmap_pivot = city_data_full.pivot_table(index="month_name", columns="day_of_month", values="index", observed=False)
            # Reindex using the short month names in the correct order
            ordered_short_months = [months_map_dict[i] for i in range(1,13)]
            heatmap_pivot = heatmap_pivot.reindex(ordered_short_months, axis=0).dropna(how='all')


            fig_heat_detail = px.imshow(
                heatmap_pivot, labels=dict(x="Day of Month", y="Month", color="AQI"),
                aspect="auto", color_continuous_scale="Inferno_r", # Reversed Inferno for higher=more intense
                text_auto=".0f"
            )
            fig_heat_detail.update_layout(
                **get_custom_plotly_layout_args(height=500, margin_t=50, margin_b=50),
                xaxis_side="top",
                xaxis=dict(tickmode='linear', tick0=1, dtick=1) # Ensure all day numbers are potential ticks
            )
            fig_heat_detail.update_traces(hovertemplate="<b>Month:</b> %{y}<br><b>Day:</b> %{x}<br><b>AQI:</b> %{z:.0f}<extra></extra>")
            st.plotly_chart(fig_heat_detail, use_container_width=True)
        
        st.markdown("<hr style='border-color: {BORDER_COLOR}; margin-top: 2rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)


# ========================================================
# =======   CITY-WISE AQI COMPARISON  ========
# ========================================================
if len(selected_cities) > 1:
    st.markdown("## 🆚 AQI COMPARISON ACROSS CITIES")
    comparison_df_list = []
    for city_comp in selected_cities:
        city_ts_comp = df_period_filtered[df_period_filtered["city"] == city_comp].copy()
        if not city_ts_comp.empty:
            city_ts_comp = city_ts_comp.sort_values("date")
            city_ts_comp["city_label"] = city_comp
            comparison_df_list.append(city_ts_comp)

    if comparison_df_list:
        combined_comp_df = pd.concat(comparison_df_list)
        fig_cmp = px.line(
            combined_comp_df, x="date", y="index", color="city_label",
            labels={"index": "AQI Index", "date": "Date", "city_label": "City"},
            markers=False, line_shape="spline", color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_cmp.update_layout(
            **get_custom_plotly_layout_args(
                height=500,
                title_text=f"AQI Trends Comparison – {selected_month_name_short}, {year}",
                margin_t=80 # More top margin for title
            ),
             hovermode="x unified",
             xaxis_title=None
        )
        fig_cmp.update_traces(hovertemplate="<b>%{fullData.name}</b><br>%{x|%Y-%m-%d}: %{y:.1f}<extra></extra>")
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.info("Not enough data or cities selected for comparison with current filters.")
    st.markdown("<hr style='border-color: {BORDER_COLOR}; margin-top: 2rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)


# ========================================================
# =======   PROMINENT POLLUTANT ANALYSIS  =====
# ========================================================
st.markdown("## 💨 PROMINENT POLLUTANT ANALYSIS")
with st.container(border=False): # Use border=False for a less card-like section wrapper
    st.markdown("##### 鑽 Yearly Dominant Pollutant Trends") # Using h5 for subtitles within sections
    
    # Ensure unique_cities is not empty before trying to find index
    default_pollutant_city_index = 0
    if default_city_val and default_city_val[0] in unique_cities:
        default_pollutant_city_index = unique_cities.index(default_city_val[0])
    elif unique_cities: # If default not found but unique_cities exist, pick first
        default_pollutant_city_index = 0

    city_pollutant_A = st.selectbox(
        "Select City for Yearly Pollutant Trend:", unique_cities,
        key="pollutant_A_city_dark", index=default_pollutant_city_index if unique_cities else 0,
        disabled=not unique_cities
    )
    if city_pollutant_A:
        pollutant_data_A = df[df["city"] == city_pollutant_A].copy()
        pollutant_data_A["year_label"] = pollutant_data_A["date"].dt.year

        if not pollutant_data_A.empty:
            grouped_poll_A = pollutant_data_A.groupby(["year_label", "pollutant"]).size().unstack(fill_value=0)
            percent_poll_A = grouped_poll_A.apply(lambda x: x / x.sum() * 100 if x.sum() > 0 else x, axis=1).fillna(0)
            percent_poll_A_long = percent_poll_A.reset_index().melt(id_vars="year_label", var_name="pollutant", value_name="percentage")

            fig_poll_A = px.bar(
                percent_poll_A_long, x="year_label", y="percentage", color="pollutant",
                labels={"percentage": "Percentage of Days (%)", "year_label": "Year", "pollutant": "Pollutant"},
                color_discrete_map=POLLUTANT_COLORS_DARK
            )
            fig_poll_A.update_layout(
                **get_custom_plotly_layout_args(
                    height=500,
                    title_text=f"Dominant Pollutants Over Years – {city_pollutant_A}",
                    margin_t=80
                ),
                xaxis_type="category", yaxis_ticksuffix="%"
            )
            fig_poll_A.update_traces(texttemplate="%{value:.1f}%", textposition="inside", textfont_size=10)
            st.plotly_chart(fig_poll_A, use_container_width=True)
        else:
            st.warning(f"No pollutant data for {city_pollutant_A} (Yearly Trend).")
    else:
        st.warning("Please select a city to view yearly pollutant trends.")


    st.markdown(f"##### ⛽ Dominant Pollutants for Selected Period ({selected_month_name_short}, {year})")
    city_pollutant_B = st.selectbox(
        "Select City for Filtered Pollutant View:", unique_cities,
        key="pollutant_B_city_dark", index=default_pollutant_city_index if unique_cities else 0, # Use same default as above
        disabled=not unique_cities
    )
    if city_pollutant_B and not df_period_filtered.empty:
        pollutant_data_B = df_period_filtered[df_period_filtered["city"] == city_pollutant_B].copy()
        if not pollutant_data_B.empty and "pollutant" in pollutant_data_B.columns:
            grouped_poll_B = pollutant_data_B.groupby("pollutant").size().reset_index(name="count")
            total_days_B = grouped_poll_B["count"].sum()
            grouped_poll_B["percentage"] = (grouped_poll_B["count"] / total_days_B * 100).round(1) if total_days_B > 0 else 0
            grouped_poll_B = grouped_poll_B[grouped_poll_B["percentage"] > 0] # Show only pollutants present

            fig_poll_B = px.bar(
                grouped_poll_B.sort_values("percentage", ascending=False), 
                x="pollutant", y="percentage", color="pollutant",
                labels={"percentage": "Percentage of Days (%)", "pollutant": "Pollutant"},
                color_discrete_map=POLLUTANT_COLORS_DARK
            )
            fig_poll_B.update_layout(
                 **get_custom_plotly_layout_args(
                    height=450,
                    title_text=f"Dominant Pollutants – {city_pollutant_B} ({selected_month_name_short}, {year})",
                    margin_t=80
                ),
                yaxis_ticksuffix="%", showlegend=False # Legend redundant with x-axis labels and color
            )
            fig_poll_B.update_traces(texttemplate="%{value:.1f}%", textposition="outside", cliponaxis=False)
            st.plotly_chart(fig_poll_B, use_container_width=True)
        else:
            st.warning(f"No pollutant data for {city_pollutant_B} for the selected period ({selected_month_name_short}, {year}).")
    elif not city_pollutant_B:
         st.warning("Please select a city to view pollutant distribution for the selected period.")
    else: # df_period_filtered is empty
        st.info(f"No data available for {selected_month_name_short}, {year} to analyze pollutants.")

st.markdown("<hr style='border-color: {BORDER_COLOR}; margin-top: 2rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)


# ========================================================
# =======   AQI FORECAST   ====================
# ========================================================
st.markdown("## 🔮 AQI FORECAST (LINEAR TREND)")
with st.container(border=False):
    forecast_city_select = st.selectbox(
        "Select City for AQI Forecast:", unique_cities,
        key="forecast_city_select_dark", index=default_pollutant_city_index if unique_cities else 0, # Use same default
        disabled=not unique_cities
    )
    if forecast_city_select and not df_period_filtered.empty:
        forecast_src_data = df_period_filtered[df_period_filtered["city"] == forecast_city_select].copy()
        if len(forecast_src_data) >= 15: # Need sufficient data points
            forecast_df = forecast_src_data.sort_values("date")[["date", "index"]].dropna()
            if len(forecast_df) >= 2: # Need at least 2 points for linear regression
                forecast_df["days_since_start"] = (forecast_df["date"] - forecast_df["date"].min()).dt.days
                
                X_train, y_train = forecast_df[["days_since_start"]], forecast_df["index"]
                model = LinearRegression().fit(X_train, y_train)
                
                last_day_num = forecast_df["days_since_start"].max()
                # Forecast for historical period + 15 days into future
                future_X_range = np.arange(0, last_day_num + 15 + 1).reshape(-1, 1)
                future_y_pred = model.predict(future_X_range)
                
                min_date_forecast = forecast_df["date"].min()
                future_dates_list = [min_date_forecast + pd.Timedelta(days=int(i)) for i in future_X_range.flatten()]

                plot_df_obs = pd.DataFrame({"date": forecast_df["date"], "AQI": y_train})
                plot_df_fcst = pd.DataFrame({"date": future_dates_list, "AQI": np.maximum(0, future_y_pred)}) # AQI cannot be negative

                fig_forecast = go.Figure()
                fig_forecast.add_trace(
                    go.Scatter(x=plot_df_obs["date"], y=plot_df_obs["AQI"], mode="lines+markers", name="Observed AQI", line=dict(color=ACCENT_COLOR))
                )
                fig_forecast.add_trace(
                    go.Scatter(x=plot_df_fcst["date"], y=plot_df_fcst["AQI"], mode="lines", name="Trend + 15d Forecast", line=dict(dash="dash", color="#FF6B6B"))
                )

                fig_forecast.update_layout(
                    **get_custom_plotly_layout_args(
                        height=450,
                        title_text=f"AQI Linear Trend & Forecast – {forecast_city_select}",
                        margin_t=80
                    ),
                    yaxis_title="AQI Index", xaxis_title=None, # Date is clear
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
            else:
                st.warning(f"Not enough valid data points (after dropping NA, found {len(forecast_df)}) for {forecast_city_select} in {selected_month_name_short}, {year} to forecast.")
        else:
            st.warning(f"Need at least 15 data points for {forecast_city_select} for forecasting in {selected_month_name_short}, {year}; found {len(forecast_src_data)}.")
    elif not forecast_city_select:
        st.warning("Please select a city to view AQI forecast.")
    else: # df_period_filtered empty
         st.info(f"No data available for {selected_month_name_short}, {year} to generate forecast.")

st.markdown("<hr style='border-color: {BORDER_COLOR}; margin-top: 2rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)


# ========================================================
# ==========  City AQI Hotspots  =========
# ========================================================
st.markdown("## 📍 City AQI Hotspots")
with st.container(border=False):
    city_coords_data = {}
    coords_file_path = "lat_long.txt"  # Ensure this file is in the correct path relative to script
    scatter_map_rendered = False

    try:
        if os.path.exists(coords_file_path):
            with open(coords_file_path, "r", encoding="utf-8") as f_coords:
                file_content_coords = f_coords.read()
            local_scope_coords = {}
            exec(file_content_coords, {}, local_scope_coords) # Safely execute to load dict
            if "city_coords" in local_scope_coords and isinstance(local_scope_coords["city_coords"], dict):
                city_coords_data = local_scope_coords["city_coords"]
    except Exception as e_exec_coords:
        st.error(f"Map Error: Error processing coordinates file '{coords_file_path}'. Scatter map cannot be displayed. Error: {e_exec_coords}")
        city_coords_data = {}

    if not df_period_filtered.empty:
        map_grouped_data = df_period_filtered.groupby("city").agg(
            avg_aqi=('index', 'mean'),
            dominant_pollutant=('pollutant', lambda x:
                x.mode().iloc[0] if not x.mode().empty and pd.notna(x.mode().iloc[0]) and x.mode().iloc[0] not in ['Other', 'nan']
                else (x[~x.isin(['Other', 'nan', np.nan])].mode().iloc[0]
                      if not x[~x.isin(['Other', 'nan', np.nan])].mode().empty else 'N/A')
            )
        ).reset_index().dropna(subset=['avg_aqi'])

        if city_coords_data and not map_grouped_data.empty:
            latlong_map_df_list = []
            for city_name_coord_map, coords_val_map in city_coords_data.items():
                if isinstance(coords_val_map, (list, tuple)) and len(coords_val_map) == 2:
                    try:
                        lat_map, lon_map = float(coords_val_map[0]), float(coords_val_map[1])
                        latlong_map_df_list.append({'city': city_name_coord_map, 'lat': lat_map, 'lon': lon_map})
                    except (ValueError, TypeError):
                        pass # Skip if coordinates are not valid numbers
            
            if latlong_map_df_list:
                latlong_map_df = pd.DataFrame(latlong_map_df_list)
                map_merged_df = pd.merge(map_grouped_data, latlong_map_df, on="city", how="inner")

                if not map_merged_df.empty:
                    map_merged_df["AQI Category"] = map_merged_df["avg_aqi"].apply(
                        lambda val: next(
                            (k_cat for k_cat, v_range_cat in {
                                'Good': (0, 50), 'Satisfactory': (51, 100), 'Moderate': (101, 200),
                                'Poor': (201, 300), 'Very Poor': (301, 400), 'Severe': (401, float('inf'))
                            }.items() if v_range_cat[0] <= val <= v_range_cat[1]), "Unknown"
                        ) if pd.notna(val) else "Unknown"
                    )

                    fig_scatter_map = px.scatter_mapbox(
                        map_merged_df, lat="lat", lon="lon",
                        size=np.maximum(map_merged_df["avg_aqi"], 1), # Ensure size is at least 1
                        size_max=20, # Slightly smaller max size for less clutter
                        color="AQI Category",
                        color_discrete_map=CATEGORY_COLORS_DARK,
                        hover_name="city",
                        custom_data=['city', 'avg_aqi', 'dominant_pollutant', 'AQI Category'],
                        text="city", # Show city names on map if space allows (zoom dependent)
                        zoom=4.2, center={"lat": 23.5, "lon": 82.0} # Adjusted center/zoom for India
                    )

                    scatter_map_layout_args = get_custom_plotly_layout_args(
                        height=650, # Adjusted height
                        title_text=f"Average AQI Hotspots - {selected_month_name_short}, {year}",
                        margin_t=80, margin_b=10, margin_l=10, margin_r=10,
                        legend_y=0.98, legend_x=0.99 # Place legend top-right
                    )
                    scatter_map_layout_args['mapbox_style'] = "carto-darkmatter"
                    scatter_map_layout_args['legend']['title_text'] = "AQI Category"
                    
                    fig_scatter_map.update_traces(
                        marker=dict(sizemin=4, opacity=0.8), # Adjusted sizemin and opacity
                        textfont=dict(size=9, color=TEXT_COLOR_DARK_THEME, family="Inter, sans-serif"),
                        hovertemplate="<b style='font-size:1.1em;'>%{customdata[0]}</b><br>" +
                                      "Avg. AQI: %{customdata[1]:.1f} (%{customdata[3]})<br>" +
                                      "Dom. Pollutant: %{customdata[2]}" +
                                      "<extra></extra>"
                    )
                    fig_scatter_map.update_layout(**scatter_map_layout_args)
                    st.plotly_chart(fig_scatter_map, use_container_width=True)
                    scatter_map_rendered = True

        if not scatter_map_rendered:
            if not city_coords_data:
                 st.warning(f"Map Warning: Coordinates file '{coords_file_path}' not found or 'city_coords' variable issue. Scatter map cannot be displayed.")
            elif 'latlong_map_df' in locals() and (not latlong_map_df_list or latlong_map_df.empty):
                 st.warning(f"Map Warning: Coordinates data from '{coords_file_path}' loaded but seems empty, malformed, or no valid coordinates found.")
            elif 'map_merged_df' in locals() and map_merged_df.empty:
                 st.info(f"No city data from your dataset matched with the loaded coordinates for the scatter map for {selected_month_name_short}, {year}.")
            else: # Generic message if map didn't render for other reasons (e.g. map_grouped_data empty)
                st.info(f"Scatter map could not be rendered. Insufficient data or coordinate mismatch for {selected_month_name_short}, {year}.")


            st.markdown("##### City AQI Overview (Map Data Unavailable/Incomplete)")
            if not map_grouped_data.empty:
                avg_aqi_cities_alt = map_grouped_data.sort_values(by='avg_aqi', ascending=False) # Show worst first
                fig_alt_bar = px.bar(
                    avg_aqi_cities_alt.head(20), # Top 20 worst AQI cities
                    x='avg_aqi', y='city', orientation='h',
                    color='avg_aqi', color_continuous_scale=px.colors.sequential.YlOrRd, # Reversed YlOrRd for worst
                    labels={'avg_aqi': 'Average AQI', 'city': 'City'}
                )
                fallback_layout = get_custom_plotly_layout_args(
                    height=max(400, len(avg_aqi_cities_alt.head(20)) * 25),
                    title_text=f"Cities with Highest Average AQI - {selected_month_name_short}, {year} (Top 20)",
                    margin_t=80, margin_l=100 # More left margin for city names
                )
                fallback_layout["yaxis"] = {'categoryorder':'total ascending'} # Shows highest bar at top
                fallback_layout["coloraxis_showscale"] = False # Hide color scale for bar chart
                fig_alt_bar.update_layout(**fallback_layout)
                st.plotly_chart(fig_alt_bar, use_container_width=True)
            else:
                st.warning(f"No city AQI data available for {selected_month_name_short}, {year} to display fallback chart.")
    else:
        st.warning(f"No air quality data available for {selected_month_name_short}, {year} to display on a map or chart.")
st.markdown("<hr style='border-color: {BORDER_COLOR}; margin-top: 2rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)


# ========================================================
# ========   DOWNLOAD FILTERED DATA   ========
# ========================================================
if export_data_list:
    st.markdown("## 📥 DOWNLOAD DATA")
    with st.container(border=False): # Using border=False for a cleaner section look
        # Check if there's actually data to concat and download
        non_empty_dfs = [df_item for df_item in export_data_list if not df_item.empty]
        if non_empty_dfs:
            combined_export_final = pd.concat(non_empty_dfs)
            from io import StringIO
            csv_buffer_final = StringIO()
            combined_export_final.to_csv(csv_buffer_final, index=False)
            
            file_name_suffix = f"{year}_{selected_month_name_short.replace(' ', '')}"
            if selected_cities:
                file_name_suffix += f"_{'_'.join(selected_cities[:2])}" # Add first few city names
                if len(selected_cities) > 2:
                    file_name_suffix += "_etc"
            
            st.download_button(
                label="📤 Download Filtered City Data (CSV)",
                data=csv_buffer_final.getvalue(),
                file_name=f"IITKGP_filtered_aqi_{file_name_suffix}.csv",
                mime="text/csv",
                use_container_width=True # Makes button wider
            )
        else:
            st.info("No data currently filtered for download based on your city selections for the chosen period.")


# ------------------- Footer -------------------
st.markdown(f"""
<div style="text-align: center; margin-top: 4rem; padding: 1.5rem; background-color: {CARD_BACKGROUND_COLOR}; border-radius: 12px; border: 1px solid {BORDER_COLOR};">
    <p style="margin:0.3em; color: {TEXT_COLOR_DARK_THEME}; font-size:0.95rem; font-weight:600;">🌬️ IIT KGP AQI Dashboard</p>
    <p style="margin:0.3em; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size:0.8rem;">Data Source: Central Pollution Control Board (CPCB), India. City coordinates are approximate.</p>
    <p style="margin:0.5em 0; color: {TEXT_COLOR_DARK_THEME}; font-size:0.85rem;">Conceptualized by: Mr. Kapil Meena & Prof. Arkopal K. Goswami, IIT Kharagpur.</p>
    <p style="margin:0.3em; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size:0.8rem;">Crafted with Streamlit.</p>
    <p style="margin-top:0.8em;">
        <a href="https://github.com/kapil2020/india-air-quality-dashboard" target="_blank" style="color:{ACCENT_COLOR}; text-decoration:none; font-weight:600; font-size:0.9rem;">
            🔗 View on GitHub
        </a>
    </p>
</div>
""", unsafe_allow_html=True)

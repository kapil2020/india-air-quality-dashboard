import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# --- Global Theme & Style Setup ---
pio.templates.default = "plotly_dark"  # Base Plotly theme

ACCENT_COLOR = "#00B0FF"                # Vibrant blue accent
TEXT_COLOR_DARK_THEME = "#E0E0E0"       # Light grey for text
SUBTLE_TEXT_COLOR_DARK_THEME = "#9E9E9E" # For less important text
BACKGROUND_COLOR = "#121212"            # Very dark grey (background)
CARD_BACKGROUND_COLOR = "#1E1E1E"       # Slightly lighter for cards
BORDER_COLOR = "#333333"                # Subtle border color

GLOBAL_FONT_FAMILY = "Inter, sans-serif"

CATEGORY_COLORS_DARK = {
    "Severe": "#D32F2F",       # Vivid Red
    "Very Poor": "#F57C00",    # Vivid Orange
    "Poor": "#FFA000",         # Amber
    "Moderate": "#FBC02D",     # Yellow
    "Satisfactory": "#7CB342", # Light Green
    "Good": "#388E3C",         # Green
    "Unknown": "#616161"       # Dark Grey for unknown
}

POLLUTANT_COLORS_DARK = {
    "PM2.5": "#FF6E40", "PM10": "#29B6F6", "NO2": "#7E57C2",
    "SO2": "#FFEE58", "CO": "#FFA726", "O3": "#66BB6A", "Other": "#BDBDBD"
}


# ------------------- Page Config -------------------
st.set_page_config(
    layout="wide",
    page_title="AuraVision AQI Dashboard",
    page_icon="💨"
)


# ------------------- Custom CSS Styling (Refined Dark Theme) -------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    body {{
        font-family: {GLOBAL_FONT_FAMILY};
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR_DARK_THEME};
        line-height: 1.6;
    }}
    .main .block-container {{
        padding: 2rem 3rem 3rem 3rem;
    }}

    /* Card Styling */
    .stPlotlyChart, .stDataFrame, .stAlert,
    div[data-testid="stExpander"], div[data-testid="stForm"] {{
        border-radius: 12px;
        border: 1px solid {BORDER_COLOR};
        background-color: {CARD_BACKGROUND_COLOR};
        padding: 1.8rem;
        margin-bottom: 2.2rem;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }}
    .stPlotlyChart:hover, div[data-testid="stExpander"]:hover {{
         transform: translateY(-3px);
         box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }}

    /* Button Styling */
    .stButton > button, .stDownloadButton > button {{
        border-radius: 8px !important;
        border: 1.5px solid {ACCENT_COLOR} !important;
        background-color: transparent !important;
        color: {ACCENT_COLOR} !important;
        padding: 0.6rem 1.3rem !important;
        font-weight: 600 !important;
        font-family: {GLOBAL_FONT_FAMILY} !important;
        transition: all 0.2s ease !important;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: {ACCENT_COLOR} !important;
        color: {BACKGROUND_COLOR} !important;
        transform: scale(1.03) !important;
        box-shadow: 0 4px 10px -2px {ACCENT_COLOR}99 !important;
    }}
    .stButton > button:active, .stDownloadButton > button:active {{
        transform: scale(0.99) !important;
        box-shadow: none !important;
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
         border-bottom: 2px solid {BORDER_COLOR};
         margin-bottom: 1.5rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 0.9rem 1.6rem;
        font-weight: 600;
        font-size: 0.95rem;
        font-family: {GLOBAL_FONT_FAMILY};
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        border-radius: 8px 8px 0 0;
        margin-right: 0.4rem;
        transition: background-color 0.3s ease, color 0.3s ease, border-bottom-color 0.3s ease;
        border-bottom: 3px solid transparent;
    }}
     .stTabs [aria-selected="true"] {{
        border-bottom: 3px solid {ACCENT_COLOR};
        color: {ACCENT_COLOR};
        background-color: {CARD_BACKGROUND_COLOR}33;
     }}

    /* Headings */
    h1 {{
        font-family: {GLOBAL_FONT_FAMILY};
        color: {TEXT_COLOR_DARK_THEME};
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        font-size: 2.8rem;
    }}
    h2 {{
        font-family: {GLOBAL_FONT_FAMILY};
        color: {ACCENT_COLOR};
        border-bottom: 2.5px solid {ACCENT_COLOR};
        padding-bottom: 0.7rem;
        margin-top: 3.5rem;
        margin-bottom: 2.2rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 1.5rem;
    }}
    h3 {{
        font-family: {GLOBAL_FONT_FAMILY};
        color: {TEXT_COLOR_DARK_THEME};
        margin-top: 1.8rem;
        margin-bottom: 1.5rem;
        font-weight: 600;
        font-size: 1.3rem;
    }}
    h4, h5 {{
        font-family: {GLOBAL_FONT_FAMILY};
        color: {TEXT_COLOR_DARK_THEME};
        margin-bottom: 1.2rem;
        font-weight: 600;
        font-size: 1.1rem;
    }}

    /* Sidebar Styling */
    .stSidebar {{
        background-color: {CARD_BACKGROUND_COLOR};
        border-right: 1px solid {BORDER_COLOR};
        padding: 2.2rem;
    }}
    .stSidebar .stMarkdown h2 {{
        color: {ACCENT_COLOR} !important;
        font-size: 1.3rem !important;
        font-family: {GLOBAL_FONT_FAMILY} !important;
        font-weight: 700 !important;
        margin-bottom: 1.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: none !important;
        text-align: left;
    }}
    .stSidebar .stSelectbox label, .stSidebar .stMultiselect label {{
        font-size: 1rem;
        font-family: {GLOBAL_FONT_FAMILY} !important;
        margin-bottom: 0.5rem !important;
        font-weight: 600 !important;
        color: {ACCENT_COLOR} !important;
    }}
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
        background-color: {BACKGROUND_COLOR} !important;
        border: 1px solid {BORDER_COLOR} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover {{
        border-color: {ACCENT_COLOR} !important;
    }}

    /* Metric Styling */
    .stMetric {{
        background-color: {CARD_BACKGROUND_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-left: 5px solid {ACCENT_COLOR};
        border-radius: 10px;
        padding: 1.3rem 1.6rem;
    }}
    .stMetric > div:nth-child(1) {{
        font-size: 0.95rem;
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        font-weight: 500;
        margin-bottom: 0.3rem;
    }}
    .stMetric > div:nth-child(2) {{
        font-size: 2.4rem;
        font-weight: 700;
        color: {ACCENT_COLOR};
        line-height: 1.2;
    }}

    /* Scrollbar Styling */
    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-track {{ background: {BACKGROUND_COLOR}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER_COLOR}; border-radius: 5px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {ACCENT_COLOR}; }}

    /* Footer Styling */
    .footer {{
        text-align: center; margin-top: 4.5rem; padding: 2.5rem;
        background-color: {CARD_BACKGROUND_COLOR}; border-radius: 12px;
        border-top: 1px solid {BORDER_COLOR};
    }}
    .footer p {{ margin: 0.5em; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 0.9rem; }}
    .footer a {{ color: {ACCENT_COLOR}; text-decoration: none; font-weight: 600; }}
    .footer a:hover {{ text-decoration: underline; filter: brightness(1.1); }}
</style>
""", unsafe_allow_html=True)


# --- Plotly Universal Layout Options ---
def get_custom_plotly_layout_args(height=None, title_text=None):
    layout_args = dict(
        font_family=GLOBAL_FONT_FAMILY.split(',')[0].strip(),
        font_color=TEXT_COLOR_DARK_THEME,
        paper_bgcolor=CARD_BACKGROUND_COLOR,
        plot_bgcolor=CARD_BACKGROUND_COLOR,
        legend=dict(
            font=dict(
                color=SUBTLE_TEXT_COLOR_DARK_THEME,
                family=GLOBAL_FONT_FAMILY.split(',')[0].strip(),
                size=10
            ),
            bgcolor=CARD_BACKGROUND_COLOR,
            bordercolor=BORDER_COLOR,
            borderwidth=0.5,
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1
        ),
        title_font_family=GLOBAL_FONT_FAMILY.split(',')[0].strip(),
        title_font_color=TEXT_COLOR_DARK_THEME,
        title_font_size=16,
        title_x=0.05,
        title_xref="container",
        xaxis=dict(
            gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, zerolinecolor=BORDER_COLOR,
            tickfont=dict(
                family=GLOBAL_FONT_FAMILY.split(',')[0].strip(),
                color=SUBTLE_TEXT_COLOR_DARK_THEME,
                size=10
            ),
            showgrid=True, gridwidth=1, griddash='solid', zeroline=False
        ),
        yaxis=dict(
            gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, zerolinecolor=BORDER_COLOR,
            tickfont=dict(
                family=GLOBAL_FONT_FAMILY.split(',')[0].strip(),
                color=SUBTLE_TEXT_COLOR_DARK_THEME,
                size=10
            ),
            showgrid=True, gridwidth=1, griddash='solid', zeroline=False
        ),
        margin=dict(l=50, r=30, t=60 if title_text else 30, b=50)
    )
    if height:
        layout_args["height"] = height
    if title_text:
        layout_args["title_text"] = title_text
        layout_args["title_font_size"] = 16
    return layout_args


# ------------------- Title -------------------
st.markdown("<h1>💨 AuraVision AQI</h1>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; margin-bottom: 2.5rem;">
    <p style="color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 1.15rem; font-weight: 400; letter-spacing: 0.5px;">
        Illuminating Air Quality Insights Across India
    </p>
</div>
""", unsafe_allow_html=True)


# ------------------- Load Data -------------------
@st.cache_data(ttl=3600)
def load_data_and_metadata(file_path="combined_air_quality.txt"):
    df_loaded = None
    load_msg = ""
    last_update_time = None

    try:
        if not os.path.exists(file_path):
            return pd.DataFrame(), f"Error: Main data file '{os.path.basename(file_path)}' not found.", None
        df_loaded = pd.read_csv(file_path, sep="\t", parse_dates=["date"])
        load_msg = f"Displaying archive data from **{os.path.basename(file_path)}**"
        try:
            last_update_time = pd.Timestamp(os.path.getmtime(file_path), unit="s").tz_localize("UTC").tz_convert("Asia/Kolkata")
        except Exception:
            last_update_time = pd.Timestamp(os.path.getmtime(file_path), unit="s")
    except Exception as e:
        return pd.DataFrame(), f"FATAL: Error loading '{os.path.basename(file_path)}': {e}.", None

    if df_loaded is not None:
        for col, default_val in [("pollutant", np.nan), ("level", "Unknown")]:
            if col not in df_loaded.columns:
                df_loaded[col] = default_val
        df_loaded["pollutant"] = (
            df_loaded["pollutant"]
            .astype(str)
            .str.split(",")
            .str[0]
            .str.strip()
            .replace(["nan", "NaN", "None", ""], np.nan)
        )
        df_loaded["level"] = df_loaded["level"].astype(str).fillna("Unknown").str.strip()
        df_loaded["pollutant"] = df_loaded["pollutant"].fillna("Other")
        # Ensure 'level' only contains known categories or 'Unknown'
        known_levels = list(CATEGORY_COLORS_DARK.keys())
        df_loaded["level"] = df_loaded["level"].apply(lambda x: x if x in known_levels else "Unknown")
    else:
        df_loaded = pd.DataFrame()
        load_msg = "Error: Dataframe is None after loading attempt."

    return df_loaded, load_msg, last_update_time


# Load the data from the combined text file
df, load_message, data_last_updated = load_data_and_metadata(file_path="combined_air_quality.txt")

if df.empty:
    st.error(f"Dashboard cannot operate without data. {load_message}")
    st.stop()

if data_last_updated:
    update_time_str = (
        data_last_updated.strftime("%B %d, %Y, %H:%M %Z")
        if hasattr(data_last_updated, "tzinfo") and data_last_updated.tzinfo is not None
        else data_last_updated.strftime("%B %d, %Y, %H:%M (UTC)")
    )
    st.markdown(
        f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 0.9rem; margin-bottom: 2.5rem;'>"
        f"Archive data last updated: {update_time_str}</p>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 0.9rem; margin-bottom: 2.5rem;'>"
        f"{load_message}</p>",
        unsafe_allow_html=True
    )


# ------------------- Sidebar Filters -------------------
st.sidebar.markdown("## 🔬 Explore Data")
st.sidebar.markdown("---", unsafe_allow_html=True)
st.sidebar.info("Adjust filters to analyze AQI trends for specific regions and times.")

unique_cities = sorted(df["city"].unique()) if "city" in df.columns else []
default_city_val = ["Delhi"] if "Delhi" in unique_cities else (unique_cities[0:1] if unique_cities else [])
selected_cities = st.sidebar.multiselect("🏙️ Select Cities", unique_cities, default=default_city_val)

years = sorted(df["date"].dt.year.unique(), reverse=True)
default_year_val = years[0] if years else None
year_selection_disabled = not bool(years)
year = st.sidebar.selectbox("🗓️ Select Year", years, index=0, disabled=year_selection_disabled)

months_display_map = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}
month_options_display = ["All Months"] + list(months_display_map.values())
selected_month_display_name = st.sidebar.selectbox(
    "🌙 Select Month (Optional)", month_options_display, index=0, disabled=year_selection_disabled
)

month_number_filter = None
if selected_month_display_name != "All Months":
    month_number_filter = list(months_display_map.keys())[
        list(months_display_map.values()).index(selected_month_display_name)
    ]

# Filter data based on global selections
if year:
    df_period_filtered = df[df["date"].dt.year == year].copy()
    if month_number_filter:
        df_period_filtered = df_period_filtered[
            df_period_filtered["date"].dt.month == month_number_filter
        ]
else:
    df_period_filtered = pd.DataFrame()


# ------------------- 💡 NATIONAL KEY INSIGHTS -------------------
st.markdown("## 🇮🇳 National Snapshot")
with st.container():
    if year:
        st.markdown(f"##### Key Metro Annual Average AQI ({year})")
        major_cities = ["Delhi", "Mumbai", "Kolkata", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Ahmedabad"]
        major_cities_annual_data = df[df["date"].dt.year == year]
        major_cities_annual_data = major_cities_annual_data[
            major_cities_annual_data["city"].isin(major_cities)
        ]

        if not major_cities_annual_data.empty:
            avg_aqi_major_cities = major_cities_annual_data.groupby("city")["index"].mean().dropna()
            present_major_cities = [c for c in major_cities if c in avg_aqi_major_cities.index]

            if present_major_cities:
                cols = st.columns(min(len(present_major_cities), 4))
                col_idx = 0
                for city_name in present_major_cities:
                    with cols[col_idx % len(cols)]:
                        aqi_val = avg_aqi_major_cities.get(city_name)
                        st.metric(label=city_name, value=f"{aqi_val:.1f}")
                    col_idx += 1
            else:
                st.info(f"No annual AQI data available for the key metro cities in {year}.")
        else:
            st.info(f"No data available for key metro cities in {year}.")

        st.markdown(f"##### General Insights for Selected Period ({selected_month_display_name}, {year})")
        if not df_period_filtered.empty:
            avg_aqi_national = df_period_filtered["index"].mean()
            city_avg_aqi_stats = df_period_filtered.groupby("city")["index"].mean().dropna()

            if not city_avg_aqi_stats.empty:
                num_cities_observed = df_period_filtered["city"].nunique()
                best_city_name, best_city_aqi = city_avg_aqi_stats.idxmin(), city_avg_aqi_stats.min()
                worst_city_name, worst_city_aqi = city_avg_aqi_stats.idxmax(), city_avg_aqi_stats.max()

                st.markdown(f"""
                <div style="font-size: 1.05rem; line-height: 1.7;">
                    Across <b>{num_cities_observed}</b> observed cities, the average AQI is 
                    <b style="color:{ACCENT_COLOR}; font-size:1.15em;">{avg_aqi_national:.2f}</b>.
                    City with best average air quality: 
                    <b style="color:{CATEGORY_COLORS_DARK.get('Good', '#FFFFFF')};">{best_city_name}</b> ({best_city_aqi:.2f}).
                    Most challenged city: 
                    <b style="color:{CATEGORY_COLORS_DARK.get('Severe', '#FFFFFF')};">{worst_city_name}</b> ({worst_city_aqi:.2f}).
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("AQI statistics could not be computed for the selected period (no valid city averages).")
        else:
            st.info("No data available for the selected period to generate general insights.")
    else:
        st.warning("Please select a year to view national insights.")
st.markdown("---")


# ------------------- 🗺️ INTERACTIVE AIR QUALITY MAP (City Hotspots) -------------------
st.markdown("## 📍 City AQI Hotspots")
with st.container():
    city_coords_data = {}
    coords_file_path = "lat_long.txt"
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
        st.error(
            f"Map Error: Error processing coordinates file '{coords_file_path}'. "
            f"Scatter map cannot be displayed. Error: {e_exec}"
        )
        city_coords_data = {}

    if not df_period_filtered.empty:
        map_grouped_data = df_period_filtered.groupby("city").agg(
            avg_aqi=("index", "mean"),
            dominant_pollutant=(
                "pollutant",
                lambda x: x.mode().iloc[0]
                if not x.mode().empty and x.mode().iloc[0] not in ["Other", "nan"]
                else (
                    x[~x.isin(["Other", "nan"])].mode().iloc[0]
                    if not x[~x.isin(["Other", "nan"])].mode().empty
                    else "N/A"
                )
            )
        ).reset_index().dropna(subset=["avg_aqi"])

        if city_coords_data and not map_grouped_data.empty:
            latlong_map_df_list = []
            for city_name_coord, coords_val in city_coords_data.items():
                if isinstance(coords_val, (list, tuple)) and len(coords_val) == 2:
                    try:
                        lat, lon = float(coords_val[0]), float(coords_val[1])
                        latlong_map_df_list.append({"city": city_name_coord, "lat": lat, "lon": lon})
                    except (ValueError, TypeError):
                        pass

            latlong_map_df = pd.DataFrame(latlong_map_df_list)

            if not latlong_map_df.empty:
                map_merged_df = pd.merge(map_grouped_data, latlong_map_df, on="city", how="inner")

                if not map_merged_df.empty:
                    map_merged_df["AQI Category"] = map_merged_df["avg_aqi"].apply(
                        lambda val: next(
                            (
                                k
                                for k, v_range in {
                                    "Good": (0, 50),
                                    "Satisfactory": (51, 100),
                                    "Moderate": (101, 200),
                                    "Poor": (201, 300),
                                    "Very Poor": (301, 400),
                                    "Severe": (401, float("inf")),
                                }.items()
                                if v_range[0] <= val <= v_range[1]
                            ),
                            "Unknown"
                        )
                        if pd.notna(val)
                        else "Unknown"
                    )

                    fig_scatter_map = px.scatter_mapbox(
                        map_merged_df,
                        lat="lat",
                        lon="lon",
                        size=np.maximum(map_merged_df["avg_aqi"], 1),
                        size_max=25,
                        color="AQI Category",
                        color_discrete_map=CATEGORY_COLORS_DARK,
                        hover_name="city",
                        custom_data=["city", "avg_aqi", "dominant_pollutant", "AQI Category"],
                        text="city",
                        zoom=4.0,
                        center={"lat": 23.0, "lon": 82.5},
                    )

                    scatter_map_layout_args = get_custom_plotly_layout_args(
                        height=700,
                        title_text=f"Average AQI Hotspots - {selected_month_display_name}, {year}"
                    )
                    scatter_map_layout_args["mapbox_style"] = "carto-darkmatter"
                    scatter_map_layout_args["margin"] = {"r": 10, "t": 60, "l": 10, "b": 10}
                    scatter_map_layout_args["legend"]["y"] = 0.98
                    scatter_map_layout_args["legend"]["x"] = 0.98
                    scatter_map_layout_args["legend"]["xanchor"] = "right"

                    fig_scatter_map.update_traces(
                        marker=dict(sizemin=5, opacity=0.75),
                        hovertemplate="<b style='font-size:1.1em;'>%{customdata[0]}</b><br>"
                                      "Avg. AQI: %{customdata[1]:.1f} (%{customdata[3]})<br>"
                                      "Dominant Pollutant: %{customdata[2]}<extra></extra>"
                    )
                    fig_scatter_map.update_layout(**scatter_map_layout_args)
                    st.plotly_chart(fig_scatter_map, use_container_width=True)
                    scatter_map_rendered = True

        if not scatter_map_rendered:
            if not city_coords_data:
                st.warning(
                    f"Map Warning: Coordinates file '{coords_file_path}' not found or 'city_coords' variable issue. "
                    "Scatter map cannot be displayed."
                )
            elif "latlong_map_df" in locals() and latlong_map_df.empty:
                st.warning(
                    f"Map Warning: Coordinates data from '{coords_file_path}' loaded but seems empty or malformed."
                )
            elif "map_merged_df" in locals() and map_merged_df.empty:
                st.info(
                    "No city data from your dataset matched with the loaded coordinates for the scatter map."
                )

            st.markdown("##### City AQI Overview (Map Data Unavailable/Incomplete)")
            if not map_grouped_data.empty:
                avg_aqi_cities_alt = map_grouped_data.sort_values(by="avg_aqi", ascending=True)
                fig_alt_bar = px.bar(
                    avg_aqi_cities_alt.tail(20),
                    x="avg_aqi",
                    y="city",
                    orientation="h",
                    color="avg_aqi",
                    color_continuous_scale=px.colors.sequential.YlOrRd_r,
                    labels={"avg_aqi": "Average AQI", "city": "City"},
                )
                fallback_layout = get_custom_plotly_layout_args(
                    height=max(400, len(avg_aqi_cities_alt.tail(20)) * 25),
                    title_text=f"Top Cities by Average AQI - {selected_month_display_name}, {year} (Map Fallback)",
                )
                fallback_layout["yaxis"] = {"categoryorder": "total ascending"}
                fig_alt_bar.update_layout(**fallback_layout)
                st.plotly_chart(fig_alt_bar, use_container_width=True)
            else:
                st.warning("No city AQI data available for the selected period to display fallback chart.")
    else:
        st.warning("No air quality data available for the selected filters to display on a map or chart.")
st.markdown("---")


# ------------------- 🆚 CITY-WISE AQI COMPARISON -------------------
if len(selected_cities) > 1 and not df_period_filtered.empty:
    st.markdown("## 📊 City-to-City AQI Analysis")

    comp_tab1, comp_tab2 = st.tabs(["📈 AQI Trend Comparison", "🌀 Seasonal AQI Radar"])

    comparison_df_list = []
    for city_comp in selected_cities:
        city_ts_comp = df_period_filtered[df_period_filtered["city"] == city_comp].copy()
        if not city_ts_comp.empty:
            comparison_df_list.append(city_ts_comp)

    if comparison_df_list:
        with comp_tab1:
            combined_comp_df = pd.concat(comparison_df_list)
            fig_cmp = px.line(
                combined_comp_df,
                x="date",
                y="index",
                color="city",
                labels={"index": "AQI Index", "date": "Date", "city": "City"},
                line_shape="spline",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_cmp.update_layout(
                **get_custom_plotly_layout_args(
                    height=500, title_text="AQI Trends Over Selected Period"
                )
            )
            fig_cmp.update_traces(line=dict(width=2.5))
            st.plotly_chart(fig_cmp, use_container_width=True)

        with comp_tab2:
            radar_fig = go.Figure()
            df_year_filtered_for_radar = df[df["date"].dt.year == year]

            cities_for_radar = [
                city_name
                for city_name in selected_cities
                if not df_year_filtered_for_radar[df_year_filtered_for_radar["city"] == city_name].empty
            ]

            if cities_for_radar:
                max_aqi_for_radar = 0
                radar_colors = px.colors.qualitative.Pastel
                for i, city_name in enumerate(cities_for_radar):
                    city_radar_data = df_year_filtered_for_radar[
                        df_year_filtered_for_radar["city"] == city_name
                    ].copy()
                    if not city_radar_data.empty:
                        city_radar_data["month"] = city_radar_data["date"].dt.month
                        monthly_avg_aqi = city_radar_data.groupby("month")["index"].mean().reindex(range(1, 13))
                        current_max = monthly_avg_aqi.max()
                        if pd.notna(current_max) and current_max > max_aqi_for_radar:
                            max_aqi_for_radar = current_max

                        radar_fig.add_trace(
                            go.Scatterpolar(
                                r=monthly_avg_aqi.values,
                                theta=[months_display_map[m][:3].upper() for m in range(1, 13)],
                                fill="toself",
                                name=city_name,
                                hovertemplate=f"<b>{city_name}</b><br>%{{theta}}: %{{r:.1f}}<extra></extra>",
                                line_color=radar_colors[i % len(radar_colors)],
                                opacity=0.7,
                            )
                        )

                radar_layout_args = get_custom_plotly_layout_args(
                    height=550, title_text="Annual Seasonal AQI Patterns"
                )
                radar_layout_args["polar"] = dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max_aqi_for_radar * 1.15 if max_aqi_for_radar > 0 else 50],
                        gridcolor=BORDER_COLOR,
                        linecolor=BORDER_COLOR,
                        tickfont_size=9,
                    ),
                    angularaxis=dict(
                        linecolor=BORDER_COLOR, gridcolor=BORDER_COLOR, tickfont_size=11
                    ),
                    bgcolor=CARD_BACKGROUND_COLOR,
                )
                radar_layout_args["legend"]["y"] = -0.1
                radar_fig.update_layout(**radar_layout_args)
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.info("Not enough annual data for selected cities to display seasonal radar patterns.")
    else:
        st.info("Not enough data for the selected cities and period to make a trend comparison. Try different filter settings.")
st.markdown("---")


# ------------------- 🏙️ CITY-SPECIFIC DEEP DIVE -------------------
if selected_cities:
    st.markdown("## 🔎 Detailed City View")
    for city in selected_cities:
        st.markdown(
            f"### {city.upper()} – "
            f"{selected_month_display_name if month_number_filter else 'Full Year'}, {year}"
        )

        # If month is selected, show that slice; otherwise use full year for the calendar
        if month_number_filter:
            city_data_for_section = df_period_filtered[df_period_filtered["city"] == city].copy()
            start_date_cal = pd.to_datetime(f"{year}-{month_number_filter:02d}-01")
            end_date_cal = start_date_cal + pd.offsets.MonthEnd(0)
        else:
            city_data_for_section = df[df["city"] == city][df["date"].dt.year == year].copy()
            start_date_cal = pd.to_datetime(f"{year}-01-01")
            end_date_cal = pd.to_datetime(f"{year}-12-31")

        if city_data_for_section.empty:
            st.warning(
                f"😔 No data available for {city} for the selected period. Try different filter settings."
            )
            continue

        with st.container():
            # --- Daily AQI Calendar ---
            full_period_dates_cal = pd.date_range(start_date_cal, end_date_cal)
            calendar_df = pd.DataFrame({"date": full_period_dates_cal})
            calendar_df["week"] = calendar_df["date"].dt.isocalendar().week
            calendar_df["day_of_week"] = calendar_df["date"].dt.dayofweek  # Monday=0, Sunday=6

            if not month_number_filter:
                # If showing full year, adjust weeks for Jan/Dec continuity
                calendar_df.loc[
                    (calendar_df["date"].dt.month == 1) & (calendar_df["week"] > 50), "week"
                ] = 0
                max_week_num = calendar_df["week"].max()
                if max_week_num < 52:
                    max_week_num = 52
                calendar_df.loc[
                    (calendar_df["date"].dt.month == 12) & (calendar_df["week"] == 1), "week"
                ] = max_week_num + 1 if 53 not in calendar_df["week"].unique() else 53

            # Label months by the first week
            month_label_df = calendar_df.copy()
            month_label_df["month_num_cal"] = month_label_df["date"].dt.month
            first_weeks_cal = month_label_df.groupby("month_num_cal")["week"].min()

            merged_cal_df = pd.merge(
                calendar_df,
                city_data_for_section[["date", "index", "level"]],
                on="date",
                how="left",
            )
            merged_cal_df["level"] = merged_cal_df["level"].fillna("Unknown")
            merged_cal_df["aqi_text"] = merged_cal_df["index"].apply(
                lambda x: f"{x:.0f}" if pd.notna(x) else "N/A"
            )

            day_labels_cal = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            sorted_levels_cal = list(CATEGORY_COLORS_DARK.keys())
            colorscale_cal = [CATEGORY_COLORS_DARK[level] for level in sorted_levels_cal]
            level_to_num_cal = {level: i for i, level in enumerate(sorted_levels_cal)}

            fig_cal = go.Figure(
                data=go.Heatmap(
                    x=merged_cal_df["week"],
                    y=merged_cal_df["day_of_week"],
                    z=merged_cal_df["level"].map(level_to_num_cal),
                    customdata=pd.DataFrame(
                        {
                            "date": merged_cal_df["date"].dt.strftime("%b %d, %Y"),
                            "level": merged_cal_df["level"],
                            "aqi": merged_cal_df["aqi_text"],
                        }
                    ),
                    hovertemplate="<b>%{customdata[0]}</b><br>AQI: %{customdata[2]} (%{customdata[1]})<extra></extra>",
                    colorscale=colorscale_cal,
                    showscale=False,
                    xgap=3,
                    ygap=3,
                    zmin=0,
                    zmax=len(sorted_levels_cal) - 1,
                )
            )

            annotations_calendar = []
            if len(calendar_df["week"].unique()) > 3 or not month_number_filter:
                annotations_calendar = [
                    go.layout.Annotation(
                        text=months_display_map[month_num][:3].upper(),
                        align="center",
                        showarrow=False,
                        xref="x domain",
                        yref="paper",
                        x=(
                            first_weeks_cal.get(month_num, 0)
                            - calendar_df["week"].min()
                            + 0.5
                        )
                        / (calendar_df["week"].max() - calendar_df["week"].min() + 1)
                        if (calendar_df["week"].max() - calendar_df["week"].min() + 1) > 0
                        else 0.5,
                        y=1.05,
                        font=dict(color=SUBTLE_TEXT_COLOR_DARK_THEME, size=10),
                    )
                    for month_num in sorted(first_weeks_cal.index)
                ]

            cal_layout = get_custom_plotly_layout_args(
                height=(160 + 20 * len(day_labels_cal))
                if month_number_filter
                else (180 + 20 * len(day_labels_cal)),
                title_text=f"Daily AQI Calendar - {city}",
            )
            cal_layout["yaxis"].update(
                dict(
                    tickmode="array",
                    tickvals=list(range(7)),
                    ticktext=day_labels_cal,
                    autorange="reversed",
                    showgrid=False,
                    zeroline=False,
                )
            )
            cal_layout["xaxis"].update(
                dict(showgrid=False, zeroline=False, ticktext=[], tickvals=[])
            )
            cal_layout["margin"]["t"] = 70 if annotations_calendar else 50
            cal_layout["annotations"] = annotations_calendar
            fig_cal.update_layout(**cal_layout)
            st.plotly_chart(fig_cal, use_container_width=True)

            # Data for tabs: use df_period_filtered for distributions and monthly heatmap
            city_data_for_tabs = df_period_filtered[
      (df_period_filtered["city"] == city)
            ].copy()
            if not city_data_for_tabs.empty:
                city_tabs_list = st.tabs(["📊 AQI Category Distribution", "📅 Monthly AQI Heatmap"])
                with city_tabs_list[0]:
                    city_data_for_tabs["level"] = city_data_for_tabs["level"].fillna("Unknown").str.strip()
                    known_levels_tab = list(CATEGORY_COLORS_DARK.keys())
                    city_data_for_tabs["level"] = city_data_for_tabs["level"].apply(
                        lambda x: x if x in known_levels_tab else "Unknown"
                    )

                    category_counts_df = (
                        city_data_for_tabs["level"]
                        .value_counts()
                        .reindex(known_levels_tab, fill_value=0)
                        .reset_index()
                    )
                    category_counts_df.columns = ["AQI Category", "Number of Days"]

                    if not category_counts_df.empty:
                        fig_dist_bar = px.bar(
                            category_counts_df,
                            x="AQI Category",
                            y="Number of Days",
                            color="AQI Category",
                            color_discrete_map=CATEGORY_COLORS_DARK,
                            text_auto=".0f",
                        )
                        fig_dist_bar.update_layout(
                            **get_custom_plotly_layout_args(
                                height=450, title_text=f"AQI Day Categories - {city}"
                            ),
                            xaxis_title=None,
                            yaxis_title="Number of Days",
                        )
                        fig_dist_bar.update_traces(marker_line_width=0)
                        st.plotly_chart(fig_dist_bar, use_container_width=True)
                    else:
                        st.info(f"No AQI category data to display for {city} in the selected period.")

                with city_tabs_list[1]:
                    if (
                        not city_data_for_tabs.empty
                        and "date" in city_data_for_tabs.columns
                        and "index" in city_data_for_tabs.columns
                    ):
                        city_data_for_tabs["month_name_heatmap"] = pd.Categorical(
                            city_data_for_tabs["date"].dt.strftime("%B"),
                            categories=[months_display_map[i] for i in sorted(city_data_for_tabs["date"].dt.month.unique())],
                            ordered=True,
                        )
                        heatmap_pivot = city_data_for_tabs.pivot_table(
                            index="month_name_heatmap",
                            columns=city_data_for_tabs["date"].dt.day,
                            values="index",
                            observed=True,
                        )

                        if not heatmap_pivot.empty:
                            fig_heat_detail = px.imshow(
                                heatmap_pivot,
                                labels=dict(x="Day of Month", y="Month", color="AQI Index"),
                                aspect="auto",
                                color_continuous_scale=px.colors.sequential.YlOrRd,
                                text_auto=".0f",
                            )
                            heat_layout_args = get_custom_plotly_layout_args(
                                height=max(350, len(heatmap_pivot.index) * 55 + 50),
                                title_text=f"Monthly AQI Heatmap - {city}",
                            )
                            heat_layout_args["xaxis_side"] = "top"
                            heat_layout_args["yaxis_title"] = None
                            heat_layout_args["xaxis"]["showgrid"] = False
                            heat_layout_args["yaxis"]["showgrid"] = False
                            fig_heat_detail.update_layout(**heat_layout_args)
                            st.plotly_chart(fig_heat_detail, use_container_width=True)
                        else:
                            st.info(
                                f"Not enough data to generate monthly AQI heatmap for {city} in the selected period."
                            )
                    else:
                        st.info(f"Monthly AQI heatmap data is incomplete for {city}.")
            else:
                st.info(
                    f"No data for {city} in the selected specific month/period for detailed distribution and monthly heatmap."
                    " The calendar above shows the broader view."
                )

elif selected_cities and df_period_filtered.empty and year:
    st.markdown("## 🔎 Detailed City View")
    st.warning("No data available for the specific month selected. Try 'All Months' or a different period to see detailed city views.")

st.markdown("---")


# ------------------- 🧪 Dominant Pollutant Insights -------------------
st.markdown("## 🧪 Dominant Pollutant Insights")
with st.container():
    if not df_period_filtered.empty:
        st.markdown(f"#### Key Pollutants ({selected_month_display_name}, {year})")

        city_options_pollutant = selected_cities if selected_cities else unique_cities
        default_pollutant_city_index = 0
        if default_city_val and default_city_val[0] in city_options_pollutant:
            default_pollutant_city_index = city_options_pollutant.index(default_city_val[0])

        if city_options_pollutant:
            city_pollutant_B = st.selectbox(
                "Select City for Pollutant Breakdown:",
                city_options_pollutant,
                key="pollutant_city_selectbox_v4",
                index=default_pollutant_city_index,
            )
            pollutant_data_B = df_period_filtered[df_period_filtered["city"] == city_pollutant_B].copy()
            if not pollutant_data_B.empty and "pollutant" in pollutant_data_B.columns and pollutant_data_B["pollutant"].notna().any():
                valid_pollutants = pollutant_data_B[
                    (pollutant_data_B["pollutant"] != "Other") & (pollutant_data_B["pollutant"].notna())
                ].copy()
                if not valid_pollutants.empty:
                    grouped_poll_B = (
                        valid_pollutants.groupby("pollutant").size().reset_index(name="count")
                    )
                    total_valid_pollutant_days = grouped_poll_B["count"].sum()
                    if total_valid_pollutant_days > 0:
                        grouped_poll_B["percentage"] = (
                            grouped_poll_B["count"] / total_valid_pollutant_days * 100
                        )
                        fig_poll_B = px.bar(
                            grouped_poll_B.sort_values("percentage", ascending=False),
                            x="pollutant",
                            y="percentage",
                            color="pollutant",
                            labels={"percentage": "Dominance (% of Days)", "pollutant": "Pollutant"},
                            color_discrete_map=POLLUTANT_COLORS_DARK,
                            text_auto=".1f%%",
                        )
                        fig_poll_B.update_layout(
                            **get_custom_plotly_layout_args(
                                height=500, title_text=f"Pollutant Dominance - {city_pollutant_B}"
                            ),
                            yaxis_ticksuffix="%",
                            xaxis_title=None,
                        )
                        fig_poll_B.update_traces(marker_line_width=0)
                        st.plotly_chart(fig_poll_B, use_container_width=True)
                    else:
                        st.info(
                            f"No specific dominant pollutant data (excluding 'Other' and missing) for {city_pollutant_B} in the selected period."
                        )
                else:
                    st.info(
                        f"No specific dominant pollutant data (excluding 'Other' and missing) for {city_pollutant_B} in the selected period."
                    )
            else:
                st.warning(f"No pollutant data found for {city_pollutant_B} for the selected period.")
        else:
            st.info(
                "No cities available for pollutant analysis with current filters (e.g., if no cities are selected globally or data is empty)."
            )
    else:
        st.info("Select a valid period (Year/Month) to see pollutant analysis.")
st.markdown("---")


# ------------------- Footer -------------------
st.markdown(f"""
<div class="footer">
    <p style="font-size: 1rem; color: {TEXT_COLOR_DARK_THEME}; font-weight: 600;">💨 AuraVision AQI Dashboard</p>
    <p>Conceptualized by: Mr. Kapil Meena & Prof. Arkopal K. Goswami, IIT Kharagpur.</p>
    <p>UI Enhancements & Debugging by Google Gemini ✨</p>
    <p style="margin-top:1.2em;">
        <a href="https://github.com/kapil2020/india-air-quality-dashboard" target="_blank" rel="noopener noreferrer">
            🔗 View on GitHub
        </a>
    </p>
</div>
""", unsafe_allow_html=True)

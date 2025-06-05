import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# --- Global Theme & Style Setup ---
pio.templates.default = "plotly_dark"

ACCENT_COLOR = "#00BCD4"                 # Vibrant Teal (unchanged)
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
    div[data-testid="stExpander"], div[data-testid="stForm"] {{
        border-radius: 12px;
        border: 1px solid {BORDER_COLOR};
        background-color: {CARD_BACKGROUND_COLOR};
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }}
    .stpyplot {{
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
    h3 {{
        font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR_DARK_THEME};
        margin-top: 0rem;
        margin-bottom: 1.2rem;
        font-weight: 600;
    }}
    h4, h5 {{
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
    .stMetric > div:nth-child(1) {{
        font-size: 1rem;
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        font-weight: 500;
    }}
    .stMetric > div:nth-child(2) {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {ACCENT_COLOR};
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
        color: {BACKGROUND_COLOR};
        border: none;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: background-color 0.3s ease;
    }}
    .stDownloadButton button:hover {{
        background-color: {ACCENT_COLOR};
        opacity: 0.85;
    }}

    /* Input Widgets */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div:first-child,
    .stMultiselect div[data-baseweb="select"] > div:first-child {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR_DARK_THEME};
        border-color: {BORDER_COLOR} !important;
    }}
    .stDateInput input {{
         background-color: {BACKGROUND_COLOR};
         color: {TEXT_COLOR_DARK_THEME};
    }}

    /* Custom Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: {BACKGROUND_COLOR};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {BORDER_COLOR};
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {ACCENT_COLOR};
    }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------
#  Helper: Common Plotly layout arguments
# --------------------------------------------
def get_custom_plotly_layout_args(height: int = None, title_text: str = None) -> dict:
    """
    Returns a dict of common Plotly layout arguments for dark-themed charts,
    using the same colors defined above.
    """
    layout_args = {
        "font": {"family": "Inter", "color": TEXT_COLOR_DARK_THEME},
        "paper_bgcolor": CARD_BACKGROUND_COLOR,
        "plot_bgcolor": CARD_BACKGROUND_COLOR,
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    }
    if height:
        layout_args["height"] = height
    if title_text:
        layout_args["title_text"] = title_text
        layout_args["title_font"] = {"color": ACCENT_COLOR, "size": 16, "family": "Inter"}
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
    csv_path = f"data/{today}.csv"
    fallback_file = "combined_air_quality.txt"
    df_loaded = None
    is_today_data = False
    load_msg = ""
    last_update_time = None

    # 1) Attempt to load today's CSV
    if os.path.exists(csv_path):
        try:
            df_loaded = pd.read_csv(csv_path)
            if "date" in df_loaded.columns:
                df_loaded["date"] = pd.to_datetime(df_loaded["date"])
                is_today_data = True
                load_msg = f"Live data from: **{today}.csv**"
                last_update_time = pd.Timestamp(os.path.getmtime(csv_path), unit="s")
            else:
                load_msg = f"Warning: '{csv_path}' found but missing 'date' column. Using fallback."
        except Exception as e:
            load_msg = f"Error loading '{csv_path}': {e}. Using fallback."

    # 2) If today's CSV is missing or invalid, load fallback
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

# ONLY show “Last data update: …” (no “Displaying archive data …”)
if data_last_updated:
    st.caption(
        f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 0.85rem;'>"
        f"Last data update: {data_last_updated.strftime('%Y-%m-%d %H:%M:%S')}"
        "</p>",
        unsafe_allow_html=True
    )


# ------------------- Sidebar Filters -------------------
st.sidebar.header("🔭 EXPLORATION CONTROLS")
st.sidebar.markdown("---", unsafe_allow_html=True)
st.sidebar.info("Fetching real-time data from CPCB, today data is available after 5:45pm.")

unique_cities = sorted(df["city"].unique()) if "city" in df.columns else []
default_city_val = ["Delhi"] if "Delhi" in unique_cities else (unique_cities[0:1] if unique_cities else [])
selected_cities = st.sidebar.multiselect("🏙️ Select Cities", unique_cities, default=default_city_val)

years = sorted(df["date"].dt.year.unique())
default_year_val = max(years) if years else None
if default_year_val:
    year_index = years.index(default_year_val)
else:
    year_index = 0
year = st.sidebar.selectbox("🗓️ Select Year", years, index=year_index if years else 0)

months_map_dict = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
}
month_options_list = ["All Months"] + list(months_map_dict.values())
selected_month_name = st.sidebar.selectbox("🌙 Select Month (Optional)", month_options_list, index=0)

month_number_filter = None
if selected_month_name != "All Months":
    month_number_filter = [k for k, v in months_map_dict.items() if v == selected_month_name][0]

# Filter data based on global selections
df_period_filtered = df[df["date"].dt.year == year].copy()
if month_number_filter:
    df_period_filtered = df_period_filtered[df_period_filtered["date"].dt.month == month_number_filter]


# ========================================================
# =========  NATIONAL KEY INSIGHTS (from app4.py)  =======
# ========================================================
st.markdown("## 🇮🇳 National Snapshot")
with st.container():
    if year:
        # --- Key Metro Annual Average AQI ---
        st.markdown(f"##### Key Metro Annual Average AQI ({year})")
        major_cities = ["Delhi", "Mumbai", "Kolkata", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Ahmedabad"]
        major_cities_annual_data = df[df["date"].dt.year == year]
        major_cities_annual_data = major_cities_annual_data[major_cities_annual_data["city"].isin(major_cities)]

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
                st.info(f"No annual AQI data available for the selected key metro cities in {year}.")
        else:
            st.info(f"No data available for key metro cities in {year}.")

        # --- General Insights ---
        st.markdown(f"##### General Insights for Selected Period ({selected_month_name}, {year})")
        if not df_period_filtered.empty:
            avg_aqi_national = df_period_filtered["index"].mean()
            city_avg_aqi_stats = df_period_filtered.groupby("city")["index"].mean().dropna()

            if not city_avg_aqi_stats.empty:
                num_cities_observed = df_period_filtered["city"].nunique()
                best_city_name, best_city_aqi = city_avg_aqi_stats.idxmin(), city_avg_aqi_stats.min()
                worst_city_name, worst_city_aqi = city_avg_aqi_stats.idxmax(), city_avg_aqi_stats.max()

                st.markdown(
                    f"""<div style="font-size: 1.05rem; line-height: 1.7;">
                    Across <b>{num_cities_observed}</b> observed cities, the average AQI is 
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
    else:
        st.warning("Please select a year to view national insights.")
st.markdown("---")


# ========================================================
# =======   CITY-SPECIFIC ANALYSIS (unchanged)   =========
# ========================================================
export_data_list = []

if not selected_cities:
    st.info("✨ Select one or more cities from the sidebar to dive into detailed analysis.")
else:
    for city in selected_cities:
        st.markdown(f"## 🏙️ {city.upper()} DEEP DIVE – {year}")

        city_data_full = df_period_filtered[df_period_filtered["city"] == city].copy()
        current_filter_period_city = f"{selected_month_name}, {year}"

        if city_data_full.empty:
            with st.container():
                st.warning(f"😔 No data available for {city} for {current_filter_period_city}. Try different filter settings.")
            continue

        city_data_full["day_of_year"] = city_data_full["date"].dt.dayofyear
        city_data_full["month_name"] = city_data_full["date"].dt.month_name()
        city_data_full["day_of_month"] = city_data_full["date"].dt.day
        export_data_list.append(city_data_full)

        tab_trend, tab_dist, tab_heatmap_detail = st.tabs(["📊 TRENDS & CALENDAR", "📈 DISTRIBUTIONS", "🗓️ DETAILED HEATMAP"])

        with tab_trend:
            st.markdown("##### 📅 Daily AQI Calendar")
            # --- Plotly-based Interactive Calendar Heatmap ---
            start_date = pd.to_datetime(f"{year}-01-01")
            end_date = pd.to_datetime(f"{year}-12-31")
            full_year_dates = pd.date_range(start_date, end_date)

            calendar_df = pd.DataFrame({"date": full_year_dates})
            calendar_df["week"] = calendar_df["date"].dt.isocalendar().week
            calendar_df["day_of_week"] = calendar_df["date"].dt.dayofweek

            # Fix week numbering for continuity:
            calendar_df.loc[
                (calendar_df["date"].dt.month == 1) & (calendar_df["week"] > 50), "week"
            ] = 0
            calendar_df.loc[
                (calendar_df["date"].dt.month == 12) & (calendar_df["week"] == 1), "week"
            ] = 53

            merged_cal_df = pd.merge(
                calendar_df,
                city_data_full[["date", "index", "level"]],
                on="date",
                how="left"
            )
            merged_cal_df["level"] = merged_cal_df["level"].fillna("Unknown")
            merged_cal_df["aqi_text"] = merged_cal_df["index"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

            day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            month_starts = calendar_df.drop_duplicates("week").set_index("week")
            month_names = month_starts["date"].dt.strftime("%b")

            fig_cal = go.Figure(
                data=go.Heatmap(
                    x=merged_cal_df["week"],
                    y=merged_cal_df["day_of_week"],
                    z=merged_cal_df["level"].map({k: i for i, k in enumerate(CATEGORY_COLORS_DARK.keys())}),
                    customdata=pd.DataFrame(
                        {
                            "date": merged_cal_df["date"].dt.strftime("%Y-%m-%d"),
                            "level": merged_cal_df["level"],
                            "aqi": merged_cal_df["aqi_text"]
                        }
                    ),
                    hovertemplate="<b>%{customdata[0]}</b><br>AQI: %{customdata[2]} (%{customdata[1]})<extra></extra>",
                    colorscale=[[i / (len(CATEGORY_COLORS_DARK) - 1), color] for i, color in enumerate(CATEGORY_COLORS_DARK.values())],
                    showscale=False,
                    xgap=2, ygap=2
                )
            )

            # Add month annotations
            annotations = []
            for week_num, month_name in month_names.items():
                annotations.append(
                    go.layout.Annotation(
                        text=month_name,
                        align="center",
                        showarrow=False,
                        xref="x", yref="y",
                        x=week_num, y=7,
                        font=dict(color=SUBTLE_TEXT_COLOR_DARK_THEME, size=12)
                    )
                )

            # Add legend manually
            for i, (level, color) in enumerate(CATEGORY_COLORS_DARK.items()):
                annotations.append(
                    go.layout.Annotation(
                        text=f"█ <span style='color:{TEXT_COLOR_DARK_THEME};'>{level}</span>",
                        align="left",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.05 + 0.12 * (i % 7),
                        y=-0.1 - 0.1 * (i // 7),
                        xanchor="left", yanchor="top",
                        font=dict(color=color, size=12)
                    )
                )

            fig_cal.update_layout(
                yaxis=dict(
                    tickmode="array",
                    tickvals=list(range(7)),
                    ticktext=day_labels,
                    showgrid=False, zeroline=False
                ),
                xaxis=dict(showgrid=False, zeroline=False, tickmode="array", ticktext=[], tickvals=[]),
                height=300,
                margin=dict(t=50, b=80, l=40, r=40),
                plot_bgcolor=CARD_BACKGROUND_COLOR,
                paper_bgcolor=CARD_BACKGROUND_COLOR,
                font_color=TEXT_COLOR_DARK_THEME,
                annotations=annotations,
                showlegend=False
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
                    marker=dict(size=4, opacity=0.7, color=SUBTLE_TEXT_COLOR_DARK_THEME),
                    line=dict(width=1.5, color=SUBTLE_TEXT_COLOR_DARK_THEME),
                    customdata=city_data_trend[["level"]],
                    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>AQI: %{y}<br>Category: %{customdata[0]}<extra></extra>"
                )
            )
            fig_trend_roll.add_trace(
                go.Scatter(
                    x=city_data_trend["date"], y=city_data_trend["rolling_avg_7day"],
                    mode="lines", name="7-Day Rolling Avg",
                    line=dict(color=ACCENT_COLOR, width=2.5, dash="dash"),
                    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>7-Day Avg AQI: %{y}<extra></extra>"
                )
            )
            fig_trend_roll.update_layout(
                yaxis_title="AQI Index", xaxis_title="Date", height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified", paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR,
                font_color=TEXT_COLOR_DARK_THEME
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
                    height=400, xaxis_title=None, yaxis_title="Number of Days",
                    paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME
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
                        height=400, margin=dict(t=20, l=20, r=20, b=20),
                        paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME
                    )
                    st.plotly_chart(fig_sunburst, use_container_width=True)
                else:
                    st.caption("No data for sunburst chart.")

            st.markdown("##### 🎻 Monthly AQI Distribution")
            month_order_cat = list(months_map_dict.values())
            city_data_full["month_name"] = pd.Categorical(city_data_full["month_name"], categories=month_order_cat, ordered=True)

            fig_violin = px.violin(
                city_data_full.sort_values("month_name"),
                x="month_name", y="index", color="month_name", color_discrete_sequence=px.colors.qualitative.Vivid,
                box=True, points="outliers",
                labels={"index": "AQI Index", "month_name": "Month"},
                hover_data=["date", "level"]
            )
            fig_violin.update_layout(
                height=450, xaxis_title=None, showlegend=False,
                paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME
            )
            fig_violin.update_traces(meanline_visible=True)
            st.plotly_chart(fig_violin, use_container_width=True)

        with tab_heatmap_detail:
            st.markdown("##### 🔥 AQI Heatmap (Month vs. Day)")
            heatmap_pivot = city_data_full.pivot_table(index="month_name", columns="day_of_month", values="index", observed=False)
            heatmap_pivot = heatmap_pivot.reindex(month_order_cat)

            fig_heat_detail = px.imshow(
                heatmap_pivot, labels=dict(x="Day of Month", y="Month", color="AQI"),
                aspect="auto", color_continuous_scale="Inferno",
                text_auto=".0f"
            )
            fig_heat_detail.update_layout(
                height=500, xaxis_side="top",
                paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME
            )
            fig_heat_detail.update_traces(hovertemplate="<b>Month:</b> %{y}<br><b>Day:</b> %{x}<br><b>AQI:</b> %{z}<extra></extra>")
            st.plotly_chart(fig_heat_detail, use_container_width=True)

# ========================================================
# =======   CITY-WISE AQI COMPARISON  (unchanged)  ========
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
            title_text=f"AQI Trends Comparison – {selected_month_name}, {year}",
            height=500, legend_title_text="City",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        with st.container():
            st.info("Not enough data or cities selected for comparison with current filters.")
st.markdown("---")


# ========================================================
# =======   PROMINENT POLLUTANT ANALYSIS  (unchanged)  =====
# ========================================================
st.markdown("## 💨 PROMINENT POLLUTANT ANALYSIS")
with st.container():
    st.markdown("#### 鑽 Yearly Dominant Pollutant Trends")
    city_pollutant_A = st.selectbox(
        "Select City for Yearly Pollutant Trend:", unique_cities,
        key="pollutant_A_city_dark", index=unique_cities.index(default_city_val[0]) if default_city_val and default_city_val[0] in unique_cities else 0
    )
    pollutant_data_A = df[df["city"] == city_pollutant_A].copy()
    pollutant_data_A["year_label"] = pollutant_data_A["date"].dt.year

    if not pollutant_data_A.empty:
        grouped_poll_A = pollutant_data_A.groupby(["year_label", "pollutant"]).size().unstack(fill_value=0)
        percent_poll_A = grouped_poll_A.apply(lambda x: x / x.sum() * 100 if x.sum() > 0 else x, axis=1).fillna(0)
        percent_poll_A_long = percent_poll_A.reset_index().melt(id_vars="year_label", var_name="pollutant", value_name="percentage")

        fig_poll_A = px.bar(
            percent_poll_A_long, x="year_label", y="percentage", color="pollutant",
            title=f"Dominant Pollutants Over Years – {city_pollutant_A}",
            labels={"percentage": "Percentage of Days (%)", "year_label": "Year", "pollutant": "Pollutant"},
            color_discrete_map=POLLUTANT_COLORS_DARK
        )
        fig_poll_A.update_layout(xaxis_type="category", yaxis_ticksuffix="%", height=500,
                                 paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME)
        fig_poll_A.update_traces(texttemplate="%{value:.1f}%", textposition="auto")
        st.plotly_chart(fig_poll_A, use_container_width=True)
    else:
        st.warning(f"No pollutant data for {city_pollutant_A} (Yearly Trend).")

with st.container():
    st.markdown(f"#### ⛽ Dominant Pollutants for Selected Period ({selected_month_name}, {year})")
    city_pollutant_B = st.selectbox(
        "Select City for Filtered Pollutant View:", unique_cities,
        key="pollutant_B_city_dark", index=unique_cities.index(default_city_val[0]) if default_city_val and default_city_val[0] in unique_cities else 0
    )
    pollutant_data_B = df_period_filtered[df_period_filtered["city"] == city_pollutant_B].copy()
    if not pollutant_data_B.empty and "pollutant" in pollutant_data_B.columns:
        grouped_poll_B = pollutant_data_B.groupby("pollutant").size().reset_index(name="count")
        total_days_B = grouped_poll_B["count"].sum()
        grouped_poll_B["percentage"] = (grouped_poll_B["count"] / total_days_B * 100).round(1) if total_days_B > 0 else 0

        fig_poll_B = px.bar(
            grouped_poll_B, x="pollutant", y="percentage", color="pollutant",
            title=f"Dominant Pollutants – {city_pollutant_B} ({selected_month_name}, {year})",
            labels={"percentage": "Percentage of Days (%)", "pollutant": "Pollutant"},
            color_discrete_map=POLLUTANT_COLORS_DARK
        )
        fig_poll_B.update_layout(yaxis_ticksuffix="%", height=450,
                                 paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME)
        fig_poll_B.update_traces(texttemplate="%{value:.1f}%", textposition="auto")
        st.plotly_chart(fig_poll_B, use_container_width=True)
    else:
        st.warning(f"No pollutant data for {city_pollutant_B} for the selected period.")


# ========================================================
# =======   AQI FORECAST (unchanged)   ====================
# ========================================================
st.markdown("## 🔮 AQI FORECAST (LINEAR TREND)")
with st.container():
    forecast_city_select = st.selectbox(
        "Select City for AQI Forecast:", unique_cities,
        key="forecast_city_select_dark", index=unique_cities.index(default_city_val[0]) if default_city_val and default_city_val[0] in unique_cities else 0
    )
    forecast_src_data = df_period_filtered[df_period_filtered["city"] == forecast_city_select].copy()
    if len(forecast_src_data) >= 15:
        forecast_df = forecast_src_data.sort_values("date")[["date", "index"]].dropna()
        if len(forecast_df) >= 2:
            forecast_df["days_since_start"] = (forecast_df["date"] - forecast_df["date"].min()).dt.days
            X_train, y_train = forecast_df[["days_since_start"]], forecast_df["index"]
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(X_train, y_train)
            last_day_num = forecast_df["days_since_start"].max()
            future_X_range = np.arange(0, last_day_num + 15 + 1)
            future_y_pred = model.predict(pd.DataFrame({"days_since_start": future_X_range}))
            min_date_forecast = forecast_df["date"].min()
            future_dates_list = [min_date_forecast + pd.Timedelta(days=int(i)) for i in future_X_range]

            plot_df_obs = pd.DataFrame({"date": forecast_df["date"], "AQI": y_train})
            plot_df_fcst = pd.DataFrame({"date": future_dates_list, "AQI": np.maximum(0, future_y_pred)})

            fig_forecast = go.Figure()
            fig_forecast.add_trace(
                go.Scatter(x=plot_df_obs["date"], y=plot_df_obs["AQI"], mode="lines+markers", name="Observed AQI", line=dict(color=ACCENT_COLOR))
            )
            fig_forecast.add_trace(
                go.Scatter(x=plot_df_fcst["date"], y=plot_df_fcst["AQI"], mode="lines", name="Forecast", line=dict(dash="dash", color="#FF6B6B"))
            )

            fig_forecast.update_layout(
                title=f"AQI Forecast – {forecast_city_select}", yaxis_title="AQI Index", xaxis_title="Date", height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.warning(f"Not enough valid data points for {forecast_city_select} to forecast.")
    else:
        st.warning(f"Need at least 15 data points for {forecast_city_select} for forecasting; found {len(forecast_src_data)}.")


# ========================================================
# ==========  City AQI Hotspots  (from app4.py)  =========
# ========================================================
st.markdown("## 📍 City AQI Hotspots")
with st.container():
    city_coords_data = {}
    coords_file_path = "lat_long.txt"  # Coordinates file
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
        st.error(f"Map Error: Error processing coordinates file '{coords_file_path}'. Scatter map cannot be displayed. Error: {e_exec}")
        city_coords_data = {}

    if not df_period_filtered.empty:
        map_grouped_data = df_period_filtered.groupby("city").agg(
            avg_aqi=('index', 'mean'),
            dominant_pollutant=('pollutant', lambda x:
                x.mode().iloc[0]
                if not x.mode().empty and x.mode().iloc[0] not in ['Other', 'nan']
                else (x[~x.isin(['Other', 'nan'])].mode().iloc[0]
                      if not x[~x.isin(['Other', 'nan'])].mode().empty else 'N/A')
            )
        ).reset_index().dropna(subset=['avg_aqi'])

        if city_coords_data and not map_grouped_data.empty:
            latlong_map_df_list = []
            for city_name_coord, coords_val in city_coords_data.items():
                if isinstance(coords_val, (list, tuple)) and len(coords_val) == 2:
                    try:
                        lat, lon = float(coords_val[0]), float(coords_val[1])
                        latlong_map_df_list.append({'city': city_name_coord, 'lat': lat, 'lon': lon})
                    except (ValueError, TypeError):
                        pass

            latlong_map_df = pd.DataFrame(latlong_map_df_list)

            if not latlong_map_df.empty:
                map_merged_df = pd.merge(map_grouped_data, latlong_map_df, on="city", how="inner")
                if not map_merged_df.empty:
                    map_merged_df["AQI Category"] = map_merged_df["avg_aqi"].apply(
                        lambda val: next(
                            (k for k, v_range in {
                                'Good': (0, 50), 'Satisfactory': (51, 100), 'Moderate': (101, 200),
                                'Poor': (201, 300), 'Very Poor': (301, 400), 'Severe': (401, float('inf'))
                            }.items() if v_range[0] <= val <= v_range[1]), "Unknown"
                        ) if pd.notna(val) else "Unknown"
                    )

                    fig_scatter_map = px.scatter_mapbox(
                        map_merged_df, lat="lat", lon="lon",
                        size=np.maximum(map_merged_df["avg_aqi"], 1),
                        size_max=25,
                        color="AQI Category",
                        color_discrete_map=CATEGORY_COLORS_DARK,
                        hover_name="city",
                        custom_data=['city', 'avg_aqi', 'dominant_pollutant', 'AQI Category'],
                        text="city",
                        zoom=4.0, center={"lat": 23.0, "lon": 82.5}
                    )

                    scatter_map_layout_args = get_custom_plotly_layout_args(
                        height=700,
                        title_text=f"Average AQI Hotspots - {selected_month_name}, {year}"
                    )
                    scatter_map_layout_args['mapbox_style'] = "carto-darkmatter"
                    scatter_map_layout_args['margin'] = {"r":10,"t":60,"l":10,"b":10}
                    scatter_map_layout_args['legend']['y'] = 0.98
                    scatter_map_layout_args['legend']['x'] = 0.98
                    scatter_map_layout_args['legend']['xanchor'] = 'right'

                    fig_scatter_map.update_traces(
                        marker=dict(sizemin=5, opacity=0.75),
                        hovertemplate="<b style='font-size:1.1em;'>%{customdata[0]}</b><br>" +
                                      "Avg. AQI: %{customdata[1]:.1f} (%{customdata[3]})<br>" +
                                      "Dominant Pollutant: %{customdata[2]}" +
                                      "<extra></extra>"
                    )
                    fig_scatter_map.update_layout(**scatter_map_layout_args)
                    st.plotly_chart(fig_scatter_map, use_container_width=True)
                    scatter_map_rendered = True

        if not scatter_map_rendered:
            if not city_coords_data:
                 st.warning(f"Map Warning: Coordinates file '{coords_file_path}' not found or 'city_coords' variable issue. Scatter map cannot be displayed.")
            elif 'latlong_map_df' in locals() and latlong_map_df.empty:
                 st.warning(f"Map Warning: Coordinates data from '{coords_file_path}' loaded but seems empty or malformed.")
            elif 'map_merged_df' in locals() and map_merged_df.empty:
                 st.info("No city data from your dataset matched with the loaded coordinates for the scatter map.")

            st.markdown("##### City AQI Overview (Map Data Unavailable/Incomplete)")
            if not map_grouped_data.empty:
                avg_aqi_cities_alt = map_grouped_data.sort_values(by='avg_aqi', ascending=True)
                fig_alt_bar = px.bar(
                    avg_aqi_cities_alt.tail(20),
                    x='avg_aqi', y='city', orientation='h',
                    color='avg_aqi', color_continuous_scale=px.colors.sequential.YlOrRd_r,
                    labels={'avg_aqi': 'Average AQI', 'city': 'City'}
                )
                fallback_layout = get_custom_plotly_layout_args(
                    height=max(400, len(avg_aqi_cities_alt.tail(20)) * 25),
                    title_text=f"Top Cities by Average AQI - {selected_month_name}, {year} (Map Fallback)"
                )
                fallback_layout["yaxis"] = {'categoryorder':'total ascending'}
                fig_alt_bar.update_layout(**fallback_layout)
                st.plotly_chart(fig_alt_bar, use_container_width=True)
            else:
                st.warning("No city AQI data available for the selected period to display fallback chart.")
    else:
        st.warning("No air quality data available for the selected filters to display on a map or chart.")
st.markdown("---")


# ========================================================
# ========   DOWNLOAD FILTERED DATA (unchanged)   ========
# ========================================================
if export_data_list:
    st.markdown("## 📥 DOWNLOAD DATA")
    with st.container():
        combined_export_final = pd.concat(export_data_list)
        from io import StringIO
        csv_buffer_final = StringIO()
        combined_export_final.to_csv(csv_buffer_final, index=False)
        st.download_button(
            label="📤 Download Filtered City Data (CSV)",
            data=csv_buffer_final.getvalue(),
            file_name=f"IITKGP_filtered_aqi_{year}_{selected_month_name.replace(' ', '')}.csv",
            mime="text/csv"
        )

# ------------------- Footer -------------------
st.markdown(f"""
<div style="text-align: center; margin-top: 4rem; padding: 1.5rem; background-color: {CARD_BACKGROUND_COLOR}; border-radius: 12px; border: 1px solid {BORDER_COLOR};">
    <p style="margin:0.3em; color: {TEXT_COLOR_DARK_THEME}; font-size:0.9rem;">🌬️ IIT KGP AQI Dashboard</p>
    <p style="margin:0.3em; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size:0.8rem;">Data Source: Central Pollution Control Board (CPCB), India. Coordinates approximate.</p>
    <p style="margin:0.5em 0; color: {TEXT_COLOR_DARK_THEME}; font-size:0.85rem;">Conceptualized by: Mr. Kapil Meena & Prof. Arkopal K. Goswami, IIT Kharagpur.</p>
    <p style="margin:0.3em; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size:0.8rem;">Made with ❤️.</p>
    <p style="margin-top:0.8em;">
        <a href="https://github.com/kapil2020/india-air-quality-dashboard" target="_blank" style="color:{ACCENT_COLOR}; text-decoration:none; font-weight:600;">
            🔗 View on GitHub
        </a>
    </p>
</div>
""", unsafe_allow_html=True)

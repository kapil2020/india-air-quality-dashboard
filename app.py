import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# --- Global Theme & Style Setup ---
pio.templates.default = "plotly_dark"

# Color Palette
ACCENT_COLOR = "#00BCD4"                 # Vibrant Teal
SECONDARY_ACCENT = "#FFD700"             # Muted Gold
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

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="IIT KGP AQI Dashboard",
    page_icon="🌬️",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    body {{
        font-family: 'Inter', sans-serif;
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR_DARK_THEME};
    }}

    /* Main container */
    .main .block-container {{
        padding: 2rem 2.5rem;
    }}

    /* Card styling */
    .stPlotlyChart, .stDataFrame, .stAlert, .stMetric,
    .stDownloadButton > button, .stButton > button,
    div[data-testid="stExpander"], div[data-testid="stForm"] {{
        border-radius: 12px;
        border: 1px solid {BORDER_COLOR};
        background-color: {CARD_BACKGROUND_COLOR};
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease;
    }}
    .stPlotlyChart:hover {{
        transform: translateY(-5px);
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        border-bottom: 2px solid {BORDER_COLOR};
        background-color: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        border-radius: 8px 8px 0 0;
        transition: all 0.3s ease;
    }}
    .stTabs [aria-selected="true"] {{
        border-bottom: 3px solid {ACCENT_COLOR};
        color: {ACCENT_COLOR};
        background-color: {BACKGROUND_COLOR};
    }}

    /* Headings */
    h1 {{
        color: {TEXT_COLOR_DARK_THEME};
        text-align: center;
        font-weight: 700;
        letter-spacing: -1px;
    }}
    h2 {{
        color: {ACCENT_COLOR};
        border-bottom: 2px solid {ACCENT_COLOR};
        padding-bottom: 0.6rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    h3 {{
        color: {TEXT_COLOR_DARK_THEME};
        font-weight: 600;
    }}

    /* Sidebar */
    .stSidebar {{
        background-color: {CARD_BACKGROUND_COLOR};
        border-right: 1px solid {BORDER_COLOR};
        padding: 1.5rem;
    }}
    .stSidebar h2 {{
        color: {ACCENT_COLOR};
        border-bottom: none;
    }}

    /* Buttons */
    .stButton > button, .stDownloadButton > button {{
        background-color: {ACCENT_COLOR};
        color: {BACKGROUND_COLOR};
        border: none;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: {SECONDARY_ACCENT};
        color: {BACKGROUND_COLOR};
    }}

    /* Inputs */
    .stSelectbox div[data-baseweb="select"] > div:first-child,
    .stMultiselect div[data-baseweb="select"] > div:first-child {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR_DARK_THEME};
        border-color: {BORDER_COLOR};
    }}

    /* Scrollbar */
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

    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    .main {{
        animation: fadeIn 0.5s ease-in;
    }}
</style>
""", unsafe_allow_html=True)

# --- Helper Function for Plotly Layouts ---
def get_custom_plotly_layout_args(height=None, title_text=None):
    layout_args = {
        "font": {"family": "Inter", "color": TEXT_COLOR_DARK_THEME},
        "paper_bgcolor": CARD_BACKGROUND_COLOR,
        "plot_bgcolor": BACKGROUND_COLOR,
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    }
    if height:
        layout_args["height"] = height
    if title_text:
        layout_args["title_text"] = title_text
        layout_args["title_font"] = {"color": ACCENT_COLOR, "size": 16}
    return layout_args

# --- Title Section ---
st.markdown("<h1>🌬️ IIT KGP AQI Dashboard</h1>", unsafe_allow_html=True)
st.markdown(f"""
<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 1.1rem; margin-bottom: 2.5rem;'>
    Illuminating Air Quality Insights Across India
</p>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data(ttl=3600)
def load_data_and_metadata():
    today = pd.to_datetime("today").date()
    csv_path = f"data/{today}.csv"
    fallback_file = "combined_air_quality.txt"
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
                load_msg = f"Live data from: **{today}.csv**"
                last_update_time = pd.Timestamp(os.path.getmtime(csv_path), unit="s")
            else:
                load_msg = f"Warning: '{csv_path}' missing 'date' column. Using fallback."
        except Exception as e:
            load_msg = f"Error loading '{csv_path}': {e}. Using fallback."

    if df_loaded is None or not is_today_data:
        try:
            if not os.path.exists(fallback_file):
                st.error(f"FATAL: '{fallback_file}' not found.")
                return pd.DataFrame(), "Error: Main data file not found.", None
            df_loaded = pd.read_csv(fallback_file, sep="\t", parse_dates=["date"])
            base_load_msg = f"Archive data from: **{fallback_file}**"
            load_msg = base_load_msg if not load_msg or is_today_data else load_msg + " " + base_load_msg
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
        .fillna("Other")
    )
    df_loaded["level"] = df_loaded["level"].astype(str).fillna("Unknown")
    return df_loaded, load_msg, last_update_time

df, load_message, data_last_updated = load_data_and_metadata()

if df.empty:
    st.error("Dashboard cannot operate without data. Please check data sources.")
    st.stop()

st.caption(
    f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 0.85rem;'>"
    f"{load_message} | Last update: {data_last_updated.strftime('%Y-%m-%d %H:%M:%S') if data_last_updated else 'N/A'}"
    "</p>",
    unsafe_allow_html=True
)

# --- Sidebar Filters ---
st.sidebar.header("🔭 Exploration Controls")
st.sidebar.markdown("---")
st.sidebar.info("Real-time data from CPCB, available after 5:45 PM daily.")

unique_cities = sorted(df["city"].unique()) if "city" in df.columns else []
default_city_val = ["Delhi"] if "Delhi" in unique_cities else (unique_cities[0:1] if unique_cities else [])
selected_cities = st.sidebar.multiselect("🏙️ Select Cities", unique_cities, default=default_city_val)

years = sorted(df["date"].dt.year.unique())
year = st.sidebar.selectbox("🗓️ Select Year", years, index=len(years)-1 if years else 0)

months_map_dict = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
}
month_options_list = ["All Months"] + list(months_map_dict.values())
selected_month_name = st.sidebar.selectbox("🌙 Select Month (Optional)", month_options_list)

month_number_filter = None
if selected_month_name != "All Months":
    month_number_filter = [k for k, v in months_map_dict.items() if v == selected_month_name][0]

df_period_filtered = df[df["date"].dt.year == year].copy()
if month_number_filter:
    df_period_filtered = df_period_filtered[df_period_filtered["date"].dt.month == month_number_filter]

# --- National Snapshot ---
st.markdown("## 🇮🇳 National Snapshot")
with st.container():
    if year:
        st.markdown(f"##### Key Metro Annual Average AQI ({year})")
        major_cities = ["Delhi", "Mumbai", "Kolkata", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Ahmedabad"]
        major_cities_data = df[df["date"].dt.year == year][df["city"].isin(major_cities)]
        
        if not major_cities_data.empty:
            avg_aqi = major_cities_data.groupby("city")["index"].mean().dropna()
            cols = st.columns(min(len(avg_aqi), 4))
            for i, (city, aqi) in enumerate(avg_aqi.items()):
                with cols[i % len(cols)]:
                    st.metric(label=city, value=f"{aqi:.1f}")
        else:
            st.info(f"No data for key metro cities in {year}.")

        st.markdown(f"##### Insights ({selected_month_name}, {year})")
        if not df_period_filtered.empty:
            avg_aqi_national = df_period_filtered["index"].mean()
            city_stats = df_period_filtered.groupby("city")["index"].mean().dropna()
            if not city_stats.empty:
                st.markdown(
                    f"""<div style='font-size: 1.05rem; line-height: 1.7;'>
                    Across <b>{df_period_filtered['city'].nunique()}</b> cities, avg AQI: 
                    <b style='color:{ACCENT_COLOR};'>{avg_aqi_national:.2f}</b>.
                    Best: <b style='color:{CATEGORY_COLORS_DARK['Good']};'>{city_stats.idxmin()}</b> ({city_stats.min():.2f}).
                    Worst: <b style='color:{CATEGORY_COLORS_DARK['Severe']};'>{city_stats.idxmax()}</b> ({city_stats.max():.2f}).
                    </div>""",
                    unsafe_allow_html=True
                )
st.markdown("---")

# --- City-Specific Analysis ---
export_data_list = []
if not selected_cities:
    st.info("✨ Select cities from the sidebar for detailed analysis.")
else:
    for city in selected_cities:
        st.markdown(f"## 🏙️ {city.upper()} Deep Dive – {year}")
        city_data = df_period_filtered[df_period_filtered["city"] == city].copy()
        
        if city_data.empty:
            st.warning(f"No data for {city} in {selected_month_name}, {year}.")
            continue

        city_data["day_of_year"] = city_data["date"].dt.dayofyear
        city_data["month_name"] = city_data["date"].dt.month_name()
        city_data["day_of_month"] = city_data["date"].dt.day
        export_data_list.append(city_data)

        tab_trend, tab_dist, tab_heatmap = st.tabs(["📊 Trends & Calendar", "📈 Distributions", "🗓️ Heatmap"])

        with tab_trend:
            st.markdown("##### 📅 Daily AQI Calendar")
            start_date = pd.to_datetime(f"{year}-01-01")
            end_date = pd.to_datetime(f"{year}-12-31")
            calendar_df = pd.DataFrame({"date": pd.date_range(start_date, end_date)})
            calendar_df["week"] = calendar_df["date"].dt.isocalendar().week
            calendar_df["day_of_week"] = calendar_df["date"].dt.dayofweek

            calendar_df.loc[(calendar_df["date"].dt.month == 1) & (calendar_df["week"] > 50), "week"] = 0
            calendar_df.loc[(calendar_df["date"].dt.month == 12) & (calendar_df["week"] == 1), "week"] = 53

            merged_cal_df = pd.merge(calendar_df, city_data[["date", "index", "level"]], on="date", how="left")
            merged_cal_df["level"] = merged_cal_df["level"].fillna("Unknown")
            merged_cal_df["aqi_text"] = merged_cal_df["index"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

            fig_cal = go.Figure(
                go.Heatmap(
                    x=merged_cal_df["week"],
                    y=merged_cal_df["day_of_week"],
                    z=merged_cal_df["level"].map({k: i for i, k in enumerate(CATEGORY_COLORS_DARK.keys())}),
                    customdata=pd.DataFrame({
                        "date": merged_cal_df["date"].dt.strftime("%Y-%m-%d"),
                        "level": merged_cal_df["level"],
                        "aqi": merged_cal_df["aqi_text"]
                    }),
                    hovertemplate="<b>%{customdata[0]}</b><br>AQI: %{customdata[2]} (%{customdata[1]})<extra></extra>",
                    colorscale=[[i / (len(CATEGORY_COLORS_DARK) - 1), color] for i, color in enumerate(CATEGORY_COLORS_DARK.values())],
                    showscale=False,
                    xgap=2, ygap=2
                )
            )
            annotations = [
                go.layout.Annotation(
                    text=month_name, x=week_num, y=7, xref="x", yref="y",
                    showarrow=False, font=dict(color=SUBTLE_TEXT_COLOR_DARK_THEME, size=12)
                )
                for week_num, month_name in calendar_df.drop_duplicates("week").set_index("week")["date"].dt.strftime("%b").items()
            ]
            for i, (level, color) in enumerate(CATEGORY_COLORS_DARK.items()):
                annotations.append(
                    go.layout.Annotation(
                        text=f"█ {level}", x=0.05 + 0.12 * (i % 7), y=-0.1 - 0.1 * (i // 7),
                        xref="paper", yref="paper", showarrow=False, font=dict(color=color, size=12)
                    )
                )
            fig_cal.update_layout(
                yaxis=dict(tickmode="array", tickvals=list(range(7)), ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
                xaxis=dict(showgrid=False, zeroline=False, ticktext=[], tickvals=[]),
                height=300, margin=dict(t=50, b=80), annotations=annotations, showlegend=False,
                **get_custom_plotly_layout_args()
            )
            st.plotly_chart(fig_cal, use_container_width=True)

            st.markdown("##### 📈 AQI Trend & Rolling Average")
            city_data["rolling_avg_7day"] = city_data["index"].rolling(window=7, center=True, min_periods=1).mean().round(2)
            fig_trend = go.Figure([
                go.Scatter(x=city_data["date"], y=city_data["index"], mode="lines+markers", name="Daily AQI",
                           marker=dict(size=4, color=SUBTLE_TEXT_COLOR_DARK_THEME),
                           line=dict(width=1.5, color=SUBTLE_TEXT_COLOR_DARK_THEME),
                           hovertemplate="<b>%{x|%Y-%m-%d}</b><br>AQI: %{y}<extra></extra>"),
                go.Scatter(x=city_data["date"], y=city_data["rolling_avg_7day"], mode="lines", name="7-Day Avg",
                           line=dict(color=ACCENT_COLOR, width=2.5, dash="dash"),
                           hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Avg: %{y}<extra></extra>")
            ])
            fig_trend.update_layout(
                yaxis_title="AQI Index", xaxis_title="Date", height=400, hovermode="x unified",
                **get_custom_plotly_layout_args()
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with tab_dist:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 📊 AQI Category (Days)")
                category_counts = city_data["level"].value_counts().reindex(CATEGORY_COLORS_DARK.keys(), fill_value=0).reset_index()
                category_counts.columns = ["Category", "Days"]
                fig_bar = px.bar(category_counts, x="Category", y="Days", color="Category",
                                 color_discrete_map=CATEGORY_COLORS_DARK, text_auto=True)
                fig_bar.update_layout(height=400, xaxis_title=None, yaxis_title="Days",
                                      **get_custom_plotly_layout_args())
                fig_bar.update_traces(textposition="outside")
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                st.markdown("##### ☀️ AQI Proportions")
                if category_counts["Days"].sum() > 0:
                    fig_sun = px.sunburst(category_counts, path=["Category"], values="Days",
                                          color="Category", color_discrete_map=CATEGORY_COLORS_DARK)
                    fig_sun.update_layout(height=400, margin=dict(t=20, l=20, r=20, b=20),
                                          **get_custom_plotly_layout_args())
                    st.plotly_chart(fig_sun, use_container_width=True)

            st.markdown("##### 🎻 Monthly AQI Distribution")
            fig_violin = px.violin(city_data, x="month_name", y="index", box=True, points="outliers",
                                   labels={"index": "AQI", "month_name": "Month"})
            fig_violin.update_layout(height=450, showlegend=False, **get_custom_plotly_layout_args())
            st.plotly_chart(fig_violin, use_container_width=True)

        with tab_heatmap:
            st.markdown("##### 🔥 AQI Heatmap")
            heatmap_pivot = city_data.pivot_table(index="month_name", columns="day_of_month", values="index")
            fig_heat = px.imshow(heatmap_pivot, labels=dict(x="Day", y="Month", color="AQI"),
                                 color_continuous_scale="Inferno", text_auto=".0f")
            fig_heat.update_layout(height=500, xaxis_side="top", **get_custom_plotly_layout_args())
            st.plotly_chart(fig_heat, use_container_width=True)

# --- City Comparison ---
if len(selected_cities) > 1:
    st.markdown("## 🆚 City AQI Comparison")
    combined_df = pd.concat([df_period_filtered[df_period_filtered["city"] == c].assign(city_label=c) for c in selected_cities])
    fig_cmp = px.line(combined_df, x="date", y="index", color="city_label", line_shape="spline",
                      labels={"index": "AQI", "date": "Date", "city_label": "City"})
    fig_cmp.update_layout(
        height=500, **get_custom_plotly_layout_args(title_text=f"AQI Trends – {selected_month_name}, {year}")
    )
    st.plotly_chart(fig_cmp, use_container_width=True)
st.markdown("---")

# --- Pollutant Analysis ---
st.markdown("## 💨 Pollutant Analysis")
with st.container():
    st.markdown("#### Yearly Pollutant Trends")
    city_pollutant = st.selectbox("Select City:", unique_cities, key="pollutant_city")
    pollutant_data = df[df["city"] == city_pollutant].copy()
    pollutant_data["year"] = pollutant_data["date"].dt.year
    grouped_poll = pollutant_data.groupby(["year", "pollutant"]).size().unstack(fill_value=0)
    percent_poll = grouped_poll.apply(lambda x: x / x.sum() * 100, axis=1).fillna(0)
    fig_poll = px.bar(percent_poll.reset_index().melt(id_vars="year"), x="year", y="value", color="pollutant",
                      color_discrete_map=POLLUTANT_COLORS_DARK, title=f"Pollutants – {city_pollutant}")
    fig_poll.update_layout(xaxis_type="category", yaxis_ticksuffix="%", height=500,
                           **get_custom_plotly_layout_args())
    fig_poll.update_traces(texttemplate="%{value:.1f}%", textposition="auto")
    st.plotly_chart(fig_poll, use_container_width=True)

# --- AQI Forecast ---
st.markdown("## 🔮 AQI Forecast")
forecast_city = st.selectbox("Select City:", unique_cities, key="forecast_city")
forecast_data = df_period_filtered[df_period_filtered["city"] == forecast_city].copy()
if len(forecast_data) >= 15:
    forecast_df = forecast_data.sort_values("date")[["date", "index"]].dropna()
    forecast_df["days"] = (forecast_df["date"] - forecast_df["date"].min()).dt.days
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(forecast_df[["days"]], forecast_df["index"])
    future_days = np.arange(0, forecast_df["days"].max() + 16)
    future_preds = model.predict(pd.DataFrame({"days": future_days}))
    future_dates = [forecast_df["date"].min() + pd.Timedelta(days=int(i)) for i in future_days]

    fig_forecast = go.Figure([
        go.Scatter(x=forecast_df["date"], y=forecast_df["index"], mode="lines+markers", name="Observed",
                   line=dict(color=ACCENT_COLOR)),
        go.Scatter(x=future_dates, y=np.maximum(0, future_preds), mode="lines", name="Forecast",
                   line=dict(dash="dash", color=SECONDARY_ACCENT))
    ])
    fig_forecast.update_layout(
        height=450, yaxis_title="AQI", xaxis_title="Date",
        **get_custom_plotly_layout_args(title_text=f"Forecast – {forecast_city}")
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.warning(f"Need at least 15 data points for {forecast_city}.")
st.markdown("---")

# --- City AQI Hotspots ---
st.markdown("## 📍 AQI Hotspots")
city_coords = {}
try:
    with open("lat_long.txt", "r", encoding="utf-8") as f:
        exec(f.read(), {}, {"city_coords": city_coords})
except Exception as e:
    st.error(f"Error loading coordinates: {e}")

if not df_period_filtered.empty and city_coords:
    map_data = df_period_filtered.groupby("city")["index"].mean().reset_index()
    latlong_df = pd.DataFrame([
        {"city": k, "lat": float(v[0]), "lon": float(v[1])}
        for k, v in city_coords.items() if isinstance(v, (list, tuple)) and len(v) == 2
    ])
    map_merged = pd.merge(map_data, latlong_df, on="city", how="inner")
    if not map_merged.empty:
        fig_map = px.scatter_mapbox(
            map_merged, lat="lat", lon="lon", size="index", color="index",
            color_continuous_scale="Viridis", hover_name="city", zoom=4, center={"lat": 23.0, "lon": 82.5}
        )
        fig_map.update_layout(
            mapbox_style="carto-darkmatter", height=700,
            **get_custom_plotly_layout_args(title_text=f"AQI Hotspots – {selected_month_name}, {year}")
        )
        st.plotly_chart(fig_map, use_container_width=True)

# --- Download Data ---
if export_data_list:
    st.markdown("## 📥 Download Data")
    combined_export = pd.concat(export_data_list)
    csv_buffer = combined_export.to_csv(index=False)
    st.download_button(
        label="📤 Download CSV",
        data=csv_buffer,
        file_name=f"IITKGP_aqi_{year}_{selected_month_name.replace(' ', '')}.csv",
        mime="text/csv"
    )

# --- Footer ---
st.markdown(f"""
<div style='text-align: center; margin-top: 4rem; padding: 1.5rem; background-color: {CARD_BACKGROUND_COLOR}; border-radius: 12px; border: 1px solid {BORDER_COLOR};'>
    <p style='color: {TEXT_COLOR_DARK_THEME};'>🌬️ IIT KGP AQI Dashboard</p>
    <p style='color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 0.8rem;'>Data: CPCB, India</p>
    <p style='color: {TEXT_COLOR_DARK_THEME};'>By: Kapil Meena & Prof. Arkopal K. Goswami, IIT Kharagpur</p>
    <a href='https://github.com/kapil2020/india-air-quality-dashboard' target='_blank' style='color:{ACCENT_COLOR};'>🔗 GitHub</a>
</div>
""", unsafe_allow_html=True)

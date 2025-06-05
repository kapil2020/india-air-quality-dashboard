import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# --- Global Theme & Style Setup ---
pio.templates.default = "plotly_dark"

ACCENT_COLOR = "#00BCD4"                 # Vibrant Teal
TEXT_COLOR_DARK_THEME = "#EAEAEA"        # Light gray for text
SUBTLE_TEXT_COLOR_DARK_THEME = "#B0B0B0" # Subtle gray
BACKGROUND_COLOR = "#121212"             # Very dark gray
CARD_BACKGROUND_COLOR = "#1E1E1E"        # Slightly lighter for cards
BORDER_COLOR = "#333333"                 # Dark border
SECONDARY_ACCENT = "#FF7043"             # Vibrant Orange-Red for forecasts

CATEGORY_COLORS_DARK = {
    "Severe": "#F44336", "Very Poor": "#FF7043", "Poor": "#FFA726",
    "Moderate": "#FFEE58", "Satisfactory": "#9CCC65", "Good": "#4CAF50",
    "Unknown": "#444444"
}

POLLUTANT_COLORS_DARK = {
    "PM2.5": "#FF6B6B", "PM10": "#4ECDC4", "NO2": "#45B7D1",
    "SO2": "#F9C74F", "CO": "#F8961E", "O3": "#90BE6D", "Other": "#B0BEC5"
}

# --- Page Config ---
st.set_page_config(layout="wide", page_title="IIT KGP AQI Dashboard", page_icon="🌬️")

# --- Custom CSS for Responsiveness and Design ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    body {{
        font-family: 'Inter', sans-serif;
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR_DARK_THEME};
        margin: 0;
        padding: 0;
    }}

    .main .block-container {{
        padding: 2rem;
    }}

    .stPlotlyChart, .stMetric {{
        border-radius: 12px;
        border: 1px solid {BORDER_COLOR};
        background-color: {CARD_BACKGROUND_COLOR};
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }}

    .stTabs [data-baseweb="tab"] {{
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        transition: background-color 0.3s ease, color 0.3s ease;
    }}
    .stTabs [aria-selected="true"] {{
        border-bottom: 3px solid {ACCENT_COLOR};
        color: {ACCENT_COLOR};
    }}

    h1 {{ color: {TEXT_COLOR_DARK_THEME}; font-weight: 700; text-align: center; }}
    h2 {{ color: {ACCENT_COLOR}; border-bottom: 2px solid {ACCENT_COLOR}; padding-bottom: 0.6rem; }}
    h3 {{ color: {TEXT_COLOR_DARK_THEME}; font-weight: 600; }}

    .stSidebar {{
        background-color: {CARD_BACKGROUND_COLOR};
        border-right: 1px solid {BORDER_COLOR};
        padding: 1.5rem;
    }}

    .stMetric > div:nth-child(2) {{
        font-size: 2rem;
        color: {ACCENT_COLOR};
    }}

    /* Responsive Design */
    @media (max-width: 768px) {{
        .main .block-container {{
            padding: 1rem;
        }}
        .stColumn {{
            width: 100% !important;
            margin-bottom: 1rem;
        }}
        .stPlotlyChart {{
            height: 300px;
        }}
    }}

    @media (min-width: 769px) and (max-width: 1024px) {{
        .stPlotlyChart {{
            height: 400px;
        }}
    }}

    @media (min-width: 1025px) {{
        .stPlotlyChart {{
            height: 500px;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# --- Helper Function ---
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

# --- Hero Section ---
st.markdown(f"""
<div style='background-color: {CARD_BACKGROUND_COLOR}; padding: 2rem; text-align: center; border-radius: 12px; margin-bottom: 2rem;'>
    <h1 style='color: {ACCENT_COLOR};'>🌬️ IIT KGP AQI Dashboard</h1>
    <p style='color: {TEXT_COLOR_DARK_THEME};'>Real-time Air Quality Insights for India</p>
</div>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data(ttl=3600)
def load_data_and_metadata():
    today = pd.to_datetime("today").date()
    csv_path = f"data/{today}.csv"
    fallback_file = "combined_air_quality.txt"
    df_loaded = None
    load_msg = ""
    last_update_time = None

    if os.path.exists(csv_path):
        try:
            df_loaded = pd.read_csv(csv_path)
            df_loaded["date"] = pd.to_datetime(df_loaded["date"])
            load_msg = f"Live data from: **{today}.csv**"
            last_update_time = pd.Timestamp(os.path.getmtime(csv_path), unit="s")
        except Exception as e:
            load_msg = f"Error loading '{csv_path}': {e}. Using fallback."

    if df_loaded is None:
        try:
            df_loaded = pd.read_csv(fallback_file, sep="\t", parse_dates=["date"])
            load_msg = f"Archive data from: **{fallback_file}**"
            last_update_time = pd.Timestamp(os.path.getmtime(fallback_file), unit="s")
        except Exception as e:
            st.error(f"FATAL: Error loading '{fallback_file}': {e}.")
            return pd.DataFrame(), f"Error: {e}", None

    for col, default_val in [("pollutant", np.nan), ("level", "Unknown")]:
        if col not in df_loaded.columns:
            df_loaded[col] = default_val

    df_loaded["pollutant"] = df_loaded["pollutant"].fillna("Other")
    return df_loaded, load_msg, last_update_time

df, load_message, data_last_updated = load_data_and_metadata()

if df.empty:
    st.error("No data available. Please check data sources.")
    st.stop()

st.caption(f"{load_message} | Last update: {data_last_updated.strftime('%Y-%m-%d %H:%M:%S') if data_last_updated else 'N/A'}")

# --- Sidebar Filters ---
st.sidebar.header("🔭 Filters")
unique_cities = sorted(df["city"].unique())
selected_cities = st.sidebar.multiselect("🏙️ Select Cities", unique_cities, default=["Delhi"])

years = sorted(df["date"].dt.year.unique())
year = st.sidebar.selectbox("🗓️ Select Year", years, index=len(years)-1)

months_map_dict = {i+1: m for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])}
month_options = ["All Months"] + list(months_map_dict.values())
selected_month = st.sidebar.selectbox("🌙 Select Month", month_options)

month_filter = [k for k, v in months_map_dict.items() if v == selected_month][0] if selected_month != "All Months" else None
df_period_filtered = df[df["date"].dt.year == year].copy()
if month_filter:
    df_period_filtered = df_period_filtered[df_period_filtered["date"].dt.month == month_filter]

# --- Key Metrics ---
st.markdown("### Key Metrics")
cols = st.columns(3)
with cols[0]:
    st.metric("Average AQI", f"{df_period_filtered['index'].mean():.2f}")
with cols[1]:
    st.metric("Cities with AQI > 100", len(df_period_filtered[df_period_filtered['index'] > 100]['city'].unique()))
with cols[2]:
    st.metric("Total Cities", df_period_filtered['city'].nunique())

# --- National Snapshot ---
st.markdown("## 🇮🇳 National Snapshot")
major_cities = ["Delhi", "Mumbai", "Kolkata", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Ahmedabad"]
major_cities_data = df_period_filtered[df_period_filtered["city"].isin(major_cities)]
if not major_cities_data.empty:
    avg_aqi = major_cities_data.groupby("city")["index"].mean().dropna()
    cols = st.columns(4)
    for i, (city, aqi) in enumerate(avg_aqi.items()):
        with cols[i % 4]:
            st.metric(city, f"{aqi:.1f}")

# --- City-Specific Analysis ---
export_data_list = []
if not selected_cities:
    st.info("✨ Select cities from the sidebar to begin.")
else:
    for city in selected_cities:
        st.markdown(f"## 🏙️ {city.upper()} Analysis – {year}")
        city_data = df_period_filtered[df_period_filtered["city"] == city].copy()
        if city_data.empty:
            st.warning(f"No data for {city}.")
            continue

        city_data["day_of_year"] = city_data["date"].dt.dayofyear
        city_data["month_name"] = city_data["date"].dt.month_name()
        city_data["day_of_month"] = city_data["date"].dt.day
        export_data_list.append(city_data)

        tab_trend, tab_dist, tab_heatmap = st.tabs(["📊 Trends", "📈 Distributions", "🗓️ Heatmap"])

        with tab_trend:
            # Daily AQI Calendar
            st.markdown("##### 📅 Daily AQI Calendar")
            calendar_df = pd.DataFrame({"date": pd.date_range(f"{year}-01-01", f"{year}-12-31")})
            calendar_df["week"] = calendar_df["date"].dt.isocalendar().week
            calendar_df["day_of_week"] = calendar_df["date"].dt.dayofweek
            calendar_df["month"] = calendar_df["date"].dt.month

            calendar_df.loc[(calendar_df["month"] == 1) & (calendar_df["week"] > 50), "week"] = 0
            calendar_df.loc[(calendar_df["month"] == 12) & (calendar_df["week"] == 1), "week"] = 53

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
                    colorscale=list(CATEGORY_COLORS_DARK.values()),
                    showscale=False,
                    xgap=2, ygap=2
                )
            )

            month_week_ranges = calendar_df.groupby("month")["week"].agg(["min", "max"])
            for month_num, (min_week, max_week) in month_week_ranges.iterrows():
                center_week = (min_week + max_week) / 2
                month_name = months_map_dict[month_num]
                fig_cal.add_shape(
                    type="rect",
                    x0=min_week - 0.5, x1=max_week + 0.5,
                    y0=7.2, y1=7.8,
                    fillcolor="rgba(255,255,255,0.1)",
                    line=dict(color="rgba(255,255,255,0.2)"),
                )
                fig_cal.add_annotation(
                    x=center_week, y=7.5,
                    text=month_name,
                    showarrow=False,
                    font=dict(color=TEXT_COLOR_DARK_THEME, size=12),
                )

            annotations = [
                go.layout.Annotation(
                    text=f"█ {level}", x=0.05 + 0.12 * (i % 7), y=-0.1 - 0.1 * (i // 7),
                    xref="paper", yref="paper", showarrow=False, font=dict(color=color, size=12)
                )
                for i, (level, color) in enumerate(CATEGORY_COLORS_DARK.items())
            ]

            fig_cal.update_layout(
                yaxis=dict(
                    tickvals=list(range(7)),
                    ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                    range=[0, 8],
                    showgrid=False,
                    zeroline=False
                ),
                xaxis=dict(showgrid=False, zeroline=False, ticktext=[], tickvals=[]),
                height=350,
                margin=dict(t=50, b=80),
                annotations=annotations,
                **get_custom_plotly_layout_args()
            )
            st.plotly_chart(fig_cal, use_container_width=True)

            # AQI Trend
            st.markdown("##### 📈 AQI Trend")
            city_data["rolling_avg"] = city_data["index"].rolling(window=7, center=True, min_periods=1).mean()
            fig_trend = go.Figure([
                go.Scatter(x=city_data["date"], y=city_data["index"], mode="lines+markers", name="Daily AQI",
                           line=dict(color=SUBTLE_TEXT_COLOR_DARK_THEME)),
                go.Scatter(x=city_data["date"], y=city_data["rolling_avg"], mode="lines", name="7-Day Avg",
                           line=dict(color=ACCENT_COLOR, dash="dash"))
            ])
            fig_trend.update_layout(yaxis_title="AQI", xaxis_title="Date", **get_custom_plotly_layout_args())
            st.plotly_chart(fig_trend, use_container_width=True)

        with tab_dist:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 📊 AQI Categories")
                category_counts = city_data["level"].value_counts().reindex(CATEGORY_COLORS_DARK.keys(), fill_value=0)
                fig_bar = px.bar(category_counts.reset_index(), x="index", y="level", color="index",
                                 color_discrete_map=CATEGORY_COLORS_DARK, text_auto=True)
                fig_bar.update_layout(xaxis_title="Category", yaxis_title="Days", **get_custom_plotly_layout_args())
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                st.markdown("##### 💨 Dominant Pollutants")
                pollutant_counts = city_data["pollutant"].value_counts().reset_index()
                fig_pie = px.pie(pollutant_counts, names="index", values="pollutant",
                                 color="index", color_discrete_map=POLLUTANT_COLORS_DARK)
                fig_pie.update_layout(**get_custom_plotly_layout_args())
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("##### 🎻 Monthly AQI Distribution")
            fig_violin = px.violin(city_data, x="month_name", y="index", box=True, points="outliers")
            fig_violin.update_layout(**get_custom_plotly_layout_args())
            st.plotly_chart(fig_violin, use_container_width=True)

        with tab_heatmap:
            st.markdown("##### 🔥 AQI Heatmap")
            heatmap_pivot = city_data.pivot_table(index="month_name", columns="day_of_month", values="index")
            fig_heat = px.imshow(heatmap_pivot, color_continuous_scale="Inferno", text_auto=".0f")
            fig_heat.update_layout(xaxis_side="top", **get_custom_plotly_layout_args())
            st.plotly_chart(fig_heat, use_container_width=True)

# --- City Comparison ---
if len(selected_cities) > 1:
    st.markdown("## 🆚 City Comparison")
    combined_df = df_period_filtered[df_period_filtered["city"].isin(selected_cities)]
    fig_cmp = px.line(combined_df, x="date", y="index", color="city", line_shape="spline")
    fig_cmp.update_layout(**get_custom_plotly_layout_args(title_text="AQI Trends Comparison"))
    st.plotly_chart(fig_cmp, use_container_width=True)

# --- Pollutant Analysis ---
st.markdown("## 💨 Pollutant Analysis")
city_pollutant = st.selectbox("Select City for Pollutants:", unique_cities, index=unique_cities.index("Delhi"))
pollutant_data = df[df["city"] == city_pollutant].copy()
pollutant_data["year"] = pollutant_data["date"].dt.year
grouped_poll = pollutant_data.groupby(["year", "pollutant"]).size().unstack(fill_value=0)
percent_poll = grouped_poll.apply(lambda x: x / x.sum() * 100, axis=1)
fig_poll = px.bar(percent_poll.reset_index().melt(id_vars="year"), x="year", y="value", color="pollutant",
                  color_discrete_map=POLLUTANT_COLORS_DARK)
fig_poll.update_layout(yaxis_ticksuffix="%", xaxis_type="category", **get_custom_plotly_layout_args())
st.plotly_chart(fig_poll, use_container_width=True)

# --- AQI Forecast ---
st.markdown("## 🔮 AQI Forecast")
forecast_city = st.selectbox("Select City for Forecast:", unique_cities, index=unique_cities.index("Delhi"))
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
    fig_forecast.update_layout(**get_custom_plotly_layout_args(title_text=f"Forecast for {forecast_city}"))
    st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.warning("Insufficient data for forecast.")

# --- AQI Hotspots ---
st.markdown("## 📍 AQI Hotspots")
city_coords = {}
coords_file = "lat_long.txt"
if not os.path.exists(coords_file):
    st.warning("Coordinates file not found. Map unavailable.")
else:
    try:
        with open(coords_file, "r", encoding="utf-8") as f:
            exec(f.read(), {}, {"city_coords": city_coords})
        map_data = df_period_filtered.groupby("city")["index"].mean().reset_index()
        latlong_df = pd.DataFrame([
            {"city": k, "lat": float(v[0]), "lon": float(v[1])}
            for k, v in city_coords.items() if len(v) == 2
        ])
        map_merged = pd.merge(map_data, latlong_df, on="city", how="inner")
        if not map_merged.empty:
            fig_map = px.scatter_mapbox(
                map_merged, lat="lat", lon="lon", size="index", color="index",
                color_continuous_scale="Viridis", hover_name="city", zoom=4
            )
            fig_map.update_layout(mapbox_style="carto-darkmatter", **get_custom_plotly_layout_args())
            st.plotly_chart(fig_map, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading map: {e}")

# --- Download Data ---
if export_data_list:
    st.markdown("## 📥 Download")
    combined_export = pd.concat(export_data_list)
    st.download_button(
        label="Download CSV",
        data=combined_export.to_csv(index=False),
        file_name=f"aqi_{year}_{selected_month}.csv",
        mime="text/csv"
    )

# --- Footer ---
st.markdown(f"""
<div style='text-align: center; padding: 1.5rem; background-color: {CARD_BACKGROUND_COLOR}; border-radius: 12px;'>
    <p style='color: {TEXT_COLOR_DARK_THEME};'>IIT KGP AQI Dashboard | Data: CPCB</p>
    <p style='color: {SUBTLE_TEXT_COLOR_DARK_THEME};'>By Kapil Meena & Prof. Arkopal K. Goswami</p>
</div>
""", unsafe_allow_html=True)

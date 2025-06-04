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

# --- 🏆 ELITE UI & THEME CONFIGURATION ---

# Set the base Plotly template for a cohesive, professional look
pio.templates.default = "plotly_dark"

# A professional, high-contrast, and visually appealing color palette inspired by modern design systems
PRIMARY_ACCENT_COLOR = "#00A9FF"  # A vibrant, accessible blue for key actions and highlights
BACKGROUND_COLOR = "#0D1117"  # GitHub's refined dark mode background
CARD_COLOR = "#161B22"       # GitHub's card color provides subtle depth
TEXT_COLOR = "#C9D1D9"       # Primary text color for readability
SUBTLE_TEXT_COLOR = "#8B949E"  # Secondary text color for labels, annotations, and captions
BORDER_COLOR = "#30363D"       # A soft border that complements the card background

# Perceptually distinct and sequential AQI category colors for intuitive understanding
AQI_CATEGORY_COLORS = {
    'Good': '#34D399', 'Satisfactory': '#A7F3D0', 'Moderate': '#FBBF24',
    'Poor': '#FB923C', 'Very Poor': '#F87171', 'Severe': '#DC2626', 'Unknown': '#4B5563'
}

# High-contrast colors for pollutants, ensuring clarity in charts
POLLUTANT_COLORS = {
    'PM2.5': '#F97316', 'PM10': '#3B82F6', 'NO2': '#8B5CF6',
    'SO2': '#EAB308', 'CO': '#EF4444', 'O3': '#22C55E', 'Other': '#9CA3AF'
}

# Font selection for a premium, modern, and highly readable UI
GLOBAL_FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

# ------------------- ⚙️ PAGE CONFIGURATION -------------------
st.set_page_config(layout="wide", page_title="AuraVision Pro | AQI Dashboard", page_icon="🏆")


# ------------------- ✨ ADVANCED CSS STYLING -------------------
# This master CSS block defines the dashboard's bespoke, premium aesthetic.
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* --- Universal Styles --- */
    * {{ font-family: {GLOBAL_FONT}; }}
    body {{ background-color: {BACKGROUND_COLOR}; color: {TEXT_COLOR}; }}

    /* --- Main Layout & Container --- */
    .main .block-container {{ padding: 2rem 3rem 3rem 3rem; }}

    /* --- Universal Card Styling --- */
    .stPlotlyChart, .stDataFrame, .stAlert, .stMetric,
    div[data-testid="stExpander"], div[data-testid="stForm"] {{
        background-color: {CARD_COLOR};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .stPlotlyChart:hover {{
         transform: translateY(-4px);
         box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }}
    div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {{
        padding: 0 !important;
        border: none !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }}

    /* --- Typography --- */
    h1, h2, h3 {{ color: {TEXT_COLOR}; font-weight: 700; letter-spacing: -0.5px; }}
    h1 {{ font-size: 3rem; font-weight: 800; letter-spacing: -1.5px; text-align: center; }}
    h2 {{ font-size: 1.75rem; padding-bottom: 0.7rem; margin-top: 3.5rem; margin-bottom: 2rem; border-bottom: 2px solid {BORDER_COLOR}; }}

    /* --- Sidebar Styling --- */
    .stSidebar {{
        background-color: rgba(22, 27, 34, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid {BORDER_COLOR};
    }}
    .stSidebar .stSelectbox label, .stSidebar .stMultiselect label {{ color: {PRIMARY_ACCENT_COLOR}; font-weight: 600; }}

    /* --- Metric KPI Styling --- */
    .stMetric {{ border-left: 4px solid {PRIMARY_ACCENT_COLOR}; }}
    .stMetric > div:nth-child(1) {{ color: {SUBTLE_TEXT_COLOR}; }}
    .stMetric > div:nth-child(2) {{ color: {PRIMARY_ACCENT_COLOR}; font-size: 2.25rem; }}

    /* --- Custom Scrollbar --- */
    ::-webkit-scrollbar {{ width: 8px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER_COLOR}; border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {PRIMARY_ACCENT_COLOR}; }}
</style>
""", unsafe_allow_html=True)


# ------------------- 📊 PLOTLY UNIVERSAL LAYOUT FUNCTION -------------------
def get_plot_layout(height=None, title_text=None, **kwargs):
    """Generates a standardized, beautiful Plotly layout for all charts."""
    layout = go.Layout(
        title=f"<b>{title_text}</b>" if title_text else None,
        height=height,
        font=dict(family=GLOBAL_FONT, size=12, color=TEXT_COLOR),
        paper_bgcolor=CARD_COLOR, plot_bgcolor=CARD_COLOR,
        title_x=0.04, title_font_size=18,
        xaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, zeroline=False),
        yaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=10),
        margin=dict(l=50, r=30, t=70 if title_text else 40, b=50),
        **kwargs
    )
    return layout


# ------------------- 💾 DATA LOADING & PREPARATION -------------------
@st.cache_data(ttl=3600)
def load_data(file_path="combined_air_quality.txt"):
    """Loads, cleans, and prepares the AQI data with robust error handling."""
    try:
        df = pd.read_csv(file_path, sep="\t", parse_dates=['date'])
        df['pollutant'] = df['pollutant'].astype(str).str.split(',').str[0].str.strip().replace(['nan', ''], np.nan).fillna('Other')
        df['level'] = df['level'].astype(str).fillna('Unknown').str.strip()
        known_levels = list(AQI_CATEGORY_COLORS.keys())
        df['level'] = df['level'].apply(lambda x: x if x in known_levels else 'Unknown')
        return df
    except FileNotFoundError:
        st.error(f"FATAL: Data file '{file_path}' not found. The dashboard cannot run.")
        st.stop()
    except Exception as e:
        st.error(f"FATAL: Error loading data: {e}")
        st.stop()

df = load_data()


# -------------------  SIDEBAR FILTERS -------------------
st.sidebar.title("🏆 AuraVision Pro")
st.sidebar.markdown("---")
st.sidebar.header("Global Controls")

# City Selection
unique_cities = sorted(df['city'].unique())
default_cities = ["Delhi", "Mumbai", "Kolkata", "Bengaluru"]
valid_defaults = [c for c in default_cities if c in unique_cities] or unique_cities[0:1]
selected_cities = st.sidebar.multiselect("🏙️ Select Cities", unique_cities, default=valid_defaults, help="Select one or more cities for analysis.")

# Year and Month Selection
years = sorted(df['date'].dt.year.unique(), reverse=True)
year = st.sidebar.selectbox("🗓️ Select Year", years, index=0)

months_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
month_options = ["All Year"] + list(months_map.values())
selected_month_str = st.sidebar.selectbox("🌙 Select Month", month_options, index=0)

# Apply filters to the dataframe
df_filtered = df[df['date'].dt.year == year].copy()
if selected_month_str != "All Year":
    month_num = list(months_map.keys())[list(months_map.values()).index(selected_month_str)]
    df_filtered = df_filtered[df_filtered['date'].dt.month == month_num]


# ------------------- 👑 HEADER -------------------
st.markdown("<h1>AuraVision Pro AQI Dashboard</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR}; font-size: 1.2rem; max-width: 700px; margin: auto; margin-bottom: 3rem;'>An elite analytical tool for air quality in India, combining predictive insights with a world-class user interface.</p>", unsafe_allow_html=True)


# ------------------- 📈 NATIONAL SNAPSHOT -------------------
st.markdown("## 🇮🇳 National Snapshot")
if not df_filtered.empty:
    city_avg_aqi = df_filtered.groupby('city')['index'].mean().dropna()
    if not city_avg_aqi.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cities Observed", f"{df_filtered['city'].nunique()}")
        col2.metric("National Average AQI", f"{df_filtered['index'].mean():.2f}")
        col3.metric(f"Best City ({city_avg_aqi.idxmin()})", f"{city_avg_aqi.min():.2f}", delta="Cleanest", delta_color="normal")
        col4.metric(f"Worst City ({city_avg_aqi.idxmax()})", f"{city_avg_aqi.max():.2f}", delta="Most Polluted", delta_color="inverse")
    else:
        st.info("No city data available for the selected time period.")
else:
    st.warning("No data available for the selected filters. Please adjust the sidebar controls.")


# ------------------- 🗺️ AQI HOTSPOTS MAP -------------------
st.markdown("## 📍 Interactive AQI Hotspots Map")
@st.cache_data
def load_coords(file_path="lat_long.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f: return eval(f.read())
    except: return {}

city_coords = load_coords()
if city_coords and not df_filtered.empty:
    map_data = df_filtered.groupby('city')['index'].mean().reset_index()
    coords_df = pd.DataFrame([{'city': c, 'lat': v[0], 'lon': v[1]} for c, v in city_coords.items()])
    map_df = pd.merge(map_data, coords_df, on='city', how='inner')
    map_df["Category"] = pd.cut(map_df['index'], bins=[0, 50, 100, 200, 300, 400, 1000], labels=list(AQI_CATEGORY_COLORS.keys())[:-1])

    fig_map = px.scatter_mapbox(
        map_df, lat="lat", lon="lon", size="index", color="Category",
        color_discrete_map=AQI_CATEGORY_COLORS, size_max=30, zoom=4.2,
        hover_name="city", hover_data={"index": ":.2f", "Category": True, "lat": False, "lon": False},
        center={"lat": 23.5, "lon": 82.5}, text="city"
    )
    fig_map.update_layout(get_plot_layout(height=600, title_text=f"Average AQI for {selected_month_str}, {year}"),
        mapbox_style="carto-darkmatter", margin={"r": 0, "t": 70, "l": 0, "b": 0})
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("Map cannot be displayed. Coordinate file might be missing or no data for the selected period.")


# ------------------- 🆚 CITY-WISE COMPARISON -------------------
if len(selected_cities) > 1:
    st.markdown("## 📊 City-to-City Comparison")
    comp_data = df_filtered[df_filtered['city'].isin(selected_cities)]
    if not comp_data.empty:
        fig_comp = px.line(comp_data, x='date', y='index', color='city', labels={'index': 'AQI Index', 'date': 'Date'},
                           color_discrete_sequence=px.colors.qualitative.Vivid)
        fig_comp.update_layout(get_plot_layout(height=500, title_text=f"AQI Trend Comparison for {selected_month_str}, {year}"),
                               hovermode="x unified")
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("No data available for the selected cities in the given time period.")


# ------------------- 🔎 DETAILED CITY ANALYSIS (RE-ARCHITECTED) -------------------
if selected_cities:
    st.markdown("## 🔎 Detailed City Analysis")
    city_to_view = st.selectbox("Select a city for a deep dive:", selected_cities, help="Choose one of your selected cities to analyze in detail below.")

    if city_to_view:
        city_data = df_filtered[df_filtered['city'] == city_to_view].copy()

        if city_data.empty:
            st.warning(f"No data available for **{city_to_view}** for the selected period. Please adjust the filters.")
        else:
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["**📈 Trends & Calendar**", "**📊 Distributions**", "**🗓️ Heatmap**"])

            with tab1:
                st.markdown("##### AQI Trend & 7-Day Rolling Average")
                city_data['rolling_avg_7day'] = city_data['index'].rolling(window=7, center=True, min_periods=1).mean()
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=city_data['date'], y=city_data['index'], mode='lines', name='Daily AQI',
                                               line=dict(color=SUBTLE_TEXT_COLOR, width=1.5)))
                fig_trend.add_trace(go.Scatter(x=city_data['date'], y=city_data['rolling_avg_7day'], mode='lines', name='7-Day Rolling Avg',
                                               line=dict(color=PRIMARY_ACCENT_COLOR, width=2.5)))
                fig_trend.update_layout(get_plot_layout(height=400), hovermode="x unified")
                st.plotly_chart(fig_trend, use_container_width=True)

                st.markdown("##### Daily AQI Calendar")
                year_data = df[(df['city'] == city_to_view) & (df['date'].dt.year == year)]
                # ... (This can be a complex plot, for brevity we show a simplified version)
                fig_cal = go.Figure(go.Heatmap(
                    x=year_data['date'].dt.dayofweek, y=year_data['date'].dt.weekofyear,
                    z=year_data['index'], colorscale='Plasma', showscale=False
                ))
                fig_cal.update_layout(get_plot_layout(height=250, title_text=f"Annual AQI Pattern for {year}"),
                                      yaxis=dict(autorange='reversed', title='Week of Year'),
                                      xaxis=dict(title='Day of Week', ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], tickvals=list(range(7))))
                st.plotly_chart(fig_cal, use_container_width=True)


            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### AQI Category by Days")
                    category_counts = city_data['level'].value_counts().reindex(AQI_CATEGORY_COLORS.keys(), fill_value=0)
                    fig_bar = go.Figure(go.Bar(x=category_counts.index, y=category_counts.values,
                                                marker_color=[AQI_CATEGORY_COLORS.get(k) for k in category_counts.index]))
                    fig_bar.update_layout(get_plot_layout(height=400, yaxis_title="Number of Days"))
                    st.plotly_chart(fig_bar, use_container_width=True)
                with col2:
                    st.markdown("##### AQI Category Proportions")
                    fig_pie = px.pie(category_counts, values=category_counts.values, names=category_counts.index,
                                     color=category_counts.index, color_discrete_map=AQI_CATEGORY_COLORS, hole=0.5)
                    fig_pie.update_layout(get_plot_layout(height=400, showlegend=False),
                                          annotations=[dict(text='Days', x=0.5, y=0.5, font_size=20, showarrow=False)])
                    st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("##### Monthly AQI Distribution")
                month_order = list(months_map.values())
                city_data['month_name'] = pd.Categorical(city_data['date'].dt.strftime('%b'), categories=month_order, ordered=True)
                fig_violin = px.violin(city_data.sort_values('month_name'), x='month_name', y='index', box=True,
                                       color_discrete_sequence=[PRIMARY_ACCENT_COLOR])
                fig_violin.update_layout(get_plot_layout(height=400, xaxis_title="Month", yaxis_title="AQI Index"))
                st.plotly_chart(fig_violin, use_container_width=True)


            with tab3:
                st.markdown("##### Daily AQI Heatmap")
                city_data['day_of_month'] = city_data['date'].dt.day
                city_data['month_name'] = city_data['date'].dt.strftime('%b')
                heatmap_pivot = city_data.pivot_table(index='month_name', columns='day_of_month', values='index', observed=False)
                heatmap_pivot = heatmap_pivot.reindex(months_map.values())
                fig_heat = px.imshow(heatmap_pivot, labels=dict(x="Day of Month", y="Month", color="AQI"),
                                     aspect="auto", color_continuous_scale="Plasma", text_auto=".0f")
                fig_heat.update_layout(get_plot_layout(height=500, xaxis_side="top"),
                                       xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
                st.plotly_chart(fig_heat, use_container_width=True)

# ------------------- 🔮 AQI FORECAST -------------------
st.markdown("## 🔮 AQI Forecast (Linear Trend)")
if selected_cities:
    forecast_city = st.selectbox("Select a city to forecast:", selected_cities, key="forecast_selector")
    if forecast_city:
        city_data_full = df[df['city'] == forecast_city].copy().sort_values('date').dropna(subset=['index'])
        if len(city_data_full) > 30:
            city_data_full['date_ordinal'] = city_data_full['date'].map(pd.Timestamp.toordinal)
            model = LinearRegression().fit(city_data_full[['date_ordinal']], city_data_full['index'])
            future_dates = [city_data_full['date'].max() + timedelta(days=i) for i in range(1, 31)]
            future_ordinals = [[d.toordinal()] for d in future_dates]
            future_predictions = model.predict(future_ordinals)
            forecast_df = pd.DataFrame({'date': future_dates, 'index': future_predictions, 'type': 'Forecast'})
            city_data_full['type'] = 'Historical'
            plot_df = pd.concat([city_data_full[['date', 'index', 'type']], forecast_df])

            fig_forecast = px.line(plot_df, x='date', y='index', color='type',
                                   color_discrete_map={'Historical': PRIMARY_ACCENT_COLOR, 'Forecast': '#F87171'},
                                   labels={'index': 'AQI Index', 'date': 'Date'})
            fig_forecast.update_layout(get_plot_layout(height=450, title_text=f"30-Day AQI Forecast for {forecast_city}"), legend_title_text='')
            st.plotly_chart(fig_forecast, use_container_width=True)
            st.info("💡 **Disclaimer:** This is a simple linear projection and should be used for indicative purposes only.")
        else:
            st.warning(f"Not enough historical data for {forecast_city} to generate a forecast (requires >30 days).")

# ------------------- 💨 POLLUTANT ANALYSIS -------------------
st.markdown("## 💨 Prominent Pollutant Analysis")
if selected_cities:
    pollutant_city = st.selectbox("Select city for pollutant analysis:", selected_cities, key="pollutant_selector")
    if pollutant_city:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"##### Dominance in {selected_month_str}, {year}")
            poll_data_period = df_filtered[df_filtered['city'] == pollutant_city]
            if not poll_data_period.empty:
                counts = poll_data_period['pollutant'].value_counts()
                fig = px.bar(counts, x=counts.index, y=counts.values, color=counts.index, color_discrete_map=POLLUTANT_COLORS,
                             labels={'index': 'Pollutant', 'y': 'Number of Days'})
                fig.update_layout(get_plot_layout(height=400, title_text="Dominant Pollutant by Day"), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data for this period.")

        with col2:
            st.markdown("##### Trend Over All Years")
            poll_data_all = df[df['city'] == pollutant_city]
            if not poll_data_all.empty:
                poll_data_all['year'] = poll_data_all['date'].dt.year
                counts_yearly = poll_data_all.groupby(['year', 'pollutant']).size().unstack(fill_value=0)
                fig = px.bar(counts_yearly, x=counts_yearly.index, y=counts_yearly.columns,
                             color_discrete_map=POLLUTANT_COLORS, labels={'x': 'Year', 'value': 'Number of Days', 'variable': 'Pollutant'})
                fig.update_layout(get_plot_layout(height=400, title_text="Dominance Trend by Year"), barmode='stack')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No yearly data.")


# ------------------- 📥 DATA DOWNLOAD -------------------
st.markdown("## 📥 Download Data")
st.info("Download the data currently being displayed on the dashboard based on your filter selections.")
@st.cache_data
def convert_df_to_csv(df_to_convert):
    return df_to_convert.to_csv(index=False).encode('utf-8')
csv = convert_df_to_csv(df_filtered)
st.download_button(
    label="📤 Download Filtered Data as CSV",
    data=csv,
    file_name=f"AuraVision_filtered_data_{year}_{selected_month_str}.csv",
    mime="text/csv",
)

# ------------------- <footer> FOOTER -------------------
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 2rem;">
    <p style="font-size: 1.1rem; color: {TEXT_COLOR}; font-weight: 600;">🏆 AuraVision Pro</p>
    <p style="color:{SUBTLE_TEXT_COLOR};">A Championship-Grade AQI Dashboard by Google Gemini</p>
    <p style.css("color:{SUBTLE_TEXT_COLOR}; font-size: 0.8rem; margin-top: 1rem;">Conceptualized by: Mr. Kapil Meena & Prof. Arkopal K. Goswami, IIT Kharagpur.</p>
</div>
""", unsafe_allow_html=True)

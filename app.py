import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
        padding: 1.5rem; /* Increased padding */
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2); /* Softer shadow for dark theme */
    }}
    .stpyplot {{ /* Matplotlib specific styling */
        border-radius: 12px;
        border: 1px solid {BORDER_COLOR};
        background-color: {CARD_BACKGROUND_COLOR}; /* Match card background */
        padding: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }}

    .stTabs [data-baseweb="tab-list"] {{
         box-shadow: none;
         border-bottom: 2px solid {BORDER_COLOR};
         padding-bottom: 0;
         background-color: transparent; /* Ensure tabs blend */
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        background-color: transparent;
        border-radius: 8px 8px 0 0; /* Rounded top corners */
        margin-right: 0.5rem;
        transition: background-color 0.3s ease, color 0.3s ease;
    }}
     .stTabs [aria-selected="true"] {{
        border-bottom: 3px solid {ACCENT_COLOR};
        color: {ACCENT_COLOR};
        background-color: {BACKGROUND_COLOR}; /* Slightly different bg for active tab */
     }}

    /* Headings */
    h1 {{
        font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR_DARK_THEME};
        text-align: center;
        margin-bottom: 1rem; /* Reduced margin */
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
        text-transform: uppercase; /* Uppercase for distinct sections */
        letter-spacing: 0.5px;
    }}
    h3 {{ /* City names or sub-section titles */
        font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR_DARK_THEME};
        margin-top: 0rem;
        margin-bottom: 1.2rem;
        font-weight: 600;
    }}
    h4, h5 {{ /* Plot titles inside cards */
        font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR_DARK_THEME};
        margin-top: 0.2rem;
        margin-bottom: 1rem;
        font-weight: 500; /* Lighter weight for plot titles */
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
        color: {ACCENT_COLOR} !important; /* Make filter labels pop */
        font-weight: 600;
    }}

    /* Metric styling */
    .stMetric {{
        background-color: {BACKGROUND_COLOR}; /* Slightly darker for emphasis */
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

    /* Expander styling */
    div[data-testid="stExpander"] summary {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {ACCENT_COLOR};
    }}
    div[data-testid="stExpander"] svg {{ /* Chevron icon color */
        fill: {ACCENT_COLOR};
    }}

    /* Specific for download button */
    .stDownloadButton button {{
        background-color: {ACCENT_COLOR};
        color: {BACKGROUND_COLOR}; /* Dark text on bright button */
        border: none;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: background-color 0.3s ease;
    }}
    .stDownloadButton button:hover {{
        background-color: {ACCENT_COLOR}; /* Keep same, or slightly lighter variant of accent */
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

    /* Custom Scrollbar for Webkit browsers */
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

# ------------------- Title -------------------
st.markdown("<h1 style='text-align: center; margin-bottom:0.5rem;'>🌬️ AuraVision AQI</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 1.1rem; margin-bottom: 2.5rem;'>Illuminating Air Quality Insights Across India</p>", unsafe_allow_html=True)


# ------------------- Load Data -------------------
@st.cache_data(ttl=3600)
def load_data_and_metadata():
    with st.spinner(f"🌬️ Initializing AuraVision Engine... Please wait."):
        today = date.today()
        csv_path = f"data/{today}.csv"
        fallback_file = "combined_air_quality.txt"
        df_loaded = None
        is_today_data = False
        load_msg = ""
        last_update_time = None

        if os.path.exists(csv_path):
            try:
                df_loaded = pd.read_csv(csv_path)
                if 'date' in df_loaded.columns:
                    df_loaded['date'] = pd.to_datetime(df_loaded['date'])
                    is_today_data = True
                    load_msg = f"Live data from: **{today}.csv**"
                    last_update_time = pd.Timestamp(os.path.getmtime(csv_path), unit='s')
                else:
                    load_msg = f"Warning: '{csv_path}' found but missing 'date' column. Using fallback."
            except Exception as e:
                load_msg = f"Error loading '{csv_path}': {e}. Using fallback."

        if df_loaded is None or not is_today_data:
            try:
                if not os.path.exists(fallback_file):
                    st.error(f"FATAL: Main data file '{fallback_file}' not found.")
                    return pd.DataFrame(), "Error: Main data file not found.", None
                df_loaded = pd.read_csv(fallback_file, sep="\t", parse_dates=['date'])
                base_load_msg = f"Displaying archive data from: **{fallback_file}**"
                load_msg = base_load_msg if not load_msg or is_today_data else load_msg + " " + base_load_msg
                last_update_time = pd.Timestamp(os.path.getmtime(fallback_file), unit='s')
            except Exception as e:
                st.error(f"FATAL: Error loading '{fallback_file}': {e}.")
                return pd.DataFrame(), f"Error loading fallback: {e}", None

        # Common processing
        for col, default_val in [('pollutant', np.nan), ('level', 'Unknown')]:
            if col not in df_loaded.columns: df_loaded[col] = default_val

        df_loaded['pollutant'] = df_loaded['pollutant'].astype(str).str.split(',').str[0].str.strip().replace(['nan', 'NaN', 'None', ''], np.nan)
        df_loaded['level'] = df_loaded['level'].astype(str).fillna('Unknown')
        df_loaded['pollutant'] = df_loaded['pollutant'].fillna('Other') # Fill after processing

        return df_loaded, load_msg, last_update_time

df, load_message, data_last_updated = load_data_and_metadata()

if df.empty:
    st.error("Dashboard cannot operate without data. Please check data sources.")
    st.stop()

if data_last_updated:
    st.caption(f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 0.85rem;'>{load_message} | Last data update: {data_last_updated.strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
else:
    st.caption(f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 0.85rem;'>{load_message}</p>", unsafe_allow_html=True)


# ------------------- Sidebar Filters -------------------
st.sidebar.header("🔭 EXPLORATION CONTROLS")
st.sidebar.markdown("---", unsafe_allow_html=True)

# --- UPDATED sidebar info message ---
st.sidebar.info("Fetching real-time data from CPCB, today data is available after 5:45pm.")

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
    month_number_filter = [k for k, v in months_map_dict.items() if v == selected_month_name][0]

# --- Filter data based on global selections ---
df_period_filtered = df[df['date'].dt.year == year].copy()
if month_number_filter:
    df_period_filtered = df_period_filtered[df_period_filtered['date'].dt.month == month_number_filter]


# ------------------- 💡 Key National Insights (Replaces National Overview) -------------------
st.markdown("## 🇮🇳 NATIONAL KEY INSIGHTS")
with st.container(): # Applies card styling
    st.markdown(f"##### Key Metro Annual Average AQI ({year})")
    major_cities = ['Delhi', 'Mumbai', 'Kolkata', 'Bengaluru', 'Chennai']
    major_cities_data = df[df['date'].dt.year == year]
    major_cities_data = major_cities_data[major_cities_data['city'].isin(major_cities)]

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
        num_cities_total = df_period_filtered['city'].nunique()

        insights = [f"🔍 Across **{num_cities_total}** observed cities, the average AQI is **{avg_aqi_national:.2f}**." if pd.notna(avg_aqi_national) else ""]

        city_avg_aqi_stats = df_period_filtered.groupby('city')['index'].mean()
        if not city_avg_aqi_stats.empty:
            best_city = city_avg_aqi_stats.idxmin()
            best_aqi = city_avg_aqi_stats.min()
            worst_city = city_avg_aqi_stats.idxmax()
            worst_aqi = city_avg_aqi_stats.max()
            insights.append(f"🌟 **{best_city}** shows the best average AQI (**{best_aqi:.2f}**).")
            insights.append(f"⚠️ **{worst_city}** records the highest average AQI (**{worst_aqi:.2f}**).")

        for insight in insights:
            if insight: st.markdown(f"<p style='font-size: 1.05rem; color: {TEXT_COLOR_DARK_THEME}; margin-bottom: 0.5rem;'>{insight}</p>", unsafe_allow_html=True)
    else:
        st.info("No data available for the selected period to generate general insights.")

# ------------------- 🏙️ City-Specific Analysis -------------------
export_data_list = []

if not selected_cities:
    st.info("✨ Select one or more cities from the sidebar to dive into detailed analysis.")
else:
    for city in selected_cities:
        st.markdown(f"## 🏙️ {city.upper()} DEEP DIVE – {year}")

        city_data_full = df_period_filtered[df_period_filtered['city'] == city].copy()
        current_filter_period_city = f"{selected_month_name}, {year}"

        if city_data_full.empty:
            with st.container(): # Card for "No Data"
                st.warning(f"😔 No data available for {city} for {current_filter_period_city}. Try different filter settings.")
            continue

        city_data_full['day_of_year'] = city_data_full['date'].dt.dayofyear
        city_data_full['month_name'] = city_data_full['date'].dt.month_name()
        city_data_full['day_of_month'] = city_data_full['date'].dt.day
        export_data_list.append(city_data_full)

        tab_trend, tab_dist, tab_heatmap_detail = st.tabs(["📊 TRENDS & CALENDAR", "📈 DISTRIBUTIONS", "🗓️ DETAILED HEATMAP"])

        with tab_trend:
            st.markdown("##### 📅 Daily AQI Calendar")
            # --- NEW: Plotly-based Interactive Calendar Heatmap ---
            start_date = pd.to_datetime(f'{year}-01-01')
            end_date = pd.to_datetime(f'{year}-12-31')
            full_year_dates = pd.date_range(start_date, end_date)
            
            calendar_df = pd.DataFrame({'date': full_year_dates})
            calendar_df['week'] = calendar_df['date'].dt.isocalendar().week
            calendar_df['day_of_week'] = calendar_df['date'].dt.dayofweek # Monday=0, Sunday=6
            
            # Adjust week numbers for years where week 53 belongs to the previous year
            calendar_df.loc[(calendar_df['date'].dt.month == 1) & (calendar_df['week'] > 50), 'week'] = 0
            # Adjust week numbers for years where week 1 belongs to the next year
            calendar_df.loc[(calendar_df['date'].dt.month == 12) & (calendar_df['week'] == 1), 'week'] = 53

            merged_cal_df = pd.merge(calendar_df, city_data_full[['date', 'index', 'level']], on='date', how='left')
            merged_cal_df['level'] = merged_cal_df['level'].fillna('Unknown')
            merged_cal_df['aqi_text'] = merged_cal_df['index'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')
            
            day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            month_starts = calendar_df.drop_duplicates('week').set_index('week')
            month_names = month_starts['date'].dt.strftime('%b')
            
            fig_cal = go.Figure(data=go.Heatmap(
                x=merged_cal_df['week'],
                y=merged_cal_df['day_of_week'],
                z=merged_cal_df['level'].map({k: i for i, k in enumerate(CATEGORY_COLORS_DARK.keys())}),
                customdata=pd.DataFrame({'date': merged_cal_df['date'].dt.strftime('%Y-%m-%d'), 'level': merged_cal_df['level'], 'aqi': merged_cal_df['aqi_text']}),
                hovertemplate="<b>%{customdata[0]}</b><br>AQI: %{customdata[2]} (%{customdata[1]})<extra></extra>",
                colorscale=[[i / (len(CATEGORY_COLORS_DARK) - 1), color] for i, color in enumerate(CATEGORY_COLORS_DARK.values())],
                showscale=False,
                xgap=2, ygap=2 # Gaps between squares
            ))

            # Add month names as annotations
            annotations = []
            for week_num, month_name in month_names.items():
                 annotations.append(go.layout.Annotation(
                        text=month_name,
                        align='center',
                        showarrow=False,
                        xref='x', yref='y',
                        x=week_num, y=7,
                        font=dict(color=SUBTLE_TEXT_COLOR_DARK_THEME, size=12)
                    ))
            
            # Add legend manually
            for i, (level, color) in enumerate(CATEGORY_COLORS_DARK.items()):
                 annotations.append(go.layout.Annotation(
                        text=f"█ <span style='color:{TEXT_COLOR_DARK_THEME};'>{level}</span>",
                        align='left',
                        showarrow=False,
                        xref='paper', yref='paper',
                        x=0.05 + 0.12 * (i % 7),
                        y=-0.1 - 0.1 * (i // 7),
                        xanchor='left', yanchor='top',
                        font=dict(color=color, size=12)
                    ))

            fig_cal.update_layout(
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(7)),
                    ticktext=day_labels,
                    showgrid=False, zeroline=False
                ),
                xaxis=dict(showgrid=False, zeroline=False, tickmode='array', ticktext=[], tickvals=[]),
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
            city_data_trend = city_data_full.sort_values('date').copy()
            city_data_trend['rolling_avg_7day'] = city_data_trend['index'].rolling(window=7, center=True, min_periods=1).mean().round(2)

            fig_trend_roll = go.Figure()
            fig_trend_roll.add_trace(go.Scatter(
                x=city_data_trend['date'], y=city_data_trend['index'], mode='lines+markers', name='Daily AQI',
                marker=dict(size=4, opacity=0.7, color=SUBTLE_TEXT_COLOR_DARK_THEME), line=dict(width=1.5, color=SUBTLE_TEXT_COLOR_DARK_THEME),
                customdata=city_data_trend[['level']],
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>AQI: %{y}<br>Category: %{customdata[0]}<extra></extra>"
            ))
            fig_trend_roll.add_trace(go.Scatter(
                x=city_data_trend['date'], y=city_data_trend['rolling_avg_7day'], mode='lines', name='7-Day Rolling Avg',
                line=dict(color=ACCENT_COLOR, width=2.5, dash='dash'),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>7-Day Avg AQI: %{y}<extra></extra>"
            ))
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
                category_counts_df = city_data_full['level'].value_counts().reindex(CATEGORY_COLORS_DARK.keys(), fill_value=0).reset_index()
                category_counts_df.columns = ['AQI Category', 'Number of Days']
                fig_dist_bar = px.bar(
                    category_counts_df, x='AQI Category', y='Number of Days', color='AQI Category',
                    color_discrete_map=CATEGORY_COLORS_DARK, text_auto=True
                )
                fig_dist_bar.update_layout(height=400, xaxis_title=None, yaxis_title="Number of Days",
                                           paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME)
                fig_dist_bar.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
                st.plotly_chart(fig_dist_bar, use_container_width=True)

            with col_sun_dist:
                st.markdown("##### ☀️ AQI Category (Proportions)")
                if category_counts_df['Number of Days'].sum() > 0:
                    fig_sunburst = px.sunburst(
                        category_counts_df, path=['AQI Category'], values='Number of Days',
                        color='AQI Category', color_discrete_map=CATEGORY_COLORS_DARK,
                    )
                    fig_sunburst.update_layout(height=400, margin=dict(t=20, l=20, r=20, b=20),
                                               paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME)
                    st.plotly_chart(fig_sunburst, use_container_width=True)
                else:
                    st.caption("No data for sunburst chart.")

            st.markdown("##### 🎻 Monthly AQI Distribution")
            month_order_cat = list(months_map_dict.values())
            city_data_full['month_name'] = pd.Categorical(city_data_full['month_name'], categories=month_order_cat, ordered=True)

            fig_violin = px.violin(
                city_data_full.sort_values('month_name'),
                x='month_name', y='index', color='month_name', color_discrete_sequence=px.colors.qualitative.Vivid,
                box=True, points="outliers",
                labels={'index': 'AQI Index', 'month_name': 'Month'},
                hover_data=['date', 'level']
            )
            fig_violin.update_layout(height=450, xaxis_title=None, showlegend=False,
                                     paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME)
            fig_violin.update_traces(meanline_visible=True)
            st.plotly_chart(fig_violin, use_container_width=True)

        with tab_heatmap_detail:
            st.markdown("##### 🔥 AQI Heatmap (Month vs. Day)")
            heatmap_pivot = city_data_full.pivot_table(index='month_name', columns='day_of_month', values='index', observed=False)
            heatmap_pivot = heatmap_pivot.reindex(month_order_cat)

            fig_heat_detail = px.imshow(
                heatmap_pivot, labels=dict(x="Day of Month", y="Month", color="AQI"),
                aspect="auto", color_continuous_scale="Inferno", # Good for dark themes
                text_auto=".0f"
            )
            fig_heat_detail.update_layout(height=500, xaxis_side="top",
                                          paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME)
            fig_heat_detail.update_traces(hovertemplate="<b>Month:</b> %{y}<br><b>Day:</b> %{x}<br><b>AQI:</b> %{z}<extra></extra>")
            st.plotly_chart(fig_heat_detail, use_container_width=True)

# ------------------- 🆚 City-wise AQI Comparison -------------------
if len(selected_cities) > 1:
    st.markdown("## 🆚 AQI COMPARISON ACROSS CITIES")
    comparison_df_list = []
    for city_comp in selected_cities:
        city_ts_comp = df_period_filtered[df_period_filtered['city'] == city_comp].copy()
        if not city_ts_comp.empty:
            city_ts_comp = city_ts_comp.sort_values('date')
            city_ts_comp['city_label'] = city_comp
            comparison_df_list.append(city_ts_comp)

    if comparison_df_list:
        combined_comp_df = pd.concat(comparison_df_list)
        fig_cmp = px.line(
            combined_comp_df, x='date', y='index', color='city_label',
            labels={'index': 'AQI Index', 'date': 'Date', 'city_label': 'City'},
            markers=False, line_shape='spline', color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_cmp.update_layout(
            title_text=f"AQI Trends Comparison – {selected_month_name}, {year}",
            height=500, legend_title_text='City',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        with st.container():
             st.info("Not enough data or cities selected for comparison with current filters.")


# ------------------- 💨 Pollutant Analysis -------------------
st.markdown("## 💨 PROMINENT POLLUTANT ANALYSIS")
with st.container():
    st.markdown("#### 鑽 Yearly Dominant Pollutant Trends")
    city_pollutant_A = st.selectbox(
        "Select City for Yearly Pollutant Trend:", unique_cities,
        key="pollutant_A_city_dark", index=unique_cities.index(default_city_val[0]) if default_city_val and default_city_val[0] in unique_cities else 0
    )
    pollutant_data_A = df[df['city'] == city_pollutant_A].copy()
    pollutant_data_A['year_label'] = pollutant_data_A['date'].dt.year

    if not pollutant_data_A.empty:
        grouped_poll_A = pollutant_data_A.groupby(['year_label', 'pollutant']).size().unstack(fill_value=0)
        percent_poll_A = grouped_poll_A.apply(lambda x: x / x.sum() * 100 if x.sum() > 0 else x, axis=1).fillna(0)
        percent_poll_A_long = percent_poll_A.reset_index().melt(id_vars='year_label', var_name='pollutant', value_name='percentage')

        fig_poll_A = px.bar(
            percent_poll_A_long, x='year_label', y='percentage', color='pollutant',
            title=f"Dominant Pollutants Over Years – {city_pollutant_A}",
            labels={'percentage': 'Percentage of Days (%)', 'year_label': 'Year', 'pollutant': 'Pollutant'},
            color_discrete_map=POLLUTANT_COLORS_DARK
        )
        fig_poll_A.update_layout(xaxis_type='category', yaxis_ticksuffix="%", height=500,
                                 paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME)
        fig_poll_A.update_traces(texttemplate='%{value:.1f}%', textposition='auto')
        st.plotly_chart(fig_poll_A, use_container_width=True)
    else:
        st.warning(f"No pollutant data for {city_pollutant_A} (Yearly Trend).")

with st.container():
    st.markdown(f"#### ⛽ Dominant Pollutants for Selected Period ({selected_month_name}, {year})")
    city_pollutant_B = st.selectbox(
        "Select City for Filtered Pollutant View:", unique_cities,
        key="pollutant_B_city_dark", index=unique_cities.index(default_city_val[0]) if default_city_val and default_city_val[0] in unique_cities else 0
    )
    pollutant_data_B = df_period_filtered[df_period_filtered['city'] == city_pollutant_B].copy()
    if not pollutant_data_B.empty and 'pollutant' in pollutant_data_B.columns:
        grouped_poll_B = pollutant_data_B.groupby('pollutant').size().reset_index(name='count')
        total_days_B = grouped_poll_B['count'].sum()
        grouped_poll_B['percentage'] = (grouped_poll_B['count'] / total_days_B * 100).round(1) if total_days_B > 0 else 0

        fig_poll_B = px.bar(
            grouped_poll_B, x='pollutant', y='percentage', color='pollutant',
            title=f"Dominant Pollutants – {city_pollutant_B} ({selected_month_name}, {year})",
            labels={'percentage': 'Percentage of Days (%)', 'pollutant': 'Pollutant'},
            color_discrete_map=POLLUTANT_COLORS_DARK
        )
        fig_poll_B.update_layout(yaxis_ticksuffix="%", height=450,
                                 paper_bgcolor=CARD_BACKGROUND_COLOR, plot_bgcolor=BACKGROUND_COLOR, font_color=TEXT_COLOR_DARK_THEME)
        fig_poll_B.update_traces(texttemplate='%{value:.1f}%', textposition='auto')
        st.plotly_chart(fig_poll_B, use_container_width=True)
    else:
        st.warning(f"No pollutant data for {city_pollutant_B} for the selected period.")

# ------------------- 🔮 AQI Forecast -------------------
st.markdown("## 🔮 AQI FORECAST (LINEAR TREND)")
with st.container():
    forecast_city_select = st.selectbox(
        "Select City for AQI Forecast:", unique_cities,
        key="forecast_city_select_dark", index=unique_cities.index(default_city_val[0]) if default_city_val and default_city_val[0] in unique_cities else 0
    )
    forecast_src_data = df_period_filtered[df_period_filtered['city'] == forecast_city_select].copy()
    if len(forecast_src_data) >= 15:
        forecast_df = forecast_src_data.sort_values('date')[['date', 'index']].dropna()
        if len(forecast_df) >= 2:
            forecast_df['days_since_start'] = (forecast_df['date'] - forecast_df['date'].min()).dt.days
            X_train, y_train = forecast_df[['days_since_start']], forecast_df['index']
            model = LinearRegression().fit(X_train, y_train)
            last_day_num = forecast_df['days_since_start'].max()
            future_X_range = np.arange(0, last_day_num + 15 + 1)
            future_y_pred = model.predict(pd.DataFrame({'days_since_start': future_X_range}))
            min_date_forecast = forecast_df['date'].min()
            future_dates_list = [min_date_forecast + timedelta(days=int(i)) for i in future_X_range]

            plot_df_obs = pd.DataFrame({'date': forecast_df['date'], 'AQI': y_train})
            plot_df_fcst = pd.DataFrame({'date': future_dates_list, 'AQI': np.maximum(0, future_y_pred)})

            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=plot_df_obs['date'], y=plot_df_obs['AQI'], mode='lines+markers', name='Observed AQI', line=dict(color=ACCENT_COLOR)))
            fig_forecast.add_trace(go.Scatter(x=plot_df_fcst['date'], y=plot_df_fcst['AQI'], mode='lines', name='Forecast', line=dict(dash='dash', color='#FF6B6B')))

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




# ------------------- 🗺️ INTERACTIVE AIR QUALITY MAP (City Hotspots) -------------------
st.markdown("## 📍 City AQI Hotspots")
with st.container():
    city_coords_data = {}
    coords_file_path = "lat_long.txt" # Uses the provided file
    scatter_map_rendered = False

    try:
        if os.path.exists(coords_file_path):
            with open(coords_file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
            
            local_scope = {}
            # For debugging if issues persist with lat_long.txt:
            # st.write(f"Attempting to execute content from: {coords_file_path}")
            # st.text_area("File Content Snippet:", file_content[:300], height=100) # Show a snippet
            exec(file_content, {}, local_scope)
            # st.write(f"Keys defined in local_scope after exec: {list(local_scope.keys())}")

            if 'city_coords' in local_scope and isinstance(local_scope['city_coords'], dict):
                city_coords_data = local_scope['city_coords']
            # else: # Error message for this case is handled by city_coords_data remaining empty
                # st.warning(f"Map Warning: 'city_coords' variable not found or not a dictionary in '{coords_file_path}'. Scatter map may be incomplete. Keys found: {list(local_scope.keys())}")
        # else: # File not found, city_coords_data remains empty
            # st.warning(f"Map Warning: Coordinates file '{coords_file_path}' not found. Scatter map cannot be displayed.")
            
    except Exception as e_exec: # Catch errors during exec or file reading
        st.error(f"Map Error: Error processing coordinates file '{coords_file_path}'. Scatter map cannot be displayed. Error: {e_exec}")
        city_coords_data = {} # Ensure it's empty on any error

    if not df_period_filtered.empty:
        map_grouped_data = df_period_filtered.groupby('city').agg(
            avg_aqi=('index', 'mean'),
            dominant_pollutant=('pollutant', lambda x: x.mode().iloc[0] if not x.mode().empty and x.mode().iloc[0] not in ['Other', 'nan'] else (x[~x.isin(['Other', 'nan'])].mode().iloc[0] if not x[~x.isin(['Other', 'nan'])].mode().empty else 'N/A'))
        ).reset_index().dropna(subset=['avg_aqi']) # Ensure avg_aqi is not NaN

        if city_coords_data and not map_grouped_data.empty:
            latlong_map_df_list = []
            for city_name_coord, coords_val in city_coords_data.items():
                if isinstance(coords_val, (list, tuple)) and len(coords_val) == 2:
                    try:
                        lat, lon = float(coords_val[0]), float(coords_val[1])
                        latlong_map_df_list.append({'city': city_name_coord, 'lat': lat, 'lon': lon})
                    except (ValueError, TypeError): pass # Skip if coords are not valid floats
            latlong_map_df = pd.DataFrame(latlong_map_df_list)

            if not latlong_map_df.empty:
                map_merged_df = pd.merge(map_grouped_data, latlong_map_df, on='city', how='inner')
                if not map_merged_df.empty:
                    map_merged_df["AQI Category"] = map_merged_df["avg_aqi"].apply(
                        lambda val: next((k for k, v_range in {
                            'Good': (0, 50), 'Satisfactory': (51, 100), 'Moderate': (101, 200),
                            'Poor': (201, 300), 'Very Poor': (301, 400), 'Severe': (401, float('inf'))
                        }.items() if v_range[0] <= val <= v_range[1]), "Unknown") if pd.notna(val) else "Unknown"
                    )
                    
                    fig_scatter_map = px.scatter_mapbox(
                        map_merged_df, lat="lat", lon="lon",
                        size=np.maximum(map_merged_df["avg_aqi"], 1), # Ensure size is at least 1 for visibility
                        size_max=25,
                        color="AQI Category",
                        color_discrete_map=CATEGORY_COLORS_DARK,
                        hover_name="city",
                        custom_data=['city', 'avg_aqi', 'dominant_pollutant', 'AQI Category'],
                        text="city", # Show city names on map if space permits
                        zoom=4.0, center={"lat": 23.0, "lon": 82.5} # Adjusted center for India
                    )
                    scatter_map_layout_args = get_custom_plotly_layout_args(height=700, title_text=f"Average AQI Hotspots - {selected_month_display_name}, {year}")
                    scatter_map_layout_args['mapbox_style'] = "carto-darkmatter"
                    scatter_map_layout_args['margin'] = {"r":10,"t":60,"l":10,"b":10} # Adjusted margins for map
                    scatter_map_layout_args['legend']['y'] = 0.98 # Legend inside top
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
                # else: st.info("No city data matched with available coordinates for the selected period.")
            # else: st.info("Coordinates data from 'lat_long.txt' is available but could not be processed into a valid format for the map.")
        
        if not scatter_map_rendered: # Fallback to bar chart if scatter map didn't render
            if not city_coords_data:
                 st.warning(f"Map Warning: Coordinates file '{coords_file_path}' not found or 'city_coords' variable issue. Scatter map cannot be displayed.")
            elif latlong_map_df.empty:
                 st.warning(f"Map Warning: Coordinates data from '{coords_file_path}' loaded but seems empty or malformed after processing.")
            elif map_merged_df.empty :
                 st.info("No city data from your dataset matched with the loaded coordinates for the scatter map.")


            st.markdown("##### City AQI Overview (Map Data Unavailable/Incomplete)")
            if not map_grouped_data.empty: # map_grouped_data has city averages, even if coords failed
                avg_aqi_cities_alt = map_grouped_data.sort_values(by='avg_aqi', ascending=True) # Show worst at top
                fig_alt_bar = px.bar(
                    avg_aqi_cities_alt.tail(20), # Show top 20 worst or all if less
                    x='avg_aqi', y='city', orientation='h',
                    color='avg_aqi', color_continuous_scale=px.colors.sequential.YlOrRd_r,
                    labels={'avg_aqi': 'Average AQI', 'city': 'City'}
                )
                fig_alt_bar.update_layout(**get_custom_plotly_layout_args(height=max(400, len(avg_aqi_cities_alt.tail(20)) * 25), title_text=f"Top Cities by Average AQI - {selected_month_display_name}, {year} (Map Fallback)"))
                fig_alt_bar.update_layout(yaxis={'categoryorder':'total ascending'}) # Ensure correct order
                st.plotly_chart(fig_alt_bar, use_container_width=True)
            else:
                st.warning("No city AQI data available for the selected period to display fallback chart.")
    else:
        st.warning("No air quality data available for the selected filters to display on a map or chart.")
st.markdown("---")


# ------------------- 📥 Download Filtered Data -------------------
if export_data_list:
    st.markdown("## 📥 DOWNLOAD DATA")
    with st.container(): # Card style for download section
        combined_export_final = pd.concat(export_data_list)
        csv_buffer_final = StringIO()
        combined_export_final.to_csv(csv_buffer_final, index=False)
        st.download_button(
            label="📤 Download Filtered City Data (CSV)",
            data=csv_buffer_final.getvalue(),
            file_name=f"AuraVision_filtered_aqi_{year}_{selected_month_name.replace(' ', '')}.csv",
            mime="text/csv"
        )

# ------------------- Footer -------------------
st.markdown(f"""
<div style="text-align: center; margin-top: 4rem; padding: 1.5rem; background-color: {CARD_BACKGROUND_COLOR}; border-radius: 12px; border: 1px solid {BORDER_COLOR};">
    <p style="margin:0.3em; color: {TEXT_COLOR_DARK_THEME}; font-size:0.9rem;">🌬️ AuraVision AQI Dashboard</p>
    <p style="margin:0.3em; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size:0.8rem;">Data Source: Central Pollution Control Board (CPCB), India (Illustrative). Coordinates approximate.</p>
    <p style="margin:0.5em 0; color: {TEXT_COLOR_DARK_THEME}; font-size:0.85rem;">Conceptualized by: Mr. Kapil Meena & Prof. Arkopal K. Goswami, IIT Kharagpur.</p>
    <p style="margin:0.3em; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size:0.8rem;">Developed with AI Assistance for advanced design and functionality.</p>
    <p style="margin-top:0.8em;"><a href="https://github.com/kapil2020/india-air-quality-dashboard" target="_blank" style="color:{ACCENT_COLOR}; text-decoration:none; font-weight:600;">🔗 View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)

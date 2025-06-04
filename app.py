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
pio.templates.default = "plotly_dark"

ACCENT_COLOR = "#00BCD4"
TEXT_COLOR = "#EAEAEA"
SUBTLE_TEXT_COLOR = "#B0B0B0"
BACKGROUND_COLOR = "#121212"
CARD_BG_COLOR = "#1E1E1E"
BORDER_COLOR = "#333333"

CATEGORY_COLORS = {
    'Severe': '#F44336',
    'Very Poor': '#FF7043',
    'Poor': '#FFA726',
    'Moderate': '#FFEE58',
    'Satisfactory': '#9CCC65',
    'Good': '#4CAF50',
    'Unknown': '#78909C'
}

POLLUTANT_COLORS = {
    'PM2.5': '#FF6B6B', 'PM10': '#4ECDC4', 'NO2': '#45B7D1',
    'SO2': '#F9C74F', 'CO': '#F8961E', 'O3': '#90BE6D', 'Other': '#B0BEC5'
}

# ------------------- Page Config -------------------
st.set_page_config(
    layout="wide",
    page_title="AuraVision AQI Dashboard",
    page_icon="🌬️",
    initial_sidebar_state="expanded"
)

# ------------------- Custom CSS for Dark Theme & Responsive Styling -------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    body {{
        font-family: 'Inter', sans-serif;
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
    }}
    .main .block-container {{
        padding: 2rem 3rem 3rem 3rem;
    }}
    .stPlotlyChart, .stDataFrame, .stAlert, .stMetric, .stDownloadButton > button, .stButton > button,
    div[data-testid="stExpander"], div[data-testid="stForm"] {{
        border-radius: 12px;
        border: 1px solid {BORDER_COLOR};
        background-color: {CARD_BG_COLOR};
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }}
    .stpyplot {{
        border-radius: 12px;
        border: 1px solid {BORDER_COLOR};
        background-color: {CARD_BG_COLOR};
        padding: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }}
    .stTabs [data-baseweb="tab-list"] {{
        border-bottom: 2px solid {BORDER_COLOR};
        background-color: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        color: {SUBTLE_TEXT_COLOR};
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
    h1 {{
        font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR};
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
        letter-spacing: -1px;
    }}
    h2 {{
        font-family: 'Inter', sans-serif;
        color: {ACCENT_COLOR};
        border-bottom: 2px solid {ACCENT_COLOR};
        padding-bottom: 0.6rem;
        margin-top: 2.5rem;
        margin-bottom: 1.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    h3 {{
        font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR};
        margin-top: 0rem;
        margin-bottom: 1.2rem;
        font-weight: 600;
    }}
    h4, h5 {{
        font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR};
        margin-top: 0.2rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }}
    .stSidebar {{
        background-color: {CARD_BG_COLOR};
        border-right: 1px solid {BORDER_COLOR};
        padding: 1.5rem;
    }}
    .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3, .stSidebar .stMarkdown p {{
        color: {TEXT_COLOR};
        text-align: left;
        border-bottom: none;
    }}
    .stSidebar .stSelectbox label, .stSidebar .stMultiselect label {{
        color: {ACCENT_COLOR} !important;
        font-weight: 600;
    }}
    .stMetric {{
        background-color: {CARD_BG_COLOR};
    }}
    .stMetric > div:nth-child(1) {{
        font-size: 1rem;
        color: {SUBTLE_TEXT_COLOR};
        font-weight: 500;
    }}
    .stMetric > div:nth-child(2) {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {ACCENT_COLOR};
    }}
    div[data-testid="stExpander"] summary {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {ACCENT_COLOR};
    }}
    div[data-testid="stExpander"] svg {{
        fill: {ACCENT_COLOR};
    }}
    .stDownloadButton button {{
        background-color: {ACCENT_COLOR};
        color: {BACKGROUND_COLOR};
        border: none;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: opacity 0.3s ease;
    }}
    .stDownloadButton button:hover {{
        opacity: 0.85;
    }}
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div:first-child,
    .stMultiselect div[data-baseweb="select"] > div:first-child {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
        border-color: {BORDER_COLOR} !important;
    }}
    .stDateInput input {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
    }}
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
    @media (max-width: 768px) {{
        .main .block-container {{
            padding: 1rem 1rem 2rem 1rem;
        }}
        h1 {{ font-size: 1.5rem; }}
        .stMetric > div:nth-child(2) {{ font-size: 1.8rem; }}
    }}
</style>
""", unsafe_allow_html=True)

# ------------------- Header Section with Logo & Title -------------------
header_col1, header_col2 = st.columns([1, 4])
with header_col1:
    st.image("https://raw.githubusercontent.com/kapil2020/india-air-quality-dashboard/main/logo.png", width=80)
with header_col2:
    st.markdown("<h1 style='margin-bottom:0;'>🌬️ AuraVision AQI Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: {SUBTLE_TEXT_COLOR}; font-size:1rem; margin-top:0;'>Illuminating Air Quality Insights Across India</p>", unsafe_allow_html=True)

st.markdown("---")

# ------------------- Data Loader -------------------
@st.cache_data(ttl=3600)
def load_data_and_metadata():
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

    for col, default_val in [('pollutant', np.nan), ('level', 'Unknown')]:
        if col not in df_loaded.columns:
            df_loaded[col] = default_val

    df_loaded['pollutant'] = df_loaded['pollutant'].astype(str).str.split(',').str[0].str.strip().replace(['nan', 'NaN', 'None', ''], np.nan)
    df_loaded['level'] = df_loaded['level'].astype(str).fillna('Unknown')
    df_loaded['pollutant'] = df_loaded['pollutant'].fillna('Other')

    return df_loaded, load_msg, last_update_time

df, load_message, data_last_updated = load_data_and_metadata()

if df.empty:
    st.error("🚫 Dashboard cannot operate without data. Please check data sources.")
    st.stop()

if data_last_updated:
    st.caption(f"<p style='text-align:center; color:{SUBTLE_TEXT_COLOR}; font-size:0.85rem;'>{load_message} | Last update: {data_last_updated.strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
else:
    st.caption(f"<p style='text-align:center; color:{SUBTLE_TEXT_COLOR}; font-size:0.85rem;'>{load_message}</p>", unsafe_allow_html=True)

# ------------------- Sidebar Filters -------------------
st.sidebar.header("🔭 EXPLORATION CONTROLS")
st.sidebar.markdown("---")

min_date = df['date'].min().date()
max_date = df['date'].max().date()
st.sidebar.info(f"Data Range: {min_date} to {max_date}")

unique_cities = sorted(df['city'].unique()) if 'city' in df.columns else []
default_city = ["Delhi"] if "Delhi" in unique_cities else (unique_cities[0:1] if unique_cities else [])
selected_cities = st.sidebar.multiselect("🏙️ Select Cities", unique_cities, default=default_city)

years = sorted(df['date'].dt.year.unique())
default_year = max(years) if years else None
year = st.sidebar.selectbox("🗓️ Select Year", years, index=years.index(default_year) if default_year in years else 0)

months_map = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
month_names = ["All Months"] + list(months_map.values())
selected_month = st.sidebar.selectbox("🌙 Select Month (Optional)", month_names, index=0)

month_filter = None
if selected_month != "All Months":
    month_filter = [num for num, name in months_map.items() if name == selected_month][0]

df_filtered = df[df['date'].dt.year == year].copy()
if month_filter:
    df_filtered = df_filtered[df_filtered['date'].dt.month == month_filter]

# ------------------- NATIONAL KEY INSIGHTS & METRICS -------------------
st.markdown("## 🇮🇳 NATIONAL KEY INSIGHTS")

if not df_filtered.empty:
    avg_aqi = df_filtered['index'].mean()
    total_cities = df_filtered['city'].nunique()
    city_means = df_filtered.groupby('city')['index'].mean()
    best_city = city_means.idxmin()
    best_val = city_means.min()
    worst_city = city_means.idxmax()
    worst_val = city_means.max()
    severe_count = df_filtered[df_filtered['level'] == 'Severe']['city'].nunique()
    good_sat_count = df_filtered[df_filtered['level'].isin(['Good', 'Satisfactory'])]['city'].nunique()

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    metric_col1.metric("Average AQI", f"{avg_aqi:.2f}")
    metric_col2.metric("Cities Monitored", f"{total_cities}")
    metric_col3.metric("Best City", f"{best_city} ({best_val:.2f})")
    metric_col4.metric("Worst City", f"{worst_city} ({worst_val:.2f})")
    metric_col5.metric("Severe Cities", f"{severe_count}")

    insights = []
    if good_sat_count > 0:
        insights.append(f"👍 **{good_sat_count}** cit{'y' if good_sat_count == 1 else 'ies'} had 'Good' or 'Satisfactory' AQI.")
    with st.container():
        for insight in insights:
            st.markdown(f"<p style='font-size:1rem; color:{TEXT_COLOR}; margin-bottom:0.5rem;'>{insight}</p>", unsafe_allow_html=True)
else:
    st.info("No data available for the selected filters.")

# ------------------- CITY-SPECIFIC ANALYSIS -------------------
export_list = []

if not selected_cities:
    st.info("✨ Please select at least one city to view detailed analysis.")
else:
    for city in selected_cities:
        st.markdown(f"## 🏙️ {city.upper()} DEEP DIVE – {year}")
        city_df = df_filtered[df_filtered['city'] == city].copy()
        period_label = f"{selected_month}, {year}" if month_filter else f"{year}"

        if city_df.empty:
            with st.container():
                st.warning(f"😔 No data for {city} in {period_label}.")
            continue

        city_df['day_of_year'] = city_df['date'].dt.dayofyear
        city_df['month_name'] = city_df['date'].dt.month_name()
        city_df['day_of_month'] = city_df['date'].dt.day
        export_list.append(city_df)

        tabs = st.tabs(["📊 TRENDS & CALENDAR", "📈 DISTRIBUTIONS", "🗓️ DETAILED HEATMAP"])

        with tabs[0]:
            st.markdown("##### 📅 Daily AQI Calendar")
            fig_cal, ax_cal = plt.subplots(figsize=(16, 2.8))
            fig_cal.patch.set_facecolor(CARD_BG_COLOR)
            ax_cal.set_facecolor(CARD_BG_COLOR)
            for _, row in city_df.iterrows():
                color = CATEGORY_COLORS.get(row['level'], "#444444")
                rect = patches.FancyBboxPatch(
                    (row['day_of_year'], 0), 0.95, 0.95,
                    boxstyle="round,pad=0.1,rounding_size=0.08",
                    linewidth=0.8, edgecolor=BORDER_COLOR, facecolor=color, alpha=0.9
                )
                ax_cal.add_patch(rect)

            ax_cal.set_xlim(0, 367); ax_cal.set_ylim(-0.1, 1.1); ax_cal.axis('off')
            month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
            month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            for d, lbl in zip(month_starts, month_labels):
                ax_cal.text(d+10, 1.3, lbl, ha='center', va='bottom', fontsize=11, color=SUBTLE_TEXT_COLOR, fontweight='500')
            legend_patches = [patches.Patch(facecolor=c, label=l, edgecolor=BORDER_COLOR, alpha=0.9) for l, c in CATEGORY_COLORS.items()]
            leg = ax_cal.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, title="AQI Category", fontsize=10, title_fontsize=11, frameon=False)
            plt.setp(leg.get_texts(), color=TEXT_COLOR)
            plt.setp(leg.get_title(), color=TEXT_COLOR, fontweight='600')
            plt.tight_layout(pad=1.8)
            st.pyplot(fig_cal, clear_figure=True)

            st.markdown("##### 📈 AQI Trend & 7-Day Rolling Average")
            trend_df = city_df.sort_values('date').copy()
            trend_df['rolling_7day'] = trend_df['index'].rolling(window=7, center=True, min_periods=1).mean().round(2)

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=trend_df['date'], y=trend_df['index'], mode='lines+markers', name='Daily AQI',
                marker=dict(size=4, opacity=0.7), line=dict(width=1.5),
                customdata=trend_df[['level']],
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>AQI: %{y}<br>Category: %{customdata[0]}<extra></extra>"
            ))
            fig_trend.add_trace(go.Scatter(
                x=trend_df['date'], y=trend_df['rolling_7day'], mode='lines', name='7-Day Avg',
                line=dict(color=ACCENT_COLOR, width=2.5, dash='dash'),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>7-Day Avg AQI: %{y}<extra></extra>"
            ))
            fig_trend.update_layout(
                yaxis_title="AQI Index", xaxis_title="Date", height=400, template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_color=TEXT_COLOR),
                hovermode="x unified", paper_bgcolor=CARD_BG_COLOR, plot_bgcolor=CARD_BG_COLOR, font_color=TEXT_COLOR
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 📊 AQI Category (Days)")
                cat_counts = city_df['level'].value_counts().reindex(CATEGORY_COLORS.keys(), fill_value=0).reset_index()
                cat_counts.columns = ['AQI Category', 'Days']
                fig_bar = px.bar(
                    cat_counts, x='AQI Category', y='Days', color='AQI Category',
                    color_discrete_map=CATEGORY_COLORS, text_auto=True
                )
                fig_bar.update_layout(height=400, xaxis_title=None, yaxis_title="Days", template="plotly_dark",
                                      paper_bgcolor=CARD_BG_COLOR, plot_bgcolor=CARD_BG_COLOR, font_color=TEXT_COLOR)
                fig_bar.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                st.markdown("##### ☀️ AQI Category (Proportions)")
                if cat_counts['Days'].sum() > 0:
                    fig_sun = px.sunburst(
                        cat_counts, path=['AQI Category'], values='Days',
                        color='AQI Category', color_discrete_map=CATEGORY_COLORS
                    )
                    fig_sun.update_layout(height=400, margin=dict(t=20,l=20,r=20,b=20), template="plotly_dark",
                                          paper_bgcolor=CARD_BG_COLOR, plot_bgcolor=CARD_BG_COLOR, font_color=TEXT_COLOR)
                    st.plotly_chart(fig_sun, use_container_width=True)
                else:
                    st.caption("No data for sunburst chart.")

            st.markdown("##### 🎻 Monthly AQI Distribution")
            city_df['month_name'] = pd.Categorical(city_df['month_name'], categories=list(months_map.values()), ordered=True)
            fig_violin = px.violin(
                city_df.sort_values('month_name'),
                x='month_name', y='index', color='month_name',
                color_discrete_sequence=px.colors.qualitative.Vivid,
                box=True, points="outliers",
                labels={'index': 'AQI Index', 'month_name': 'Month'},
                hover_data=['date', 'level']
            )
            fig_violin.update_layout(height=450, xaxis_title=None, showlegend=False, template="plotly_dark",
                                     paper_bgcolor=CARD_BG_COLOR, plot_bgcolor=CARD_BG_COLOR, font_color=TEXT_COLOR)
            fig_violin.update_traces(meanline_visible=True)
            st.plotly_chart(fig_violin, use_container_width=True)

        with tabs[2]:
            st.markdown("##### 🔥 AQI Heatmap (Month vs. Day)")
            heatmap_data = city_df.pivot_table(index='month_name', columns='day_of_month', values='index', observed=False)
            heatmap_data = heatmap_data.reindex(list(months_map.values()))
            fig_heat = px.imshow(
                heatmap_data, labels=dict(x="Day", y="Month", color="AQI"),
                aspect="auto", color_continuous_scale="Inferno", text_auto=".0f"
            )
            fig_heat.update_layout(height=500, xaxis_side="top", template="plotly_dark",
                                   paper_bgcolor=CARD_BG_COLOR, plot_bgcolor=CARD_BG_COLOR, font_color=TEXT_COLOR)
            fig_heat.update_traces(hovertemplate="<b>Month:</b> %{y}<br><b>Day:</b> %{x}<br><b>AQI:</b> %{z}<extra></extra>")
            st.plotly_chart(fig_heat, use_container_width=True)

# ------------------- CITY COMPARISON -------------------
if len(selected_cities) > 1:
    st.markdown("## 🆚 AQI COMPARISON ACROSS CITIES")
    comp_list = []
    for city_comp in selected_cities:
        temp_df = df_filtered[df_filtered['city'] == city_comp].copy()
        if not temp_df.empty:
            temp_df = temp_df.sort_values('date')
            temp_df['city_label'] = city_comp
            comp_list.append(temp_df)
    if comp_list:
        comp_df = pd.concat(comp_list)
        fig_cmp = px.line(
            comp_df, x='date', y='index', color='city_label',
            labels={'index': 'AQI Index', 'date': 'Date', 'city_label': 'City'},
            markers=False, line_shape='spline', color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_cmp.update_layout(
            title_text=f"AQI Trends Comparison – {selected_month}, {year}" if month_filter else f"AQI Trends Comparison – {year}",
            height=500, legend_title_text='City', template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_color=TEXT_COLOR),
            paper_bgcolor=CARD_BG_COLOR, plot_bgcolor=CARD_BG_COLOR, font_color=TEXT_COLOR
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.info("Not enough data for comparison.")

# ------------------- PROMINENT POLLUTANT ANALYSIS -------------------
st.markdown("## 💨 PROMINENT POLLUTANT ANALYSIS")

# Yearly Pollutant Trends
with st.container():
    st.markdown("#### 🗓️ Yearly Dominant Pollutant Trends")
    pollutant_city_A = st.selectbox(
        "Select City:", unique_cities,
        key="pollutant_A_city", index=unique_cities.index(default_city[0]) if default_city and default_city[0] in unique_cities else 0
    )
    pol_df_A = df[df['city'] == pollutant_city_A].copy()
    if not pol_df_A.empty:
        pol_df_A['year'] = pol_df_A['date'].dt.year
        grp = pol_df_A.groupby(['year', 'pollutant']).size().unstack(fill_value=0)
        pct = grp.apply(lambda x: x / x.sum() * 100 if x.sum() > 0 else x, axis=1).fillna(0)
        pct_long = pct.reset_index().melt(id_vars='year', var_name='pollutant', value_name='percentage')

        fig_poll_A = px.bar(
            pct_long, x='year', y='percentage', color='pollutant',
            title=f"Dominant Pollutants Over Years – {pollutant_city_A}",
            labels={'percentage': 'Percentage of Days (%)', 'year': 'Year', 'pollutant': 'Pollutant'},
            color_discrete_map=POLLUTANT_COLORS
        )
        fig_poll_A.update_layout(xaxis_type='category', yaxis_ticksuffix="%", height=500, template="plotly_dark",
                                 paper_bgcolor=CARD_BG_COLOR, plot_bgcolor=CARD_BG_COLOR, font_color=TEXT_COLOR)
        fig_poll_A.update_traces(texttemplate='%{value:.1f}%', textposition='auto')
        st.plotly_chart(fig_poll_A, use_container_width=True)
    else:
        st.warning(f"No pollutant data for {pollutant_city_A}.")

# Period Pollutant Distribution
with st.container():
    st.markdown(f"#### ⛽ Dominant Pollutants for Selected Period ({selected_month}, {year})" if month_filter else f"#### ⛽ Dominant Pollutants for {year}")
    pollutant_city_B = st.selectbox(
        "Select City:", unique_cities,
        key="pollutant_B_city", index=unique_cities.index(default_city[0]) if default_city and default_city[0] in unique_cities else 0
    )
    pol_df_B = df_filtered[df_filtered['city'] == pollutant_city_B].copy()
    if not pol_df_B.empty and 'pollutant' in pol_df_B.columns:
        grp_B = pol_df_B.groupby('pollutant').size().reset_index(name='count')
        total_days = grp_B['count'].sum()
        grp_B['percentage'] = (grp_B['count'] / total_days * 100).round(1) if total_days > 0 else 0

        fig_poll_B = px.bar(
            grp_B, x='pollutant', y='percentage', color='pollutant',
            title=f"Dominant Pollutants – {pollutant_city_B} ({selected_month}, {year})" if month_filter else f"Dominant Pollutants – {pollutant_city_B} ({year})",
            labels={'percentage': 'Percentage of Days (%)', 'pollutant': 'Pollutant'},
            color_discrete_map=POLLUTANT_COLORS
        )
        fig_poll_B.update_layout(yaxis_ticksuffix="%", height=450, template="plotly_dark",
                                 paper_bgcolor=CARD_BG_COLOR, plot_bgcolor=CARD_BG_COLOR, font_color=TEXT_COLOR)
        fig_poll_B.update_traces(texttemplate='%{value:.1f}%', textposition='auto')
        st.plotly_chart(fig_poll_B, use_container_width=True)
    else:
        st.warning(f"No pollutant data for {pollutant_city_B} for selected period.")

# ------------------- AQI FORECAST -------------------
st.markdown("## 🔮 AQI FORECAST (LINEAR TREND)")
with st.container():
    forecast_city = st.selectbox(
        "Select City for Forecast:", unique_cities,
        key="forecast_city", index=unique_cities.index(default_city[0]) if default_city and default_city[0] in unique_cities else 0
    )
    fc_df = df_filtered[df_filtered['city'] == forecast_city].copy()
    if len(fc_df) >= 15:
        fc_ready = fc_df.sort_values('date')[['date', 'index']].dropna()
        if len(fc_ready) >= 2:
            fc_ready['day_num'] = (fc_ready['date'] - fc_ready['date'].min()).dt.days
            X_train, y_train = fc_ready[['day_num']], fc_ready['index']
            model = LinearRegression().fit(X_train, y_train)
            last_day = fc_ready['day_num'].max()
            future_range = np.arange(0, last_day + 15 + 1)
            future_pred = model.predict(pd.DataFrame({'day_num': future_range}))
            start_date = fc_ready['date'].min()
            future_dates = [start_date + timedelta(days=int(i)) for i in future_range]

            obs_df = pd.DataFrame({'date': fc_ready['date'], 'AQI': y_train})
            fcst_df = pd.DataFrame({'date': future_dates, 'AQI': np.maximum(0, future_pred)})

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=obs_df['date'], y=obs_df['AQI'], mode='lines+markers', name='Observed AQI', line=dict(color=ACCENT_COLOR)))
            fig_fc.add_trace(go.Scatter(x=fcst_df['date'], y=fcst_df['AQI'], mode='lines', name='Forecast', line=dict(dash='dash', color='#FF6B6B')))
            fig_fc.update_layout(
                title=f"AQI Forecast – {forecast_city}", yaxis_title="AQI Index", xaxis_title="Date", height=450, template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_color=TEXT_COLOR),
                paper_bgcolor=CARD_BG_COLOR, plot_bgcolor=CARD_BG_COLOR, font_color=TEXT_COLOR
            )
            st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.warning(f"Not enough valid data for {forecast_city} to forecast.")
    else:
        st.warning(f"Need at least 15 data points for {forecast_city}; found {len(fc_df)}.")

# ------------------- INTERACTIVE AIR QUALITY MAP -------------------
st.markdown("## 📍 AIR QUALITY MAP VISUALIZATION")
with st.container():
    city_coords = {}
    try:
        with open("lat_long.txt", "r") as f:
            text = "".join(f.readlines())
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                city_coords = eval(text[start:end+1])
            else:
                st.error("Map Error: Invalid structure in lat_long.txt.")
    except FileNotFoundError:
        st.error("Map Error: `lat_long.txt` not found.")
    except Exception as e:
        st.error(f"Map Error: Parsing `lat_long.txt`: {e}.")

    if city_coords and not df_filtered.empty:
        coords_df = pd.DataFrame([{'city': k, 'lat': v[0], 'lon': v[1]} for k, v in city_coords.items()])
        grp_map = df_filtered.groupby('city').agg(
            avg_aqi=('index', 'mean'),
            dominant_pollutant=('pollutant', lambda x: x.mode().iloc[0] if not x.mode().empty and pd.notna(x.mode().iloc[0]) else 'N/A')
        ).reset_index()
        map_df = pd.merge(grp_map, coords_df, on='city', how='inner')

        def classify_aqi(val):
            if pd.isna(val): return "Unknown"
            if val <= 50: return "Good"
            if val <= 100: return "Satisfactory"
            if val <= 200: return "Moderate"
            if val <= 300: return "Poor"
            if val <= 400: return "Very Poor"
            return "Severe"

        map_df["AQI Category"] = map_df["avg_aqi"].apply(classify_aqi)
        map_filter = st.selectbox("🧪 Filter Map by AQI Category", ["All Categories"] + list(CATEGORY_COLORS.keys()), key="map_filter")

        disp_df = map_df.copy()
        if map_filter != "All Categories":
            disp_df = disp_df[disp_df["AQI Category"] == map_filter]

        if not disp_df.empty:
            fig_map = px.scatter_mapbox(
                disp_df, lat="lat", lon="lon",
                size=np.maximum(1, disp_df["avg_aqi"]), size_max=30,
                color="AQI Category", color_discrete_map=CATEGORY_COLORS,
                hover_name="city", text="city",
                hover_data={"avg_aqi": ":.2f", "dominant_pollutant": True, "AQI Category": True, "lat": False, "lon": False, "city": False},
                zoom=4.5, center={"lat": 22.8, "lon": 82.5}, height=700
            )
            fig_map.update_layout(
                mapbox_style="carto-darkmatter",
                legend_title="AQI Category", title_text=f"Average AQI by City ({selected_month}, {year})" if month_filter else f"Average AQI by City ({year})",
                margin={"r":0,"t":40,"l":0,"b":0}, template="plotly_dark",
                paper_bgcolor=CARD_BG_COLOR, plot_bgcolor=CARD_BG_COLOR, font_color=TEXT_COLOR,
                legend=dict(font_color=TEXT_COLOR)
            )
            fig_map.update_traces(marker=dict(sizemin=5, sizemode='diameter', opacity=0.8))
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No cities match the map filter.")
    elif df_filtered.empty:
        st.info("No data to display on map for selected period.")
    else:
        st.warning("Map cannot be displayed due to missing coordinates.")

# ------------------- DOWNLOAD FILTERED DATA -------------------
if export_list:
    st.markdown("## 📥 DOWNLOAD DATA")
    with st.container():
        export_df = pd.concat(export_list)
        buffer = StringIO()
        export_df.to_csv(buffer, index=False)
        st.download_button(
            label="📤 Download Filtered Data (CSV)",
            data=buffer.getvalue(),
            file_name=f"AuraVision_filtered_{year}_{selected_month.replace(' ', '') if month_filter else year}.csv",
            mime="text/csv"
        )

# ------------------- FOOTER -------------------
st.markdown(f"""
<div style="text-align:center; margin-top:3rem; padding:1.5rem; background-color:{CARD_BG_COLOR}; border-radius:12px; border:1px solid {BORDER_COLOR};">
    <p style="margin:0.3em; color:{TEXT_COLOR}; font-size:0.9rem;">🌬️ AuraVision AQI Dashboard</p>
    <p style="margin:0.3em; color:{SUBTLE_TEXT_COLOR}; font-size:0.8rem;">Data Source: Central Pollution Control Board (CPCB), India. Coordinates approximate.</p>
    <p style="margin:0.5em 0; color:{TEXT_COLOR}; font-size:0.85rem;">Conceptualized by: Mr. Kapil Meena & Prof. Arkopal K. Goswami, IIT Kharagpur.</p>
    <p style="margin:0.3em; color:{SUBTLE_TEXT_COLOR}; font-size:0.8rem;">Developed for Apple/Google Dashboard Challenge with Award-Winning Design & Functionality.</p>
    <p style="margin-top:0.8em;"><a href="https://github.com/kapil2020/india-air-quality-dashboard" target="_blank" style="color:{ACCENT_COLOR}; text-decoration:none; font-weight:600;">🔗 View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)

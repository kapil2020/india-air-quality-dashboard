import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import StringIO
from sklearn.linear_model import LinearRegression
import os
from datetime import date, timedelta
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Set default Plotly template for a clean look
pio.templates.default = "plotly_white"

# ------------------- Page Config -------------------
st.set_page_config(layout="wide", page_title="India AQI Dashboard", page_icon="🇮🇳")

# ------------------- Custom CSS Styling -------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');

    body {
        font-family: 'Nunito', sans-serif;
        background-color: #f0f2f6; /* Light gray background */
    }

    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }

    /* Card-like styling for sections/charts */
    .stPlotlyChart, .stDataFrame, .stAlert, .stMetric, .stDownloadButton, .stButton > button, .stTabs [data-baseweb="tab-list"] {
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        padding: 1.2rem;
        background-color: #ffffff; /* White background for cards */
        margin-bottom: 1.5rem; /* Space below cards */
    }
    .stpyplot { /* Matplotlib specific styling */
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        padding: 0.8rem; /* Adjust padding for matplotlib figures */
        background-color: #ffffff;
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
         box-shadow: none; /* Remove shadow from tab list itself */
         border-bottom: 2px solid #e0e0e0;
         padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        font-weight: 600;
    }
     .stTabs [aria-selected="true"] {
        border-bottom-color: #007E00; /* Theme color for selected tab */
        color: #007E00;
     }


    /* Headings */
    h1 {
        font-family: 'Nunito', sans-serif;
        color: #1A2A3A; /* Dark blue-gray */
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
    }
    h2 {
        font-family: 'Nunito', sans-serif;
        color: #007E00; /* Theme green */
        border-bottom: 3px solid #007E00;
        padding-bottom: 0.5rem;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
    }
    h3 { /* For city names primarily */
        font-family: 'Nunito', sans-serif;
        color: #2c3e50; /* Slightly lighter dark blue */
        margin-top: 0rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    h4, h5 { /* For plot titles inside cards */
        font-family: 'Nunito', sans-serif;
        color: #333;
        margin-top: 0.2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    /* Sidebar styling */
    .stSidebar {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
        box-shadow: 2px 0 8px rgba(0,0,0,0.05);
    }
    .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3 {
        color: #007E00;
        text-align: left;
        border-bottom: none;
    }

    /* Metric styling */
    .stMetric > div:nth-child(1) { /* Label */
        font-size: 0.9rem;
        color: #555;
    }
    .stMetric > div:nth-child(2) { /* Value */
        font-size: 1.8rem;
        font-weight: 700;
        color: #007E00;
    }
    
    /* Horizontal Rule */
    hr {
      display: none; /* Hide default hr, use h2 for section breaks */
    }

    /* Responsive adjustments */
    @media screen and (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .stPlotlyChart, .stpyplot, .stDataFrame, .stAlert, .stMetric {
            padding: 0.8rem;
        }
        h1 { font-size: 1.8rem; }
        h2 { font-size: 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# ------------------- Title -------------------
st.title("🇮🇳 India Air Quality Dashboard")

# ------------------- Introduction (Collapsible) -------------------
with st.expander("ℹ️ About this Dashboard & How to Use", expanded=False):
    st.markdown("""
    Welcome to the enhanced **India Air Quality Dashboard**! This platform provides a comprehensive look at air quality across various Indian cities.

    **🔍 Features & Navigation:**
    - **Sidebar Filters:** Use the controls on the left to select cities, year, and optionally, a specific month.
    - **National Overview:** Get a high-level summary of AQI across the country for the selected period.
    - **City-Specific Deep Dive:** For each selected city, explore:
        - **Trends & Calendar:** Daily AQI progression and a visual calendar heatmap.
        - **Distributions:** Breakdown of AQI categories (bar, sunburst charts) and monthly AQI distribution (violin plots).
        - **Detailed Heatmap:** Month vs. Day of Month AQI patterns.
    - **Comparative Analysis:** Compare AQI trends across multiple selected cities.
    - **Pollutant Insights:** Analyze dominant pollutants year-wise and for filtered periods.
    - **AQI Forecast:** A simple linear trend forecast for the next 15 days.
    - **Interactive Map:** Visualize city-wise average AQI, categories, and dominant pollutants.

    **📤 Data Download:** A button is available at the bottom to download the filtered data as a CSV.

    This dashboard aims to provide actionable insights into air quality trends in India.
    """)

# ------------------- Load Data -------------------
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_data_and_metadata():
    with st.spinner("🛰️ Fetching and processing air quality data... Please wait."):
        today = date.today()
        csv_path = f"data/{today}.csv" # Assumes a 'data' subdirectory
        fallback_file = "combined_air_quality.txt"
        df_loaded = None
        is_today_data = False
        load_msg = ""
        last_update_time = None

        # Attempt to load today's data
        if os.path.exists(csv_path):
            try:
                df_loaded = pd.read_csv(csv_path)
                if 'date' in df_loaded.columns:
                    df_loaded['date'] = pd.to_datetime(df_loaded['date'])
                    is_today_data = True
                    load_msg = f"Displaying data from today's report: **{today}.csv**"
                    last_update_time = pd.Timestamp(os.path.getmtime(csv_path), unit='s')
                else:
                    load_msg = f"Warning: '{csv_path}' found but missing 'date' column. Using fallback."
            except Exception as e:
                load_msg = f"Error loading '{csv_path}': {e}. Using fallback."

        # Fallback or initial load
        if df_loaded is None or not is_today_data:
            try:
                if not os.path.exists(fallback_file):
                    st.error(f"FATAL: Main data file '{fallback_file}' not found. Dashboard cannot operate.")
                    return pd.DataFrame(), "Error: Main data file not found.", None
                
                df_loaded = pd.read_csv(fallback_file, sep="\t", parse_dates=['date'])
                load_msg = f"Using data from **{fallback_file}**."
                if is_today_data: # if today's load failed after initially being true
                     load_msg = f"Failed to load today's data. " + load_msg
                last_update_time = pd.Timestamp(os.path.getmtime(fallback_file), unit='s')

            except Exception as e:
                st.error(f"FATAL: Error loading '{fallback_file}': {e}. Dashboard cannot operate.")
                return pd.DataFrame(), f"Error loading fallback: {e}", None
        
        # Common processing for any loaded df
        if 'pollutant' in df_loaded.columns:
            df_loaded['pollutant'] = df_loaded['pollutant'].astype(str).str.split(',').str[0].str.strip()
            df_loaded['pollutant'].replace(['nan', 'NaN', 'None', ''], np.nan, inplace=True)
        else:
            df_loaded['pollutant'] = np.nan # Add column if missing

        if 'level' in df_loaded.columns:
             df_loaded['level'] = df_loaded['level'].astype(str) # Ensure string type before fillna
        else:
            df_loaded['level'] = 'Unknown' # Add column if missing
        
        return df_loaded, load_msg, last_update_time


df, load_message, data_last_updated = load_data_and_metadata()

if df.empty:
    st.stop()

# Display data source and last update time
if data_last_updated:
    st.caption(f"ℹ️ {load_message} Last data file update: **{data_last_updated.strftime('%Y-%m-%d %H:%M:%S')}**")
else:
    st.caption(f"ℹ️ {load_message}")


# ------------------- Sidebar Filters -------------------
st.sidebar.header("⚙️ Dashboard Controls")

if 'date' not in df.columns or df['date'].isnull().all():
    st.error("Date column is missing or empty. Cannot proceed.")
    st.stop()

min_date_available = df['date'].min().date()
max_date_available = df['date'].max().date()

st.sidebar.info(
    f"Data available from **{min_date_available}** to **{max_date_available}**"
)

unique_cities = sorted(df['city'].unique()) if 'city' in df.columns else []
default_city = ["Delhi"] if "Delhi" in unique_cities else (unique_cities[0:1] if unique_cities else [])
selected_cities = st.sidebar.multiselect("🏙️ Select Cities", unique_cities, default=default_city)

years = sorted(df['date'].dt.year.unique())
if not years:
    st.error("No year data available. Cannot proceed.")
    st.stop()

default_year_val = max(years) if years else None
year = st.sidebar.selectbox("🗓️ Select Year", years, index=years.index(default_year_val) if default_year_val in years else 0)

months_map = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
month_options_list = ["All Months"] + list(months_map.values())
selected_month_name = st.sidebar.selectbox("🌙 Select Month (Optional)", month_options_list, index=0)

month_number_filter = None
if selected_month_name != "All Months":
    month_number_filter = [k for k, v in months_map.items() if v == selected_month_name][0]

# AQI Category Colors & Pollutant Colors
category_colors = {
    'Severe': '#7E0023', 'Very Poor': '#FF0000', 'Poor': '#FF7E00',
    'Moderate': '#FFFF00', 'Satisfactory': '#00E400', 'Good': '#007E00', 'Unknown': '#D3D3D3'
}
df['level'] = df['level'].fillna('Unknown') # Ensure all levels are handled

pollutant_colors = {
    'PM2.5': '#F8766D', 'PM10': '#7CAE00', 'NO2': '#00BFC4',
    'SO2': '#C77CFF', 'CO': '#E69F00', 'O3': '#619CFF', 'Other': '#A9A9A9'
}
df['pollutant'] = df['pollutant'].fillna('Other')


# --- Filter data based on selections for national overview and subsequent processing ---
df_period_filtered = df[df['date'].dt.year == year].copy()
if month_number_filter:
    df_period_filtered = df_period_filtered[df_period_filtered['date'].dt.month == month_number_filter]

# ------------------- 🌍 National Overview -------------------
st.markdown("## 🌍 National AQI Snapshot")
if not df_period_filtered.empty:
    national_avg_aqi = df_period_filtered['index'].mean()
    city_avg_aqi = df_period_filtered.groupby('city')['index'].mean().sort_values()
    
    col_nat1, col_nat2, col_nat3 = st.columns(3)
    with col_nat1:
        st.metric("National Average AQI", f"{national_avg_aqi:.2f}" if pd.notna(national_avg_aqi) else "N/A")
    with col_nat2:
        if not city_avg_aqi.empty:
            st.metric("City with Best Avg. AQI", f"{city_avg_aqi.index[0]} ({city_avg_aqi.iloc[0]:.2f})")
        else:
            st.metric("City with Best Avg. AQI", "N/A")
    with col_nat3:
        if not city_avg_aqi.empty and len(city_avg_aqi) > 1 :
            st.metric("City with Worst Avg. AQI", f"{city_avg_aqi.index[-1]} ({city_avg_aqi.iloc[-1]:.2f})")
        elif not city_avg_aqi.empty:
             st.metric("City with Worst Avg. AQI", f"{city_avg_aqi.index[0]} ({city_avg_aqi.iloc[0]:.2f})") # Only one city
        else:
            st.metric("City with Worst Avg. AQI", "N/A")

    if not city_avg_aqi.empty:
        fig_nat_avg = px.bar(
            city_avg_aqi.reset_index(), x='city', y='index',
            title=f"Average AQI Across Cities ({selected_month_name}, {year})",
            labels={'index': 'Average AQI', 'city': 'City'},
            color='index', color_continuous_scale=px.colors.sequential.YlOrRd
        )
        fig_nat_avg.update_layout(height=450, xaxis_title=None)
        st.plotly_chart(fig_nat_avg, use_container_width=True)
    else:
        st.info("No city data available for the selected period to display National Overview chart.")
else:
    st.warning("No data available for the selected period to display National Overview.")


# ------------------- 🏙️ City-Specific Analysis -------------------
export_data_list = []

if not selected_cities:
    st.info("✨ Select one or more cities from the sidebar to dive into detailed analysis.")
else:
    for city in selected_cities:
        st.markdown(f"## 🏙️ {city} Analysis – {year}")
        
        city_data_full = df_period_filtered[df_period_filtered['city'] == city].copy()

        current_filter_period_city = f"{selected_month_name}, {year}"
        st.markdown(f"### Showing data for **{current_filter_period_city}**")

        if city_data_full.empty:
            st.warning(f"😔 No data available for {city} for {current_filter_period_city}. Try different filter settings.")
            continue

        city_data_full['day_of_year'] = city_data_full['date'].dt.dayofyear
        city_data_full['month_name'] = city_data_full['date'].dt.month_name()
        city_data_full['day_of_month'] = city_data_full['date'].dt.day
        export_data_list.append(city_data_full)

        tab_trend, tab_dist, tab_heatmap_detail = st.tabs(["📈 Trends & Calendar", "📊 Distributions", "🗓️ Detailed Heatmap"])

        with tab_trend:
            st.markdown("##### 📅 Daily AQI Calendar")
            # Using a slightly larger figure for Matplotlib to give legend space
            fig_cal, ax_cal = plt.subplots(figsize=(16, 2.5)) # Wider and slightly taller
            plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style for better aesthetics

            for _, row in city_data_full.iterrows():
                color = category_colors.get(row['level'], '#FFFFFF')
                rect = patches.FancyBboxPatch(
                    (row['day_of_year'], 0), 0.95, 0.95, # Slightly smaller for spacing
                    boxstyle="round,pad=0.1,rounding_size=0.05", 
                    linewidth=0.5, edgecolor='#cccccc', facecolor=color, alpha=0.8
                )
                ax_cal.add_patch(rect)

            ax_cal.set_xlim(0, 367)
            ax_cal.set_ylim(-0.1, 1.1) # Adjusted ylim for better text placement
            ax_cal.axis('off')
            month_starts_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
            month_labels_cal = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for day_val, label_val in zip(month_starts_days, month_labels_cal):
                ax_cal.text(day_val + 10, 1.25, label_val, ha='center', va='bottom', fontsize=10, color="#555")

            legend_elements = [patches.Patch(facecolor=color, label=label, edgecolor='#ccc', alpha=0.8) for label, color in category_colors.items()]
            ax_cal.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(category_colors)//2, title="AQI Category", fontsize=9, title_fontsize=10)
            plt.tight_layout(pad=1.5) # Add padding
            st.pyplot(fig_cal, clear_figure=True)


            st.markdown("##### 📈 AQI Trend & 7-Day Rolling Average")
            city_data_trend = city_data_full.sort_values('date').copy()
            city_data_trend['rolling_avg_7day'] = city_data_trend['index'].rolling(window=7, center=True, min_periods=1).mean().round(2)

            fig_trend_roll = px.line(city_data_trend, x='date', y='index', labels={'index': 'Daily AQI'})
            fig_trend_roll.update_traces(
                name='Daily AQI', showlegend=True,
                hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>AQI:</b> %{y}<br><b>Category:</b> %{customdata[0]}<extra></extra>",
                customdata=city_data_trend[['level']]
            )
            fig_trend_roll.add_scatter(
                x=city_data_trend['date'], y=city_data_trend['rolling_avg_7day'], 
                mode='lines', name='7-Day Rolling Avg', line=dict(color='rgba(255,165,0,0.8)', width=2.5), # Orange, slightly thicker
                hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>7-Day Avg AQI:</b> %{y}<extra></extra>"
            )
            fig_trend_roll.update_layout(
                yaxis_title="AQI Index", xaxis_title="Date", height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_trend_roll, use_container_width=True)

        with tab_dist:
            col_bar, col_sun = st.columns(2)
            with col_bar:
                st.markdown("##### 📊 AQI Category (Days)")
                category_counts_df = city_data_full['level'].value_counts().reindex(category_colors.keys(), fill_value=0).reset_index()
                category_counts_df.columns = ['AQI Category', 'Number of Days']
                fig_dist_bar = px.bar(
                    category_counts_df, x='AQI Category', y='Number of Days', color='AQI Category',
                    color_discrete_map=category_colors,
                    text_auto=True
                )
                fig_dist_bar.update_layout(height=400, xaxis_title=None, yaxis_title="Number of Days")
                st.plotly_chart(fig_dist_bar, use_container_width=True)
            
            with col_sun:
                st.markdown("##### ☀️ AQI Category (Proportions)")
                if category_counts_df['Number of Days'].sum() > 0:
                    fig_sunburst = px.sunburst(
                        category_counts_df, path=['AQI Category'], values='Number of Days',
                        color='AQI Category', color_discrete_map=category_colors,
                    )
                    fig_sunburst.update_layout(height=400, margin=dict(t=20, l=20, r=20, b=20))
                    st.plotly_chart(fig_sunburst, use_container_width=True)
                else:
                    st.caption("No data for sunburst chart.")

            st.markdown("##### 🎻 Monthly AQI Distribution")
            month_order_cat = list(months_map.values())
            city_data_full['month_name'] = pd.Categorical(city_data_full['month_name'], categories=month_order_cat, ordered=True)
            
            fig_violin = px.violin(
                city_data_full.sort_values('month_name'), 
                x='month_name', y='index', color='month_name',
                box=True, points="outliers", # Show box plot inside violin and outliers
                labels={'index': 'AQI Index', 'month_name': 'Month'},
                hover_data=['date', 'level']
            )
            fig_violin.update_layout(height=450, xaxis_title=None, showlegend=False)
            fig_violin.update_traces(meanline_visible=True)
            st.plotly_chart(fig_violin, use_container_width=True)
        
        with tab_heatmap_detail:
            st.markdown("##### 🔥 AQI Heatmap (Month vs. Day)")
            heatmap_pivot = city_data_full.pivot_table(index='month_name', columns='day_of_month', values='index', observed=False)
            heatmap_pivot = heatmap_pivot.reindex(month_order_cat) # Ensure correct month order

            fig_heat_detail = px.imshow(
                heatmap_pivot, labels=dict(x="Day of Month", y="Month", color="AQI"),
                aspect="auto", color_continuous_scale="YlOrRd",
                text_auto=".0f" # Show AQI values on cells
            )
            fig_heat_detail.update_layout(height=500, xaxis_side="top")
            fig_heat_detail.update_traces(hovertemplate="<b>Month:</b> %{y}<br><b>Day:</b> %{x}<br><b>AQI:</b> %{z}<extra></extra>")
            st.plotly_chart(fig_heat_detail, use_container_width=True)

# ------------------- 🆚 City-wise AQI Comparison -------------------
if len(selected_cities) > 1:
    st.markdown("## 🆚 AQI Comparison Across Selected Cities")
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
            markers=False, line_shape='spline' # Smoothed lines
        )
        fig_cmp.update_layout(
            title_text=f"AQI Trends Comparison – {selected_month_name}, {year}",
            height=500, legend_title_text='City',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.info("Not enough data or cities selected for comparison with current filters.")

# ------------------- 💨 Pollutant Analysis -------------------
st.markdown("## 💨 Prominent Pollutant Analysis")

# Chart A: Year-wise Prominent Pollutants (ALL years for a selected city)
with st.container(): # Using container for card-like appearance from CSS
    st.markdown("#### 钻 Yearly Dominant Pollutant Trends")
    city_pollutant_A = st.selectbox(
        "Select City for Yearly Pollutant Trend:", unique_cities, 
        key="pollutant_A_city", index=unique_cities.index(default_city[0]) if default_city and default_city[0] in unique_cities else 0
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
            color_discrete_map=pollutant_colors, text_auto=".1f"
        )
        fig_poll_A.update_layout(xaxis_type='category', yaxis_ticksuffix="%", height=500)
        st.plotly_chart(fig_poll_A, use_container_width=True)
    else:
        st.warning(f"No pollutant data for {city_pollutant_A} (Yearly Trend).")

# Chart B: Pollutants for selected period and city
with st.container():
    st.markdown(f"#### ⛽ Dominant Pollutants for Selected Period ({selected_month_name}, {year})")
    city_pollutant_B = st.selectbox(
        "Select City for Filtered Pollutant View:", unique_cities, 
        key="pollutant_B_city", index=unique_cities.index(default_city[0]) if default_city and default_city[0] in unique_cities else 0
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
            color_discrete_map=pollutant_colors, text_auto=True
        )
        fig_poll_B.update_layout(yaxis_ticksuffix="%", height=450)
        st.plotly_chart(fig_poll_B, use_container_width=True)
    else:
        st.warning(f"No pollutant data for {city_pollutant_B} for the selected period.")


# ------------------- 🔮 AQI Forecast -------------------
st.markdown("## 🔮 AQI Forecast (Linear Trend)")
with st.container():
    forecast_city_select = st.selectbox(
        "Select City for AQI Forecast:", unique_cities, 
        key="forecast_city_select", index=unique_cities.index(default_city[0]) if default_city and default_city[0] in unique_cities else 0
    )
    forecast_src_data = df_period_filtered[df_period_filtered['city'] == forecast_city_select].copy()

    if len(forecast_src_data) >= 15:
        forecast_df = forecast_src_data.sort_values('date')[['date', 'index']].dropna()
        if len(forecast_df) >= 2:
            forecast_df['days_since_start'] = (forecast_df['date'] - forecast_df['date'].min()).dt.days
            X_train = forecast_df[['days_since_start']]
            y_train = forecast_df['index']

            model = LinearRegression()
            model.fit(X_train, y_train)

            last_day_num = forecast_df['days_since_start'].max()
            future_X_range = np.arange(0, last_day_num + 15 + 1) # Predict 15 days into future
            future_X_df_pred = pd.DataFrame({'days_since_start': future_X_range})
            future_y_pred = model.predict(future_X_df_pred)
            
            min_date_forecast = forecast_df['date'].min()
            future_dates_list = [min_date_forecast + timedelta(days=int(i)) for i in future_X_range]
            
            plot_df_obs = pd.DataFrame({'date': forecast_df['date'], 'AQI': y_train})
            plot_df_fcst = pd.DataFrame({'date': future_dates_list, 'AQI': np.maximum(0, future_y_pred)}) # AQI >= 0

            fig_forecast = px.line(plot_df_obs, x='date', y='AQI', title=f"AQI Forecast – {forecast_city_select}")
            fig_forecast.data[0].name = 'Observed AQI'; fig_forecast.data[0].showlegend = True
            fig_forecast.add_scatter(
                x=plot_df_fcst['date'], y=plot_df_fcst['AQI'], mode='lines',
                name='Forecast (Linear Trend)', line=dict(dash='dash', color='firebrick', width=2)
            )
            fig_forecast.update_layout(
                yaxis_title="AQI Index", xaxis_title="Date", height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.warning(f"Not enough valid data points (after dropping NaN) for {forecast_city_select} to create a forecast.")
    else:
        st.warning(f"Need at least 15 data points for {forecast_city_select} for forecasting; found {len(forecast_src_data)}. Try a broader period or different city.")


# ------------------- 🗺️ Interactive AQI Map -------------------
st.markdown("## 🗺️ Interactive Air Quality Map")
with st.container():
    city_coords_data = {}
    try:
        with open("lat_long.txt", "r") as f:
            dict_text_full = ''.join(f.readlines())
            start_brace, end_brace = dict_text_full.find('{'), dict_text_full.rfind('}')
            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                city_coords_data = eval(dict_text_full[start_brace : end_brace+1])
            else:
                st.error("Map Error: Invalid dictionary structure in lat_long.txt.")
    except FileNotFoundError:
        st.error("Map Error: `lat_long.txt` not found.")
    except Exception as e:
        st.error(f"Map Error: Parsing `lat_long.txt`: {e}.")

    if city_coords_data and not df_period_filtered.empty:
        latlong_map_df = pd.DataFrame([
            {'city': city_name, 'lat': coords[0], 'lon': coords[1]}
            for city_name, coords in city_coords_data.items()
        ])
        
        map_grouped_data = df_period_filtered.groupby('city').agg(
            avg_aqi=('index', 'mean'),
            dominant_pollutant=('pollutant', lambda x: x.mode().iloc[0] if not x.mode().empty and pd.notna(x.mode().iloc[0]) else 'N/A')
        ).reset_index()

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
        
        map_aqi_cat_filter = st.selectbox("🧪 Filter Map by AQI Category", ["All Categories"] + list(category_colors.keys()), key="map_aqi_cat_filter")
        
        display_map_df = map_merged_df.copy()
        if map_aqi_cat_filter != "All Categories":
            display_map_df = display_map_df[display_map_df["AQI Category"] == map_aqi_cat_filter]
        
        if not display_map_df.empty:
            fig_map_final = px.scatter_mapbox(
                display_map_df, lat="lat", lon="lon", size=np.maximum(1, display_map_df["avg_aqi"]), # Ensure size is at least 1
                size_max=25, color="AQI Category", color_discrete_map=category_colors,
                hover_name="city", text="city", # Show city name on map
                hover_data={
                    "avg_aqi": ":.2f", "dominant_pollutant": True, "AQI Category": True, 
                    "lat": False, "lon": False, "city": False
                },
                zoom=4.5, center={"lat": 22.8, "lon": 82.5}, height=700
            )
            fig_map_final.update_layout(
                mapbox_style="carto-positron", mapbox_accesstoken=st.secrets.get("pk.eyJ1Ijoic2FkaXR5YTkyMTEiLCJhIjoiY2xidzNvcWQ2MXlrazNucW5rcGxnc2RncCJ9.1GMKNUsQUmXSxvrOqlDnsw"), # Optional: if you have a Mapbox token
                legend_title="AQI Category", title_text=f"Average AQI by City ({selected_month_name}, {year})",
                margin={"r": 0, "t": 40, "l": 0, "b": 0}
            )
            fig_map_final.update_traces(cluster=dict(enabled=True)) # Basic clustering for overlapping points
            st.plotly_chart(fig_map_final, use_container_width=True)
        else:
            st.info("No cities match the map filter for the selected period.")
    elif df_period_filtered.empty:
        st.info("No data available for the selected period to display the map.")
    else:
        st.warning("Map cannot be displayed due to missing coordinate data.")

# ------------------- 📥 Download Filtered Data -------------------
if export_data_list:
    st.markdown("## 📥 Download Data")
    with st.container():
        combined_export_final = pd.concat(export_data_list)
        csv_buffer_final = StringIO()
        combined_export_final.to_csv(csv_buffer_final, index=False)
        st.download_button(
            label="📤 Download Filtered City Data as CSV",
            data=csv_buffer_final.getvalue(),
            file_name=f"filtered_aqi_data_{year}_{selected_month_name.replace(' ', '')}.csv",
            mime="text/csv"
        )

# ------------------- Footer -------------------
st.markdown("---") # Using markdown hr as it's not styled by default CSS
st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding: 1rem; background-color: #fff; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
    <p style="margin:0.2em;">📊 Data Source: Central Pollution Control Board (CPCB), India (Illustrative).</p>
    <p style="margin:0.2em;">Latitude/Longitude are approximate and for visualization purposes.</p>
    <p style="margin:0.5em 0;"><strong>Conceptualized by:</strong> Mr. Kapil Meena & Prof. Arkopal K. Goswami, IIT Kharagpur.</p>
    <p style="margin:0.2em;"><strong>Enhanced & Developed with AI Assistance.</strong></p>
    <p style="margin-top:0.5em;"><a href="https://github.com/kapil2020/india-air-quality-dashboard" target="_blank">🔗 View Example Structure on GitHub</a></p>
</div>
""", unsafe_allow_html=True)

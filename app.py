import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import StringIO
from sklearn.linear_model import LinearRegression
import os
from datetime import date, timedelta # Added timedelta
from sklearn.metrics import mean_squared_error
import plotly.express as px
# import osmnx as ox # Not actively used in the final plots, can be commented if not needed for data prep
# import geopandas as gpd # Same as osmnx
# from shapely.geometry import Point # Same as osmnx
# from shapely.ops import unary_union # Same as osmnx

# ------------------- Page Config -------------------
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# ------------------- Title -------------------
st.title("🇮🇳 India Air Quality Dashboard")

# ------------------- Introduction -------------------
st.info("""
Welcome to the **India Air Quality Dashboard** 🇮🇳

🔍 Use the **sidebar** to:
- Select one or more cities
- Choose a specific year
- (Optional) Select a month for detailed analysis

📊 Explore:
- **Interactive charts** for AQI trends, distributions, heatmaps, and pollutant breakdowns.
- **AQI forecast** with linear trendline predictions.
- **Interactive AQI map** with city-wise averages and dominant pollutants.

📤 Download the filtered dataset as a CSV using the button at the bottom.
""")

# ------------------- Load Data -------------------
@st.cache_data(ttl=3600)
def load_data():
    with st.spinner("Loading air quality data..."):
        today = date.today()
        csv_path = f"data/{today}.csv" # Assumes a 'data' subdirectory

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # Ensure 'date' column is datetime
                if 'date' not in df.columns: # Or if it's not the primary date column
                     st.warning(f"'{csv_path}' does not contain a 'date' column. Attempting to use file date.")
                     df['date'] = pd.to_datetime(today) # Fallback, might need adjustment
                else:
                     df['date'] = pd.to_datetime(df['date'])
                return df, True, f"Displaying data from today's report: {today}.csv"
            except Exception as e:
                st.error(f"Error loading today's CSV ({csv_path}): {e}. Falling back to main dataset.")
                # Fall through to fallback
        
        # Fallback to combined_air_quality.txt
        try:
            df = pd.read_csv("combined_air_quality.txt", sep="\t", parse_dates=['date'])
            # Clean pollutant column earlier
            df['pollutant'] = df['pollutant'].astype(str).str.split(',').str[0].str.strip()
            df['pollutant'].replace(['nan', 'NaN', 'None', ''], np.nan, inplace=True)
            return df, False, f"No report for {today}. Displaying data from 'combined_air_quality.txt'."
        except FileNotFoundError:
            st.error("FATAL: 'combined_air_quality.txt' not found. Please ensure the data file is present.")
            return pd.DataFrame(), False, "Error: Main data file 'combined_air_quality.txt' not found."
        except Exception as e:
            st.error(f"Error loading 'combined_air_quality.txt': {e}")
            return pd.DataFrame(), False, f"Error loading 'combined_air_quality.txt': {e}"


df, is_today, load_message = load_data()

if df.empty:
    st.stop()

st.sidebar.info(load_message)

# ------------------- Sidebar Filters -------------------
st.sidebar.markdown("---")
st.sidebar.header("📊 Filters")

if 'date' not in df.columns or df['date'].isnull().all():
    st.error("Date column is missing or empty in the DataFrame. Cannot proceed with filtering.")
    st.stop()

min_date = df['date'].min().date()
max_date = df['date'].max().date()

st.sidebar.markdown(
    f"Data available from **{min_date}** to **{max_date}**"
)

# Ensure 'city' column exists
if 'city' not in df.columns:
    st.error("'city' column not found in the data. Please check your data file.")
    st.sidebar.warning("Cannot select cities.")
    selected_cities = []
else:
    unique_cities = sorted(df['city'].unique())
    default_city = ["Delhi"] if "Delhi" in unique_cities else (unique_cities[0:1] if unique_cities else [])
    selected_cities = st.sidebar.multiselect("Select Cities", unique_cities, default=default_city)

years = sorted(df['date'].dt.year.unique())
if not years:
    st.error("No year data available. Cannot proceed.")
    st.stop()

default_year = max(years) if years else None
year = st.sidebar.selectbox("Select a Year", years, index=years.index(default_year) if default_year in years else 0)

months_dict = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
month_options = ["All"] + [months_dict[m] for m in range(1, 13)]
selected_month_name = st.sidebar.selectbox("Select Month (optional)", month_options, index=0)

# Define month_number based on sidebar selection (globally accessible)
month_number_filter = None
if selected_month_name != "All":
    month_number_filter = [k for k, v in months_dict.items() if v == selected_month_name][0]

# AQI Category Colors
category_colors = {
    'Severe': '#7E0023',
    'Very Poor': '#FF0000',
    'Poor': '#FF7E00',
    'Moderate': '#FFFF00',
    'Satisfactory': '#00E400',
    'Good': '#007E00',
    'Unknown': '#D3D3D3' # For missing levels
}
df['level'] = df['level'].fillna('Unknown')


# ------------------- Dashboard Body -------------------
export_data_list = [] # Renamed to avoid conflict

if not selected_cities:
    st.warning("Please select at least one city from the sidebar to view data.")
else:
    for city in selected_cities:
        st.markdown(f"<hr style='border:1px solid #DDD'>", unsafe_allow_html=True)
        st.markdown(f"## 🏙️ {city} – {year}")

        city_data = df[
            (df['city'] == city) &
            (df['date'].dt.year == year)
        ].copy()

        current_filter_period = f"Full Year {year}"
        if month_number_filter:
            city_data = city_data[city_data['date'].dt.month == month_number_filter]
            current_filter_period = f"{selected_month_name} {year}"
        
        st.markdown(f"#### Showing data for **{current_filter_period}**")

        if city_data.empty:
            st.warning(f"No data available for {city} for {current_filter_period}.")
            continue # Skip to next city if no data

        city_data['day_of_year'] = city_data['date'].dt.dayofyear
        city_data['month'] = city_data['date'].dt.month_name() # Use month name for Plotly boxplot
        city_data['day'] = city_data['date'].dt.day
        export_data_list.append(city_data)

        tab1, tab2, tab3 = st.tabs(["📈 Trends & Calendar", "📊 Distributions", "🗓️ Detailed Heatmap"])

        with tab1:
            st.markdown("##### Daily AQI Calendar Heatmap")
            fig_cal, ax_cal = plt.subplots(figsize=(18, 2)) # Adjusted fixed size
            for _, row in city_data.iterrows():
                color = category_colors.get(row['level'], '#FFFFFF') # Default to white if level is missing
                rect = patches.FancyBboxPatch((row['day_of_year'], 0), 1, 1, boxstyle="round,pad=0.1", linewidth=0.5, edgecolor='gray', facecolor=color)
                ax_cal.add_patch(rect)

            ax_cal.set_xlim(0, 367) # Start from 0 for day_of_year
            ax_cal.set_ylim(0, 1)
            ax_cal.axis('off')
            month_starts_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335] # Approximate day of year for month starts
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for day, label in zip(month_starts_days, month_labels):
                ax_cal.text(day, 1.15, label, ha='left', va='bottom', fontsize=9)

            legend_elements = [patches.Patch(facecolor=color, label=label, edgecolor='gray') for label, color in category_colors.items()]
            ax_cal.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="AQI Category", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_cal)

            st.markdown("##### AQI Trend & 7-Day Rolling Average")
            city_data_trend = city_data.sort_values('date').copy()
            city_data_trend['rolling_avg_7day'] = city_data_trend['index'].rolling(window=7, center=True, min_periods=1).mean()

            fig_trend_roll_plotly = px.line(city_data_trend, x='date', y='index', labels={'index': 'Daily AQI'}, custom_data=['level'])
            fig_trend_roll_plotly.update_traces(hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>AQI:</b> %{y}<br><b>Category:</b> %{customdata[0]}<extra></extra>")
            fig_trend_roll_plotly.add_scatter(x=city_data_trend['date'], y=city_data_trend['rolling_avg_7day'], mode='lines', name='7-Day Rolling Avg', line=dict(color='orange'))
            
            fig_trend_roll_plotly.update_layout(
                yaxis_title="AQI Index", xaxis_title="Date", height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_trend_roll_plotly, use_container_width=True)

        with tab2:
            col_dist1, col_dist2 = st.columns(2)
            with col_dist1:
                st.markdown("##### AQI Category Distribution")
                category_counts_df = city_data['level'].value_counts().reindex(category_colors.keys(), fill_value=0).reset_index()
                category_counts_df.columns = ['AQI Category', 'Number of Days']
                fig_dist_plotly = px.bar(
                    category_counts_df, x='AQI Category', y='Number of Days', color='AQI Category',
                    color_discrete_map=category_colors, title="Category Breakdown"
                )
                fig_dist_plotly.update_layout(height=350, xaxis_title=None)
                st.plotly_chart(fig_dist_plotly, use_container_width=True)
            
            with col_dist2:
                st.markdown("##### AQI Category Share")
                if category_counts_df['Number of Days'].sum() > 0:
                    fig_pie_plotly = px.pie(
                        category_counts_df, names='AQI Category', values='Number of Days', color='AQI Category',
                        color_discrete_map=category_colors, title="Category Share", hole=0.3
                    )
                    fig_pie_plotly.update_layout(height=350, legend_title_text='Category')
                    fig_pie_plotly.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie_plotly, use_container_width=True)
                else:
                    st.caption("No data for pie chart.")

            st.markdown("##### Monthly AQI Distribution (Boxplot)")
            # Order months correctly for boxplot
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            city_data_boxplot = city_data.copy()
            city_data_boxplot['month'] = pd.Categorical(city_data_boxplot['month'], categories=month_order, ordered=True)
            city_data_boxplot = city_data_boxplot.sort_values('month')

            fig_box_plotly = px.box(
                city_data_boxplot, x='month', y='index', color='month',
                labels={'index': 'AQI Index', 'month': 'Month'},
                title=f"Monthly AQI Boxplot – {city} {year}",
                notched=True # Optional: for a different style
            )
            fig_box_plotly.update_layout(height=450, xaxis_title=None)
            st.plotly_chart(fig_box_plotly, use_container_width=True)
        
        with tab3:
            st.markdown("##### AQI Heatmap (Month x Day of Month)")
            if not city_data.empty:
                heatmap_data = city_data.pivot_table(index='month', columns='day', values='index', observed=False)
                # Reorder months for heatmap
                heatmap_data = heatmap_data.reindex(month_order)

                fig_heat_plotly = px.imshow(
                    heatmap_data,
                    labels=dict(x="Day of Month", y="Month", color="AQI"),
                    x=heatmap_data.columns, 
                    y=heatmap_data.index,
                    aspect="auto",
                    color_continuous_scale="YlOrRd"
                )
                fig_heat_plotly.update_layout(title=f"AQI Heatmap – {city} {year}", height=500)
                fig_heat_plotly.update_xaxes(side="top")
                st.plotly_chart(fig_heat_plotly, use_container_width=True)
            else:
                st.caption("No data for detailed heatmap.")


# ------------------- 📈 City-wise AQI Time Series Comparison -------------------
if len(selected_cities) > 1:
    st.markdown(f"<hr style='border:1px solid #DDD'>", unsafe_allow_html=True)
    st.markdown("## 📈 AQI Comparison Across Selected Cities")

    comparison_data_list = []
    for city_comp in selected_cities:
        city_ts = df[
            (df['city'] == city_comp) &
            (df['date'].dt.year == year)
        ].copy()
        if month_number_filter:
            city_ts = city_ts[city_ts['date'].dt.month == month_number_filter]
        city_ts = city_ts.sort_values('date')
        if not city_ts.empty:
            city_ts['city_label'] = city_comp # For Plotly legend
            comparison_data_list.append(city_ts)

    if comparison_data_list:
        combined_comparison_df = pd.concat(comparison_data_list)
        fig_cmp_plotly = px.line(
            combined_comparison_df,
            x='date',
            y='index',
            color='city_label', # Use the new column for color
            labels={'index': 'AQI Index', 'date': 'Date', 'city_label': 'City'},
            markers=False # Can set to True if desired
        )
        title_suffix_comp = f"{selected_month_name} {year}" if selected_month_name != "All" else f"Full Year {year}"
        fig_cmp_plotly.update_layout(
            title=f"AQI Trends Across Cities – {title_suffix_comp}",
            height=450,
            legend_title_text='City'
        )
        st.plotly_chart(fig_cmp_plotly, use_container_width=True)
    else:
        st.warning("No data available for comparison with the current filters.")

# ------------------- Pollutant Colors -------------------
pollutant_colors = {
    'PM2.5': '#F8766D', 'PM10': '#7CAE00', 'NO2': '#00BFC4',
    'SO2': '#C77CFF', 'CO': '#E69F00', 'O3': '#619CFF', 'Other': '#A9A9A9'
}
# Ensure 'Other' is in df['pollutant'] if it's a category, or handle np.nan
df['pollutant'] = df['pollutant'].fillna('Other')


# ------------------- 📊 Chart A: Year-wise Prominent Pollutants (ALL years) -------------------
st.markdown(f"<hr style='border:1px solid #DDD'>", unsafe_allow_html=True)
st.markdown("## 📊 Prominent Pollutants by Year (Overall Trend)")

if 'city' in df.columns:
    city_for_pollutant_A = st.selectbox(
        "Select a city for overall year-wise pollutant view:",
        unique_cities if 'unique_cities' in locals() else sorted(df['city'].unique()),
        key="pollutant_city_A",
        index=unique_cities.index(default_city[0]) if default_city and default_city[0] in unique_cities else 0
    )

    yearly_data_A = df[df['city'] == city_for_pollutant_A].copy()
    yearly_data_A = yearly_data_A.dropna(subset=['pollutant']) # Still drop if pollutant became 'Other' from actual NaN
    yearly_data_A['year_label'] = yearly_data_A['date'].dt.year # Use 'year_label' to avoid conflict

    if not yearly_data_A.empty:
        grouped_yearly = yearly_data_A.groupby(['year_label', 'pollutant']).size().unstack(fill_value=0)
        percent_yearly = grouped_yearly.apply(lambda x: x / x.sum() * 100, axis=1).fillna(0)
        
        percent_yearly_df_long = percent_yearly.reset_index().melt(id_vars='year_label', var_name='pollutant', value_name='percentage')
        
        fig_yearly_plotly = px.bar(
            percent_yearly_df_long, x='year_label', y='percentage', color='pollutant',
            title=f"Prominent Pollutants Over the Years – {city_for_pollutant_A}",
            labels={'percentage': 'Percentage (%)', 'year_label': 'Year', 'pollutant': 'Pollutant'},
            color_discrete_map=pollutant_colors
        )
        fig_yearly_plotly.update_layout(xaxis_type='category', yaxis_ticksuffix="%", height=500)
        st.plotly_chart(fig_yearly_plotly, use_container_width=True)
    else:
        st.warning(f"No pollutant data to display for {city_for_pollutant_A} (Overall Trend).")
else:
    st.warning("City information unavailable for pollutant analysis.")


# ------------------- 📊 Chart B: Sidebar-Filtered Pollutants by Year (Selected) -------------------
st.markdown(f"<hr style='border:1px solid #DDD'>", unsafe_allow_html=True)
st.markdown("## 📊 Prominent Pollutants – Based on Sidebar Filters")

if 'city' in df.columns:
    city_for_pollutant_B = st.selectbox(
        "Select a city for filter-based pollutant view:",
        unique_cities if 'unique_cities' in locals() else sorted(df['city'].unique()),
        key="pollutant_city_B",
        index=unique_cities.index(default_city[0]) if default_city and default_city[0] in unique_cities else 0
    )

    filtered_data_B = df[
        (df['city'] == city_for_pollutant_B) &
        (df['date'].dt.year == year)
    ].copy()

    if month_number_filter:
        filtered_data_B = filtered_data_B[filtered_data_B['date'].dt.month == month_number_filter]

    if not filtered_data_B.empty:
        filtered_data_B = filtered_data_B.dropna(subset=['pollutant'])
        filtered_data_B['year_label'] = filtered_data_B['date'].dt.year # Use 'year_label'

        if not filtered_data_B.empty:
            grouped_filtered = filtered_data_B.groupby(['year_label', 'pollutant']).size().unstack(fill_value=0)
            percent_filtered = grouped_filtered.apply(lambda x: x / x.sum() * 100, axis=1).fillna(0)
            percent_filtered_df_long = percent_filtered.reset_index().melt(id_vars='year_label', var_name='pollutant', value_name='percentage')

            title_suffix_B = f"{selected_month_name} {year}" if selected_month_name != "All" else f"{year}"
            fig_filtered_plotly = px.bar(
                percent_filtered_df_long, x='year_label', y='percentage', color='pollutant',
                title=f"Prominent Pollutants – {city_for_pollutant_B}, {title_suffix_B}",
                labels={'percentage': 'Percentage (%)', 'year_label': 'Year', 'pollutant': 'Pollutant'},
                color_discrete_map=pollutant_colors
            )
            fig_filtered_plotly.update_layout(xaxis_type='category', yaxis_ticksuffix="%", height=500)
            st.plotly_chart(fig_filtered_plotly, use_container_width=True)
        else:
            st.warning(f"No pollutant data available for {city_for_pollutant_B} with selected filters (Year: {year}, Month: {selected_month_name}).")
    else:
        st.warning(f"No data available for {city_for_pollutant_B} with selected filters (Year: {year}, Month: {selected_month_name}).")
else:
    st.warning("City information unavailable for pollutant analysis.")


# ------------------- 📈 AQI Trendline Forecast -------------------
st.markdown(f"<hr style='border:1px solid #DDD'>", unsafe_allow_html=True)
st.markdown("## 📈 AQI Forecast – Linear Trendline (Next 15 Days)")

if 'city' in df.columns:
    forecast_city = st.selectbox(
        "Select a city for AQI forecast:",
        unique_cities if 'unique_cities' in locals() else sorted(df['city'].unique()),
        key="forecast_city",
        index=unique_cities.index(default_city[0]) if default_city and default_city[0] in unique_cities else 0
    )

    forecast_data_source = df[
        (df['city'] == forecast_city) &
        (df['date'].dt.year == year)
    ].copy()

    if month_number_filter:
        forecast_data_source = forecast_data_source[forecast_data_source['date'].dt.month == month_number_filter]

    if len(forecast_data_source) >= 15: # Threshold for meaningful forecast
        forecast_data = forecast_data_source.sort_values('date').copy()
        forecast_data = forecast_data[['date', 'index']].dropna()

        if len(forecast_data) >= 2: # Need at least 2 points for regression
            forecast_data['days_since_start'] = (forecast_data['date'] - forecast_data['date'].min()).dt.days
            X = forecast_data[['days_since_start']]
            y = forecast_data['index']

            model = LinearRegression()
            model.fit(X, y)

            future_days_count = 15
            last_day_in_data = forecast_data['days_since_start'].max()
            
            future_X_values = np.arange(0, last_day_in_data + future_days_count + 1)
            future_X_df = pd.DataFrame({'days_since_start': future_X_values})
            future_y_pred = model.predict(future_X_df)
            
            min_date_for_forecast = forecast_data['date'].min()
            future_dates = [min_date_for_forecast + timedelta(days=int(i)) for i in future_X_values]
            
            plot_df_observed = pd.DataFrame({'date': forecast_data['date'], 'AQI': y, 'Type': 'Observed'})
            plot_df_forecast = pd.DataFrame({'date': future_dates, 'AQI': future_y_pred, 'Type': 'Forecast'})
            
            # Clip forecast to not go below 0
            plot_df_forecast['AQI'] = np.maximum(0, plot_df_forecast['AQI'])

            fig_forecast_plotly = px.line(plot_df_observed, x='date', y='AQI', title=f"AQI Forecast – {forecast_city}")
            fig_forecast_plotly.data[0].name = 'Observed AQI' # Rename trace
            fig_forecast_plotly.data[0].showlegend = True

            fig_forecast_plotly.add_scatter(
                x=plot_df_forecast['date'], y=plot_df_forecast['AQI'], mode='lines',
                name='Forecast (Linear Trend)', line=dict(dash='dash', color='red')
            )
            
            forecast_period_title = f"{selected_month_name} {year}" if selected_month_name != "All" else f"Full Year {year}"
            fig_forecast_plotly.update_layout(
                title=f"AQI Forecast – {forecast_city} ({forecast_period_title})",
                yaxis_title="AQI Index", xaxis_title="Date", height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_forecast_plotly, use_container_width=True)
        else:
            st.warning(f"Not enough valid data points (after dropping NaN) for {forecast_city} in the selected period to create a forecast.")
    else:
        st.warning(f"Not enough data points (found {len(forecast_data_source)}, need at least 15) for {forecast_city} in the selected period to create a forecast. Select a different city or time range.")
else:
    st.warning("City information unavailable for AQI forecast.")


# ------------------- Load City Coordinates -------------------
city_coords = {} # Initialize
try:
    with open("lat_long.txt", "r") as f:
        lines = f.readlines()
        # Robust parsing: find the first '{' and last '}'
        dict_text_full = ''.join(lines)
        start_brace = dict_text_full.find('{')
        end_brace = dict_text_full.rfind('}')
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            dict_text = dict_text_full[start_brace : end_brace+1]
            city_coords = eval(dict_text)
        else:
            st.error("Could not find valid dictionary structure in lat_long.txt")

except FileNotFoundError:
    st.error("`lat_long.txt` not found. Map cannot be displayed.")
except Exception as e:
    st.error(f"Error parsing `lat_long.txt`: {e}. Map cannot be displayed.")

if city_coords:
    latlong_df = pd.DataFrame([
        {'city': city_name, 'lat': coords[0], 'lon': coords[1]}
        for city_name, coords in city_coords.items()
    ])

    # ------------------- 🗺️ Improved Interactive AQI Map -------------------
    st.markdown(f"<hr style='border:1px solid #DDD'>", unsafe_allow_html=True)
    st.markdown("## 🗺️ Interactive Air Quality Map – City-wise")

    def classify_aqi(val): # Keep original classification for map
        if pd.isna(val): return "Unknown"
        if val <= 50: return "Good"
        if val <= 100: return "Satisfactory"
        if val <= 200: return "Moderate"
        if val <= 300: return "Poor"
        if val <= 400: return "Very Poor"
        return "Severe"

    map_data_filter = df.copy()
    map_data_filter = map_data_filter[map_data_filter['date'].dt.year == year]

    if month_number_filter:
        map_data_filter = map_data_filter[map_data_filter['date'].dt.month == month_number_filter]

    if not map_data_filter.empty:
        map_grouped = map_data_filter.groupby('city').agg(
            avg_aqi=('index', 'mean'),
            dominant_pollutant=('pollutant', lambda x: x.mode().iloc[0] if not x.mode().empty and not x.mode().isnull().all() else 'N/A')
        ).reset_index()

        map_merged = pd.merge(map_grouped, latlong_df, on='city', how='inner')
        map_merged["AQI Category"] = map_merged["avg_aqi"].apply(classify_aqi)
        
        map_aqi_categories = ["All"] + list(category_colors.keys())
        selected_aqi_category_map = st.selectbox("🧪 Filter Map by AQI Category", map_aqi_categories, index=0, key="map_aqi_filter")

        if selected_aqi_category_map != "All":
            map_merged_display = map_merged[map_merged["AQI Category"] == selected_aqi_category_map]
        else:
            map_merged_display = map_merged.copy()
        
        if not map_merged_display.empty:
            fig_map = px.scatter_mapbox(
                map_merged_display, lat="lat", lon="lon", size="avg_aqi", size_max=25,
                color="AQI Category", color_discrete_map=category_colors,
                hover_name="city",
                hover_data={
                    "avg_aqi": ":.2f", "dominant_pollutant": True,
                    "AQI Category": True, "lat": False, "lon": False
                },
                zoom=4.5, center={"lat": 22.5, "lon": 82.0}, # Centered more on India
                height=700 
            )
            fig_map.update_layout(
                mapbox_style="carto-positron", # Light map style
                title_text=f"Average AQI by City ({selected_month_name} {year})",
                legend_title="AQI Category",
                margin={"r": 0, "t": 40, "l": 0, "b": 0}
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("No cities match the selected AQI category filter for the map.")
    else:
        st.warning(f"No data available for the map for the selected period (Year: {year}, Month: {selected_month_name}).")
else:
    st.warning("City coordinates not loaded. Map cannot be displayed.")


# ------------------- Download Filtered Data -------------------
if export_data_list: # Check if list is populated
    st.markdown(f"<hr style='border:1px solid #DDD'>", unsafe_allow_html=True)
    combined_export_df = pd.concat(export_data_list)
    csv_buffer = StringIO()
    combined_export_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📤 Download Filtered City Data as CSV",
        data=csv_buffer.getvalue(),
        file_name=f"filtered_air_quality_{year}_{selected_month_name if selected_month_name != 'All' else 'AllMonths'}.csv",
        mime="text/csv"
    )

# ------------------- Footer -------------------
st.markdown("---")
st.caption("📊 Data Source: Central Pollution Control Board (CPCB), India (Data is illustrative). Latitude/Longitude are approximate.")
st.markdown("""
**Developed by:** Mr. [Kapil Meena](https://sites.google.com/view/kapil-lab/home) 
Doctoral Scholar, IIT Kharagpur  
📧 kapil.meena@kgpian.iitkgp.ac.in 

**With guidance from:** [Prof. Arkopal K. Goswami, PhD](https://www.mustlab.in/faculty) 
""")
st.markdown("🔗 [View on GitHub (Source code)](https://github.com/kapil2020/india-air-quality-dashboard)")

# ------------------- Mobile Friendly Styles -------------------
st.markdown("""
<style>
@media screen and (max-width: 768px) {
    .stPlotlyChart { /* Target Plotly charts specifically for min-height on mobile */
        min-height: 350px;
    }
    /* Reduce padding slightly on mobile for main container */
    .main .block-container { 
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
/* Consistent header styles */
h2 {
    border-bottom: 2px solid #4CAF50; /* A green accent */
    padding-bottom: 0.3rem;
    color: #333;
}
h4 {
    color: #555;
}
h5 {
    color: #007E00; /* Green for sub-section titles like plot titles */
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}
/* Custom styling for horizontal rule */
hr {
  margin-top: 2rem !important;
  margin-bottom: 1.5rem !important;
}
</style>
""", unsafe_allow_html=True)

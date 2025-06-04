import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import StringIO
from sklearn.linear_model import LinearRegression
import os
from datetime import date
from sklearn.metrics import mean_squared_error
import plotly.express as px
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

# Detect screen width for responsive design
def get_device_type():
    return "mobile" if st.session_state.get("screen_width", 1000) < 768 else "desktop"

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
- **Calendar-style daily AQI heatmaps** for a visual snapshot of air quality
- **Daily AQI trends** to monitor changes over time
- **AQI category breakdowns** with bar and pie charts
- **Monthly AQI boxplots** to see distribution across months
- **7-day rolling average AQI line** for smoothed trends
- **Day vs. Month heatmap** to spot seasonal patterns
- **100% Stacked Bar Charts** showing prominent pollutants (yearly and filtered views)
- **AQI forecast** with linear trendline predictions
- **Interactive AQI map** with city-wise averages, categories, and dominant pollutants

📤 Download the filtered dataset as a CSV using the button below the charts. 
""")

# ------------------- Load Data -------------------
@st.cache_data(ttl=3600)
def load_data():
    today = date.today()
    csv_path = f"data/{today}.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(today)
        return df, True
    else:
        st.warning(f"No AQI report found for today ({today}).")
        df = pd.read_csv("combined_air_quality.txt", sep="\t", parse_dates=['date'])
        return df, False

df, is_today = load_data()

# ------------------- Sidebar Filters -------------------
st.sidebar.markdown(
    f"📊 Data available from **{df['date'].min().date()}** to **{df['date'].max().date()}**"
)

selected_cities = st.sidebar.multiselect("Select Cities", sorted(df['city'].unique()), default=["Delhi"])

years = sorted(df['date'].dt.year.unique())
default_year = max(years)
year = st.sidebar.selectbox("Select a Year", years, index=years.index(default_year))

# Month Dropdown
months_dict = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
month_options = ["All"] + [months_dict[m] for m in range(1, 13)]
selected_month = st.sidebar.selectbox("Select Month (optional)", month_options, index=0)

# AQI Category Colors
category_colors = {
    'Severe': '#7E0023',
    'Very Poor': '#FF0000',
    'Poor': '#FF7E00',
    'Moderate': '#FFFF00',
    'Satisfactory': '#00E400',
    'Good': '#007E00'
}

# ------------------- Dashboard Body -------------------
export_data = []

for city in selected_cities:
    st.markdown(f"## {city} – {year}")
    
    city_data = df[
        (df['city'] == city) &
        (df['date'].dt.year == year)
    ].copy()

    if selected_month != "All":
        month_number = [k for k, v in months_dict.items() if v == selected_month][0]
        city_data = city_data[city_data['date'].dt.month == month_number]
        st.markdown(f"### Showing data for **{selected_month} {year}**")
    else:
        st.markdown(f"### Showing data for **Full Year {year}**")
    
    city_data['day_of_year'] = city_data['date'].dt.dayofyear
    city_data['month'] = city_data['date'].dt.month
    city_data['day'] = city_data['date'].dt.day
    export_data.append(city_data)

    fig_width = 20 if st.session_state.get("device", "desktop") == "desktop" else 10

    # Calendar Heatmap
    st.markdown("#### Calendar Heatmap")
    fig, ax = plt.subplots(figsize=(fig_width, 2))
    for _, row in city_data.iterrows():
        color = category_colors.get(row['level'], '#FFFFFF')
        rect = patches.FancyBboxPatch((row['day_of_year'], 0), 1, 1, boxstyle="round,pad=0.1", linewidth=0, facecolor=color)
        ax.add_patch(rect)

    ax.set_xlim(1, 367)
    ax.set_ylim(0, 1)
    ax.axis('off')
    for day, label in zip([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
                          ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
        ax.text(day, 1.05, label, ha='center', fontsize=10)

    legend_elements = [patches.Patch(facecolor=color, label=label) for label, color in category_colors.items()]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="AQI Category")
    st.pyplot(fig)

    # AQI Trend
    st.markdown("#### AQI Trend")
    fig2, ax2 = plt.subplots(figsize=(fig_width * 0.8, 3))
    ax2.plot(city_data['date'], city_data['index'], marker='o', linestyle='-', markersize=3)
    ax2.set_ylabel("AQI Index")
    ax2.set_xlabel("Date")
    ax2.set_title(f"AQI Trend for {city} in {year}")
    ax2.grid(True)
    st.pyplot(fig2)


    # Rolling Average
    st.markdown("#### 7-Day Rolling Average AQI")
    fig_roll, ax_roll = plt.subplots(figsize=(fig_width * 0.8, 3))
    city_data['rolling'] = city_data['index'].rolling(window=7).mean()
    ax_roll.plot(city_data['date'], city_data['rolling'], color='orange')
    ax_roll.set_title(f"7-Day Rolling AQI Average – {city}")
    ax_roll.set_ylabel("AQI")
    ax_roll.set_xlabel("Date")
    ax_roll.grid(True)
    st.pyplot(fig_roll)

    # AQI Category Distribution
    st.markdown("#### AQI Category Distribution")
    category_counts = city_data['level'].value_counts().reindex(category_colors.keys(), fill_value=0)
    fig3, ax3 = plt.subplots(figsize=(fig_width * 0.5, 3))
    ax3.bar(category_counts.index, category_counts.values, color=[category_colors[k] for k in category_counts.index])
    ax3.set_ylabel("Number of Days")
    ax3.set_title(f"AQI Category Breakdown - {city} ({year})")
    st.pyplot(fig3)

    # Pie Chart
    st.markdown("#### AQI Category Share (Pie Chart)")
    fig_pie, ax_pie = plt.subplots(figsize=(fig_width * 0.4, fig_width * 0.4))
    ax_pie.pie(category_counts.values, labels=category_counts.index, autopct="%1.1f%%", colors=[category_colors[k] for k in category_counts.index])
    ax_pie.set_title(f"AQI Category Proportions – {city} {year}")
    st.pyplot(fig_pie)

    # Box Plot by Month
    st.markdown("#### Monthly AQI Distribution (Boxplot)")
    fig_box, ax_box = plt.subplots(figsize=(fig_width * 0.5, 4))
    city_data.boxplot(column='index', by='month', ax=ax_box)
    ax_box.set_title(f"Monthly AQI Boxplot – {city} {year}")
    ax_box.set_ylabel("AQI")
    ax_box.set_xlabel("Month")
    plt.suptitle("")
    st.pyplot(fig_box)

    # Heatmap by Month & Day
    st.markdown("#### AQI Heatmap (Month x Day)")
    heatmap_data = city_data.pivot_table(index='month', columns='day', values='index')
    fig_heat, ax_heat = plt.subplots(figsize=(fig_width * 0.6, 4))
    c = ax_heat.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', origin='lower')
    ax_heat.set_title(f"AQI Heatmap – {city} {year}")
    ax_heat.set_xlabel("Day of Month")
    ax_heat.set_ylabel("Month")
    fig_heat.colorbar(c, ax=ax_heat, label='AQI')
    st.pyplot(fig_heat)

# ------------------- 📈 City-wise AQI Time Series Comparison -------------------
if len(selected_cities) > 1:
    st.markdown("## 📈 AQI Comparison Across Selected Cities")

    fig_cmp, ax_cmp = plt.subplots(figsize=(14, 4))

    for city in selected_cities:
        city_ts = df[
            (df['city'] == city) &
            (df['date'].dt.year == year)
        ]
        if selected_month != "All":
            month_number = [k for k, v in months_dict.items() if v == selected_month][0]
            city_ts = city_ts[city_ts['date'].dt.month == month_number]
        city_ts = city_ts.sort_values('date')
        ax_cmp.plot(city_ts['date'], city_ts['index'], marker='o', label=city, linewidth=1.5, markersize=2)

    ax_cmp.set_title(f"AQI Trends Across Cities – {year}" + (f", {selected_month}" if selected_month != "All" else ""), fontsize=14)
    ax_cmp.set_xlabel("Date")
    ax_cmp.set_ylabel("AQI Index")
    ax_cmp.grid(True, linestyle='--', alpha=0.6)
    ax_cmp.legend(loc='upper right')
    st.pyplot(fig_cmp)
# ------------------- Pollutant Colors (ggplot2 style) -------------------
pollutant_colors = {
    'PM2.5': '#F8766D',
    'PM10': '#7CAE00',
    'NO2': '#00BFC4',
    'SO2': '#C77CFF',
    'CO': '#E69F00',
    'O3': '#619CFF'
}

# Clean pollutant column
df['pollutant'] = df['pollutant'].astype(str).str.split(',').str[0].str.strip()
df['pollutant'].replace(['nan', 'NaN', 'None', ''], np.nan, inplace=True)

# ------------------- 📊 Chart A: Year-wise Prominent Pollutants (ALL years) -------------------
st.markdown("## 📊 Prominent Pollutants by Year (Overall Trend)")

city_for_pollutant_plot = st.selectbox("Select a city for overall year-wise view:", sorted(df['city'].unique()))

yearly_data = df[df['city'] == city_for_pollutant_plot].copy()
yearly_data = yearly_data.dropna(subset=['pollutant'])
yearly_data['year'] = yearly_data['date'].dt.year

grouped_yearly = yearly_data.groupby(['year', 'pollutant']).size().unstack(fill_value=0)
percent_yearly = grouped_yearly.div(grouped_yearly.sum(axis=1), axis=0) * 100

fig_yearly, ax_yearly = plt.subplots(figsize=(10, 5))
bottoms = np.zeros(len(percent_yearly))

for pollutant in pollutant_colors:
    if pollutant in percent_yearly.columns:
        vals = percent_yearly[pollutant].values
        ax_yearly.bar(percent_yearly.index, vals, bottom=bottoms, label=pollutant, color=pollutant_colors[pollutant])
        bottoms += vals

ax_yearly.set_title(f"Prominent Pollutants Over the Years – {city_for_pollutant_plot}")
ax_yearly.set_ylabel("Percentage")
ax_yearly.set_xlabel("Year")
ax_yearly.set_ylim(0, 100)
ax_yearly.set_xticks(percent_yearly.index)
ax_yearly.set_xticklabels(percent_yearly.index, rotation=45)
ax_yearly.legend(title="Pollutant", bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig_yearly)

# ------------------- 📊 Chart B: Sidebar-Filtered Pollutants by Year (Selected) -------------------
st.markdown("## 📊 Prominent Pollutants – Based on Sidebar Filters")

filtered_data = df[
    (df['city'] == city_for_pollutant_plot) &
    (df['date'].dt.year == year)
].copy()

if selected_month != "All":
    filtered_data = filtered_data[filtered_data['date'].dt.month == month_number]

if not filtered_data.empty:
    filtered_data = filtered_data.dropna(subset=['pollutant'])
    filtered_data['year'] = filtered_data['date'].dt.year

    grouped_filtered = filtered_data.groupby(['year', 'pollutant']).size().unstack(fill_value=0)
    percent_filtered = grouped_filtered.div(grouped_filtered.sum(axis=1), axis=0) * 100

    fig_filtered, ax_filtered = plt.subplots(figsize=(10, 5))
    bottoms = np.zeros(len(percent_filtered))

    for pollutant in pollutant_colors:
        if pollutant in percent_filtered.columns:
            vals = percent_filtered[pollutant].values
            ax_filtered.bar(percent_filtered.index, vals, bottom=bottoms, label=pollutant, color=pollutant_colors[pollutant])
            bottoms += vals

    title_suffix = f"{selected_month} {year}" if selected_month != "All" else f"{year}"
    ax_filtered.set_title(f"Prominent Pollutants – {city_for_pollutant_plot}, {title_suffix}")
    ax_filtered.set_ylabel("Percentage")
    ax_filtered.set_xlabel("Year")
    ax_filtered.set_ylim(0, 100)
    ax_filtered.set_xticks(percent_filtered.index)
    ax_filtered.set_xticklabels(percent_filtered.index, rotation=45)
    ax_filtered.legend(title="Pollutant", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig_filtered)
else:
    st.warning("No data available for the selected city and time period.")



# ------------------- 📈 AQI Trendline Forecast -------------------
st.markdown("## 📈 AQI Forecast – Linear Trendline")

forecast_city = st.selectbox("Select a city for AQI forecast:", sorted(df['city'].unique()), index=0)

# Filter data
forecast_data = df[
    (df['city'] == forecast_city) &
    (df['date'].dt.year == year)
].copy()

if selected_month != "All":
    forecast_data = forecast_data[forecast_data['date'].dt.month == month_number]

# Continue only if there's enough data
if len(forecast_data) >= 15:
    forecast_data = forecast_data.sort_values('date')
    forecast_data = forecast_data[['date', 'index']].dropna()

    # Prepare data for regression
    forecast_data['days_since_start'] = (forecast_data['date'] - forecast_data['date'].min()).dt.days
    X = forecast_data[['days_since_start']]
    y = forecast_data['index']

    # Linear Regression
    model = LinearRegression()
    model.fit(X, y)

    # Predict existing & future days
    future_days = 15
    total_days = forecast_data['days_since_start'].max() + future_days
    future_X = pd.DataFrame({'days_since_start': np.arange(0, total_days + 1)})
    future_y_pred = model.predict(future_X)

    # Dates for future prediction
    future_dates = [forecast_data['date'].min() + pd.Timedelta(days=int(i)) for i in future_X['days_since_start']]
    
    # Plot
    fig_forecast, ax_forecast = plt.subplots(figsize=(10, 4))
    ax_forecast.plot(forecast_data['date'], y, 'bo-', label='Observed AQI')
    ax_forecast.plot(future_dates, future_y_pred, 'r--', label='Forecast (Linear Trend)', linewidth=2)

    ax_forecast.set_title(f"AQI Forecast – {forecast_city} ({selected_month if selected_month != 'All' else 'Full Year'} {year})")
    ax_forecast.set_xlabel("Date")
    ax_forecast.set_ylabel("AQI Index")
    ax_forecast.legend()
    ax_forecast.grid(True)
    st.pyplot(fig_forecast)
else:
    st.warning("Not enough data for forecast. Select a different city or time range.")

# ------------------- Load City Coordinates -------------------
with open(r"lat_long.txt", "r") as f:
    lines = f.readlines()
    dict_text = ''.join(lines[1:])  # Skip the first line (e.g., 'city_coords = {')
    city_coords = eval("{" + dict_text)  # Add the opening brace back

# Convert to DataFrame
latlong_df = pd.DataFrame([
    {'city': city, 'lat': coords[0], 'lon': coords[1]}
    for city, coords in city_coords.items()
])

# ------------------- 🗺️ Improved Interactive AQI Map -------------------
st.markdown("## 🗺️ Interactive Air Quality Map – City-wise")

# Assign AQI category to map_merged
def classify_aqi(val):
    if val <= 50:
        return "Good"
    elif val <= 100:
        return "Satisfactory"
    elif val <= 200:
        return "Moderate"
    elif val <= 300:
        return "Poor"
    elif val <= 400:
        return "Very Poor"
    else:
        return "Severe"

map_data = df.copy()
map_data['year'] = map_data['date'].dt.year
map_data = map_data[map_data['year'] == year]

if selected_month != "All":
    map_data = map_data[map_data['date'].dt.month == month_number]

map_grouped = map_data.groupby('city').agg({
    'index': 'mean',
    'pollutant': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
}).reset_index().rename(columns={'index': 'avg_aqi', 'pollutant': 'dominant_pollutant'})

map_merged = pd.merge(map_grouped, latlong_df, on='city', how='inner')
map_merged["AQI Category"] = map_merged["avg_aqi"].apply(classify_aqi)
# Dropdown filter for AQI Category
# Custom color scale similar to CPCB categories
aqi_colors = {
    "Good": "#007E00",
    "Satisfactory": "#00E400",
    "Moderate": "#FFFF00",
    "Poor": "#FF7E00",
    "Very Poor": "#FF0000",
    "Severe": "#7E0023"
}
aqi_categories = ["All"] + list(aqi_colors.keys())
selected_aqi_category = st.selectbox("🧪 Filter by AQI Category", aqi_categories, index=0)

# Filter data if category is selected
if selected_aqi_category != "All":
    map_merged = map_merged[map_merged["AQI Category"] == selected_aqi_category]


fig_map = px.scatter_mapbox(
    map_merged,
    lat="lat",
    lon="lon",
    size="avg_aqi",
    size_max=30,
    color="AQI Category",
    color_discrete_map=aqi_colors,
    hover_name="city",
    hover_data={
        "avg_aqi": True,
        "dominant_pollutant": True,
        "AQI Category": True,
        "lat": False,
        "lon": False
    },
    zoom=4,
    height=800,
)

fig_map.update_layout(
    mapbox_style="carto-positron",
    title="Average AQI by City with Categories",
    legend_title="AQI Category",
    margin={"r": 0, "t": 40, "l": 0, "b": 0}
)

st.plotly_chart(fig_map, use_container_width=True)





# ------------------- Download Filtered Data -------------------
if export_data:
    combined_export = pd.concat(export_data)
    csv_buffer = StringIO()
    combined_export.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📤 Download Filtered Data as CSV",
        data=csv_buffer.getvalue(),
        file_name="filtered_air_quality_data.csv",
        mime="text/csv"
    )

# ------------------- Footer -------------------
st.markdown("---")
st.caption("📊 Data Source: Central Pollution Control Board (India)")
st.markdown("""
**Developed by:**  
Mr. [Kapil Meena](https://sites.google.com/view/kapil-lab/home)  
Doctoral Scholar, IIT Kharagpur  
📧 kapil.meena@kgpian.iitkgp.ac.in  

**With guidance from:**  
[Prof. Arkopal K. Goswami, PhD](https://www.mustlab.in/faculty)  
Associate Professor, Chairperson  
RCGSIDM, IIT Kharagpur  
📧 akgoswami@infra.iitkgp.ac.in
""")
st.markdown("🔗 [View on GitHub](https://github.com/kapil2020/india-air-quality-dashboard)")

# ------------------- Mobile Friendly Styles -------------------
st.markdown("""
<style>
@media screen and (max-width: 768px) {
    .element-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


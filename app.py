import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import StringIO
import scipy.interpolate as interp
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Set page config
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# ------------------- Title -------------------
st.title("🇮🇳 India Air Quality Dashboard")

# ------------------- Introduction -------------------
st.info("""
Welcome to the **India Air Quality Dashboard** 🇮🇳

🔍 Use the **sidebar** to:
- Select one or more cities
- Choose a specific year (defaults to 2024)

📊 View:
- Calendar-style daily AQI heatmaps
- Daily AQI trends
- AQI category breakdowns
- Monthly AQI boxplots
- Rolling average AQI line
- Category pie chart
- Day vs. Month heatmap

📤 Download the filtered dataset using the button below the charts.
""")

# ------------------- Load Data -------------------
data_path = "combined_air_quality.txt"

@st.cache_data(ttl=600)
def load_data():
    return pd.read_csv(data_path, sep='\t', parse_dates=['date'])

df = load_data()

# ------------------- Sidebar Filters -------------------
selected_cities = st.sidebar.multiselect("Select Cities", sorted(df['city'].unique()), default=["Delhi"])
years = sorted(df['date'].dt.year.unique())
year = st.sidebar.selectbox("Select a Year", years, index=years.index(2024) if 2024 in years else 0)

# AQI Category Colors
category_colors = {
    'Severe': '#7E0023',
    'Very Poor': '#FF0000',
    'Poor': '#FF7E00',
    'Moderate': '#FFFF00',
    'Satisfactory': '#00E400',
    'Good': '#007E00'
}

# City coordinates (latitude, longitude)
city_coords = {
    "Agra": [27.1767, 78.0081],
    "Bengaluru": [12.9716, 77.5946],
    "Chennai": [13.0827, 80.2707],
    "Delhi": [28.7041, 77.1025],
    "Faridabad": [28.4089, 77.3178],
    "Gaya": [24.7955, 85.0119],
    "Haldia": [22.0257, 88.0583],
    "Hyderabad": [17.3850, 78.4867],
    "Jodhpur": [26.2389, 73.0243],
    "Kanpur": [26.4499, 80.3319],
    "Lucknow": [26.8467, 80.9462],
    "Mumbai": [19.0760, 72.8777],
    "Muzaffarpur": [26.1209, 85.3647],
    "Navi Mumbai": [19.0330, 73.0297],
    "Panchkula": [30.6942, 76.8606],
    "Patna": [25.5941, 85.1376],
    "Pune": [18.5204, 73.8567],
    "Varanasi": [25.3176, 82.9739],
    "Chandrapur": [19.9615, 79.2961],
    "Jaipur": [26.9124, 75.7873]
}

# ------------------- Spatial Map -------------------
st.markdown(f"### 🗺️ India AQI Heatmap – {year}")

# Collect data for the selected year
map_data = []
for city, coords in city_coords.items():
    city_data = df[(df['city'] == city) & (df['date'].dt.year == year)]
    if not city_data.empty:
        lat, lon = coords
        avg_aqi = city_data['index'].mean()
        map_data.append([lon, lat, avg_aqi, city])  # [longitude, latitude, AQI, city]

if map_data:
    map_array = np.array(map_data, dtype=object)
    x = map_array[:, 0].astype(float)
    y = map_array[:, 1].astype(float)
    z = map_array[:, 2].astype(float)
    cities = map_array[:, 3]

    # Create interpolation grid
    grid_x, grid_y = np.mgrid[67:98:300j, 6:37:300j]  # Full India coverage
    
    # Perform interpolation
    grid_z = interp.griddata((x, y), z, (grid_x, grid_y), method='linear')

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Set map boundaries
    ax.set_extent([67, 98, 6, 37], crs=ccrs.PlateCarree())

    # Add geographic features
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5')
    ax.add_feature(cfeature.OCEAN, facecolor='#a3ccff')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.STATES, linestyle=':', linewidth=0.3)

    # Plot interpolated data
    contour = ax.contourf(grid_x, grid_y, grid_z, 60,
                         cmap='RdYlGn_r',
                         transform=ccrs.PlateCarree(),
                         alpha=0.7)

    # Add city markers and labels
    for lon, lat, aqi, city in map_data:
        ax.plot(lon, lat, 'o', markersize=8, color='black', transform=ccrs.PlateCarree())
        ax.text(lon + 0.15, lat + 0.1, city,
                transform=ccrs.PlateCarree(),
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.5)
    cbar.set_label('AQI Index', fontsize=12)

    # Add grid lines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title(f'AQI Distribution Across India - {year}', fontsize=16, pad=20)
    st.pyplot(fig)

else:
    st.warning(f"No data available for {year}")

# ------------------- Dashboard Body -------------------
export_data = []

for city in selected_cities:
    st.markdown(f"## {city} – {year}")
    city_data = df[(df['city'] == city) & (df['date'].dt.year == year)].copy()
    city_data['day_of_year'] = city_data['date'].dt.dayofyear
    city_data['month'] = city_data['date'].dt.month
    city_data['day'] = city_data['date'].dt.day
    export_data.append(city_data)

    # Set responsive figure width
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

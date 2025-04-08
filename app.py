import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import StringIO
import matplotlib

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

# ------------------- Spatial Map -------------------
import scipy.interpolate as interp

st.markdown(f"### 🗺️ India AQI Heatmap – {year}")
map_data = []
for city in city_coords.keys():
    city_data = df[(df['city'] == city) & (df['date'].dt.year == year)]
    if not city_data.empty:
        lat, lon = city_coords[city]
        avg_aqi = city_data['index'].mean()
        map_data.append([lon, lat, avg_aqi, city])  # Added city name

if map_data:
    map_array = np.array(map_data, dtype=object)
    x = map_array[:, 0].astype(float)  # Longitudes
    y = map_array[:, 1].astype(float)  # Latitudes
    z = map_array[:, 2].astype(float)  # AQI values
    cities = map_array[:, 3]           # City names

    # Create a grid covering all of India
    grid_x, grid_y = np.mgrid[67:98:200j, 6:37:200j]  # Expanded India bounds
    
    # Perform IDW interpolation
    grid_z = interp.griddata((x, y), z, (grid_x, grid_y), method='linear')

    # Create the map plot
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set map extent to cover entire India
    ax.set_extent([67, 98, 6, 37], crs=ccrs.PlateCarree())  # Expanded bounds
    
    # Add map features
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')
    ax.add_feature(cfeature.OCEAN, facecolor='#a0d8ef')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    # Plot the heatmap
    heatmap = ax.contourf(grid_x, grid_y, grid_z, 60, 
                         cmap='RdYlGn_r', 
                         transform=ccrs.PlateCarree(),
                         alpha=0.7,
                         vmin=np.nanmin(z), 
                         vmax=np.nanmax(z))

    # Add city markers and labels
    for xi, yi, zi, city in zip(x, y, z, cities):
        ax.plot(xi, yi, 'ko', markersize=5, transform=ccrs.PlateCarree())
        ax.text(xi + 0.2, yi + 0.1, city, 
                transform=ccrs.PlateCarree(),
                fontsize=8,
                ha='left',
                va='center',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax, shrink=0.5)
    cbar.set_label('AQI Index')

    # Add title and grid
    ax.set_title(f"AQI Distribution Across India – {year}", fontsize=14, pad=20)
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)

    st.pyplot(fig)
else:
    st.warning(f"No data available for any cities in {year}")

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


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import StringIO

st.set_page_config(layout="wide")
st.title("🇮🇳 India Air Quality Dashboard")

# Load combined data from .txt file
data_path = "combined_air_quality.txt"
df = pd.read_csv(data_path, sep='\t', parse_dates=['date'])

# Sidebar filters
selected_cities = st.sidebar.multiselect("Select Cities", sorted(df['city'].unique()), default=["Delhi"])
years = sorted(df['date'].dt.year.unique())
year = st.sidebar.selectbox("Select a Year", years)

# AQI category colors
category_colors = {
    'Severe': '#7E0023',
    'Very Poor': '#FF0000',
    'Poor': '#FF7E00',
    'Moderate': '#FFFF00',
    'Satisfactory': '#00E400',
    'Good': '#007E00'
}

# Data to collect for export
export_data = []

# Side-by-side comparison layout
cols = st.columns(len(selected_cities)) if selected_cities else []

for idx, city in enumerate(selected_cities):
    city_data = df[(df['city'] == city) & (df['date'].dt.year == year)].copy()
    city_data['day_of_year'] = city_data['date'].dt.dayofyear
    export_data.append(city_data)

    with cols[idx]:
        st.markdown(f"### {city}")

        # Category distribution chart
        category_counts = city_data['level'].value_counts().reindex(category_colors.keys(), fill_value=0)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(category_counts.index, category_counts.values,
               color=[category_colors[k] for k in category_counts.index])
        ax.set_title("Category Breakdown")
        ax.set_xticklabels(category_counts.index, rotation=45)
        st.pyplot(fig)

# Ranking table
st.markdown("## 🏆 City AQI Rankings")
ranking_data = df[df['date'].dt.year == year].groupby('city')['index'].mean().sort_values()
rank_df = ranking_data.reset_index().rename(columns={'index': 'Average AQI'})
st.dataframe(rank_df.style.background_gradient(cmap='RdYlGn_r'), use_container_width=True)

# Per-city detailed plots
for city in selected_cities:
    st.markdown(f"## 📊 Detailed View – {city} – {year}")
    city_data = df[(df['city'] == city) & (df['date'].dt.year == year)].copy()
    city_data['day_of_year'] = city_data['date'].dt.dayofyear

    # Calendar heatmap
    st.markdown("#### Calendar Heatmap")
    fig, ax = plt.subplots(figsize=(20, 2))
    for _, row in city_data.iterrows():
        color = category_colors.get(row['level'], '#FFFFFF')
        rect = patches.FancyBboxPatch(
            (row['day_of_year'], 0), 1, 1,
            boxstyle="round,pad=0.1",
            linewidth=0, facecolor=color
        )
        ax.add_patch(rect)

    ax.set_xlim(1, 367)
    ax.set_ylim(0, 1)
    ax.axis('off')
    for month_day, label in zip([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
                                 ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
        ax.text(month_day, 1.05, label, ha='center', fontsize=10)

    legend_elements = [patches.Patch(facecolor=color, label=label) for label, color in category_colors.items()]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="AQI Category")
    st.pyplot(fig)

    # AQI trend chart
    st.markdown("#### AQI Trend")
    fig2, ax2 = plt.subplots(figsize=(16, 3))
    ax2.plot(city_data['date'], city_data['index'], marker='o', linestyle='-', markersize=3)
    ax2.set_ylabel("AQI Index")
    ax2.set_xlabel("Date")
    ax2.set_title(f"AQI Trend for {city} in {year}")
    ax2.grid(True)
    st.pyplot(fig2)

# Export button
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

st.markdown("---")
st.caption("Data Source: Central Pollution Control Board (India)")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import StringIO

# Dark mode toggle
mode = st.sidebar.radio("Choose Theme", ["🌞 Light Mode", "🌙 Dark Mode"])
if mode == "🌙 Dark Mode":
    st.markdown("""
        <style>
            body, .css-18e3th9, .css-1d391kg, .css-1kyxreq, .st-bx, .st-b3 {
                background-color: #111 !important;
                color: #f0f0f0 !important;
            }
        </style>
    """, unsafe_allow_html=True)

st.set_page_config(layout="wide")
st.title("🇮🇳 India Air Quality Dashboard")

# Load combined data from .txt file
data_path = "combined_air_quality.txt"
@st.cache_data(ttl=600)
def load_data():
    return pd.read_csv(data_path, sep='\t', parse_dates=['date'])

df = load_data()

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

for city in selected_cities:
    st.markdown(f"## {city} – {year}")
    city_data = df[(df['city'] == city) & (df['date'].dt.year == year)].copy()
    city_data['day_of_year'] = city_data['date'].dt.dayofyear

    # Append to export list
    export_data.append(city_data)

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

    # Category distribution
    st.markdown("#### AQI Category Distribution")
    category_counts = city_data['level'].value_counts().reindex(category_colors.keys(), fill_value=0)
    fig3, ax3 = plt.subplots()
    ax3.bar(category_counts.index, category_counts.values, color=[category_colors[k] for k in category_counts.index])
    ax3.set_ylabel("Number of Days")
    ax3.set_title(f"AQI Category Breakdown - {city} ({year})")
    st.pyplot(fig3)

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

# Mobile layout tip
st.markdown("""
<style>
    @media screen and (max-width: 768px) {
        .element-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    }
</style>
""", unsafe_allow_html=True)

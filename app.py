import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import StringIO

# Set page config at the very top
st.set_page_config(layout="wide")

# 🌙 Theme toggle + rerun trigger
mode = st.sidebar.radio("Choose Theme", ["🌞 Light Mode", "🌙 Dark Mode"])
if "theme" not in st.session_state:
    st.session_state.theme = mode
elif mode != st.session_state.theme:
    st.session_state.theme = mode
    st.stop()  # safer than st.experimental_rerun() for Streamlit Cloud

# Inject CSS based on theme
# 🌙 Dark/Light Mode Toggle with Session Persistence
default_mode = "🌞 Light Mode"
if "theme" not in st.session_state:
    st.session_state.theme = default_mode

mode = st.sidebar.radio("Choose Theme", ["🌞 Light Mode", "🌙 Dark Mode"], index=0 if st.session_state.theme == "🌞 Light Mode" else 1)

if mode != st.session_state.theme:
    st.session_state.theme = mode
    st.experimental_rerun()

# Inject styles
if st.session_state.theme == "🌙 Dark Mode":
    st.markdown("""
        <style>
        html, body, [class*="css"] {
            background-color: #0e1117 !important;
            color: #FAFAFA !important;
        }
        .stApp { background-color: #0e1117; }
        .markdown-text-container { color: white !important; }
        .css-1v0mbdj, .st-bx, .st-b3 {
            background-color: #1c1f26 !important;
        }
        .css-1q8dd3e, .css-ffhzg2 { color: white !important; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        html, body, [class*="css"] {
            background-color: white !important;
            color: black !important;
        }
        </style>
    """, unsafe_allow_html=True)


# Dashboard Title
st.title("🇮🇳 India Air Quality Dashboard")

# Load data
data_path = "combined_air_quality.txt"

@st.cache_data(ttl=600)
def load_data():
    return pd.read_csv(data_path, sep='\t', parse_dates=['date'])

df = load_data()

# Sidebar: Filters
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

# Collect data for export
export_data = []

for city in selected_cities:
    st.markdown(f"## {city} – {year}")
    city_data = df[(df['city'] == city) & (df['date'].dt.year == year)].copy()
    city_data['day_of_year'] = city_data['date'].dt.dayofyear
    export_data.append(city_data)

    # Calendar Heatmap
    st.markdown("#### Calendar Heatmap")
    fig, ax = plt.subplots(figsize=(20, 2))
    for _, row in city_data.iterrows():
        color = category_colors.get(row['level'], '#FFFFFF')
        rect = patches.FancyBboxPatch((row['day_of_year'], 0), 1, 1, boxstyle="round,pad=0.1", linewidth=0, facecolor=color)
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

    # AQI Trend
    st.markdown("#### AQI Trend")
    fig2, ax2 = plt.subplots(figsize=(16, 3))
    ax2.plot(city_data['date'], city_data['index'], marker='o', linestyle='-', markersize=3)
    ax2.set_ylabel("AQI Index")
    ax2.set_xlabel("Date")
    ax2.set_title(f"AQI Trend for {city} in {year}")
    ax2.grid(True)
    st.pyplot(fig2)

    # AQI Category Distribution
    st.markdown("#### AQI Category Distribution")
    category_counts = city_data['level'].value_counts().reindex(category_colors.keys(), fill_value=0)
    fig3, ax3 = plt.subplots()
    ax3.bar(category_counts.index, category_counts.values, color=[category_colors[k] for k in category_counts.index])
    ax3.set_ylabel("Number of Days")
    ax3.set_title(f"AQI Category Breakdown - {city} ({year})")
    st.pyplot(fig3)

# 📤 Export Button
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

# Footer
st.markdown("---")
st.caption("📊 Data Source: Central Pollution Control Board (India)")
st.caption("👨‍🎓 Developed by **Kapil**, PhD Scholar, Indian Institute of Technology Kharagpur")

# Mobile layout adjustments
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

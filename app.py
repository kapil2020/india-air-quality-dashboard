# Responsive & Improved India Air Quality Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.express as px
from datetime import date
from io import StringIO
from sklearn.linear_model import LinearRegression
import os

# Set Plotting Theme
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# Streamlit Setup
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("🇮🇳 India Air Quality Dashboard")

# Sidebar & Device Detection
def get_device_type():
    return "mobile" if st.session_state.get("screen_width", 1000) < 768 else "desktop"

# Load Data
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
st.sidebar.markdown(f"📊 Data from **{df['date'].min().date()}** to **{df['date'].max().date()}**")

# Sidebar Filters
cities = sorted(df['city'].dropna().unique())
selected_cities = st.sidebar.multiselect("Select Cities", cities, default=["Delhi"])

years = sorted(df['date'].dt.year.unique())
default_year = max(years)
year = st.sidebar.selectbox("Select Year", years, index=years.index(default_year))

months_dict = {i: name for i, name in enumerate(
    ['January', 'February', 'March', 'April', 'May', 'June',
     'July', 'August', 'September', 'October', 'November', 'December'], 1)}

month_options = ["All"] + list(months_dict.values())
selected_month = st.sidebar.selectbox("Select Month (optional)", month_options, index=0)
if selected_month != "All":
    month_number = [k for k, v in months_dict.items() if v == selected_month][0]

# Color Palettes
category_colors = {
    'Severe': '#7E0023', 'Very Poor': '#FF0000', 'Poor': '#FF7E00',
    'Moderate': '#FFFF00', 'Satisfactory': '#00E400', 'Good': '#007E00'
}
pollutant_colors = {
    'PM2.5': '#F8766D', 'PM10': '#7CAE00', 'NO2': '#00BFC4',
    'SO2': '#C77CFF', 'CO': '#E69F00', 'O3': '#619CFF'
}

# Process + Plot per city
export_data = []
for city in selected_cities:
    st.markdown(f"## {city} – {year}")
    city_data = df[(df['city'] == city) & (df['date'].dt.year == year)].copy()
    if selected_month != "All":
        city_data = city_data[city_data['date'].dt.month == month_number]
        st.markdown(f"### Showing data for **{selected_month} {year}**")
    else:
        st.markdown(f"### Showing data for **Full Year {year}**")

    if city_data.empty:
        st.warning("No data for this selection.")
        continue

    export_data.append(city_data)

    # KPI
    avg_aqi = city_data['index'].mean()
    st.metric("Average AQI", f"{avg_aqi:.1f}")

    # AQI Trend - Plotly
    fig_trend = px.line(city_data, x='date', y='index', title=f"AQI Trend – {city}", markers=True)
    st.plotly_chart(fig_trend, use_container_width=True)

    # 7-Day Rolling
    city_data['rolling'] = city_data['index'].rolling(7).mean()
    fig_roll = px.line(city_data, x='date', y='rolling', title=f"7-Day Rolling AQI – {city}")
    st.plotly_chart(fig_roll, use_container_width=True)

    # AQI Category Bar + Pie
    cat_counts = city_data['level'].value_counts().reindex(category_colors.keys(), fill_value=0)
    col1, col2 = st.columns(2)
    with col1:
        fig_bar = px.bar(cat_counts, x=cat_counts.index, y=cat_counts.values,
                         color=cat_counts.index, color_discrete_map=category_colors,
                         labels={"x": "AQI Category", "y": "Days"}, title="AQI Category Count")
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        fig_pie = px.pie(names=cat_counts.index, values=cat_counts.values,
                        color=cat_counts.index, color_discrete_map=category_colors,
                        title="AQI Category Share")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Monthly Boxplot
    city_data['month'] = city_data['date'].dt.month
    fig_box = px.box(city_data, x='month', y='index', title="Monthly AQI Boxplot",
                     labels={"month": "Month", "index": "AQI"},
                     color_discrete_sequence=['#2E91E5'])
    fig_box.update_xaxes(tickvals=list(months_dict.keys()), ticktext=list(months_dict.values()))
    st.plotly_chart(fig_box, use_container_width=True)

# Export Filtered CSV
if export_data:
    combined_export = pd.concat(export_data)
    csv_buffer = StringIO()
    combined_export.to_csv(csv_buffer, index=False)
    st.download_button("\ud83d\udce4 Download Filtered Data", data=csv_buffer.getvalue(),
                       file_name="filtered_air_quality_data.csv", mime="text/csv")

# Footer
st.markdown("---")
st.caption("\ud83d\udcca Data Source: Central Pollution Control Board (India)")
st.markdown("""
**Developed by:**  
Mr. [Kapil Meena](https://sites.google.com/view/kapil-lab/home)  
Doctoral Scholar, IIT Kharagpur  
**With guidance from:**  
[Prof. Arkopal K. Goswami](https://www.mustlab.in/faculty)  
Chairperson, RCGSIDM, IIT Kharagpur
""")

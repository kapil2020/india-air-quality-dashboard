# 🌬️ IIT KGP AQI Dashboard

A dark‐themed, interactive Streamlit application that visualizes and analyzes air quality data (AQI) across major Indian cities. Built with Plotly, Pandas, and Scikit‐Learn, the dashboard provides:

- **National Key Insights**: Aggregate AQI metrics for major metros and overall trends.
- **City Deep Dives**: Calendar heatmaps, daily trends, rolling averages, category distributions, and monthly heatmaps for each city.
- **City‐to‐City Comparisons**: Side‐by‐side AQI trend lines and seasonal radar charts.
- **Pollutant Analysis**: Yearly and period‐specific dominant pollutant breakdowns.
- **Linear AQI Forecasts**: Simple linear regression forecasts for selected cities.
- **Interactive AQI Hotspots Map**: Scatter‐map of average AQI by city (with fallback bar charts if coordinates are unavailable).
- **Downloadable CSV**: Export filtered city data for offline analysis.

<br />

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Demo Screenshot](#demo-screenshot)  
4. [Tech Stack](#tech-stack)  
5. [Data Sources](#data-sources)  
6. [Installation & Setup](#installation--setup)  
7. [Running Locally](#running-locally)  
8. [Project Structure](#project-structure)  
9. [Usage](#usage)  
10. [Contributing](#contributing)  
11. [License](#license)  
12. [Acknowledgments](#acknowledgments)  

---

## 📖 Project Overview

Air quality is a critical public health metric. The **IIT KGP AQI Dashboard** (“AuraVision”) was conceptualized by Mr. Kapil Meena & Prof. Arkopal K. Goswami (IIT Kharagpur) and developed to:

- Aggregate and visualize historical AQI data from the Central Pollution Control Board (CPCB), India.
- Provide intuitive, interactive charts and maps to explore air quality at both national and city levels.
- Offer pollutant breakdowns and simple forecasting to highlight trends and areas of concern.
- Empower users to download filtered data for their own analyses.

This repository contains the complete Streamlit application (`final_app.py`), auxiliary data files, and instructions to reproduce and customize the dashboard.

---

## ⚙️ Features

1. **National Key Insights**  
   - **Annual Average AQI** for eight major metros (Delhi, Mumbai, Kolkata, Bengaluru, Chennai, Hyderabad, Pune, Ahmedabad).  
   - **General Period Insights**: Overall average AQI, best‐performing city, and worst‐performing city during the selected year/month.

2. **City Deep Dive**  
   - **Calendar Heatmap**: Daily AQI levels displayed on a full‐year calendar.  
   - **Trend & Rolling Average**: Daily AQI line plot + 7‐day rolling average band.  
   - **Category Distribution**: Bar chart & sunburst showing number/proportion of “Good”, “Moderate”, “Poor”, etc., days.  
   - **Monthly Violin Plot**: AQI distribution per month (with overlaid boxplots and outliers).  
   - **Monthly Heatmap**: Grid visualization of day‐by‐month AQI values.

3. **City‐to‐City Comparisons**  
   - **Trend Comparison**: Overlayed line charts for selected cities, highlighting relative AQI trajectories.  
   - **Seasonal Radar Chart**: Monthly average AQI by city (full‐year), enabling visual comparison of seasonal patterns.

4. **Prominent Pollutant Analysis**  
   - **Yearly Pollutant Trends**: Stacked‐bar percentages of dominant pollutants (PM2.5, PM10, NO₂, SO₂, CO, O₃, etc.) over multiple years.  
   - **Filtered Period Pollutant Breakdown**: Bar chart showing the proportion of days dominated by each pollutant in the selected period.

5. **AQI Forecast (Linear Trend)**  
   - Simple linear regression forecast using historical AQI data for the selected city.  
   - Overlay of observed vs. predicted AQI values for the next 15 days.

6. **City AQI Hotspots (Map)**  
   - **Scatter‐Mapbox**: Plots each city’s latitude/longitude with circle size proportional to average AQI and color coded by AQI category.  
   - **Hover Info**: City name, average AQI, AQI category, and dominant pollutant.  
   - **Fallback Bar Chart**: If latitude/longitude data is missing or malformed, a horizontal bar chart of top‐20 average‐AQI cities is shown.

7. **Download Filtered Data**  
   - Single‐click CSV download of the concatenated, filtered city‐level data for offline use.

<br />

---



---

## 🛠️ Tech Stack

- **Language**: Python 3.x  
- **Framework**: [Streamlit](https://streamlit.io/)  
- **Visualization**: [Plotly](https://plotly.com/python/), [Matplotlib](https://matplotlib.org/)  
- **Data**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)  
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/) (LinearRegression)  
- **Dependencies**: Listed in `requirements.txt` (see below)

---

## 🌐 Data Sources

1. **Central Pollution Control Board (CPCB), India**  
   - Historical AQI data (daily city‐level) stored in `combined_air_quality.txt` (tab‐separated).  
   - The app automatically attempts to load a “today’s CSV” named `data/YYYY-MM-DD.csv` if it exists; otherwise, it falls back to `combined_air_quality.txt`.

2. **City Coordinates**  
   - `lat_long.txt` must define a Python dictionary named `city_coords` mapping each city name (string) to a `[latitude, longitude]` pair.  
   - Example format inside `lat_long.txt`:
     ```python
     city_coords = {
         "Delhi": [28.7041, 77.1025],
         "Mumbai": [19.0760, 72.8777],
         "Kolkata": [22.5726, 88.3639],
         # …additional cities
     }
     ```

---

## ⚙️ Installation & Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/iitkgp-aqi-dashboard.git
   cd iitkgp-aqi-dashboard


---

## 🗂️ Project Structure

india-air-quality-dashboard/ ├── .github/workflows/ # GitHub Actions: auto-fetch CPCB data daily │ └── fetch_aqi.yml ├── app.py # Streamlit dashboard source code ├── fetch_cpcb_aqi.jl # Julia script to download and clean CPCB PDF ├── combined_air_quality.txt # Historical AQI data fallback ├── lat_long.txt # Coordinates for cities in the dashboard ├── data/ # Folder where daily AQI CSVs are saved │ └── YYYY-MM-DD.csv ├── requirements.txt # Python dependencies for Streamlit app └── README.md # This file


---

## ⚙️ Setup Instructions

### 🔧 Local Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/kapil2020/india-air-quality-dashboard.git
   cd india-air-quality-dashboard

2. Create and activate a virtual environment:
      ```
      python -m venv venv
      source venv/bin/activate  # On Windows: .\venv\Scripts\activate

3. Install dependencies:
     ```
      pip install -r requirements.txt

4. Run the app:
    ```
    streamlit run app.py


Data Automation via GitHub Actions
The repository includes a GitHub Action that:

Runs daily at 5:45 PM IST

Fetches CPCB's latest AQI bulletin PDF

Converts it to a cleaned .csv using tabula-py via Julia

Commits the data to the data/ directory

All .csv files follow the format: data/YYYY-MM-DD.csv

If CPCB hasn’t uploaded the bulletin yet, the workflow exits gracefully and skips the update.

📊 Data Source
📌 CPCB Daily AQI Bulletin
https://cpcb.nic.in/air-quality-monitoring/

👨‍💻 Author
Kapil Meena
Doctoral Scholar, IIT Kharagpur
🌐 Website, https://sites.google.com/view/kapil-lab/home
📧 kapil.meena@kgpian.iitkgp.ac.in


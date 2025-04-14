# 🇮🇳 India Air Quality Dashboard

A real-time, interactive dashboard built with **Streamlit** to visualize and analyze daily air quality data across major Indian cities using **CPCB AQI Bulletins**.

[![Streamlit App](https://img.shields.io/badge/🚀%20Launch%20App-Click%20Here-brightgreen)](https://kapil2020-india-air-quality-dashboard.streamlit.app/)
[![GitHub Workflow](https://github.com/kapil2020/india-air-quality-dashboard/actions/workflows/fetch_aqi.yml/badge.svg)](https://github.com/kapil2020/india-air-quality-dashboard/actions)

---

## 📊 Features

- **📆 Calendar Heatmap**: Daily AQI levels color-coded by CPCB categories
- **📈 AQI Trends**: Daily, monthly, and 7-day rolling averages
- **📊 Category Distributions**: Bar + pie charts by AQI level
- **📦 Pollutant Composition**: 100% stacked bar plots of PM2.5, PM10, NO₂, etc.
- **🗺️ Interactive Map**: City-wise average AQI and dominant pollutants
- **📉 Forecasting**: Linear AQI trendline for near-term projections
- **🧠 Automation**: GitHub Actions fetch daily CPCB bulletins at 5:45 PM IST

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


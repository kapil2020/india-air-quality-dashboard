# India Air Quality Dashboard 🌬️

A Streamlit-powered interactive dashboard for visualizing air quality trends across Indian cities. Features calendar heatmaps, time series trends, and air quality category breakdowns.

![image](https://github.com/user-attachments/assets/d7de1e4a-f3b5-4589-b565-509d6bd84ab0)


## Features ✨

- **Multi-City Comparison**: Compare air quality trends across multiple cities
- **Calendar Heatmap**: Visualize daily AQI levels in an intuitive yearly calendar format
- **Interactive Trends**: Explore AQI fluctuations with interactive time series charts
- **Category Breakdown**: View distribution of air quality categories (Good to Severe)
- **Dark/Light Mode**: Toggle between different color themes
- **Data Export**: Download filtered datasets in CSV format
- **Responsive Design**: Optimized for both desktop and mobile viewing

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/yourusername/india-air-quality-dashboard.git
cd india-air-quality-dashboard

Install required packages:

bash
pip install streamlit pandas matplotlib numpy

Prepare your data file:

Create a combined_air_quality.txt file with tab-separated columns:

city    date    index   level
Delhi   2023-01-01  342    Poor
Mumbai  2023-01-01  298    Poor


Run the app:

bash
streamlit run app.py

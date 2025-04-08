# 🇮🇳 India Air Quality Dashboard

An interactive web-based dashboard built with **Streamlit** to visualize and explore air quality trends across major Indian cities.

## 🔍 Features

- **City & Year Selector** via sidebar
- **Calendar-style AQI heatmap**
- **Daily AQI trend line**
- **7-day rolling average plot**
- **AQI category distribution (bar + pie)**
- **Monthly boxplot of AQI**
- **Month × Day heatmap**
- **Export filtered data as CSV**

## 📦 Dataset

The app uses a combined `.txt` file generated from daily AQI bulletins published by the **Central Pollution Control Board (CPCB), India**.  
You can place your file at the root level as:



![image](https://github.com/user-attachments/assets/d7de1e4a-f3b5-4589-b565-509d6bd84ab0)


## 📦 Dataset

The app uses a combined `.txt` file generated from daily AQI bulletins published by the **Central Pollution Control Board (CPCB), India**.  
You can place your file at the root level as:


The file should be tab-delimited (`.tsv`) and must include at least the following columns:
- `city`
- `date`
- `index` (AQI value)
- `level` (AQI category)

## 🚀 Running the App Locally

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/india-air-quality-dashboard.git
cd india-air-quality-dashboard

2. **Install dependencies**

pip install -r requirements.txt

3. Run the dashboard
streamlit run app.py

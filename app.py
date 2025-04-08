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
    "Jaipur": [26.9124, 75.7873],
    "Solapur": [17.6874, 75.9059],
    "Rohtak": [28.9067, 76.6069],
    "Gurgaon": [28.4595, 77.0266],
    "Kollam": [8.8932, 76.6141],
    "Koppal": [15.3524, 76.1561],
    "Korba": [22.3504, 82.6855],
    "Kota": [25.2138, 75.8648],
    "Shillong": [25.5768, 91.8800],
    "Shivamogga": [13.9299, 75.5609],
    "Sikar": [27.6119, 75.1399],
    "Agartala": [23.8314, 91.2868],
    "Ahmedabad": [23.0225, 72.5714],
    "Ahmednagar": [19.0952, 74.7477],
    "Aizawl": [23.7271, 92.7176],
    "Ajmer": [26.4499, 74.6399],
    "Akola": [20.7000, 77.0000],
    "Alwar": [27.5666, 76.6131],
    "Amaravati": [16.5796, 80.7965],
     "Amravati": [20.9319, 77.7798],
    "Amritsar": [31.6340, 74.8723],
    "Anantapur": [14.6815, 77.5901],
    "Angul": [20.8525, 85.1195],
    "Ankleshwar": [21.6359, 73.0135],
    "Arrah": [25.5606, 84.6684],
    "Asansol": [23.6834, 86.9853],
    "Aurangabad": [19.8762, 75.3433],
    "Baddi": [30.9572, 76.7933],
    "Bagalkot": [16.1754, 75.2974],
    "Bahadurgarh": [28.6886, 76.9202],
    "Baigachi": [22.8253, 88.4905],
    "Balaghat": [21.8054, 80.1797],
    "Ballabgarh": [28.3518, 77.3270],
    "Ballarpur": [19.9700, 79.3400],
    "Banda": [25.4869, 80.3355],
    "Bangalore": [12.9716, 77.5946],
    "Bankura": [23.2400, 87.0700],
    "Baranagar": [22.6500, 88.3700],
    "Bardhaman": [23.2500, 87.8600],
    "Bareilly": [28.3643, 79.4304],
    "Bargarh": [21.3671, 83.7575],
    "Bathinda": [30.2175, 74.9500],
    "Belagavi": [15.8500, 74.5000],
    "Berhampur": [19.3100, 84.7900],
    "Bhagalpur": [25.2400, 87.0200],
    "Bhandara": [21.1600, 79.6600],
    "Bharatpur": [27.2200, 77.4900],
    "Bharuch": [21.7000, 72.9700],
    "Bhatinda": [30.2175, 74.9500],
    "Bhavnagar": [21.7700, 72.1500],
    "Bhilai": [21.2200, 81.4300],
    "Bhilwara": [25.3500, 74.6400],
    "Bhiwadi": [28.2089, 77.0869],
    "Bhopal": [23.2500, 77.4100],
    "Bhubaneswar": [20.2700, 85.8400],
    "Bidar": [17.9200, 77.5200],
    "Bihar Sharif": [25.1900, 85.5100],
    "Bikaner": [28.0100, 73.3100],
    "Bilaspur": [22.0900, 82.0100],
     "Bileipada": [21.7600, 84.8300],
    "Boisar": [19.7000, 72.7500],
    "Bokaro": [23.6700, 86.1500],
    "Bongaigaon": [26.4800, 90.5700],
    "Brajrajnagar": [21.8167, 83.9167],
    "Budge Budge": [22.4700, 88.1900],
    "Bulandshahr": [28.4100, 77.7500],
    "Burdwan": [23.2500, 87.8600],
    "Burhanpur": [21.3000, 76.2300],
    "Buxar": [25.5800, 83.9800],
    "Byasanagar": [20.8469, 86.2311],
    "Byrnihat": [25.7500, 91.8800],
    "Chamarajanagar": [11.9000, 76.9500],
    "Chandigarh": [30.7400, 76.7900],
    "Chapra": [25.7800, 84.7300],
    "Chas": [23.6500, 86.1600],
    "Chhatarpur": [24.9189, 79.6083],
    "Chhindwara": [22.0600, 78.9400],
    "Chikkaballapur": [13.4200, 77.7300],
    "Chikkamagaluru": [13.3100, 75.7700],
    "Chilika": [19.6500, 85.2000],
    "Chitradurga": [14.2300, 76.3900],
    "Chittorgarh": [24.8900, 74.6300],
    "Churu": [28.3000, 74.9500],
    "Coimbatore": [11.0000, 76.9600],
    "Cuttack": [20.4600, 85.8800],
    "Damanjodi": [18.8200, 82.1800],
    "Darbhanga": [26.1700, 85.9000],
    "Daudnagar": [25.0500, 84.4000],
    "Davangere": [14.4600, 75.9200],
    "Deeg": [27.4700, 77.2900],
    "Dehradun": [30.3200, 78.0300],
    "Dehri": [24.8500, 84.0000],
    "Delhi NCR": [28.6139, 77.2090],
    "Dharamsala": [32.2200, 76.3200],
    "Dharmapuri": [12.1300, 78.1600],
    "Dharuhera": [28.2097, 76.8770],
    "Dhenkanal": [20.7000, 85.5900],
    "Dholpur": [26.7000, 77.9100],
    "Dhubri": [26.0200, 89.9800],
    "Dibrugarh": [27.4800, 94.9000],
    "Durgapur": [23.5500, 87.3200],
    "Erode": [11.3400, 77.7300],
    "Gadag": [15.4100, 75.6300],
    "Gandhinagar": [23.2200, 72.6400],
    "Ganjam": [19.3900, 85.0700],
    "Garhwa": [24.3600, 83.8700],
    "Gaya": [24.7955, 85.0119],
    "Ghaziabad": [28.6700, 77.4200],
    "Giridih": [24.1800, 86.3000],
    "Godda": [25.0000, 87.2200],
    "Gondia": [21.7500, 80.2000],
    "Gorakhpur": [26.7600, 83.3700],
    "Greater Noida": [28.5000, 77.5000],
    "Gummidipoondi": [13.4300, 80.2300],
    "Guna": [24.6500, 77.3200],
    "Guntakal": [15.1600, 77.3800],
    "Guntur": [16.3000, 80.4400],
    "Guwahati": [26.1800, 91.7400],
    "Gwalior": [26.2100, 78.1800],
    "Hajipur": [25.6800, 85.2100],
    "Hassan": [13.0000, 76.1000],
    "Hathras": [27.6000, 78.0500],
    "Haveri": [14.8000, 75.4000],
    "Hazaribagh": [23.9900, 85.3600],
    "Hindupur": [13.8200, 77.4900],
    "Hoshiarpur": [31.5300, 75.9100],
    "Hosur": [12.7300, 77.8800],
    "Hubballi": [15.3600, 75.1200],
    "Irinjalakuda": [10.3300, 76.2300],
    "Itanagar": [27.0800, 93.6200],
    "Jabalpur": [23.1800, 79.9900],
    "Jajpur": [20.8300, 86.3300],
    "Jaisalmer": [26.9200, 70.9100],
    "Jajpur Road": [20.9600, 85.9800],
    "Jalandhar": [31.3256, 75.5794],
    "Jalgaon": [21.0100, 75.5600],
    "Jalna": [19.0300, 75.8800],
    "Jalore": [25.3500, 72.6200],
    "Jamshedpur": [22.8000, 86.1800],
    "Jamui": [24.9200, 86.2200],
    "Jangaon": [17.5200, 79.1800],
    "Jatani": [20.1700, 85.7300],
    "Jehanabad": [25.1200, 84.9700],
    "Jeypore": [18.8600, 82.5900],
    "Jhansi": [25.4500, 78.5700],
    "Jharsuguda": [21.8200, 84.0300],
         "Jind": [29.3200, 76.3200],
    "Jorapokhar": [23.7000, 86.4100],
    "Jowai": [25.1700, 92.2800],
    "Kadapa": [14.4700, 78.8200],
    "Kaithal": [29.8000, 76.3900],
    "Kakinada": [16.9300, 82.2300],
    "Kalaburagi": [17.3000, 76.8300],
    "Kalamassery": [10.0300, 76.3500],
    "Kalamboli": [19.0183, 73.0883],
    "Kalyan": [19.2502, 73.1349],
    "Kanchipuram": [12.8300, 79.7000],
    "Kandhar": [19.1500, 77.3200],
    "Kanhangad": [12.3100, 75.0700],
    "Kanjirappally": [9.5600, 76.5100],
    "Kannauj": [27.0500, 79.9200],
    "Kannur": [11.8700, 75.3700],
    "Karaikkudi": [10.0700, 78.7800],
    "Karauli": [26.4900, 77.0100],
    "Karimnagar": [18.4300, 79.1300],
    "Karnal": [29.6700, 76.9700],
    "Karur": [10.9700, 78.0800],
    "Karwar": [14.8000, 74.1300],
    "Kashipur": [29.2200, 79.5500],
    "Katihar": [25.2400, 87.5800],
    "Katni": [23.8300, 80.3900],
    "Kavaratti": [10.3200, 72.6400],
    "Kawardha": [22.0100, 81.2500],
    "Kendujhar": [21.6500, 85.5900],
    "Khagaria": [25.3000, 86.4800],
    "Khambhat": [22.3000, 72.6200],
    "Khandwa": [21.8300, 76.3500],
    "Kharagpur": [22.3400, 87.3200],
    "Kharsawan": [22.7900, 85.8000],
    "Khed Brahma": [24.0200, 73.0200],
    "Koderma": [24.4600, 85.5900],
    "Kohima": [25.6700, 94.1000],
    "Kolar": [13.1300, 78.1300],
    "Kollam": [8.8932, 76.6141],
    "Koppal": [15.3524, 76.1561],
    "Koraput": [18.8200, 82.1800],
    "Koriya": [23.0500, 82.5200],
    "Kosamba": [21.5300, 73.0500],
    "Kottayam": [9.5900, 76.5200],
    "Kozhikode": [11.2500, 75.7700],
    "Krishnanagar": [23.3900, 88.5000],
    "Kullu": [31.9500, 77.1100],
    "Kurnool": [15.8300, 78.0300],
    "Kurukshetra": [29.9700, 76.8300],
    "Lachhmangarh": [27.3300, 75.1400],
    "Ladwa": [30.1300, 76.6000],
    "Laharipur": [27.7300, 81.0800],
    "Lakhisarai": [25.0700, 86.0700],
    "Lalitpur": [24.6700, 78.4200],
    "Latur": [18.4000, 75.0000],
    "Leh": [34.1700, 77.5800],
    "Lohardaga": [23.4300, 84.7200],
    "Loni": [28.7100, 77.2800],
    "Luckeesarai": [25.1700, 86.0600],
    "Ludhiana": [30.9100, 75.8500],
    "Machilipatnam": [16.1700, 81.1300],
    "Madanapalle": [13.5500, 78.5000],
    "Madgaon": [15.2900, 73.9700],
    "Madikeri": [12.4300, 75.7400],
    "Madurai": [9.9200, 78.1100],
    "Mahabubnagar": [16.7300, 77.9900],
    "Mahe": [11.7000, 75.5300],
    "Mahesana": [23.6000, 72.4000],
    "Mahnar Bazar": [25.7500, 85.3500],
    "Mahoba": [25.2900, 79.8700],
    "Mainaguri": [26.5400, 88.7100],
    "Majitar": [27.2000, 88.2900],
    "Malappuram": [11.0600, 76.0800],
    "Malda": [25.0000, 88.1400],
    "Malkangiri": [18.1900, 82.0700],
    "Mancherial": [18.8700, 79.4700],
    "Mandi": [31.7200, 76.9200],
    "Mandideep": [23.2500, 77.5800],
    "Mandla": [22.6000, 80.3900],
    "Mandsaur": [24.0700, 75.0700],
    "Mangaldoi": [26.3500, 92.1400],
    "Mangalore": [12.8700, 74.8800],
    "Manihari": [25.2600, 87.6300],
    "Manipal": [13.3500, 74.7900],
    "Margherita": [27.5900, 95.7000],
    "Markapur": [15.7300, 79.2700],
    "Mathura": [27.4900, 77.6700],
    "Medininagar": [24.0300, 84.0000],
    "Meerut": [28.9800, 77.7100],
    "Mhow Cantt": [22.5600, 75.7800],
    "Midnapore": [22.4200, 87.3500],
    "Mirzapur": [25.1500, 82.5800],
    "Modasa": [23.7700, 73.3000],
    "Moga": [30.8200, 75.1700],
    "Monghyr": [25.3600, 86.4800],
    "Motihari": [26.6500, 84.9200],
    "Mughalsarai": [25.2700, 83.0200],
    "Mukerian": [31.9500, 75.6100],
    "Munger": [25.3600, 86.4800],
    "Muradnagar": [28.7700, 77.5000],
    "Murliganj": [25.9200, 86.9700],
    "Murshidabad": [24.1800, 88.2600],
    "Murtijapur": [20.7600, 77.6300],
    "Murwara": [23.8500, 80.9000],
    "Mysore": [12.3000, 76.6500],
    "Nabadwip": [23.4100, 88.3700],
    "Nabha": [30.3700, 76.2600],
    "Nadiad": [22.7000, 72.8700],
    "Nagaon": [26.3500, 92.6800],
    "Nagapattinam": [10.7700, 79.8400],
    "Nagercoil": [8.1800, 77.4300],
    "Nagpur": [21.1500, 79.0900],
    "Nahan": [30.5600, 77.3100],
    "Naihati": [22.9200, 88.4200],
    "Naila Janjgir": [22.0300, 82.6700],
    "Nainital": [29.3800, 79.4600],
    "Nalanda": [25.1500, 85.5300],
    "Nalbari": [26.4100, 91.4300],
    "Nalgonda": [16.4500, 79.2700],
    "Namakkal": [11.2300, 78.1700],
    "Nanded": [19.1500, 77.3300],
    "Nandurbar": [21.3700, 74.2500],
    "Nandyal": [15.4800, 78.4800],
    "Narayangarh": [22.1500, 87.4100],
    "Narnaul": [28.0500, 76.1100],
    "Narsinghpur": [22.9500, 79.2100],
    "Narsingpur": [23.1600, 80.9800],
    "Nashik": [19.9900, 73.7800],
    "Nathdwara": [24.9300, 73.8200],
    "Navsari": [20.8500, 72.9500],
    "Nawada": [24.8800, 85.5300],
    "Neemuch": [24.4500, 75.1400],
    "Nellore": [14.4400, 79.9700],
    "New Delhi": [28.6139, 77.2090],
    "Neyveli": [11.4900, 79.4900],
    "Nizamabad": [18.6700, 78.1000],
    "Nohar": [29.1800, 75.0100],
    "Noida": [28.5700, 77.3200],
    "North Barrackpur": [22.7500, 88.3500],
    "Nowgong": [26.3300, 93.3300],
    "Obra": [24.5000, 82.9800],
    "Oddanchatram": [10.4300, 77.8000],
    "Ongole": [15.5000, 80.0500],
    "Orai": [25.9900, 79.4600],
    "Osmanabad": [18.1700, 76.0500],
    "Ottappalam": [10.7700, 76.3900],
    "Pachora": [20.6700, 75.3500],
    "Padrauna": [26.8900, 83.9900],
    "Paithan": [19.4800, 75.4000],
    "Pakur": [24.6300, 87.8500],
    "Palakkad": [10.7900, 76.6600],
    "Palanpur": [24.1800, 72.4400],
    "Palghar": [19.7000, 72.7500],
    "Pali": [25.7700, 73.3300],
    "Palkalaiperur": [10.0700, 78.7800],
    "Palwal": [28.1500, 77.3300],
    "Panagar": [23.1800, 80.1200],
    "Panagarh": [23.5100, 87.4300],
    "Panaji": [15.4900, 73.8200],
    "Panapana": [22.4000, 85.0500],
    "Panchmahal-Godhra": [22.7700, 73.6500],
    "Pandharpur": [18.0700, 75.3300],
    "Pandhurna": [21.6100, 78.5200],
    "Panipat": [29.3900, 76.9700],
    "Panna": [24.7200, 79.2000],
         "Paradeep": [20.3100, 86.6700],
    "Paramakudi": [9.5300, 78.6600],
    "Parasia": [22.2500, 78.6700],
    "Parbhani": [19.2700, 76.7700],
    "Parnera": [20.5700, 72.9500],
    "Pasighat": [28.0700, 95.3300],
    "Patan": [23.8500, 72.1200],
    "Pathanamthitta": [9.2500, 76.7800],
    "Pathankot": [32.2700, 75.6400],
    "Patiala": [30.3300, 76.4000],
    "Pattambi": [10.8000, 76.2000],
    "Pattukkottai": [10.4200, 79.3200],
    "Pendra Road": [22.7800, 81.9700],
    "Phagwara": [31.2200, 75.7700],
    "Phalodi": [26.9300, 72.3700],
    "Phaltan": [18.1000, 74.4300],
    "Phulabani": [20.4700, 84.5200],
    "Pilibanga": [29.2500, 75.3900],
    "Pilibhit": [28.6400, 79.8000],
    "Pimpri Chinchwad": [18.6278, 73.8028],
    "Porbandar": [21.6400, 69.6000],
    "Port Blair": [11.6700, 92.7500],
    "Pratapgarh": [25.9000, 81.9500],
    "Puducherry": [11.9400, 79.8300],
    "Pudukkottai": [10.3800, 78.8200],
    "Pulivendla": [14.3200, 78.2300],
    "Punalur": [9.0300, 76.9200],
    "Purba Medinipur": [22.0000, 87.7500],
    "Puri": [19.8000, 85.8200],
    "Purnea": [25.7800, 87.4700],
    "Purulia": [23.3300, 86.3700],
    "Pusad": [20.3700, 77.6000],
    "Rae Bareli": [26.2300, 81.2400],
    "Raichur": [16.2100, 77.3400],
    "Raiganj": [25.6100, 88.1200],
    "Raigarh": [21.9000, 83.3500],
    "Raipur": [21.2500, 81.6300],
    "Rajahmundry": [17.0000, 81.7800],
    "Rajampet": [14.1800, 79.1600],
    "Rajgarh": [24.0700, 76.7000],
    "Rajgir": [25.0300, 85.4200],
    "Rajkot": [22.2900, 70.8000],
    "Rajnandgaon": [21.0900, 80.9900],
    "Rajouri": [33.0000, 74.3000],
    "Rajsamand": [25.0700, 74.1200],
    "Rajura": [19.7800, 79.3700],
    "Ramanagara": [12.7200, 77.3200],
    "Ramanathapuram": [9.3700, 79.3100],
    "Ramgarh": [23.6300, 85.5100],
    "Ramjibanpur": [22.7100, 87.8000],
    "Ramnagar": [29.4000, 79.1500],
    "Ramngarh": [23.6300, 85.5100],
    "Rampur": [28.8100, 79.0200],
    "Ranchi": [23.3400, 85.3200],
    "Rangia": [26.5000, 91.6300],
    "Ratangarh": [27.9700, 74.6100],
    "Rathjhan": [26.6800, 77.1700],
    "Ratlam": [23.3200, 75.0300],
    "Ratnagiri": [16.9900, 73.3000],
    "Rayagada": [19.1600, 83.4200],
    "Reengus": [27.4200, 75.5800],
    "Rewa": [24.5300, 81.3000],
    "Rewari": [28.1900, 76.6200],
    "Rishikesh": [30.1000, 78.2900],
    "Robertsonpet": [12.9500, 78.2800],
    "Roorkee": [29.8600, 77.8900],
    "Rourkela": [22.2500, 84.8900],
    "Sagar": [23.8500, 78.7600],
    "Saharanpur": [29.9700, 77.5500],
    "Saharsa": [25.8800, 86.6000],
    "Saiha": [22.4800, 93.2200],
    "Salem": [11.6600, 78.1400],
    "Samastipur": [25.8600, 85.7800],
    "Sambalpur": [21.4600, 83.9800],
    "Sambhal": [28.5700, 78.5500],
    "Sangamner": [19.5700, 74.2100],
    "Sangareddy": [17.6300, 78.0800],
    "Sangli": [16.8600, 74.5700],
    "Sangrur": [30.2500, 75.8500],
    "Sankarankoil": [9.1700, 77.5000],
    "Santiniketan": [23.6800, 87.6900],
    "Sasaram": [24.9500, 84.0200],
    "Satara": [17.6800, 74.0200],
    "Satna": [24.5800, 80.8300],
    "Sawai Madhopur": [26.0200, 76.4000],
    "Sehore": [23.2000, 77.0900],
    "Seoni": [22.4800, 79.5300],
    "Serampore": [22.7500, 88.3400],
    "Shahabad": [27.6200, 79.9100],
    "Shahdol": [23.3000, 81.3600],
    "Shahjahanpur": [27.8800, 79.9100],
    "Shajapur": [23.4300, 76.2700],
    "Sheopur": [25.6700, 76.7000],
    "Sherghati": [24.6300, 84.8600],
    "Shillong": [25.5768, 91.8800],
    "Shimla": [31.1000, 77.1700],
    "Shivamogga": [13.9299, 75.5609],
    "Shivpuri": [25.6500, 77.6500],
    "Sholapur": [17.6874, 75.9059],
    "Sibsagar": [26.9900, 94.6400],
    "Sidhi": [24.4000, 81.8800],
    "Sikar": [27.6119, 75.1399],
    "Silchar": [24.8200, 92.7900],
    "Siliguri": [26.7200, 88.4200],
    "Singrauli": [24.2000, 82.8700],
    "Sirohi": [25.2700, 72.8600],
    "Sirsa": [29.5300, 75.0200],
    "Sirsi": [14.8400, 74.8400],
    "Sirsilla": [18.3900, 78.8400],
    "Sitamarhi": [26.6000, 85.4800],
    "Sitapur": [27.5800, 80.6800],
    "Siwan": [26.2100, 84.3600],
    "Sohagpur": [23.3800, 81.2800],
    "Sohna": [28.2400, 77.0700],
    "Sonamura": [23.4700, 91.2400],
    "Sonbhadra": [24.6700, 83.0000],
    "Sonepur": [20.8500, 84.9100],
    "Sopore": [34.3000, 74.4500],
    "South Dumdum": [22.6300, 88.3800],
    "Sri Ganganagar": [29.9100, 73.8800],
    "Sri Muktsar Sahib": [30.7100, 74.5100],
    "Sri Vijaya Puram": [16.5200, 80.8200],
    "Srikakulam": [18.3000, 83.9000],
    "Srinagar": [34.0900, 74.8000],
    "Srirampore": [22.7500, 88.3400],
    "Sujangarh": [27.6900, 74.4500],
    "Sultanpur": [26.2500, 82.0700],
    "Sunam": [30.1200, 75.7900],
    "Supaul": [26.1300, 86.6000],
    "Surat": [21.1700, 72.8300],
    "Surendranagar": [22.7300, 71.6500],
    "Suri": [23.9000, 87.5300],
    "Suryapet": [17.6000, 79.6200],
    "Talcher": [20.9500, 85.2100],
    "Tamluk": [22.3200, 87.9300],
    "Tanda": [22.5600, 88.3500],
    "Tandur": [17.2400, 77.5900],
    "Tanuku": [16.8500, 81.6900],
    "Tarakeswar": [22.8700, 88.0100],
    "Tehri": [30.3700, 78.4900],
    "Tellicherry": [11.7500, 75.4900],
    "Tenali": [16.2400, 80.6400],
    "Tensa": [22.1300, 85.2400],
    "Thalassery": [11.7500, 75.4900],
    "Thane": [19.1800, 72.9600],
    "Thanjavur": [10.7900, 79.1300],
    "Thatipur": [26.5200, 78.2500],
    "Theni": [10.0000, 77.4700],
    "Thiruvalla": [9.3900, 76.5900],
    "Thoothukudi": [8.7800, 78.1300],
    "Thoubal": [24.7800, 93.9700],
    "Thrissur": [10.5300, 76.2100],
    "Tindivanam": [12.2500, 79.6400],
    "Tinsukia": [27.4900, 95.3600],
    "Tiruchirappalli": [10.8000, 78.6900],
    "Tirunelveli": [8.7300, 77.7000],
    "Tirupati": [13.6300, 79.4200],
    "Tirupur": [11.1100, 77.3400],
    "Tiruvannamalai": [12.2300, 79.0700],
    "Titagarh": [22.7400, 88.3600],
    "Tonk": [26.1600, 75.7900],
    "Tumakuru": [13.3400, 77.1100],
     "Tura": [25.5200, 90.2000],
    "Udaipur": [24.5800, 73.7100],
    "Udgir": [18.3900, 77.1100],
    "Udhagamandalam": [11.4000, 76.6900],
    "Udhampur": [32.9300, 75.1400],
    "Udupi": [13.3400, 74.7500],
    "Ujjain": [23.1800, 75.7700],
    "Ulhasnagar": [19.2200, 73.1500],
    "Una": [31.8800, 76.2700],
    "Unnao": [26.5500, 80.6400],
    "Upper Tadong": [27.3300, 88.6300],
    "Uttarpara Kotrung": [22.6700, 88.3400],
    "Vadakara": [11.6000, 75.5900],
    "Vadodara": [22.3000, 73.1800],
    "Vaishali": [25.6700, 85.1000],
    "Valsad": [20.6000, 72.9300],
    "Vapi": [20.3900, 72.9100],
    "Varanasi": [25.3176, 82.9739],
    "Vatakara": [11.6000, 75.5900],
    "Vatva": [22.9800, 72.6400],
    "Vellore": [12.9200, 79.1300],
    "Veraval": [20.9100, 70.3700],
    "Vidisha": [23.5300, 77.8000],
    "Vijayapura": [16.8300, 75.7000],
    "Vijayawada": [16.5200, 80.6500],
    "Vikarabad": [17.3300, 77.9000],
    "Viluppuram": [11.9300, 79.5000],
    "Vinukonda": [16.0500, 80.0800],
    "Virar": [19.4500, 72.7900],
    "Virudhunagar": [9.5000, 77.9500],
    "Visakhapatnam": [17.6900, 83.2200],
    "Vizianagaram": [18.1100, 83.3900],
    "Vrindavan": [27.5700, 77.7000],
    "Vyara": [21.1300, 73.4000],
    "Waidhan": [24.0700, 82.6700],
    "Wanaparthy": [16.3500, 78.0700],
    "Wani": [20.0700, 79.0700],
    "Warangal": [18.0000, 79.5900],
    "Wardha": [20.7500, 78.6000],
    "Washim": [20.1000, 77.1500],
    "Wokha": [26.2700, 94.2700],
    "Yadgir": [16.7700, 77.1200],
    "Yamunanagar": [30.1300, 77.2700],
    "Yavatmal": [20.3900, 78.1200],
    "Zaidpur": [26.6000, 81.3300],
    "Zamania": [25.2200, 83.6000]
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

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
# from sklearn.linear_model import LinearRegression # Note: This import is present but not used.
import os
# from datetime import date, timedelta # Not directly used, pandas handles dates
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json # For loading GeoJSON

# --- Global Theme & Style Setup ---
pio.templates.default = "plotly_dark"
ACCENT_COLOR = "#00E0C6" # A slightly brighter, more vibrant teal
TEXT_COLOR_DARK_THEME = "#F0F0F0" # Brighter text for better contrast
SUBTLE_TEXT_COLOR_DARK_THEME = "#A0A0A0" # Softer subtle text
BACKGROUND_COLOR = "#0A0A0A" # Deep, near-black
CARD_BACKGROUND_COLOR = "#141414" # Dark cards, distinct from BG
BORDER_COLOR = "#2A2A2A"
GLOBAL_FONT_FAMILY = "Inter, sans-serif"

CATEGORY_COLORS_DARK = {
    'Severe': '#E53935', 'Very Poor': '#F57C00', 'Poor': '#FFA000',
    'Moderate': '#FBC02D', 'Satisfactory': '#7CB342', 'Good': '#388E3C',
    'Unknown': '#545454'
}
POLLUTANT_COLORS_DARK = {
    'PM2.5': '#EF5350', 'PM10': '#26A69A', 'NO2': '#29B6F6',
    'SO2': '#FFEE58', 'CO': '#FFA726', 'O3': '#9CCC65', 'Other': '#B0BEC5'
}

# ------------------- Page Config -------------------
st.set_page_config(layout="wide", page_title="AuraVision AQI Dashboard", page_icon="💨")

# ------------------- Custom CSS Styling (Dark Theme Focus) -------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    body {{
        font-family: {GLOBAL_FONT_FAMILY};
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR_DARK_THEME};
        line-height: 1.65; /* Enhanced readability */
    }}
    .main .block-container {{
        padding: 2rem 3.5rem 3.5rem 3.5rem; /* Generous padding */
    }}
    .stPlotlyChart, .stDataFrame, .stAlert,
    div[data-testid="stExpander"], div[data-testid="stForm"] {{
        border-radius: 18px; /* More pronounced rounding */
        border: 1px solid {BORDER_COLOR};
        background-color: {CARD_BACKGROUND_COLOR};
        padding: 2.2rem; /* Increased padding for cards */
        margin-bottom: 2.8rem; /* Increased margin */
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.35); /* Softer, more modern shadow */
        transition: transform 0.25s ease-in-out, box-shadow 0.25s ease-in-out;
    }}
    .stPlotlyChart:hover, div[data-testid="stExpander"]:hover {{
         transform: translateY(-4px); /* More lift on hover */
         box-shadow: 0 14px 35px -5px rgba(0,0,0,0.4);
    }}
    .stButton > button, .stDownloadButton > button {{
        border-radius: 12px !important; /* Softer radius */
        border: 2px solid {ACCENT_COLOR} !important;
        background-color: transparent !important;
        color: {ACCENT_COLOR} !important;
        padding: 0.8rem 1.8rem !important;
        font-weight: 700 !important; /* Bolder button text */
        font-family: {GLOBAL_FONT_FAMILY} !important;
        letter-spacing: 0.5px;
        transition: all 0.25s ease !important;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: {ACCENT_COLOR} !important;
        color: {BACKGROUND_COLOR} !important;
        transform: scale(1.05) translateY(-1px) !important;
        box-shadow: 0 6px 15px -3px {ACCENT_COLOR}aa !important;
    }}
    .stButton > button:active, .stDownloadButton > button:active {{
        transform: scale(0.98) translateY(0px) !important;
        box-shadow: none !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
         border-bottom: 2.5px solid {BORDER_COLOR};
         margin-bottom: 1.8rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 1.1rem 2rem; /* Even larger tabs */
        font-weight: 700;
        font-size: 1.05rem;
        font-family: {GLOBAL_FONT_FAMILY};
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        border-radius: 12px 12px 0 0; /* More rounded top */
        margin-right: 0.6rem;
        transition: background-color 0.3s ease, color 0.3s ease, border-bottom-color 0.3s ease;
        border-bottom: 4px solid transparent; /* Thicker indicator */
    }}
     .stTabs [aria-selected="true"] {{
        border-bottom: 4px solid {ACCENT_COLOR};
        color: {ACCENT_COLOR};
        background-color: {CARD_BACKGROUND_COLOR}22; /* Subtle bg for active tab */
     }}
    h1 {{
        font-family: {GLOBAL_FONT_FAMILY};
        color: {TEXT_COLOR_DARK_THEME};
        text-align: center;
        margin-bottom: 0.4rem;
        font-weight: 900; /* Heaviest weight */
        letter-spacing: -2px; /* Tighter spacing */
        font-size: 3.4rem; /* Very prominent */
        line-height: 1.2;
    }}
    h2 {{ /* Main section titles */
        font-family: {GLOBAL_FONT_FAMILY};
        color: {ACCENT_COLOR};
        border-bottom: 3.5px solid {ACCENT_COLOR}; /* Thicker border */
        padding-bottom: 1rem; /* More padding */
        margin-top: 4.5rem; /* Max separation */
        margin-bottom: 3rem; /* More space below */
        font-weight: 800; /* Bolder H2 */
        text-transform: uppercase;
        letter-spacing: 1.5px; /* Wider letter spacing */
        font-size: 1.75rem; /* Larger H2 */
    }}
    h3 {{ /* Sub-section titles for cities, etc. */
        font-family: {GLOBAL_FONT_FAMILY};
        color: {TEXT_COLOR_DARK_THEME};
        margin-top: 2rem;
        margin-bottom: 1.8rem;
        font-weight: 700;
        font-size: 1.45rem;
    }}
    h4, h5 {{ /* Plot titles, etc. */
        font-family: {GLOBAL_FONT_FAMILY};
        color: {TEXT_COLOR_DARK_THEME};
        margin-bottom: 1.4rem;
        font-weight: 600;
        font-size: 1.2rem;
    }}
    .stSidebar {{
        background-color: {CARD_BACKGROUND_COLOR}dd; /* Slightly transparent sidebar */
        border-right: 1.5px solid {BORDER_COLOR};
        padding: 2.8rem; /* More padding */
        backdrop-filter: blur(5px); /* Frosted glass effect if browser supports */
    }}
    .stSidebar .stMarkdown h2 {{ /* Sidebar Header */
        color: {ACCENT_COLOR} !important;
        font-size: 1.5rem !important;
        font-family: {GLOBAL_FONT_FAMILY} !important;
        font-weight: 800 !important;
        margin-bottom: 2.2rem !important;
    }}
    .stSidebar .stSelectbox label, .stSidebar .stMultiselect label {{
        font-size: 1.05rem; /* Clearer labels */
        font-family: {GLOBAL_FONT_FAMILY} !important;
        margin-bottom: 0.6rem !important;
        font-weight: 600 !important;
        color: {ACCENT_COLOR}e0 !important;
    }}
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
        background-color: {BACKGROUND_COLOR}cc !important; /* Slightly transparent inputs */
        border-color: {BORDER_COLOR} !important;
        border-radius: 10px !important; /* Softer input radius */
    }}
    .stMetric {{
        background-color: {CARD_BACKGROUND_COLOR};
        border: 1.5px solid {BORDER_COLOR};
        border-left: 7px solid {ACCENT_COLOR}; /* Thicker accent */
        border-radius: 14px; /* Consistent rounding */
        padding: 1.6rem 2rem; /* Ample padding */
    }}
    .stMetric > div:nth-child(1) {{ /* Label */
        font-size: 1.05rem;
        color: {SUBTLE_TEXT_COLOR_DARK_THEME};
        font-weight: 500;
        margin-bottom: 0.5rem;
    }}
    .stMetric > div:nth-child(2) {{ /* Value */
        font-size: 3rem; /* Impactful value */
        font-weight: 800;
        color: {ACCENT_COLOR};
        line-height: 1.15;
    }}
    ::-webkit-scrollbar {{ width: 12px; height: 12px; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER_COLOR}; border-radius: 6px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {ACCENT_COLOR}; }}
    .footer {{ /* Class for footer div */
        text-align: center; margin-top: 5.5rem; padding: 3rem;
        background-color: {CARD_BACKGROUND_COLOR}; border-radius: 18px;
        border-top: 1.5px solid {BORDER_COLOR}; box-shadow: 0 -6px 15px -5px rgba(0,0,0,0.2);
    }}
    .footer p {{ margin: 0.6em; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 1rem; }}
    .footer a {{ color: {ACCENT_COLOR}; text-decoration: none; font-weight: 700; }}
    .footer a:hover {{ text-decoration: underline; filter: brightness(1.2); }}
</style>
""", unsafe_allow_html=True)

# --- Plotly Universal Layout Options ---
def get_custom_plotly_layout_args(height=None):
    layout_args = dict(
        font_family=GLOBAL_FONT_FAMILY.split(',')[0].strip(),
        font_color=TEXT_COLOR_DARK_THEME,
        paper_bgcolor=CARD_BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR, # Darker plot area
        legend=dict(
            font=dict(color=SUBTLE_TEXT_COLOR_DARK_THEME, family=GLOBAL_FONT_FAMILY.split(',')[0].strip(), size=11),
            bgcolor=CARD_BACKGROUND_COLOR, bordercolor=BORDER_COLOR, borderwidth=1,
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1 # Top right legend
        ),
        title_font_family=GLOBAL_FONT_FAMILY.split(',')[0].strip(),
        title_font_color=TEXT_COLOR_DARK_THEME,
        title_font_size=18, # Standardized plot title size
        xaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, zerolinecolor=BORDER_COLOR, tickfont=dict(family=GLOBAL_FONT_FAMILY.split(',')[0].strip(), color=SUBTLE_TEXT_COLOR_DARK_THEME, size=11), showgrid=True, gridwidth=0.5, griddash='dot'), # Subtle grid
        yaxis=dict(gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR, zerolinecolor=BORDER_COLOR, tickfont=dict(family=GLOBAL_FONT_FAMILY.split(',')[0].strip(), color=SUBTLE_TEXT_COLOR_DARK_THEME, size=11), showgrid=True, gridwidth=0.5, griddash='dot'),
        margin=dict(l=60, r=40, t=70, b=60) # Consistent margins
    )
    if height:
        layout_args['height'] = height
    return layout_args

# ------------------- Title -------------------
st.markdown("<h1>🌬️ AuraVision AQI</h1>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; margin-bottom: 2.5rem;">
    <p style="color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 1.25rem; font-weight: 400; letter-spacing: 0.7px;">
        Illuminating Air Quality Insights Across India
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------- Load Data -------------------
@st.cache_data(ttl=3600)
def load_data_and_metadata():
    fallback_file = "combined_air_quality.txt" #
    df_loaded = None
    load_msg = ""
    last_update_time = None
    try:
        if not os.path.exists(fallback_file):
            return pd.DataFrame(), f"Error: Main data file '{fallback_file}' not found.", None
        df_loaded = pd.read_csv(fallback_file, sep="\t", parse_dates=['date']) #
        load_msg = f"Displaying archive data from **{os.path.basename(fallback_file)}**" #
        try:
            last_update_time = pd.Timestamp(os.path.getmtime(fallback_file), unit='s').tz_localize('UTC').tz_convert('Asia/Kolkata')
        except Exception:
            last_update_time = pd.Timestamp(os.path.getmtime(fallback_file), unit='s')
    except Exception as e:
        return pd.DataFrame(), f"FATAL: Error loading '{fallback_file}': {e}.", None

    if df_loaded is not None:
        for col, default_val in [('pollutant', np.nan), ('level', 'Unknown')]:
            if col not in df_loaded.columns: df_loaded[col] = default_val
        df_loaded['pollutant'] = df_loaded['pollutant'].astype(str).str.split(',').str[0].str.strip().replace(['nan', 'NaN', 'None', ''], np.nan)
        df_loaded['level'] = df_loaded['level'].astype(str).fillna('Unknown')
        df_loaded['pollutant'] = df_loaded['pollutant'].fillna('Other')
    else:
        df_loaded = pd.DataFrame()
        load_msg = "Error: Dataframe is None after loading attempt."
    return df_loaded, load_msg, last_update_time

df, load_message, data_last_updated = load_data_and_metadata()

if df.empty:
    st.error(f"Dashboard cannot operate without data. {load_message}")
    st.stop()

if data_last_updated:
    update_time_str = data_last_updated.strftime('%B %d, %Y, %H:%M %Z') if hasattr(data_last_updated, 'tzinfo') and data_last_updated.tzinfo is not None else data_last_updated.strftime('%B %d, %Y, %H:%M (UTC)')
    st.markdown(f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 1rem; margin-bottom: 3rem;'>Archive data last updated: {update_time_str}</p>", unsafe_allow_html=True)
else:
    st.markdown(f"<p style='text-align: center; color: {SUBTLE_TEXT_COLOR_DARK_THEME}; font-size: 1rem; margin-bottom: 3rem;'>{load_message}</p>", unsafe_allow_html=True)

# ------------------- Sidebar Filters -------------------
st.sidebar.markdown("## 🔬 EXPLORE DATA")
st.sidebar.markdown("---", unsafe_allow_html=True)
st.sidebar.info("Adjust filters to analyze AQI trends for specific regions and times.")

unique_cities = sorted(df['city'].unique()) if 'city' in df.columns else []
default_city_val = ["Delhi"] if "Delhi" in unique_cities else (unique_cities[0:1] if unique_cities else [])
selected_cities = st.sidebar.multiselect("🏙️ Select Cities", unique_cities, default=default_city_val)

years = sorted(df['date'].dt.year.unique(), reverse=True)
default_year_val = years[0] if years else None
year_selection_disabled = not bool(years)
year = st.sidebar.selectbox("🗓️ Select Year", years, index=0, disabled=year_selection_disabled)

months_map_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
month_options_list = ["All Months"] + list(months_map_dict.values())
selected_month_name_short = st.sidebar.selectbox("🌙 Select Month (Optional)", month_options_list, index=0, disabled=year_selection_disabled)

month_number_filter = None
full_selected_month_name = "All Months"
if selected_month_name_short != "All Months":
    month_number_filter = list(months_map_dict.keys())[list(months_map_dict.values()).index(selected_month_name_short)]
    full_month_names_long = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    full_selected_month_name = full_month_names_long[month_number_filter]


if year:
    df_period_filtered = df[df['date'].dt.year == year].copy()
    if month_number_filter:
        df_period_filtered = df_period_filtered[df_period_filtered['date'].dt.month == month_number_filter]
else:
    df_period_filtered = pd.DataFrame()

# ------------------- 💡 NATIONAL KEY INSIGHTS -------------------
st.markdown("## 🇮🇳 NATIONAL SNAPSHOT")
with st.container():
    if year:
        st.markdown(f"##### Key Metro Annual Average AQI ({year})")
        major_cities = ['Delhi', 'Mumbai', 'Kolkata', 'Bengaluru', 'Chennai', 'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
        major_cities_data_for_year = df[df['date'].dt.year == year]
        major_cities_data = major_cities_data_for_year[major_cities_data_for_year['city'].isin(major_cities)]
        if not major_cities_data.empty:
            avg_aqi_major_cities = major_cities_data.groupby('city')['index'].mean()
            present_major_cities = [city for city in major_cities if city in avg_aqi_major_cities.index and pd.notna(avg_aqi_major_cities[city])]
            if present_major_cities:
                cols = st.columns(min(len(present_major_cities), 5)) # Max 5 cols for readability
                col_idx = 0
                for city_name in present_major_cities:
                    with cols[col_idx % len(cols)]:
                        aqi_val = avg_aqi_major_cities.get(city_name)
                        st.metric(label=city_name, value=f"{aqi_val:.1f}")
                    col_idx += 1
            else: st.info(f"No data for key metro cities in {year}.")
        else: st.info(f"No annual data for key metro cities in {year}.")

        st.markdown(f"##### General Insights for Selected Period ({full_selected_month_name}, {year})")
        if not df_period_filtered.empty:
            avg_aqi_national = df_period_filtered['index'].mean()
            city_avg_aqi_stats = df_period_filtered.groupby('city')['index'].mean().dropna()
            if not city_avg_aqi_stats.empty:
                num_cities_observed = df_period_filtered['city'].nunique()
                best_city_name, best_city_aqi = city_avg_aqi_stats.idxmin(), city_avg_aqi_stats.min()
                worst_city_name, worst_city_aqi = city_avg_aqi_stats.idxmax(), city_avg_aqi_stats.max()
                st.markdown(f"""<div style="font-size: 1.1rem; line-height: 1.75;">
                    Across <b>{num_cities_observed}</b> observed cities, the average AQI is <b style="color:{ACCENT_COLOR}; font-size:1.2em;">{avg_aqi_national:.2f}</b>.
                    Best performer: <b style="color:{CATEGORY_COLORS_DARK['Good']};">{best_city_name}</b> ({best_city_aqi:.2f}).
                    Most challenged: <b style="color:{CATEGORY_COLORS_DARK['Severe']};">{worst_city_name}</b> ({worst_city_aqi:.2f}).
                </div>""", unsafe_allow_html=True)
            else: st.info("AQI statistics could not be computed for the selection.")
        else: st.info("No data for general insights in the selected period.")
    else: st.warning("Please select a year for national insights.")
st.markdown("---")
# ------------------- 🗺️ GEOSPATIAL AQI ANALYSIS OF INDIA -------------------
st.markdown("## 🗺️ INDIA AQI HEATMAP (STATE-LEVEL)")
with st.container():
    geojson_path = "india_states.geojson" # User MUST provide this file
    # IMPORTANT: User needs to expand this mapping significantly for their data
    city_to_state_mapping = {
        'Agartala': 'Tripura', 'Ahmedabad': 'Gujarat', 'Aizawl': 'Mizoram', 'Ajmer': 'Rajasthan',
        'Alwar': 'Rajasthan', 'Amaravati': 'Andhra Pradesh', 'Ambala': 'Haryana', 'Amritsar': 'Punjab',
        'Ankleshwar': 'Gujarat', 'Asansol': 'West Bengal', 'Aurangabad': 'Maharashtra', 'Bagalkot': 'Karnataka',
        'Bahadurgarh': 'Haryana', 'Ballabgarh': 'Haryana', 'Bathinda': 'Punjab', 'Begusarai': 'Bihar',
        'Belgaum': 'Karnataka', 'Bengaluru': 'Karnataka', 'Bhiwadi': 'Rajasthan', 'Bhiwani': 'Haryana',
        'Bhopal': 'Madhya Pradesh', 'Bhubaneswar': 'Odisha', 'Bidar': 'Karnataka',
        'Bihar Sharif': 'Bihar', 'Bilaspur': 'Chhattisgarh', 'Brajrajnagar': 'Odisha', 'Bulandshahr': 'Uttar Pradesh',
        'Buxar': 'Bihar', 'Chamarajanagar': 'Karnataka', 'Chandigarh': 'Chandigarh', 'Chandrapur': 'Maharashtra',
        'Charkhi Dadri': 'Haryana', 'Chennai': 'Tamil Nadu', 'Chhapra': 'Bihar', 'Chikkaballapur': 'Karnataka',
        'Chikkamagaluru': 'Karnataka', 'Coimbatore': 'Tamil Nadu', 'Damoh': 'Madhya Pradesh',
        'Darbhanga': 'Bihar', 'Davanagere': 'Karnataka', 'Dehradun': 'Uttarakhand', 'Dewas': 'Madhya Pradesh',
        'Delhi': 'Delhi', 'Dharuhera': 'Haryana', 'Durgapur': 'West Bengal', 'Ernakulam': 'Kerala',
        'Erode': 'Tamil Nadu', 'Faridabad': 'Haryana', 'Fatehabad': 'Haryana', 'Firozabad': 'Uttar Pradesh',
        'Gadag': 'Karnataka', 'Gandhinagar': 'Gujarat', 'Gangtok': 'Sikkim', 'Gaya': 'Bihar', 'Ghaziabad': 'Uttar Pradesh',
        'Gorakhpur': 'Uttar Pradesh', 'Greater Noida': 'Uttar Pradesh', 'Gummidipoondi': 'Tamil Nadu',
        'Guntur': 'Andhra Pradesh', 'Gurugram': 'Haryana', 'Guwahati': 'Assam', 'Gwalior': 'Madhya Pradesh',
        'Hajipur': 'Bihar', 'Haldia': 'West Bengal', 'Hapur': 'Uttar Pradesh', 'Hassan': 'Karnataka',
        'Hisar': 'Haryana', 'Howrah': 'West Bengal', 'Hubballi': 'Karnataka', 'Hyderabad': 'Telangana',
        'Imphal': 'Manipur', 'Indore': 'Madhya Pradesh', 'Jabalpur': 'Madhya Pradesh', 'Jaipur': 'Rajasthan',
        'Jalandhar': 'Punjab', 'Jind': 'Haryana', 'Jodhpur': 'Rajasthan', 'Kadapa': 'Andhra Pradesh',
        'Kolar': 'Karnataka', 'Kalaburagi': 'Karnataka', 'Kalyan': 'Maharashtra', 'Kannur': 'Kerala',
        'Kanpur': 'Uttar Pradesh', 'Karnal': 'Haryana', 'Katihar': 'Bihar', 'Katni': 'Madhya Pradesh',
        'Khanna': 'Punjab', 'Khurja': 'Uttar Pradesh', 'Kochi': 'Kerala', 'Kohima': 'Nagaland',
        'Kolhapur': 'Maharashtra', 'Kolkata': 'West Bengal', 'Kollam': 'Kerala', 'Korba': 'Chhattisgarh',
        'Kota': 'Rajasthan', 'Kozhikode': 'Kerala', 'Kurnool': 'Andhra Pradesh', 'Kurukshetra': 'Haryana',
        'Lucknow': 'Uttar Pradesh', 'Ludhiana': 'Punjab', 'Maihar': 'Madhya Pradesh', 'Mandideep': 'Madhya Pradesh',
        'Mandikhera': 'Haryana', 'Manesar': 'Haryana', 'Mangalore': 'Karnataka', 'Mathura': 'Uttar Pradesh',
        'Medikeri': 'Karnataka', 'Meerut': 'Uttar Pradesh', 'Moradabad': 'Uttar Pradesh', 'Motihari': 'Bihar',
        'Mumbai': 'Maharashtra', 'Munger': 'Bihar', 'Muzaffarnagar': 'Uttar Pradesh', 'Muzaffarpur': 'Bihar',
        'Mysuru': 'Karnataka', 'Nagpur': 'Maharashtra', 'Nalbari': 'Assam', 'Nandesari': 'Gujarat',
        'Nashik': 'Maharashtra', 'Navi Mumbai': 'Maharashtra', 'Noida': 'Uttar Pradesh', 'Palwal': 'Haryana',
        'Panipat': 'Haryana', 'Patiala': 'Punjab', 'Patna': 'Bihar', 'Pithampur': 'Madhya Pradesh',
        'Prayagraj': 'Uttar Pradesh', 'Puducherry': 'Puducherry', 'Pune': 'Maharashtra', 'Purnia': 'Bihar',
        'Raichur': 'Karnataka', 'Raipur': 'Chhattisgarh', 'Rajamahendravaram': 'Andhra Pradesh', 'Rajgir': 'Bihar',
        'Ramanagara': 'Karnataka', 'Ranchi': 'Jharkhand', 'Ratlam': 'Madhya Pradesh', 'Rohtak': 'Haryana',
        'Rupnagar': 'Punjab', 'Sagar': 'Madhya Pradesh', 'Saharsa': 'Bihar', 'Salem': 'Tamil Nadu',
        'Samastipur': 'Bihar', 'Sasaram': 'Bihar', 'Satna': 'Madhya Pradesh', 'Shillong': 'Meghalaya',
        'Shimla': 'Himachal Pradesh', 'Shivamogga': 'Karnataka', 'Silchar': 'Assam', 'Singrauli': 'Madhya Pradesh',
        'Sirsa': 'Haryana', 'Siwan': 'Bihar', 'Solapur': 'Maharashtra', 'Sonipat': 'Haryana', 'Sri Ganganagar': 'Rajasthan',
        'Srinagar': 'Jammu and Kashmir', 'Talcher': 'Odisha', 'Thane': 'Maharashtra', 'Thiruvananthapuram': 'Kerala',
        'Thrissur': 'Kerala', 'Thoothukudi': 'Tamil Nadu', 'Tiruchirappalli': 'Tamil Nadu', 'Tirupati': 'Andhra Pradesh',
        'Udaipur': 'Rajasthan', 'Udupi': 'Karnataka', 'Ujjain': 'Madhya Pradesh', 'Vadodara': 'Gujarat',
        'Vapi': 'Gujarat', 'Varanasi': 'Uttar Pradesh', 'Vatva': 'Gujarat', 'Vijayapura': 'Karnataka',
        'Vijayawada': 'Andhra Pradesh', 'Visakhapatnam': 'Andhra Pradesh', 'Yamunanagar': 'Haryana', 'Warangal': 'Telangana'
        # ADD MORE CITIES AND THEIR STATES HERE
    }
    choropleth_rendered = False
    try:
        with open(geojson_path, 'r') as f:
            india_geojson = json.load(f)
        
        if not df_period_filtered.empty:
            df_period_filtered['state'] = df_period_filtered['city'].map(city_to_state_mapping)
            state_aqi_data = df_period_filtered.dropna(subset=['state']).groupby('state')['index'].mean().reset_index()
            state_aqi_data.columns = ['state', 'avg_aqi']

            if not state_aqi_data.empty:
                fig_choropleth = px.choropleth_mapbox(
                    state_aqi_data,
                    geojson=india_geojson,
                    locations='state', # Name of the column in your DataFrame
                    featureidkey="properties.ST_NM", # Path to state names in GeoJSON properties, MIGHT NEED ADJUSTMENT
                    color='avg_aqi',
                    color_continuous_scale="Plasma_r", # Or "Viridis_r", "Inferno_r"
                    range_color=(state_aqi_data['avg_aqi'].min(), state_aqi_data['avg_aqi'].max()),
                    mapbox_style="carto_darkmatter", # Consistent dark style
                    zoom=3.8,
                    center={"lat": 23.5, "lon": 80.5}, # Adjusted center for India
                    opacity=0.7,
                    hover_name='state',
                    hover_data={'avg_aqi': ':.2f', 'state': False}
                )
                custom_map_layout = get_custom_plotly_layout_args(height=750)
                custom_map_layout['mapbox_style'] = "carto-darkmatter" # Ensure mapbox style is dark
                custom_map_layout['margin'] = {"r":0,"t":30,"l":0,"b":0} # Minimal margins for map
                custom_map_layout['legend']['y'] = 0.95 # Adjust legend if needed
                
                fig_choropleth.update_layout(**custom_map_layout)
                fig_choropleth.update_traces(hovertemplate="<b>%{hovertext}</b><br>Avg. AQI: %{z:.2f}<extra></extra>")
                
                st.plotly_chart(fig_choropleth, use_container_width=True)
                st.caption("ℹ️ State-level average AQI. Ensure 'india_states.geojson' is accurate and city-to-state mapping is comprehensive. GeoJSON state name property used: `properties.ST_NM`.")
                choropleth_rendered = True
            else:
                st.info("No state-level AQI data to display for the selected period after mapping. Please check the city-to-state mapping and data availability.")
        else:
            st.info("No data available for the selected period to generate the India AQI Heatmap.")

    except FileNotFoundError:
        st.warning(f"⚠️ **GeoJSON file for India states (`{geojson_path}`) not found.** Cannot display the spatial India plot. Please add the required GeoJSON file to your project directory.")
    except Exception as e:
        st.error(f"❌ Could not render spatial India plot: {e}. This might be due to issues with the GeoJSON file structure or data processing.")

    # Fallback to scatter plot if choropleth wasn't rendered and data exists
    if not choropleth_rendered and not df_period_filtered.empty:
        st.markdown("##### City Hotspots (Fallback View)")
        city_coords_data = {}
        coords_file_path = "lat_long.txt" #
        try:
            if os.path.exists(coords_file_path):
                with open(coords_file_path, "r", encoding="utf-8") as f: file_content = f.read() #
                local_scope = {}
                # For debugging: st.write(f"Content of {coords_file_path} read. Attempting exec...")
                exec(file_content, {}, local_scope)
                # For debugging: st.write(f"Keys in local_scope after exec: {list(local_scope.keys())}")
                if 'city_coords' in local_scope and isinstance(local_scope['city_coords'], dict):
                    city_coords_data = local_scope['city_coords']
                # else: # Error for this handled by city_coords_data remaining empty
                #    st.caption(f"Debug: 'city_coords' not found or not dict in {coords_file_path}. Keys found: {list(local_scope.keys())}")


            if city_coords_data: # Proceed only if coordinates were loaded
                map_grouped_data = df_period_filtered.groupby('city').agg(
                    avg_aqi=('index', 'mean'),
                    dominant_pollutant=('pollutant', lambda x: x.mode().iloc[0] if not x.mode().empty and x.mode().iloc[0] != 'Other' else (x[x != 'Other'].mode().iloc[0] if not x[x != 'Other'].mode().empty else 'N/A'))
                ).reset_index()

                latlong_map_df_list = []
                for city_name, coords in city_coords_data.items():
                    if isinstance(coords, (list, tuple)) and len(coords) == 2:
                        try:
                            lat, lon = float(coords[0]), float(coords[1])
                            latlong_map_df_list.append({'city': city_name, 'lat': lat, 'lon': lon})
                        except (ValueError, TypeError): pass
                latlong_map_df = pd.DataFrame(latlong_map_df_list)

                if not latlong_map_df.empty and not map_grouped_data.empty:
                    map_merged_df = pd.merge(map_grouped_data, latlong_map_df, on='city', how='inner')
                    map_merged_df["AQI Category"] = map_merged_df["avg_aqi"].apply(lambda val: next((k for k, v_range in {'Good': (0, 50), 'Satisfactory': (51, 100), 'Moderate': (101, 200), 'Poor': (201, 300), 'Very Poor': (301, 400), 'Severe': (401, float('inf'))}.items() if v_range[0] <= val <= v_range[1]), "Unknown") if pd.notna(val) else "Unknown")
                    
                    if not map_merged_df.empty:
                        fig_scatter_map = px.scatter_mapbox(map_merged_df, lat="lat", lon="lon", size=np.maximum(map_merged_df["avg_aqi"], 1), size_max=28, color="AQI Category", color_discrete_map=CATEGORY_COLORS_DARK, hover_name="city", custom_data=['city', 'avg_aqi', 'dominant_pollutant', 'AQI Category'], text="city", zoom=4, center={"lat": 23.5, "lon": 82.0})
                        fig_scatter_map.update_traces(textposition='top right', marker=dict(sizemin=6, opacity=0.8), hovertemplate="<b style='font-size:1.1em;'>%{customdata[0]}</b><br>Avg. AQI: %{customdata[1]:.1f} (%{customdata[3]})<br>Dominant: %{customdata[2]}<extra></extra>")
                        scatter_map_layout = get_custom_plotly_layout_args(height=700)
                        scatter_map_layout['mapbox_style'] = "carto-darkmatter"
                        scatter_map_layout['margin'] = {"r":0,"t":10,"l":0,"b":0}
                        fig_scatter_map.update_layout(**scatter_map_layout)
                        st.plotly_chart(fig_scatter_map, use_container_width=True)
                        choropleth_rendered = True # To prevent bar chart fallback if this succeeds
                    # else: st.info("No city data matched with coordinates for scatter map.")
            # else: st.info("City coordinates from 'lat_long.txt' could not be loaded for scatter map fallback.")
        except Exception as e_scatter:
            st.error(f"Error rendering city scatter map: {e_scatter}")


    # Fallback to Bar Chart if no map was rendered
    if not choropleth_rendered and not df_period_filtered.empty:
        st.markdown("##### City AQI Overview (Map Unavailable)")
        st.info("Map visualization could not be rendered. Displaying average AQI by city instead.")
        avg_aqi_cities_alt = df_period_filtered.groupby('city')['index'].mean().dropna().sort_values(ascending=False).reset_index()
        if not avg_aqi_cities_alt.empty:
            fig_alt_bar = px.bar(avg_aqi_cities_alt, x='index', y='city', orientation='h', color='index', color_continuous_scale=px.colors.sequential.Plasma_r, labels={'index': 'Average AQI', 'city': 'City'}, height=max(400, len(avg_aqi_cities_alt['city']) * 30))
            fig_alt_bar.update_layout(**get_custom_plotly_layout_args(), yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_alt_bar, use_container_width=True)
        else: st.warning("No city AQI data available for bar chart fallback.")

    elif not choropleth_rendered and df_period_filtered.empty:
         st.warning("No data available for the selected period to display on any map or chart.")
st.markdown("---")

# ------------------- 🆚 CITY-WISE AQI COMPARISON -------------------
if len(selected_cities) > 1 and not df_period_filtered.empty:
    st.markdown("## 📊 CITY-TO-CITY AQI ANALYSIS")
    comp_tab1, comp_tab2 = st.tabs(["📈 AQI Trend Comparison", "🌀 Seasonal AQI Radar"])
    comparison_df_list = [df_period_filtered[df_period_filtered['city'] == city_comp].copy() for city_comp in selected_cities if not df_period_filtered[df_period_filtered['city'] == city_comp].empty]

    if comparison_df_list:
        with comp_tab1:
            st.markdown("##### AQI Trends Over Selected Period")
            combined_comp_df = pd.concat(comparison_df_list)
            fig_cmp = px.line(combined_comp_df, x='date', y='index', color='city', labels={'index': 'AQI Index', 'date': 'Date', 'city': 'City'}, line_shape='spline', color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_cmp.update_layout(**get_custom_plotly_layout_args(height=550))
            fig_cmp.update_traces(line=dict(width=3))
            st.plotly_chart(fig_cmp, use_container_width=True)
        with comp_tab2:
            st.markdown("##### Annual Seasonal AQI Patterns")
            radar_fig = go.Figure()
            df_year_radar = df[df['date'].dt.year == year]
            cities_for_radar = [city for city in selected_cities if not df_year_radar[df_year_radar['city'] == city].empty]
            if cities_for_radar:
                max_aqi_radar, radar_colors = 0, px.colors.qualitative.Vivid
                for i, city_name in enumerate(cities_for_radar):
                    monthly_avg_aqi = df_year_radar[df_year_radar['city'] == city_name].groupby(df_year_radar['date'].dt.month)['index'].mean().reindex(range(1, 13))
                    max_aqi_radar = max(max_aqi_radar, monthly_avg_aqi.max() if pd.notna(monthly_avg_aqi.max()) else 0)
                    radar_fig.add_trace(go.Scatterpolar(r=monthly_avg_aqi.values, theta=[m.upper() for m in months_map_dict.values()], fill='toself', name=city_name, hovertemplate=f"<b>{city_name}</b><br>%{{theta}}: %{{r:.1f}}<extra></extra>", line_color=radar_colors[i % len(radar_colors)], opacity=0.75))
                radar_layout_args = get_custom_plotly_layout_args(height=600)
                radar_layout_args['polar'] = dict(radialaxis=dict(visible=True, range=[0, max_aqi_radar * 1.1 if max_aqi_radar > 0 else 50], gridcolor=BORDER_COLOR, linecolor=BORDER_COLOR), angularaxis=dict(linecolor=BORDER_COLOR, gridcolor=BORDER_COLOR, tickfont_size=12), bgcolor=BACKGROUND_COLOR)
                radar_layout_args['legend']['y'] = -0.15 # Below plot
                radar_fig.update_layout(**radar_layout_args)
                st.plotly_chart(radar_fig, use_container_width=True)
            else: st.info("Insufficient data for seasonal radar patterns for selected cities in this year.")
    elif len(selected_cities) > 1: st.info("Not enough data for comparison with current filters.")
st.markdown("---")

# ------------------- 🏙️ CITY-SPECIFIC DEEP DIVE -------------------
if selected_cities and (not df_period_filtered.empty or year):
    st.markdown("## 🔎 DETAILED CITY VIEW")
    for city in selected_cities:
        city_data_section = df_period_filtered[df_period_filtered['city'] == city].copy() if month_number_filter else df[(df['city'] == city) & (df['date'].dt.year == year)].copy()
        st.markdown(f"### {city.upper()} – {full_selected_month_name if month_number_filter else 'Full Year'}, {year}")
        if city_data_section.empty:
            st.warning(f"😔 No data for {city} in the selected period."); continue
        with st.container():
            st.markdown("##### 📅 Daily AQI Calendar")
            start_cal, end_cal = (pd.to_datetime(f'{year}-{month_number_filter:02d}-01'), pd.to_datetime(f'{year}-{month_number_filter:02d}-01') + pd.offsets.MonthEnd(0)) if month_number_filter else (pd.to_datetime(f'{year}-01-01'), pd.to_datetime(f'{year}-12-31'))
            cal_df = pd.DataFrame({'date': pd.date_range(start_cal, end_cal)})
            cal_df['week'], cal_df['day_of_week'] = cal_df['date'].dt.isocalendar().week, cal_df['date'].dt.dayofweek
            if not month_number_filter:
                cal_df.loc[(cal_df['date'].dt.month == 1) & (cal_df['week'] > 50), 'week'] = 0
                max_w = cal_df['week'].max(); cal_df.loc[(cal_df['date'].dt.month == 12) & (cal_df['week'] == 1), 'week'] = max_w + 1 if max_w < 53 else 53
            
            first_weeks = cal_df.groupby(cal_df['date'].dt.month)['week'].min()
            merged_cal = pd.merge(cal_df, city_data_section[['date', 'index', 'level']], on='date', how='left')
            merged_cal['level'].fillna('Unknown', inplace=True); merged_cal['aqi_text'] = merged_cal['index'].apply(lambda x: f'{x:.0f}' if pd.notna(x) else 'N/A')
            
            sorted_levels, colors_cal, lvl_map = list(CATEGORY_COLORS_DARK.keys()), [CATEGORY_COLORS_DARK[lvl] for lvl in CATEGORY_COLORS_DARK.keys()], {lvl: i for i, lvl in enumerate(CATEGORY_COLORS_DARK.keys())}
            fig_cal = go.Figure(data=go.Heatmap(x=merged_cal['week'], y=merged_cal['day_of_week'], z=merged_cal['level'].map(lvl_map), customdata=pd.DataFrame({'date': merged_cal['date'].dt.strftime('%b %d, %Y'), 'level': merged_cal['level'], 'aqi': merged_cal['aqi_text']}), hovertemplate="<b>%{customdata[0]}</b><br>AQI: %{customdata[2]} (%{customdata[1]})<extra></extra>", colorscale=colors_cal, showscale=False, xgap=3.5, ygap=3.5, zmin=0, zmax=len(sorted_levels)-1))
            
            annotations_cal = [go.layout.Annotation(text=months_map_dict[mn][:3].upper(), align='center', showarrow=False, xref='x domain', yref='paper', x=(wk - cal_df['week'].min() + 0.5) / (cal_df['week'].max() - cal_df['week'].min() + 1) if (cal_df['week'].max() - cal_df['week'].min() + 1) > 0 else 0.5, y=1.07, font=dict(color=SUBTLE_TEXT_COLOR_DARK_THEME, size=12, family=GLOBAL_FONT_FAMILY.split(',')[0].strip())) for mn, wk in first_weeks.items() if (len(cal_df['week'].unique()) > 3 or not month_number_filter)]
            
            cal_layout_args = get_custom_plotly_layout_args(height=(180 + 20*7) if month_number_filter else (200 + 20*7)) # Fixed height by days
            cal_layout_args['yaxis'].update(dict(tickmode='array', tickvals=list(range(7)), ticktext=['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'], autorange="reversed", linecolor=BACKGROUND_COLOR, gridcolor=BACKGROUND_COLOR, showgrid=False, zeroline=False))
            cal_layout_args['xaxis'].update(dict(tickmode='array', ticktext=[], tickvals=[], linecolor=BACKGROUND_COLOR, gridcolor=BACKGROUND_COLOR, showgrid=False, zeroline=False))
            cal_layout_args['plot_bgcolor'] = CARD_BACKGROUND_COLOR
            cal_layout_args['margin'] = dict(t=70 if annotations_cal else 40, b=30, l=60, r=30)
            cal_layout_args['annotations'] = annotations_cal
            fig_cal.update_layout(**cal_layout_args)
            st.plotly_chart(fig_cal, use_container_width=True)

            city_data_tabs = df_period_filtered[df_period_filtered['city'] == city].copy()
            if not city_data_tabs.empty:
                city_tabs_list = st.tabs(["📊 AQI Category Distribution", "📅 Monthly AQI Heatmap"])
                with city_tabs_list[0]:
                    cat_counts = city_data_tabs['level'].value_counts().reindex(CATEGORY_COLORS_DARK.keys(), fill_value=0).reset_index(); cat_counts.columns = ['AQI Category', 'Number of Days']
                    fig_dist = px.bar(cat_counts, x='AQI Category', y='Number of Days', color='AQI Category', color_discrete_map=CATEGORY_COLORS_DARK, text_auto='.0f')
                    fig_dist.update_layout(**get_custom_plotly_layout_args(height=480), xaxis_title=None, yaxis_title="Number of Days")
                    fig_dist.update_traces(marker_line_width=0, marker_line_color=BACKGROUND_COLOR)
                    st.plotly_chart(fig_dist, use_container_width=True)
                with city_tabs_list[1]:
                    city_data_tabs['month_name'] = pd.Categorical(city_data_tabs['date'].dt.strftime('%B'), categories=[full_month_names_long[i] for i in sorted(city_data_tabs['date'].dt.month.unique())], ordered=True)
                    heat_pivot = city_data_tabs.pivot_table(index='month_name', columns=city_data_tabs['date'].dt.day, values='index', observed=True)
                    fig_heat = px.imshow(heat_pivot, labels=dict(x="Day of Month", y="Month", color="AQI Index"), aspect="auto", color_continuous_scale=px.colors.sequential.YlOrRd_r, text_auto=".0f") # Different colors
                    heat_layout_args = get_custom_plotly_layout_args(height=max(380, len(heat_pivot.index)*65))
                    heat_layout_args['xaxis_side'] = "top"; heat_layout_args['yaxis_title'] = None
                    heat_layout_args['xaxis']['showgrid'] = False; heat_layout_args['yaxis']['showgrid'] = False;
                    fig_heat.update_layout(**heat_layout_args)
                    st.plotly_chart(fig_heat, use_container_width=True)
            # else: st.info(f"Detailed distribution and heatmap not available for {city} for the specific month selected. Full year calendar shown above.")
st.markdown("---")
# ------------------- 💨 POLLUTANT ANALYSIS -------------------
st.markdown("## 🧪 DOMINANT POLLUTANT INSIGHTS")
with st.container():
    if not df_period_filtered.empty:
        st.markdown(f"#### Key Pollutants ({full_selected_month_name}, {year})")
        city_opts_pollutant = selected_cities if selected_cities else unique_cities
        def_poll_city_idx = 0
        if default_city_val and default_city_val[0] in city_opts_pollutant: def_poll_city_idx = city_opts_pollutant.index(default_city_val[0])
        
        if city_opts_pollutant:
            city_pollutant_sel = st.selectbox("Select City for Pollutant Breakdown:", city_opts_pollutant, key="pollutant_city_sel_v3", index=def_poll_city_idx)
            poll_data = df_period_filtered[df_period_filtered['city'] == city_pollutant_sel].copy()
            if not poll_data.empty and 'pollutant' in poll_data.columns and poll_data['pollutant'].notna().any():
                valid_polls = poll_data[(poll_data['pollutant'] != 'Other') & (poll_data['pollutant'].notna())].copy()
                if not valid_polls.empty:
                    grouped_polls = valid_polls.groupby('pollutant').size().reset_index(name='count')
                    total_val_poll_days = grouped_polls['count'].sum()
                    if total_val_poll_days > 0:
                        grouped_polls['percentage'] = (grouped_polls['count'] / total_val_poll_days * 100)
                        fig_poll = px.bar(grouped_polls.sort_values('percentage', ascending=False), x='pollutant', y='percentage', color='pollutant', labels={'percentage': 'Dominance (% of Days with pollutant data)', 'pollutant': 'Pollutant'}, color_discrete_map=POLLUTANT_COLORS_DARK, text_auto='.1f%%')
                        fig_poll.update_layout(**get_custom_plotly_layout_args(height=520), yaxis_ticksuffix="%", xaxis_title=None)
                        fig_poll.update_traces(marker_line_width=0)
                        st.plotly_chart(fig_poll, use_container_width=True)
                    else: st.info(f"No specific dominant pollutant data (excluding 'Other' and missing) for {city_pollutant_sel}.")
                else: st.info(f"No specific dominant pollutant data (excluding 'Other' and missing) for {city_pollutant_sel}.")
            else: st.warning(f"No pollutant data for {city_pollutant_sel} in this period.")
        else: st.info("No cities available for pollutant analysis with current filters.")
    else: st.info("Select a valid period to see pollutant analysis.")
st.markdown("---")
# ------------------- Footer -------------------
st.markdown(f"""
<div class="footer">
    <p style="font-size: 1.1rem; color: {TEXT_COLOR_DARK_THEME}; font-weight: 700;">💨 AuraVision AQI Dashboard</p>
    <p>Conceptualized by: Mr. Kapil Meena & Prof. Arkopal K. Goswami, IIT Kharagpur.</p>
    <p>Enhanced with AI by Google Gemini ✨</p>
    <p style="margin-top:1.3em;"><a href="https://github.com/kapil2020/india-air-quality-dashboard" target="_blank" rel="noopener noreferrer">🔗 View on GitHub</a></p>
</div>
""", unsafe_allow_html=True)

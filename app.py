import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta
import ast
import warnings

# --- 1. APP CONFIGURATION & THEME ---
st.set_page_config(
    page_title="India AQI Analytics | Pro Dashboard",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings("ignore")
pio.templates.default = "plotly_dark"

# --- 2. CUSTOM CSS (Glassmorphism & Typography) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #050505;
            color: #E0E0E0;
        }
        
        /* Gradient Background */
        .stApp {
            background: radial-gradient(circle at 10% 20%, #0d1117 0%, #000000 90%);
        }

        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 20px;
            margin-bottom: 20px;
            transition: transform 0.3s ease, border-color 0.3s ease;
        }
        .glass-card:hover {
            transform: translateY(-4px);
            border-color: #00ADB5;
        }

        /* Metric Styling */
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            background: -webkit-linear-gradient(45deg, #00ADB5, #EEEEEE);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-label {
            color: #9CA3AF;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0A0A0A;
            border-right: 1px solid #1F1F1F;
        }
        
        /* Custom Headers */
        h1, h2, h3 {
            color: #FFFFFF;
            font-weight: 700;
        }
        .highlight {
            color: #00ADB5;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. DATA LOADING FUNCTIONS ---

@st.cache_data
def load_data():
    """Loads and preprocesses AQI data with robust caching."""
    try:
        # Check if file exists in current directory or subfolders
        file_path = "combined_air_quality.txt"
        
        # Load Data (Tab Separated)
        df = pd.read_csv(file_path, sep="\t", parse_dates=["date"])
        
        # Data Cleaning
        df['index'] = pd.to_numeric(df['index'], errors='coerce')
        df.dropna(subset=['index', 'date', 'city'], inplace=True)
        df.sort_values('date', inplace=True)
        
        # Extract temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['month_name'] = df['date'].dt.strftime('%B')
        
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}. Please ensure 'combined_air_quality.txt' is uploaded.")
        return pd.DataFrame()

@st.cache_data
def load_coordinates():
    """Parses the python-dictionary formatted lat_long.txt file safely."""
    default_coords = {
        "Delhi": [28.7041, 77.1025], "Mumbai": [19.0760, 72.8777], "Bengaluru": [12.9716, 77.5946],
        "Chennai": [13.0827, 80.2707], "Hyderabad": [17.3850, 78.4867], "Kolkata": [22.5726, 88.3639]
    }
    try:
        with open("lat_long.txt", "r") as f:
            content = f.read()
            # Find the start of the dictionary
            start_index = content.find('{')
            if start_index != -1:
                dict_str = content[start_index:]
                return ast.literal_eval(dict_str)
    except Exception:
        pass
    return default_coords

# Load Data
df = load_data()
city_coords = load_coordinates()

# --- 4. HELPER FUNCTIONS ---

def get_aqi_color(aqi):
    if aqi <= 50: return "#00E400"  # Good
    elif aqi <= 100: return "#FFFF00" # Satisfactory
    elif aqi <= 200: return "#FF7E00" # Moderate
    elif aqi <= 300: return "#FF0000" # Poor
    elif aqi <= 400: return "#99004C" # Very Poor
    else: return "#7E0023" # Severe

def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

# --- 5. SIDEBAR FILTERS (PRESERVED & ENHANCED) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3208/3208705.png", width=60)
    st.title("Control Panel")
    st.markdown("---")

    if not df.empty:
        # 1. Year Filter
        available_years = sorted(df['year'].unique(), reverse=True)
        selected_year = st.selectbox("Select Year", available_years, index=0)
        
        # 2. Month Filter
        available_months = ["All Months"] + list(df['month_name'].unique())
        selected_month = st.selectbox("Select Month", available_months, index=0)
        
        # Filter Logic
        df_filtered = df[df['year'] == selected_year]
        if selected_month != "All Months":
            df_filtered = df_filtered[df_filtered['month_name'] == selected_month]
            
        # 3. City Filter
        all_cities = sorted(df_filtered['city'].unique())
        default_cities = ["Delhi", "Mumbai", "Bengaluru", "Chennai"]
        # Intersect defaults with available to avoid errors
        default_selection = [c for c in default_cities if c in all_cities]
        if not default_selection and all_cities:
            default_selection = [all_cities[0]]
            
        selected_cities = st.multiselect("Select Cities", all_cities, default=default_selection)
        
        if not selected_cities:
            st.warning("Please select at least one city.")
            st.stop()
            
        df_city_filtered = df_filtered[df_filtered['city'].isin(selected_cities)]
    else:
        st.stop()
    
    st.markdown("---")
    st.markdown("### 📊 Data Export")
    if not df_city_filtered.empty:
        csv = df_city_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("Download Filtered Data", csv, "aqi_data.csv", "text/csv")

# --- 6. MAIN DASHBOARD UI ---

# Header Section
col_head1, col_head2 = st.columns([0.7, 0.3])
with col_head1:
    st.markdown(f"# 🇮🇳 India Air Quality <span class='highlight'>Pulse</span>", unsafe_allow_html=True)
    st.markdown(f"**Period:** {selected_month} {selected_year} | **Cities Selected:** {len(selected_cities)}")

# Metrics Row
if not df_city_filtered.empty:
    avg_aqi = df_city_filtered['index'].mean()
    max_aqi = df_city_filtered['index'].max()
    min_aqi = df_city_filtered['index'].min()
    dominant_pollutant = df_city_filtered['pollutant'].mode()[0] if not df_city_filtered['pollutant'].empty else "N/A"
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class='glass-card'>
            <div class='metric-label'>Avg AQI (Selected)</div>
            <div class='metric-value' style='color:{get_aqi_color(avg_aqi)}'>{avg_aqi:.0f}</div>
            <small>{get_aqi_category(avg_aqi)}</small>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class='glass-card'>
            <div class='metric-label'>Max AQI Recorded</div>
            <div class='metric-value' style='color:#FF4B4B'>{max_aqi:.0f}</div>
            <small>Peak Pollution</small>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class='glass-card'>
            <div class='metric-label'>Cleanest Record</div>
            <div class='metric-value' style='color:#00E400'>{min_aqi:.0f}</div>
            <small>Best Air Day</small>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class='glass-card'>
            <div class='metric-label'>Primary Pollutant</div>
            <div class='metric-value' style='font-size:1.8rem'>{dominant_pollutant}</div>
            <small>Most Frequent</small>
        </div>""", unsafe_allow_html=True)

# TABS FOR ORGANIZATION
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview & Trends", "🗺️ Geospatial Intelligence", "⚔️ Comparative Analysis", "🔮 AI Forecast"])

# --- TAB 1: OVERVIEW & TRENDS ---
with tab1:
    col_t1, col_t2 = st.columns([2, 1])
    
    with col_t1:
        st.markdown("### 🗓️ Daily AQI Trends")
        # Rolling Average for smooth lines
        df_city_filtered['Rolling_AQI'] = df_city_filtered.groupby('city')['index'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        
        fig_trend = px.line(
            df_city_filtered, x='date', y='Rolling_AQI', color='city',
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={'Rolling_AQI': 'AQI (3-Day Moving Avg)', 'date': 'Date'}
        )
        fig_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            hovermode="x unified", legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_t2:
        st.markdown("### 🍩 Severity Distribution")
        pie_data = df_city_filtered['level'].value_counts().reset_index()
        pie_data.columns = ['Level', 'Count']
        
        color_map = {
            'Good': '#00E400', 'Satisfactory': '#FFFF00', 'Moderate': '#FF7E00',
            'Poor': '#FF0000', 'Very Poor': '#99004C', 'Severe': '#7E0023'
        }
        
        fig_pie = px.donut(pie_data, values='Count', names='Level', color='Level',
                           color_discrete_map=color_map, hole=0.5)
        fig_pie.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)',
                              margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Heatmap Section
    st.markdown("### 🔥 Pollution Intensity Calendar")
    if len(selected_cities) == 1:
        # Create a calendar-like heatmap for the single selected city
        city_single = df_city_filtered[df_city_filtered['city'] == selected_cities[0]].copy()
        city_single['Week'] = city_single['date'].dt.isocalendar().week
        city_single['Day'] = city_single['date'].dt.day_name()
        
        # Pivot for heatmap
        heatmap_data = city_single.pivot_table(index='Day', columns='Week', values='index', aggfunc='mean')
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex(days_order)
        
        fig_heat = px.imshow(
            heatmap_data, 
            labels=dict(x="Week of Year", y="Day of Week", color="AQI"),
            color_continuous_scale='RdYlGn_r',
            title=f"AQI Heatmap for {selected_cities[0]} ({selected_year})"
        )
        fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Select exactly one city in the sidebar to view its detailed Calendar Heatmap.")

# --- TAB 2: GEOSPATIAL INTELLIGENCE ---
with tab2:
    st.markdown("### 🌍 3D Pollution Map")
    
    # Prepare map data
    map_data = []
    for city in selected_cities:
        if city in city_coords:
            c_data = df_city_filtered[df_city_filtered['city'] == city]
            if not c_data.empty:
                avg_val = c_data['index'].mean()
                map_data.append({
                    'city': city, 'lat': city_coords[city][0], 'lon': city_coords[city][1],
                    'aqi': avg_val, 'color': get_aqi_color(avg_val)
                })
    
    map_df = pd.DataFrame(map_data)
    
    if not map_df.empty:
        # 3D Scatter Mapbox
        fig_map = px.scatter_mapbox(
            map_df, lat="lat", lon="lon", size="aqi", color="aqi",
            hover_name="city", hover_data={"lat": False, "lon": False, "aqi": True},
            color_continuous_scale=["#00E400", "#FFFF00", "#FF7E00", "#FF0000", "#99004C", "#7E0023"],
            range_color=[0, 500], zoom=4, center={"lat": 22.0, "lon": 80.0},
            height=600, size_max=40
        )
        fig_map.update_layout(
            mapbox_style="carto-darkmatter",
            paper_bgcolor='rgba(0,0,0,0)',
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("No coordinate data found for selected cities.")

# --- TAB 3: COMPARATIVE ANALYSIS ---
with tab3:
    st.markdown("### ⚔️ City vs City Performance")
    
    if len(selected_cities) > 1:
        col_c1, col_c2 = st.columns(2)
        
        # 1. Box Plot for Distribution Comparison
        with col_c1:
            st.markdown("#### AQI Distribution & Variability")
            fig_box = px.box(
                df_city_filtered, x='city', y='index', color='city',
                color_discrete_sequence=px.colors.qualitative.Bold,
                points="outliers"
            )
            fig_box.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_box, use_container_width=True)
            
        # 2. Radar Chart
        with col_c2:
            st.markdown("#### Radar: Multi-Metric Comparison")
            radar_data = []
            for city in selected_cities:
                subset = df_city_filtered[df_city_filtered['city'] == city]
                radar_data.append({
                    'City': city,
                    'Average': subset['index'].mean(),
                    'Max Peak': subset['index'].max(),
                    'Volatility (Std)': subset['index'].std()
                })
            radar_df = pd.DataFrame(radar_data)
            
            # Normalize for Radar Chart visibility
            categories = ['Average', 'Max Peak', 'Volatility (Std)']
            fig_radar = go.Figure()
            
            for city in selected_cities:
                city_vals = radar_df[radar_df['City'] == city].iloc[0]
                values = [city_vals['Average'], city_vals['Max Peak'], city_vals['Volatility (Std)']]
                fig_radar.add_trace(go.Scatterpolar(
                    r=values, theta=categories, fill='toself', name=city
                ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
                showlegend=True, paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        # 3. Pollutant Composition Stacked Bar
        st.markdown("#### 🧪 Pollutant Makeup")
        poll_comp = df_city_filtered.groupby(['city', 'pollutant']).size().reset_index(name='count')
        fig_bar = px.bar(poll_comp, x='city', y='count', color='pollutant', title="Dominant Pollutants by City", barmode='stack')
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)
        
    else:
        st.info("Select at least two cities in the sidebar to unlock Comparative Analysis.")

# --- TAB 4: AI FORECAST ---
with tab4:
    st.markdown("### 🔮 Predictive Analytics (Polynomial Regression)")
    st.markdown("Using historical data patterns to predict AQI trends for the next 30 days.")
    
    forecast_city = st.selectbox("Select City to Forecast", selected_cities)
    
    if forecast_city:
        city_df = df[df['city'] == forecast_city].copy()
        
        # Need enough data points
        if len(city_df) > 50:
            # Feature Engineering
            city_df['days_ordinal'] = city_df['date'].map(pd.Timestamp.toordinal)
            
            # Train on data
            X = city_df[['days_ordinal']]
            y = city_df['index']
            
            # Polynomial Regression (Degree 4 for better seasonality capture)
            poly = PolynomialFeatures(degree=4)
            X_poly = poly.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Predict Next 30 Days
            last_date = city_df['date'].max()
            future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
            future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
            future_poly = poly.transform(future_ordinals)
            predictions = model.predict(future_poly)
            
            # Visualization
            fig_pred = go.Figure()
            
            # Historical (Last 90 Days)
            recent = city_df.sort_values('date').tail(90)
            fig_pred.add_trace(go.Scatter(
                x=recent['date'], y=recent['index'], mode='markers', 
                name='Historical Data', marker=dict(color='gray', opacity=0.5)
            ))
            
            # Forecast Line
            fig_pred.add_trace(go.Scatter(
                x=future_dates, y=predictions, mode='lines+markers', 
                name='AI Forecast', line=dict(color='#00ADB5', width=3)
            ))
            
            fig_pred.update_layout(
                title=f"30-Day Forecast for {forecast_city}",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Date", yaxis_title="Predicted AQI",
                hovermode="x unified"
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Insights
            avg_pred = np.mean(predictions)
            trend = "Increasing" if predictions[-1] > predictions[0] else "Decreasing"
            st.markdown(f"""
                <div class='glass-card'>
                    <h4>Forecast Insights</h4>
                    <ul>
                        <li><b>Projected Trend:</b> {trend} over the next 30 days.</li>
                        <li><b>Average Predicted AQI:</b> {avg_pred:.0f} ({get_aqi_category(avg_pred)})</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        else:
            st.warning("Insufficient historical data for this city to generate a reliable forecast.")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280;'>
    <small>Engineered with ❤️ | Data Source: CPCB</small>
</div>
""", unsafe_allow_html=True)

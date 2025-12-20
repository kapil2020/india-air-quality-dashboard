import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from io import StringIO
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION & THEME SETUP
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="AuraVision | Air Quality Analytics",
    page_icon="🌬️",
    initial_sidebar_state="expanded"
)

# Color Palette (Modern Dark Theme)
THEME = {
    "bg": "#0E1117",
    "card_bg": "#1A1C24",
    "card_border": "#2D2F3B",
    "text": "#E0E0E0",
    "sub_text": "#A0A0A0",
    "accent": "#00D4FF",  # Cyan
    "highlight": "#FF4B4B", # Red
    "success": "#00C853", # Green
    "warning": "#FFD600", # Yellow
}

# AQI Color Scale (Standardized)
AQI_COLORS = {
    "Good": "#00E400",          # Green
    "Satisfactory": "#FFFF00",  # Yellow
    "Moderate": "#FF7E00",      # Orange
    "Poor": "#FF0000",          # Red
    "Very Poor": "#8F3F97",     # Purple
    "Severe": "#7E0023",        # Maroon
    "Unknown": "#444444"
}

# Pollutant Colors
POLLUTANT_COLORS = {
    "PM2.5": "#FF4B4B", "PM10": "#00D4FF", "NO2": "#7F00FF",
    "SO2": "#FFFF00", "CO": "#FFA500", "O3": "#00FF00", "Other": "#808080"
}

# Apply Custom CSS
st.markdown(f"""
<style>
    /* Global Font & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: {THEME['bg']};
        color: {THEME['text']};
    }}

    /* Custom Cards */
    .metric-card {{
        background-color: {THEME['card_bg']};
        border: 1px solid {THEME['card_border']};
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        transition: transform 0.2s;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,212,255,0.1);
        border-color: {THEME['accent']};
    }}
    
    /* Headers */
    h1, h2, h3 {{
        color: {THEME['text']};
        font-weight: 700;
    }}
    h1 {{
        background: -webkit-linear-gradient(0deg, {THEME['accent']}, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {THEME['card_bg']};
        border-right: 1px solid {THEME['card_border']};
    }}
    
    /* Plotly Chart Container */
    .stPlotlyChart {{
        background-color: {THEME['card_bg']};
        border-radius: 12px;
        padding: 10px;
        border: 1px solid {THEME['card_border']};
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        border-bottom-color: {THEME['card_border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {THEME['sub_text']};
    }}
    .stTabs [aria-selected="true"] {{
        color: {THEME['accent']} !important;
        border-bottom-color: {THEME['accent']} !important;
    }}

    /* Remove padding/margins from standard containers */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_aqi_category(aqi):
    if pd.isna(aqi): return "Unknown"
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Satisfactory"
    if aqi <= 200: return "Moderate"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"

def apply_chart_theme(fig):
    """Applies a consistent dark theme to Plotly figures"""
    fig.update_layout(
        paper_bgcolor=THEME['card_bg'],
        plot_bgcolor=THEME['card_bg'],
        font={"color": THEME['text'], "family": "Inter"},
        margin={"t": 40, "b": 40, "l": 40, "r": 20},
        xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor=THEME['card_border']),
        yaxis=dict(showgrid=True, gridcolor=THEME['card_border'], zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    return fig

# ==========================================
# 3. DATA LOADING
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
    today = pd.to_datetime("today").date()
    # Paths
    csv_path = f"data/{today}.csv"
    fallback_file = "combined_air_quality.txt"
    
    df = None
    last_updated = None
    source = ""

    # Try loading today's data
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                source = "Live Data (Today)"
                last_updated = pd.Timestamp(os.path.getmtime(csv_path), unit="s")
        except:
            pass

    # Fallback
    if df is None:
        try:
            if os.path.exists(fallback_file):
                df = pd.read_csv(fallback_file, sep="\t", parse_dates=["date"])
                source = "Archived Data"
                last_updated = pd.Timestamp(os.path.getmtime(fallback_file), unit="s")
            else:
                return pd.DataFrame(), "Data file not found", None
        except Exception as e:
            return pd.DataFrame(), str(e), None

    # Cleaning
    for col, default in [("pollutant", "Other"), ("level", "Unknown")]:
        if col not in df.columns: df[col] = default
    
    df["pollutant"] = df["pollutant"].astype(str).str.split(",").str[0].str.strip()
    df["pollutant"] = df["pollutant"].replace(["nan", "NaN", "None", ""], "Other")
    
    # Exclude future data if any (2025 safeguard logic from original)
    if 2025 in df["date"].dt.year.unique():
         df = df[~((df["date"].dt.year == 2025) & (df["date"].dt.month > 5))]
         
    return df, source, last_updated

df, data_source, last_updated_time = load_data()

if df.empty:
    st.error("⚠️ Application could not load data. Please check source files.")
    st.stop()

# ==========================================
# 4. SIDEBAR & FILTERS
# ==========================================
with st.sidebar:
    st.title("🔭 Controls")
    st.markdown(f"<p style='color:{THEME['sub_text']}; font-size:0.8rem'>Source: {data_source}</p>", unsafe_allow_html=True)
    
    # City Filter
    cities = sorted(df["city"].unique()) if "city" in df.columns else []
    default_city = ["Delhi"] if "Delhi" in cities else cities[:1]
    selected_cities = st.multiselect("Select Cities", cities, default=default_city)
    
    # Date Filters
    years = sorted(df["date"].dt.year.unique(), reverse=True)
    selected_year = st.selectbox("Select Year", years)
    
    months = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 
              7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
    
    # Month Filter logic (handling 2025 limit)
    available_months = list(months.values())
    if selected_year == 2025:
        available_months = available_months[:5]
        
    selected_month = st.selectbox("Select Month", ["All Months"] + available_months)
    
    # Filter Logic
    df_filtered = df[df["date"].dt.year == selected_year].copy()
    month_num = None
    if selected_month != "All Months":
        month_num = [k for k,v in months.items() if v == selected_month][0]
        df_filtered = df_filtered[df_filtered["date"].dt.month == month_num]
        
    st.divider()
    st.markdown("Developed by **IIT Kharagpur**")

# ==========================================
# 5. MAIN DASHBOARD
# ==========================================

# Header
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("AuraVision Analytics")
    st.markdown("### Real-time Air Quality Monitoring System")
with col_head2:
    if last_updated_time:
        st.caption(f"Last Updated:\n{last_updated_time.strftime('%Y-%m-%d %H:%M')}")

# --- KEY METRICS ROW ---
st.markdown("### 🇮🇳 National Snapshot")
col1, col2, col3, col4 = st.columns(4)

# Calculate Metrics
avg_aqi = df_filtered["index"].mean()
total_cities = df_filtered["city"].nunique()
worst_city = df_filtered.groupby("city")["index"].mean().idxmax() if not df_filtered.empty else "N/A"
best_city = df_filtered.groupby("city")["index"].mean().idxmin() if not df_filtered.empty else "N/A"

def render_metric_card(container, label, value, sublabel, color_hex):
    with container:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: {THEME['sub_text']}; font-size: 0.9rem; margin-bottom: 5px;">{label}</div>
            <div style="color: {color_hex}; font-size: 2rem; font-weight: 700;">{value}</div>
            <div style="color: {THEME['sub_text']}; font-size: 0.8rem;">{sublabel}</div>
        </div>
        """, unsafe_allow_html=True)

render_metric_card(col1, "Average AQI", f"{avg_aqi:.0f}", f"National Avg ({selected_year})", THEME['accent'])
render_metric_card(col2, "Cities Monitored", total_cities, "Active Stations", THEME['text'])
render_metric_card(col3, "Cleanest City", best_city, "Lowest Avg AQI", THEME['success'])
render_metric_card(col4, "Most Polluted", worst_city, "Highest Avg AQI", THEME['highlight'])

st.markdown("---")

# --- RANKINGS SECTION ---
st.markdown("### 🏆 City Rankings")
col_rank1, col_rank2 = st.columns(2)

if not df_filtered.empty:
    city_avgs = df_filtered.groupby("city")["index"].mean().sort_values()
    
    with col_rank1:
        st.markdown("**Top 5 Cleanest Cities**")
        top5 = city_avgs.head(5).reset_index()
        fig_top = px.bar(top5, x="index", y="city", orientation='h', 
                         color="index", color_continuous_scale="Blugrn_r")
        fig_top.update_layout(xaxis_title="Avg AQI", yaxis_title=None, coloraxis_showscale=False, height=250)
        st.plotly_chart(apply_chart_theme(fig_top), use_container_width=True)
        
    with col_rank2:
        st.markdown("**Top 5 Polluted Cities**")
        bot5 = city_avgs.tail(5).reset_index()
        fig_bot = px.bar(bot5, x="index", y="city", orientation='h',
                         color="index", color_continuous_scale="Reds")
        fig_bot.update_layout(xaxis_title="Avg AQI", yaxis_title=None, coloraxis_showscale=False, height=250)
        st.plotly_chart(apply_chart_theme(fig_bot), use_container_width=True)
else:
    st.info("No data available for rankings.")

# --- CITY DEEP DIVE ---
st.markdown("---")
st.markdown(f"### 🏙️ City Deep Dive: {', '.join(selected_cities[:2])} {'...' if len(selected_cities)>2 else ''}")

if not selected_cities:
    st.warning("Please select at least one city from the sidebar.")
else:
    tabs = st.tabs(["📈 Trends", "📊 Distributions", "🌡️ Heatmaps", "📅 Calendar", "🔮 Forecast", "🏥 Health"])
    
    # Filter data for selected cities
    city_df = df_filtered[df_filtered["city"].isin(selected_cities)]
    
    # 1. Trends
    with tabs[0]:
        if not city_df.empty:
            fig_trend = px.line(city_df, x="date", y="index", color="city", markers=True,
                                title=f"AQI Trend ({selected_year})", labels={"index": "AQI", "date": "Date"})
            st.plotly_chart(apply_chart_theme(fig_trend), use_container_width=True)
        else:
            st.info("No data for trends.")

    # 2. Distributions
    with tabs[1]:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig_hist = px.histogram(city_df, x="index", color="city", barmode="overlay", nbins=30,
                                    title="AQI Distribution Histogram")
            st.plotly_chart(apply_chart_theme(fig_hist), use_container_width=True)
        with col_d2:
            fig_box = px.box(city_df, x="city", y="index", color="city", title="AQI Spread (Boxplot)")
            st.plotly_chart(apply_chart_theme(fig_box), use_container_width=True)

    # 3. Heatmaps (Month vs Day)
    with tabs[2]:
        if len(selected_cities) == 1:
            city_single = city_df.copy()
            city_single["day"] = city_single["date"].dt.day
            city_single["month"] = city_single["date"].dt.month_name()
            # Sort months
            month_order = ["January", "February", "March", "April", "May", "June", 
                           "July", "August", "September", "October", "November", "December"]
            
            pivot = city_single.pivot_table(index="month", columns="day", values="index")
            pivot = pivot.reindex(month_order).dropna(how='all')
            
            fig_heat = px.imshow(pivot, color_continuous_scale="Turbo", title=f"Daily Intensity Heatmap - {selected_cities[0]}")
            st.plotly_chart(apply_chart_theme(fig_heat), use_container_width=True)
        else:
            st.info("Select exactly one city to view the detailed density heatmap.")

    # 4. Calendar View
    with tabs[3]:
        st.caption("Visualizing AQI severity across the weeks of the year.")
        if len(selected_cities) == 1:
            cal_df = city_df.copy()
            cal_df['week'] = cal_df['date'].dt.isocalendar().week
            cal_df['weekday'] = cal_df['date'].dt.dayofweek
            
            fig_cal = go.Figure(data=go.Heatmap(
                z=cal_df['index'],
                x=cal_df['week'],
                y=cal_df['weekday'],
                colorscale='RdYlGn_r',
                hoverongaps=False
            ))
            fig_cal.update_layout(
                title=f"Calendar Heatmap - {selected_cities[0]}",
                yaxis=dict(tickmode='array', tickvals=[0,1,2,3,4,5,6], ticktext=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']),
                xaxis_title="Week of Year"
            )
            st.plotly_chart(apply_chart_theme(fig_cal), use_container_width=True)
        else:
            st.info("Select exactly one city to view the calendar.")

    # 5. Forecast
    with tabs[4]:
        st.markdown("#### Short-term AQI Prediction (Linear Regression)")
        if len(selected_cities) == 1:
            if len(city_df) > 15:
                # Prepare data
                fc_data = city_df.sort_values("date")[["date", "index"]].dropna()
                fc_data["days"] = (fc_data["date"] - fc_data["date"].min()).dt.days
                
                X = fc_data[["days"]]
                y = fc_data["index"]
                
                # Model
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                model = LinearRegression().fit(X_poly, y)
                
                # Predict next 15 days
                last_day = fc_data["days"].max()
                future_days = np.arange(last_day, last_day + 15).reshape(-1, 1)
                future_dates = [fc_data["date"].min() + pd.Timedelta(days=int(d)) for d in future_days.flatten()]
                future_aqi = model.predict(poly.transform(future_days))
                
                # Plot
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=fc_data["date"], y=y, mode='lines', name='Historical', line=dict(color=THEME['sub_text'])))
                fig_fc.add_trace(go.Scatter(x=future_dates, y=future_aqi, mode='lines+markers', name='Forecast', line=dict(color=THEME['accent'], dash='dash')))
                
                fig_fc.update_layout(title=f"15-Day AQI Forecast for {selected_cities[0]}")
                st.plotly_chart(apply_chart_theme(fig_fc), use_container_width=True)
            else:
                st.warning("Not enough data points to generate a forecast.")
        else:
            st.info("Select exactly one city to generate a forecast.")

    # 6. Health & Pollutants
    with tabs[5]:
        col_h1, col_h2 = st.columns([1, 2])
        
        with col_h1:
            st.markdown("#### Dominant Pollutant")
            if "pollutant" in city_df.columns:
                poll_counts = city_df["pollutant"].value_counts().reset_index()
                poll_counts.columns = ["Pollutant", "Days"]
                fig_pie = px.pie(poll_counts, values="Days", names="Pollutant", color="Pollutant", 
                                 color_discrete_map=POLLUTANT_COLORS, hole=0.4)
                fig_pie.update_layout(showlegend=False, margin=dict(t=0,b=0,l=0,r=0), height=300)
                st.plotly_chart(apply_chart_theme(fig_pie), use_container_width=True)
        
        with col_h2:
            st.markdown("#### Health Recommendations")
            if not city_df.empty:
                latest = city_df.sort_values("date").iloc[-1]
                aqi_curr = latest["index"]
                cat_curr = get_aqi_category(aqi_curr)
                
                rec_map = {
                    "Good": "Ideal for outdoor activities. Enjoy the fresh air!",
                    "Satisfactory": "Good day for a walk, but sensitive groups should monitor.",
                    "Moderate": "Sensitive individuals should limit prolonged outdoor exertion.",
                    "Poor": "Wear a mask if outside. Reduce heavy exertion.",
                    "Very Poor": "Avoid outdoor activities. Keep windows closed. Use air purifiers.",
                    "Severe": "Health Alert: Serious risk. Stay indoors."
                }
                
                st.markdown(f"""
                <div style="background:{THEME['card_bg']}; padding:20px; border-radius:12px; border-left: 5px solid {AQI_COLORS.get(cat_curr, '#fff')};">
                    <h3 style="margin:0; color:{AQI_COLORS.get(cat_curr, '#fff')}">{cat_curr} (AQI: {aqi_curr:.0f})</h3>
                    <p style="margin-top:10px; font-size:1.1rem;">{rec_map.get(cat_curr, "No data")}</p>
                    <small>Based on data from {latest['date'].strftime('%Y-%m-%d')}</small>
                </div>
                """, unsafe_allow_html=True)

# --- GEOSPATIAL VIEW ---
st.markdown("---")
st.markdown("### 🗺️ Geospatial Hotspots")

# Load Lat/Long
coords_file = "lat_long.txt"
if os.path.exists(coords_file):
    try:
        # Safe execution of the dictionary string
        with open(coords_file, "r") as f:
            content = f.read()
            local_scope = {}
            exec(content, {}, local_scope)
            city_coords = local_scope.get("city_coords", {})
            
        # Prepare Map Data
        map_df = df_filtered.groupby("city")["index"].mean().reset_index()
        map_df["lat"] = map_df["city"].map(lambda x: city_coords.get(x, [None, None])[0])
        map_df["lon"] = map_df["city"].map(lambda x: city_coords.get(x, [None, None])[1])
        map_df = map_df.dropna()
        map_df["category"] = map_df["index"].apply(get_aqi_category)
        
        if not map_df.empty:
            fig_map = px.scatter_mapbox(
                map_df, lat="lat", lon="lon", size="index", color="category",
                color_discrete_map=AQI_COLORS, hover_name="city",
                size_max=30, zoom=3.5, center={"lat": 22, "lon": 82},
                mapbox_style="carto-darkmatter"
            )
            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500, paper_bgcolor="#000000")
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("Not enough coordinate data to display map.")
            
    except Exception as e:
        st.error(f"Error loading map coordinates: {e}")
else:
    st.info("Map coordinates file not found.")

# --- FOOTER ---
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 20px; color: {THEME['sub_text']};">
    <p>© {datetime.now().year} IIT Kharagpur | Air Quality Research Group</p>
    <p style="font-size: 0.8rem;">Data sourced from Central Pollution Control Board (CPCB)</p>
</div>
""", unsafe_allow_html=True)

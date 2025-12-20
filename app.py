import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
from io import StringIO
import requests
import json
from datetime import datetime, timedelta
import base64
from PIL import Image
import time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# --- Global Theme & Style Setup ---
pio.templates.default = "plotly_dark"

# Color palette for dark theme
ACCENT_COLOR = "#00BCD4"                 # Vibrant Teal
TEXT_COLOR_DARK_THEME = "#EAEAEA"        # Light gray for text
SUBTLE_TEXT_COLOR_DARK_THEME = "#B0B0B0" # Subtle gray
BACKGROUND_COLOR = "#121212"             # Very dark gray (almost black)
CARD_BACKGROUND_COLOR = "#1E1E1E"        # Slightly lighter for cards
BORDER_COLOR = "#333333"                 # Dark border
HIGHLIGHT_COLOR = "#FF6B6B"              # For alerts and highlights

# Color palette for light theme
TEXT_COLOR_LIGHT_THEME = "#212121"       # Dark gray for text
SUBTLE_TEXT_COLOR_LIGHT_THEME = "#757575" # Subtle gray
BACKGROUND_COLOR_LIGHT = "#FFFFFF"       # White
CARD_BACKGROUND_COLOR_LIGHT = "#F5F5F5"  # Light gray for cards
BORDER_COLOR_LIGHT = "#E0E0E0"           # Light border

# AQI Category Colors
CATEGORY_COLORS_DARK = {
    "Severe": "#F44336",      # Vivid Red
    "Very Poor": "#FF7043",   # Vivid Orange-Red
    "Poor": "#FFA726",        # Vivid Orange
    "Moderate": "#FFEE58",    # Vivid Yellow
    "Satisfactory": "#9CCC65",# Vivid Light Green
    "Good": "#4CAF50",        # Vivid Green
    "Unknown": "#444444"      # Dark gray for unknown days
}

CATEGORY_COLORS_LIGHT = {
    "Severe": "#D32F2F",      # Darker Red
    "Very Poor": "#F57C00",   # Darker Orange
    "Poor": "#FFA000",        # Darker Orange
    "Moderate": "#FBC02D",    # Darker Yellow
    "Satisfactory": "#689F38",# Darker Green
    "Good": "#388E3C",        # Darker Green
    "Unknown": "#757575"      # Gray for unknown days
}

# Pollutant Colors
POLLUTANT_COLORS_DARK = {
    "PM2.5": "#FF6B6B", "PM10": "#4ECDC4", "NO2": "#45B7D1",
    "SO2": "#F9C74F", "CO": "#F8961E", "O3": "#90BE6D", "Other": "#B0BEC5"
}

POLLUTANT_COLORS_LIGHT = {
    "PM2.5": "#E53935", "PM10": "#00897B", "NO2": "#039BE5",
    "SO2": "#FDD835", "CO": "#FB8C00", "O3": "#7CB342", "Other": "#90A4AE"
}

# Health recommendations based on AQI levels
HEALTH_RECOMMENDATIONS = {
    "Good": "Perfect day for outdoor activities!",
    "Satisfactory": "Sensitive individuals should consider reducing prolonged/heavy exertion.",
    "Moderate": "Sensitive groups should reduce outdoor activities.",
    "Poor": "Everyone should reduce prolonged/heavy exertion.",
    "Very Poor": "Avoid outdoor activities, especially for sensitive groups.",
    "Severe": "Avoid all outdoor activities, keep windows closed.",
    "Unknown": "Air quality data unavailable - take precautions."
}

# Extended health recommendations
EXTENDED_HEALTH_RECOMMENDATIONS = {
    "Good": {
        "General": "Air quality is satisfactory and poses little or no risk.",
        "Sensitive": "Enjoy your outdoor activities.",
        "Outdoor": "Great day for outdoor sports and activities.",
        "Indoor": "No special precautions needed.",
        "Children": "Safe for children to play outside.",
        "Elderly": "Safe for all outdoor activities."
    },
    "Satisfactory": {
        "General": "Air quality is acceptable for most people.",
        "Sensitive": "Unusually sensitive people should consider limiting prolonged outdoor exertion.",
        "Outdoor": "Outdoor activities are generally safe.",
        "Indoor": "No special precautions needed.",
        "Children": "Children can play outside, but monitor for symptoms.",
        "Elderly": "Generally safe, but monitor for any respiratory issues."
    },
    "Moderate": {
        "General": "Members of sensitive groups may experience health effects.",
        "Sensitive": "Sensitive groups should reduce prolonged outdoor exertion.",
        "Outdoor": "Limit prolonged outdoor exertion.",
        "Indoor": "Consider keeping windows closed during peak pollution hours.",
        "Children": "Children should limit prolonged outdoor exertion.",
        "Elderly": "Elderly people should limit prolonged outdoor exertion."
    },
    "Poor": {
        "General": "Everyone may begin to experience health effects.",
        "Sensitive": "Sensitive groups should avoid prolonged outdoor exertion.",
        "Outdoor": "Avoid prolonged outdoor exertion.",
        "Indoor": "Keep windows closed and use air purifiers if available.",
        "Children": "Children should avoid prolonged outdoor exertion.",
        "Elderly": "Elderly people should avoid prolonged outdoor exertion."
    },
    "Very Poor": {
        "General": "Health warnings of emergency conditions.",
        "Sensitive": "Sensitive groups should remain indoors and avoid exertion.",
        "Outdoor": "Avoid all outdoor activities.",
        "Indoor": "Keep all windows closed and use air purifiers.",
        "Children": "Children should remain indoors.",
        "Elderly": "Elderly people should remain indoors."
    },
    "Severe": {
        "General": "Emergency conditions: everyone is likely to be affected.",
        "Sensitive": "Sensitive groups should remain indoors and avoid any exertion.",
        "Outdoor": "Avoid all outdoor activities.",
        "Indoor": "Keep all windows closed, use air purifiers, and avoid indoor pollution sources.",
        "Children": "Children should remain indoors at all times.",
        "Elderly": "Elderly people should remain indoors at all times."
    },
    "Unknown": {
        "General": "Air quality data unavailable.",
        "Sensitive": "Take precautions as if air quality is moderate.",
        "Outdoor": "Limit prolonged outdoor exertion.",
        "Indoor": "Keep windows closed during peak pollution hours.",
        "Children": "Children should limit prolonged outdoor exertion.",
        "Elderly": "Elderly people should limit prolonged outdoor exertion."
    }
}

# ------------------- Page Config -------------------
st.set_page_config(
    layout="wide",
    page_title="IIT KGP AQI Dashboard",
    page_icon="🌬️",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'theme' not in st.session_state:
    st.session_state.theme = "dark"
if 'selected_cities' not in st.session_state:
    st.session_state.selected_cities = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# ------------------- Helper Functions -------------------
def get_theme_colors():
    """Return color scheme based on current theme"""
    if st.session_state.theme == "dark":
        return {
            "text": TEXT_COLOR_DARK_THEME,
            "subtle_text": SUBTLE_TEXT_COLOR_DARK_THEME,
            "background": BACKGROUND_COLOR,
            "card_background": CARD_BACKGROUND_COLOR,
            "border": BORDER_COLOR,
            "accent": ACCENT_COLOR,
            "highlight": HIGHLIGHT_COLOR,
            "category_colors": CATEGORY_COLORS_DARK,
            "pollutant_colors": POLLUTANT_COLORS_DARK
        }
    else:
        return {
            "text": TEXT_COLOR_LIGHT_THEME,
            "subtle_text": SUBTLE_TEXT_COLOR_LIGHT_THEME,
            "background": BACKGROUND_COLOR_LIGHT,
            "card_background": CARD_BACKGROUND_COLOR_LIGHT,
            "border": BORDER_COLOR_LIGHT,
            "accent": ACCENT_COLOR,
            "highlight": HIGHLIGHT_COLOR,
            "category_colors": CATEGORY_COLORS_LIGHT,
            "pollutant_colors": POLLUTANT_COLORS_LIGHT
        }

def get_custom_plotly_layout_args(height: int = None, title_text: str = None) -> dict:
    """
    Returns a dict of common Plotly layout arguments based on current theme
    """
    colors = get_theme_colors()
    
    layout_args = {
        "font": {"family": "Inter", "color": colors["text"], "size": 14},
        "paper_bgcolor": colors["card_background"],
        "plot_bgcolor": colors["card_background"],
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 12}
        },
        "margin": {"l": 40, "r": 20, "t": 60, "b": 40},
        "hoverlabel": {
            "bgcolor": colors["card_background"],
            "font_size": 12,
            "font_family": "Inter",
            "bordercolor": colors["border"]
        }
    }
    if height:
        layout_args["height"] = height
    if title_text:
        layout_args["title_text"] = title_text
        layout_args["title_font"] = {"color": colors["accent"], "size": 18, "family": "Inter"}
        layout_args["title_x"] = 0.03
        layout_args["title_y"] = 0.95
    return layout_args

def format_number(num):
    if num > 1000000:
        return f"{num/1000000:.1f}M"
    if num > 1000:
        return f"{num/1000:.1f}K"
    return str(num)

def get_category(aqi_val):
    """Map AQI value to category"""
    if pd.isna(aqi_val):
        return "Unknown"
    if aqi_val <= 50:
        return "Good"
    elif aqi_val <= 100:
        return "Satisfactory"
    elif aqi_val <= 200:
        return "Moderate"
    elif aqi_val <= 300:
        return "Poor"
    elif aqi_val <= 400:
        return "Very Poor"
    else:
        return "Severe"

def get_aqi_color(aqi_val):
    """Return color based on AQI value"""
    category = get_category(aqi_val)
    colors = get_theme_colors()
    return colors["category_colors"].get(category, colors["category_colors"]["Unknown"])

def get_pollutant_color(pollutant):
    """Return color based on pollutant"""
    colors = get_theme_colors()
    return colors["pollutant_colors"].get(pollutant, colors["pollutant_colors"]["Other"])

def detect_anomalies(data, column='index', contamination=0.05):
    """Detect anomalies using Isolation Forest"""
    try:
        model = IsolationForest(contamination=contamination, random_state=42)
        data_clean = data[[column]].dropna()
        if len(data_clean) < 10:  # Not enough data for anomaly detection
            return data, []
        
        anomalies = model.fit_predict(data_clean)
        data_clean['is_anomaly'] = anomalies == -1
        
        # Merge back with original data
        result = data.copy()
        result['is_anomaly'] = False
        result.loc[data_clean.index, 'is_anomaly'] = data_clean['is_anomaly']
        
        anomaly_dates = result[result['is_anomaly']]['date'].tolist()
        return result, anomaly_dates
    except Exception as e:
        st.error(f"Error in anomaly detection: {e}")
        return data, []

def generate_forecast(data, periods=7):
    """Generate AQI forecast using polynomial regression"""
    try:
        if len(data) < 10:  # Not enough data for forecasting
            return None, None, None
        
        forecast_df = data.sort_values("date")[["date", "index"]].dropna()
        forecast_df["days_since_start"] = (forecast_df["date"] - forecast_df["date"].min()).dt.days
        
        X = forecast_df["days_since_start"].values.reshape(-1, 1)
        y = forecast_df["index"].values
        
        # Try polynomial degrees from 1 to 3 and select the best
        best_degree = 1
        best_mae = float('inf')
        best_model = None
        best_poly = None
        
        for degree in range(1, 4):
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X)
            model = LinearRegression().fit(X_poly, y)
            
            # Calculate MAE on training data
            y_pred = model.predict(X_poly)
            mae = mean_absolute_error(y, y_pred)
            
            if mae < best_mae:
                best_mae = mae
                best_degree = degree
                best_model = model
                best_poly = poly
        
        # Generate forecast
        last_day_num = forecast_df["days_since_start"].max()
        future_X_range = np.arange(last_day_num + 1, last_day_num + periods + 1)
        future_X_poly = best_poly.transform(future_X_range.reshape(-1, 1))
        future_y_pred = best_model.predict(future_X_poly)
        
        # Calculate confidence intervals (simplified approach)
        residuals = y - best_model.predict(best_poly.transform(X))
        std_error = np.std(residuals)
        upper_bound = future_y_pred + 1.96 * std_error
        lower_bound = future_y_pred - 1.96 * std_error
        
        # Ensure forecast values are non-negative
        future_y_pred = np.maximum(0, future_y_pred)
        lower_bound = np.maximum(0, lower_bound)
        
        # Generate future dates
        min_date_forecast = forecast_df["date"].min()
        future_dates = [min_date_forecast + pd.Timedelta(days=int(i)) for i in future_X_range]
        
        forecast_data = pd.DataFrame({
            "date": future_dates,
            "forecast": future_y_pred,
            "upper_bound": upper_bound,
            "lower_bound": lower_bound
        })
        
        return forecast_data, best_degree, best_mae
    except Exception as e:
        st.error(f"Error in forecast generation: {e}")
        return None, None, None

def calculate_pollution_trend(data, window=30):
    """Calculate pollution trend direction and strength"""
    if len(data) < window:
        return "Insufficient data", 0
    
    # Calculate rolling average
    data_sorted = data.sort_values("date")
    data_sorted["rolling_avg"] = data_sorted["index"].rolling(window=window, min_periods=1).mean()
    
    # Get first and last values of the rolling average
    first_value = data_sorted["rolling_avg"].iloc[window-1]
    last_value = data_sorted["rolling_avg"].iloc[-1]
    
    # Calculate percentage change
    if first_value > 0:
        percent_change = ((last_value - first_value) / first_value) * 100
    else:
        percent_change = 0
    
    # Determine trend direction
    if percent_change > 5:
        trend = "Improving"
    elif percent_change < -5:
        trend = "Deteriorating"
    else:
        trend = "Stable"
    
    return trend, percent_change

def create_comparison_chart(data, cities, metric='index'):
    """Create a comparison chart between selected cities"""
    comparison_data = data[data['city'].isin(cities)].copy()
    
    # Group by city and calculate average for the metric
    city_avg = comparison_data.groupby('city')[metric].mean().reset_index()
    city_avg = city_avg.sort_values(metric, ascending=False)
    
    # Create a horizontal bar chart
    colors = get_theme_colors()
    fig = px.bar(
        city_avg,
        x=metric,
        y='city',
        orientation='h',
        color=metric,
        color_continuous_scale='RdYlGn_r',
        labels={metric: 'Average AQI', 'city': 'City'},
        height=max(300, len(cities) * 50)
    )
    
    fig.update_layout(
        title=f"Average AQI Comparison",
        xaxis_title="Average AQI",
        yaxis_title=None,
        coloraxis_showscale=False,
        **get_custom_plotly_layout_args()
    )
    
    return fig

def create_correlation_heatmap(data):
    """Create a correlation heatmap between pollutants"""
    # Filter for common pollutants
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    colors = get_theme_colors()
    
    # Create a pivot table with pollutants as columns
    pivot_data = data.pivot_table(
        index='date',
        columns='pollutant',
        values='index',
        aggfunc='mean'
    )
    
    # Filter for available pollutants
    available_pollutants = [p for p in pollutants if p in pivot_data.columns]
    if len(available_pollutants) < 2:
        return None
    
    pivot_data = pivot_data[available_pollutants].dropna()
    
    # Calculate correlation matrix
    corr_matrix = pivot_data.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu_r',
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        height=500
    )
    
    fig.update_layout(
        title="Pollutant Correlation Matrix",
        **get_custom_plotly_layout_args()
    )
    
    return fig

def create_pollution_sources_chart(data):
    """Create a chart showing pollution sources breakdown"""
    # This is a placeholder function as actual pollution source data might not be available
    # We'll create a simulated breakdown based on pollutant types
    
    colors = get_theme_colors()
    
    # Count occurrences of each pollutant
    pollutant_counts = data['pollutant'].value_counts()
    
    # Create a pie chart
    fig = px.pie(
        values=pollutant_counts.values,
        names=pollutant_counts.index,
        color_discrete_map=colors["pollutant_colors"],
        hole=0.4,
        height=500
    )
    
    fig.update_layout(
        title="Pollution Sources Breakdown",
        **get_custom_plotly_layout_args()
    )
    
    return fig

def create_weather_correlation_chart(data):
    """Create a chart showing correlation between AQI and weather parameters"""
    # This is a placeholder function as actual weather data might not be available
    # We'll create a simulated correlation
    
    colors = get_theme_colors()
    
    # Create simulated data for demonstration
    dates = pd.date_range(start=data['date'].min(), end=data['date'].max(), freq='D')
    np.random.seed(42)
    
    # Simulate temperature data (higher in summer, lower in winter)
    temp = 20 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 2, len(dates))
    
    # Simulate humidity data (inverse correlation with temperature)
    humidity = 70 - 0.5 * temp + np.random.normal(0, 5, len(dates))
    
    # Simulate wind speed data
    wind_speed = 5 + np.random.normal(0, 2, len(dates))
    
    # Create a DataFrame with weather data
    weather_df = pd.DataFrame({
        'date': dates,
        'temperature': temp,
        'humidity': humidity,
        'wind_speed': wind_speed
    })
    
    # Merge with AQI data
    merged_df = pd.merge(data, weather_df, on='date', how='inner')
    
    # Create scatter plots
    fig = go.Figure()
    
    # Temperature vs AQI
    fig.add_trace(go.Scatter(
        x=merged_df['temperature'],
        y=merged_df['index'],
        mode='markers',
        name='Temperature',
        marker=dict(color=colors["accent"], size=5),
        text=merged_df['date'].dt.strftime('%Y-%m-%d'),
        hovertemplate='Date: %{text}<br>Temperature: %{x:.1f}°C<br>AQI: %{y}<extra></extra>'
    ))
    
    # Humidity vs AQI
    fig.add_trace(go.Scatter(
        x=merged_df['humidity'],
        y=merged_df['index'],
        mode='markers',
        name='Humidity',
        marker=dict(color=colors["highlight"], size=5),
        text=merged_df['date'].dt.strftime('%Y-%m-%d'),
        hovertemplate='Date: %{text}<br>Humidity: %{x:.1f}%<br>AQI: %{y}<extra></extra>',
        visible='legendonly'
    ))
    
    # Wind Speed vs AQI
    fig.add_trace(go.Scatter(
        x=merged_df['wind_speed'],
        y=merged_df['index'],
        mode='markers',
        name='Wind Speed',
        marker=dict(color=colors["category_colors"]["Good"], size=5),
        text=merged_df['date'].dt.strftime('%Y-%m-%d'),
        hovertemplate='Date: %{text}<br>Wind Speed: %{x:.1f} m/s<br>AQI: %{y}<extra></extra>',
        visible='legendonly'
    ))
    
    fig.update_layout(
        title="AQI vs Weather Parameters",
        xaxis_title="Value",
        yaxis_title="AQI",
        height=500,
        **get_custom_plotly_layout_args()
    )
    
    return fig

def create_diurnal_pattern_chart(data):
    """Create a chart showing diurnal (hourly) AQI patterns"""
    # This is a placeholder function as hourly data might not be available
    # We'll create a simulated diurnal pattern
    
    colors = get_theme_colors()
    
    # Create simulated hourly data for demonstration
    hours = list(range(24))
    np.random.seed(42)
    
    # Simulate AQI pattern (higher in morning and evening rush hours)
    base_aqi = data['index'].mean()
    hourly_aqi = base_aqi + 30 * np.sin(np.array(hours) * np.pi / 12 - np.pi/2) + np.random.normal(0, 10, 24)
    hourly_aqi = np.maximum(0, hourly_aqi)  # Ensure non-negative
    
    # Create a DataFrame with hourly data
    hourly_df = pd.DataFrame({
        'hour': hours,
        'aqi': hourly_aqi,
        'category': [get_category(aqi) for aqi in hourly_aqi]
    })
    
    # Create a line chart
    fig = px.line(
        hourly_df,
        x='hour',
        y='aqi',
        color='category',
        color_discrete_map=colors["category_colors"],
        markers=True,
        height=500,
        labels={'hour': 'Hour of Day', 'aqi': 'Average AQI', 'category': 'AQI Category'}
    )
    
    fig.update_layout(
        title="Diurnal AQI Pattern",
        xaxis_title="Hour of Day",
        yaxis_title="Average AQI",
        **get_custom_plotly_layout_args()
    )
    
    # Add annotations for rush hours
    fig.add_annotation(
        x=8, y=hourly_df.loc[hourly_df['hour'] == 8, 'aqi'].values[0],
        text="Morning Rush",
        showarrow=True,
        arrowhead=1,
        arrowcolor=colors["accent"],
        ax=20,
        ay=-30
    )
    
    fig.add_annotation(
        x=18, y=hourly_df.loc[hourly_df['hour'] == 18, 'aqi'].values[0],
        text="Evening Rush",
        showarrow=True,
        arrowhead=1,
        arrowcolor=colors["accent"],
        ax=-20,
        ay=-30
    )
    
    return fig

def create_health_impact_scorecard(data):
    """Create a health impact scorecard"""
    colors = get_theme_colors()
    
    # Calculate health impact metrics
    total_days = len(data)
    good_days = len(data[data['level'] == 'Good'])
    satisfactory_days = len(data[data['level'] == 'Satisfactory'])
    moderate_days = len(data[data['level'] == 'Moderate'])
    poor_days = len(data[data['level'] == 'Poor'])
    very_poor_days = len(data[data['level'] == 'Very Poor'])
    severe_days = len(data[data['level'] == 'Severe'])
    
    # Calculate health impact score (0-100, higher is worse)
    health_impact_score = (
        (moderate_days * 10) +
        (poor_days * 25) +
        (very_poor_days * 50) +
        (severe_days * 75)
    ) / total_days if total_days > 0 else 0
    
    # Determine health impact level
    if health_impact_score < 10:
        impact_level = "Low"
        impact_color = colors["category_colors"]["Good"]
    elif health_impact_score < 25:
        impact_level = "Moderate"
        impact_color = colors["category_colors"]["Moderate"]
    elif health_impact_score < 50:
        impact_level = "High"
        impact_color = colors["category_colors"]["Poor"]
    else:
        impact_level = "Very High"
        impact_color = colors["category_colors"]["Severe"]
    
    # Create a gauge chart for health impact
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_impact_score,
        title={'text': f"Health Impact Score: {impact_level}"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': impact_color},
            'steps': [
                {'range': [0, 10], 'color': colors["category_colors"]["Good"]},
                {'range': [10, 25], 'color': colors["category_colors"]["Satisfactory"]},
                {'range': [25, 50], 'color': colors["category_colors"]["Moderate"]},
                {'range': [50, 75], 'color': colors["category_colors"]["Poor"]},
                {'range': [75, 100], 'color': colors["category_colors"]["Severe"]}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        **get_custom_plotly_layout_args()
    )
    
    return fig, health_impact_score, impact_level

def create_pollution_calendar(data, year):
    """Create an enhanced calendar heatmap"""
    colors = get_theme_colors()
    
    # Create a complete date range for the year
    start_date = pd.to_datetime(f"{year}-01-01")
    end_date = pd.to_datetime(f"{year}-12-31")
    full_year_dates = pd.date_range(start_date, end_date, freq="D")
    
    # Create a DataFrame with all dates
    calendar_df = pd.DataFrame({"date": full_year_dates})
    calendar_df["week"] = calendar_df["date"].dt.isocalendar().week
    calendar_df["day_of_week"] = calendar_df["date"].dt.dayofweek
    calendar_df["month"] = calendar_df["date"].dt.month
    
    # Adjust week numbers for display
    calendar_df.loc[(calendar_df["date"].dt.month == 1) & (calendar_df["week"] > 50), "week"] = 0
    calendar_df.loc[(calendar_df["date"].dt.month == 12) & (calendar_df["week"] == 1), "week"] = calendar_df["week"].max() + 1
    
    # Merge with AQI data
    merged_cal_df = pd.merge(
        calendar_df,
        data[["date", "index", "level"]],
        on="date",
        how="left"
    )
    
    # Fill missing values
    merged_cal_df["level"] = merged_cal_df["level"].fillna("Unknown")
    merged_cal_df["aqi_text"] = merged_cal_df["index"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    
    # Map AQI categories to numeric values for the heatmap
    level_to_code = {level: idx for idx, level in enumerate(colors["category_colors"].keys())}
    z_values = merged_cal_df["level"].map(level_to_code)
    
    # Day labels
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    # Create month labels
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            x=merged_cal_df["week"],
            y=merged_cal_df["day_of_week"],
            z=z_values,
            customdata=pd.DataFrame({
                "date": merged_cal_df["date"].dt.strftime("%Y-%m-%d"),
                "level": merged_cal_df["level"],
                "aqi": merged_cal_df["aqi_text"]
            }),
            hovertemplate="<b>%{customdata[0]}</b><br>AQI: %{customdata[2]} (%{customdata[1]})<extra></extra>",
            colorscale=[[i / (len(colors["category_colors"]) - 1), color] for i, color in enumerate(colors["category_colors"].values())],
            showscale=False,
            xgap=3, ygap=3
        )
    )
    
    # Add month annotations
    month_starts = merged_cal_df.groupby("month")["week"].min().reset_index()
    for _, row in month_starts.iterrows():
        month_idx = int(row["month"]) - 1
        if 0 <= month_idx < len(month_labels):
            fig.add_annotation(
                text=month_labels[month_idx],
                x=row["week"],
                y=7,
                showarrow=False,
                font=dict(color=colors["subtle_text"], size=12)
            )
    
    # Add legend annotations
    for i, (level, color) in enumerate(colors["category_colors"].items()):
        fig.add_annotation(
            text=f"█ <span style='color:{colors["text"]};'>{level}</span>",
            x=0.05 + 0.12 * (i % 7),
            y=-0.15 - 0.1 * (i // 7),
            showarrow=False,
            font=dict(color=color, size=12),
            xref="paper",
            yref="paper"
        )
    
    fig.update_layout(
        yaxis=dict(tickmode="array", tickvals=list(range(7)), ticktext=day_labels, showgrid=False, zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False, tickmode="array", ticktext=[], tickvals=[]),
        height=350,
        margin=dict(t=50, b=100, l=40, r=40),
        **get_custom_plotly_layout_args()
    )
    
    return fig

def create_aqi_forecast_chart(data, forecast_data):
    """Create an enhanced AQI forecast chart"""
    colors = get_theme_colors()
    
    # Sort data by date
    data_sorted = data.sort_values("date")
    
    # Create the figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=data_sorted["date"],
        y=data_sorted["index"],
        mode="lines+markers",
        name="Historical AQI",
        line=dict(color=colors["subtle_text"], width=2),
        marker=dict(size=4),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>AQI: %{y}<extra></extra>"
    ))
    
    # Add forecast data
    if forecast_data is not None:
        fig.add_trace(go.Scatter(
            x=forecast_data["date"],
            y=forecast_data["forecast"],
            mode="lines+markers",
            name="Forecast AQI",
            line=dict(color=colors["accent"], width=2, dash="dash"),
            marker=dict(size=5),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Forecast AQI: %{y}<extra></extra>"
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_data["date"],
            y=forecast_data["upper_bound"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_data["date"],
            y=forecast_data["lower_bound"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=f"rgba({int(colors['accent'][1:3], 16)}, {int(colors['accent'][3:5], 16)}, {int(colors['accent'][5:7], 16)}, 0.2)",
            name="Confidence Interval",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Upper Bound: %{y}<extra></extra>"
        ))
    
    # Add vertical line to separate historical data from forecast
    if forecast_data is not None and len(data_sorted) > 0:
        fig.add_vline(
            x=data_sorted["date"].max(),
            line_dash="dash",
            line_color=colors["accent"],
            annotation_text="Forecast Start",
            annotation_position="top"
        )
    
    fig.update_layout(
        title="AQI Forecast",
        xaxis_title="Date",
        yaxis_title="AQI",
        height=500,
        hovermode="x unified",
        **get_custom_plotly_layout_args()
    )
    
    return fig

def create_enhanced_distribution_chart(data):
    """Create an enhanced AQI distribution chart"""
    colors = get_theme_colors()
    
    # Count occurrences of each AQI category
    category_counts = data["level"].value_counts().reindex(colors["category_colors"].keys(), fill_value=0)
    
    # Create a DataFrame for plotting
    category_df = pd.DataFrame({
        "Category": category_counts.index,
        "Count": category_counts.values
    })
    
    # Create a bar chart with enhanced styling
    fig = px.bar(
        category_df,
        x="Category",
        y="Count",
        color="Category",
        color_discrete_map=colors["category_colors"],
        height=500
    )
    
    # Add percentage annotations
    total_days = len(data)
    for i, count in enumerate(category_counts.values):
        if count > 0:
            percentage = (count / total_days) * 100
            fig.add_annotation(
                x=i,
                y=count + max(category_counts) * 0.02,
                text=f"{percentage:.1f}%",
                showarrow=False,
                font=dict(color=colors["text"], size=12)
            )
    
    fig.update_layout(
        title="AQI Distribution",
        xaxis_title="AQI Category",
        yaxis_title="Number of Days",
        showlegend=False,
        **get_custom_plotly_layout_args()
    )
    
    return fig

def create_monthly_comparison_chart(data, cities):
    """Create a monthly comparison chart between cities"""
    colors = get_theme_colors()
    
    # Filter data for selected cities
    city_data = data[data['city'].isin(cities)].copy()
    
    # Extract month and year
    city_data['month'] = city_data['date'].dt.month
    city_data['month_name'] = city_data['date'].dt.month_name()
    
    # Group by city and month
    monthly_avg = city_data.groupby(['city', 'month', 'month_name'])['index'].mean().reset_index()
    
    # Create a line chart
    fig = px.line(
        monthly_avg,
        x='month_name',
        y='index',
        color='city',
        markers=True,
        height=500,
        labels={'index': 'Average AQI', 'month_name': 'Month'}
    )
    
    fig.update_layout(
        title="Monthly AQI Comparison",
        xaxis_title="Month",
        yaxis_title="Average AQI",
        **get_custom_plotly_layout_args()
    )
    
    return fig

def create_pollutant_breakdown_chart(data):
    """Create a pollutant breakdown chart"""
    colors = get_theme_colors()
    
    # Count occurrences of each pollutant
    pollutant_counts = data['pollutant'].value_counts()
    
    # Create a DataFrame for plotting
    pollutant_df = pd.DataFrame({
        "Pollutant": pollutant_counts.index,
        "Count": pollutant_counts.values
    })
    
    # Create a bar chart with enhanced styling
    fig = px.bar(
        pollutant_df,
        x="Pollutant",
        y="Count",
        color="Pollutant",
        color_discrete_map=colors["pollutant_colors"],
        height=500
    )
    
    # Add percentage annotations
    total_days = len(data)
    for i, count in enumerate(pollutant_counts.values):
        if count > 0:
            percentage = (count / total_days) * 100
            fig.add_annotation(
                x=i,
                y=count + max(pollutant_counts) * 0.02,
                text=f"{percentage:.1f}%",
                showarrow=False,
                font=dict(color=colors["text"], size=12)
            )
    
    fig.update_layout(
        title="Pollutant Distribution",
        xaxis_title="Pollutant",
        yaxis_title="Number of Days",
        showlegend=False,
        **get_custom_plotly_layout_args()
    )
    
    return fig

def create_weekday_pattern_chart(data):
    """Create a weekday pattern chart"""
    colors = get_theme_colors()
    
    # Extract day of week
    data_copy = data.copy()
    data_copy['day_of_week'] = data_copy['date'].dt.day_name()
    
    # Define order of days
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    data_copy['day_of_week'] = pd.Categorical(data_copy['day_of_week'], categories=weekday_order, ordered=True)
    
    # Group by day of week
    weekday_avg = data_copy.groupby('day_of_week')['index'].mean().reset_index()
    
    # Create a line chart
    fig = px.line(
        weekday_avg,
        x='day_of_week',
        y='index',
        markers=True,
        height=500,
        labels={'index': 'Average AQI', 'day_of_week': 'Day of Week'}
    )
    
    # Add annotations for weekdays and weekends
    fig.add_annotation(
        x="Friday",
        y=weekday_avg[weekday_avg['day_of_week'] == 'Friday']['index'].values[0],
        text="Weekday Peak",
        showarrow=True,
        arrowhead=1,
        arrowcolor=colors["accent"],
        ax=0,
        ay=-30
    )
    
    fig.add_annotation(
        x="Sunday",
        y=weekday_avg[weekday_avg['day_of_week'] == 'Sunday']['index'].values[0],
        text="Weekend Low",
        showarrow=True,
        arrowhead=1,
        arrowcolor=colors["accent"],
        ax=0,
        ay=-30
    )
    
    fig.update_layout(
        title="Weekday AQI Pattern",
        xaxis_title="Day of Week",
        yaxis_title="Average AQI",
        **get_custom_plotly_layout_args()
    )
    
    return fig

def create_aqi_trend_chart(data):
    """Create an enhanced AQI trend chart with trend line and annotations"""
    colors = get_theme_colors()
    
    # Sort data by date
    data_sorted = data.sort_values("date")
    
    # Calculate 7-day rolling average
    data_sorted["rolling_avg_7day"] = data_sorted["index"].rolling(window=7, center=True, min_periods=1).mean()
    
    # Calculate 30-day rolling average for trend
    data_sorted["rolling_avg_30day"] = data_sorted["index"].rolling(window=30, center=True, min_periods=1).mean()
    
    # Detect anomalies
    data_with_anomalies, anomaly_dates = detect_anomalies(data_sorted)
    
    # Create the figure
    fig = go.Figure()
    
    # Add daily AQI values
    fig.add_trace(go.Scatter(
        x=data_sorted["date"],
        y=data_sorted["index"],
        mode="lines+markers",
        name="Daily AQI",
        line=dict(color=colors["subtle_text"], width=1),
        marker=dict(size=3),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>AQI: %{y}<extra></extra>"
    ))
    
    # Add 7-day rolling average
    fig.add_trace(go.Scatter(
        x=data_sorted["date"],
        y=data_sorted["rolling_avg_7day"],
        mode="lines",
        name="7-Day Average",
        line=dict(color=colors["accent"], width=2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>7-Day Avg AQI: %{y}<extra></extra>"
    ))
    
    # Add 30-day rolling average
    fig.add_trace(go.Scatter(
        x=data_sorted["date"],
        y=data_sorted["rolling_avg_30day"],
        mode="lines",
        name="30-Day Trend",
        line=dict(color=colors["highlight"], width=2, dash="dash"),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>30-Day Trend AQI: %{y}<extra></extra>"
    ))
    
    # Add anomaly markers
    if anomaly_dates:
        anomaly_data = data_sorted[data_sorted['date'].isin(anomaly_dates)]
        fig.add_trace(go.Scatter(
            x=anomaly_data["date"],
            y=anomaly_data["index"],
            mode="markers",
            name="Anomalies",
            marker=dict(color=colors["highlight"], size=8, symbol="x"),
            hovertemplate="<b>Anomaly Detected!</b><br>Date: %{x|%Y-%m-%d}<br>AQI: %{y}<extra></extra>"
        ))
    
    # Calculate trend direction and strength
    trend_direction, trend_strength = calculate_pollution_trend(data_sorted)
    
    # Add trend annotation
    fig.add_annotation(
        x=data_sorted["date"].max() - pd.Timedelta(days=30),
        y=data_sorted["rolling_avg_30day"].iloc[-1],
        text=f"Trend: {trend_direction} ({trend_strength:.1f}%)",
        showarrow=True,
        arrowhead=1,
        arrowcolor=colors["accent"],
        ax=-50,
        ay=0
    )
    
    fig.update_layout(
        title="AQI Trend Analysis",
        xaxis_title="Date",
        yaxis_title="AQI",
        height=500,
        hovermode="x unified",
        **get_custom_plotly_layout_args()
    )
    
    return fig, trend_direction, trend_strength

def create_enhanced_calendar_heatmap(data, year):
    """Create an enhanced calendar heatmap with month view"""
    colors = get_theme_colors()
    
    # Create a complete date range for the year
    start_date = pd.to_datetime(f"{year}-01-01")
    end_date = pd.to_datetime(f"{year}-12-31")
    full_year_dates = pd.date_range(start_date, end_date, freq="D")
    
    # Create a DataFrame with all dates
    calendar_df = pd.DataFrame({"date": full_year_dates})
    calendar_df["week"] = calendar_df["date"].dt.isocalendar().week
    calendar_df["day_of_week"] = calendar_df["date"].dt.dayofweek
    calendar_df["month"] = calendar_df["date"].dt.month
    calendar_df["day_of_month"] = calendar_df["date"].dt.day
    
    # Adjust week numbers for display
    calendar_df.loc[(calendar_df["date"].dt.month == 1) & (calendar_df["week"] > 50), "week"] = 0
    calendar_df.loc[(calendar_df["date"].dt.month == 12) & (calendar_df["week"] == 1), "week"] = calendar_df["week"].max() + 1
    
    # Merge with AQI data
    merged_cal_df = pd.merge(
        calendar_df,
        data[["date", "index", "level"]],
        on="date",
        how="left"
    )
    
    # Fill missing values
    merged_cal_df["level"] = merged_cal_df["level"].fillna("Unknown")
    merged_cal_df["aqi_text"] = merged_cal_df["index"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    
    # Map AQI categories to numeric values for the heatmap
    level_to_code = {level: idx for idx, level in enumerate(colors["category_colors"].keys())}
    z_values = merged_cal_df["level"].map(level_to_code)
    
    # Create month heatmaps
    months = merged_cal_df["month"].unique()
    fig = go.Figure()
    
    for month in months:
        month_data = merged_cal_df[merged_cal_df["month"] == month]
        
        # Create a subplot for each month
        fig.add_trace(go.Heatmap(
            x=month_data["week"],
            y=month_data["day_of_week"],
            z=z_values[month_data.index],
            customdata=pd.DataFrame({
                "date": month_data["date"].dt.strftime("%Y-%m-%d"),
                "level": month_data["level"],
                "aqi": month_data["aqi_text"]
            }),
            hovertemplate="<b>%{customdata[0]}</b><br>AQI: %{customdata[2]} (%{customdata[1]})<extra></extra>",
            colorscale=[[i / (len(colors["category_colors"]) - 1), color] for i, color in enumerate(colors["category_colors"].values())],
            showscale=False if month != months[0] else True,
            xgap=3, ygap=3,
            name=f"Month {month}"
        ))
    
    # Day labels
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    # Create month labels
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # Update layout
    fig.update_layout(
        yaxis=dict(tickmode="array", tickvals=list(range(7)), ticktext=day_labels, showgrid=False, zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False, tickmode="array", ticktext=[], tickvals=[]),
        height=700,
        margin=dict(t=50, b=100, l=40, r=40),
        **get_custom_plotly_layout_args()
    )
    
    return fig

def create_city_comparison_radar(data, cities):
    """Create a radar chart comparing cities across different metrics"""
    colors = get_theme_colors()
    
    # Filter data for selected cities
    city_data = data[data['city'].isin(cities)].copy()
    
    # Calculate metrics for each city
    metrics = {}
    for city in cities:
        city_df = city_data[city_data['city'] == city]
        
        # Average AQI
        avg_aqi = city_df['index'].mean()
        
        # Percentage of days in each category
        total_days = len(city_df)
        good_pct = (len(city_df[city_df['level'] == 'Good']) / total_days) * 100 if total_days > 0 else 0
        satisfactory_pct = (len(city_df[city_df['level'] == 'Satisfactory']) / total_days) * 100 if total_days > 0 else 0
        moderate_pct = (len(city_df[city_df['level'] == 'Moderate']) / total_days) * 100 if total_days > 0 else 0
        poor_pct = (len(city_df[city_df['level'] == 'Poor']) / total_days) * 100 if total_days > 0 else 0
        very_poor_pct = (len(city_df[city_df['level'] == 'Very Poor']) / total_days) * 100 if total_days > 0 else 0
        severe_pct = (len(city_df[city_df['level'] == 'Severe']) / total_days) * 100 if total_days > 0 else 0
        
        # Calculate health score (lower is better)
        health_score = (
            (moderate_pct * 1) +
            (poor_pct * 2) +
            (very_poor_pct * 3) +
            (severe_pct * 4)
        ) / 100
        
        # Normalize metrics to 0-100 scale
        metrics[city] = {
            'Avg AQI': min(100, (avg_aqi / 500) * 100),  # Normalize to 0-100
            'Good Days': good_pct,
            'Satisfactory Days': satisfactory_pct,
            'Moderate Days': moderate_pct,
            'Poor Days': poor_pct,
            'Very Poor Days': very_poor_pct,
            'Severe Days': severe_pct,
            'Health Score': min(100, health_score * 25)  # Normalize to 0-100
        }
    
    # Create radar chart
    fig = go.Figure()
    
    # Add traces for each city
    for i, city in enumerate(cities):
        fig.add_trace(go.Scatterpolar(
            r=list(metrics[city].values()),
            theta=list(metrics[city].keys()),
            fill='toself',
            name=city,
            line_color=list(colors["pollutant_colors"].values())[i % len(colors["pollutant_colors"])]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title="City Comparison Radar Chart",
        height=600,
        **get_custom_plotly_layout_args()
    )
    
    return fig

def create_pollution_trend_animation(data):
    """Create an animated pollution trend chart"""
    colors = get_theme_colors()
    
    # Sort data by date
    data_sorted = data.sort_values("date")
    
    # Extract year and month for animation
    data_sorted['year_month'] = data_sorted['date'].dt.to_period('M')
    
    # Group by city and year_month
    monthly_data = data_sorted.groupby(['city', 'year_month'])['index'].mean().reset_index()
    monthly_data['year_month_str'] = monthly_data['year_month'].dt.strftime('%Y-%m')
    
    # Create animated bar chart
    fig = px.bar(
        monthly_data,
        x='city',
        y='index',
        color='city',
        animation_frame='year_month_str',
        range_y=[0, monthly_data['index'].max() * 1.1],
        labels={'index': 'Average AQI', 'city': 'City'},
        height=500
    )
    
    # Update layout
    fig.update_layout(
        title="Monthly AQI Trends by City",
        xaxis_title="City",
        yaxis_title="Average AQI",
        showlegend=False,
        **get_custom_plotly_layout_args()
    )
    
    # Update animation settings
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500
    
    return fig

def create_aqi_gauge_chart(aqi_value, level):
    """Create a gauge chart for current AQI"""
    colors = get_theme_colors()
    
    # Determine color based on AQI level
    gauge_color = colors["category_colors"].get(level, colors["category_colors"]["Unknown"])
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        title={'text': f"Current AQI: {level}"},
        gauge={
            'axis': {'range': [None, 500]},
            'bar': {'color': gauge_color},
            'steps': [
                {'range': [0, 50], 'color': colors["category_colors"]["Good"]},
                {'range': [50, 100], 'color': colors["category_colors"]["Satisfactory"]},
                {'range': [100, 200], 'color': colors["category_colors"]["Moderate"]},
                {'range': [200, 300], 'color': colors["category_colors"]["Poor"]},
                {'range': [300, 400], 'color': colors["category_colors"]["Very Poor"]},
                {'range': [400, 500], 'color': colors["category_colors"]["Severe"]}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': aqi_value
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        **get_custom_plotly_layout_args()
    )
    
    return fig

def create_health_recommendations_card(level):
    """Create a health recommendations card based on AQI level"""
    colors = get_theme_colors()
    
    # Get extended health recommendations
    recommendations = EXTENDED_HEALTH_RECOMMENDATIONS.get(level, EXTENDED_HEALTH_RECOMMENDATIONS["Unknown"])
    
    # Create HTML for the card
    card_html = f"""
    <div style="background-color: {colors['card_background']}; border-radius: 16px; padding: 1.5rem; border: 1px solid {colors['border']}; margin-bottom: 1.5rem;">
        <h3 style="color: {colors['accent']}; margin-bottom: 1rem;">Health Recommendations</h3>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;">
            <div style="background-color: {colors['background']}; border-radius: 8px; padding: 1rem;">
                <h4 style="color: {colors['text']}; margin-bottom: 0.5rem;">General Public</h4>
                <p style="color: {colors['subtle_text']}; margin: 0;">{recommendations['General']}</p>
            </div>
            
            <div style="background-color: {colors['background']}; border-radius: 8px; padding: 1rem;">
                <h4 style="color: {colors['text']}; margin-bottom: 0.5rem;">Sensitive Groups</h4>
                <p style="color: {colors['subtle_text']}; margin: 0;">{recommendations['Sensitive']}</p>
            </div>
            
            <div style="background-color: {colors['background']}; border-radius: 8px; padding: 1rem;">
                <h4 style="color: {colors['text']}; margin-bottom: 0.5rem;">Outdoor Activities</h4>
                <p style="color: {colors['subtle_text']}; margin: 0;">{recommendations['Outdoor']}</p>
            </div>
            
            <div style="background-color: {colors['background']}; border-radius: 8px; padding: 1rem;">
                <h4 style="color: {colors['text']}; margin-bottom: 0.5rem;">Indoor Precautions</h4>
                <p style="color: {colors['subtle_text']}; margin: 0;">{recommendations['Indoor']}</p>
            </div>
            
            <div style="background-color: {colors['background']}; border-radius: 8px; padding: 1rem;">
                <h4 style="color: {colors['text']}; margin-bottom: 0.5rem;">Children</h4>
                <p style="color: {colors['subtle_text']}; margin: 0;">{recommendations['Children']}</p>
            </div>
            
            <div style="background-color: {colors['background']}; border-radius: 8px; padding: 1rem;">
                <h4 style="color: {colors['text']}; margin-bottom: 0.5rem;">Elderly</h4>
                <p style="color: {colors['subtle_text']}; margin: 0;">{recommendations['Elderly']}</p>
            </div>
        </div>
    </div>
    """
    
    return card_html

def create_pollution_sources_sankey(data):
    """Create a Sankey diagram showing pollution sources and their impacts"""
    colors = get_theme_colors()
    
    # This is a placeholder function as actual pollution source data might not be available
    # We'll create a simulated Sankey diagram
    
    # Define source, target, and value for the Sankey diagram
    sources = ["Industrial", "Vehicular", "Construction", "Agricultural", "Natural"]
    targets = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
    
    # Create random connections between sources and pollutants
    np.random.seed(42)
    connections = []
    for i, source in enumerate(sources):
        for j, target in enumerate(targets):
            # Create a random value for the connection
            value = np.random.randint(1, 20)
            if value > 10:  # Only include significant connections
                connections.append({
                    "source": i,
                    "target": len(sources) + j,
                    "value": value
                })
    
    # Create the Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color=colors["border"], width=0.5),
            label=sources + targets,
            color=[colors["accent"] if i < len(sources) else colors["highlight"] for i in range(len(sources) + len(targets))]
        ),
        link=dict(
            source=[conn["source"] for conn in connections],
            target=[conn["target"] for conn in connections],
            value=[conn["value"] for conn in connections],
            hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>Value: %{value}<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title="Pollution Sources and Impacts",
        height=600,
        **get_custom_plotly_layout_args()
    )
    
    return fig

def create_aqi_prediction_model(data):
    """Create an AQI prediction model using machine learning"""
    colors = get_theme_colors()
    
    # Sort data by date
    data_sorted = data.sort_values("date")
    
    # Extract features
    data_sorted["day_of_year"] = data_sorted["date"].dt.dayofyear
    data_sorted["month"] = data_sorted["date"].dt.month
    data_sorted["day_of_week"] = data_sorted["date"].dt.dayofweek
    
    # Create lag features
    for lag in [1, 7, 30]:
        data_sorted[f"lag_{lag}"] = data_sorted["index"].shift(lag)
    
    # Create rolling features
    for window in [7, 30]:
        data_sorted[f"rolling_{window}"] = data_sorted["index"].rolling(window=window).mean()
    
    # Drop rows with NaN values
    model_data = data_sorted.dropna()
    
    if len(model_data) < 50:  # Not enough data for modeling
        return None
    
    # Define features and target
    features = ["day_of_year", "month", "day_of_week", "lag_1", "lag_7", "lag_30", "rolling_7", "rolling_30"]
    X = model_data[features]
    y = model_data["index"]
    
    # Split data into train and test sets
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train a polynomial regression model
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_poly)
    
    # Calculate model performance
    mae = mean_absolute_error(y_test, y_pred)
    
    # Create a DataFrame with actual and predicted values
    results_df = pd.DataFrame({
        "date": model_data["date"].iloc[split_idx:].values,
        "actual": y_test.values,
        "predicted": y_pred
    })
    
    # Create a plot showing actual vs predicted values
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=results_df["date"],
        y=results_df["actual"],
        mode="lines+markers",
        name="Actual AQI",
        line=dict(color=colors["subtle_text"], width=2),
        marker=dict(size=4),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Actual AQI: %{y}<extra></extra>"
    ))
    
    # Add predicted values
    fig.add_trace(go.Scatter(
        x=results_df["date"],
        y=results_df["predicted"],
        mode="lines+markers",
        name="Predicted AQI",
        line=dict(color=colors["accent"], width=2, dash="dash"),
        marker=dict(size=4),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Predicted AQI: %{y}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"AQI Prediction Model (MAE: {mae:.2f})",
        xaxis_title="Date",
        yaxis_title="AQI",
        height=500,
        hovermode="x unified",
        **get_custom_plotly_layout_args()
    )
    
    return fig, mae

def create_enhanced_css():
    """Generate enhanced CSS styling for the dashboard"""
    colors = get_theme_colors()
    
    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

        /* =================================
           1. GENERAL RESETS & DEFAULTS
           ================================= */
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        html, body, #root, .stApp {{
            background-color: {colors['background']} !important;
            color: {colors['text']} !important;
        }}

        body {{
            font-family: 'Inter', sans-serif;
            line-height: 1.7;
            font-size: 16px;
        }}

        a {{
            color: {colors['accent']};
            text-decoration: none;
            transition: color 0.3s ease;
        }}

        a:hover {{
            color: #00E5FF;
        }}

        /* =================================
           2. LAYOUT & CONTAINERS
           ================================= */
        .main .block-container {{
            padding: 2rem;
        }}

        /* Card-like styling for sections/charts */
        .stPlotlyChart, .stDataFrame, .stAlert, .stMetric,
        .stDownloadButton > button, .stButton > button,
        div[data-testid="stExpander"], div[data-testid="stForm"] {{
            border-radius: 16px;
            border: 1px solid {colors['border']};
            background-color: {colors['card_background']};
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .stPlotlyChart:hover, .stDataFrame:hover, .stMetric:hover,
        div[data-testid="stExpander"]:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 30px rgba(0, 188, 212, 0.25);
            border-color: #555555;
        }}

        /* Custom insight card with more padding */
        .insight-card {{
            background: linear-gradient(145deg, #1a1a1a, #232323);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            border: 1px solid {colors['border']};
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            height: 100%;
        }}

        /* =================================
           3. TYPOGRAPHY
           ================================= */
        h1 {{
            font-family: 'Inter', sans-serif;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.5rem;
            letter-spacing: -0.5px;
            font-size: 3rem;
            background: linear-gradient(90deg, {colors['accent']}, #00E5FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        h2 {{
            font-family: 'Inter', sans-serif;
            color: {colors['accent']};
            border-bottom: 2px solid {colors['border']};
            padding-bottom: 0.75rem;
            margin-top: 3rem;
            margin-bottom: 2rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            position: relative;
            font-size: 2rem;
        }}
        
        h2:after {{
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            width: 100px;
            height: 3px;
            background: linear-gradient(90deg, {colors['accent']}, transparent);
        }}
        
        h3 {{
            font-family: 'Inter', sans-serif;
            color: {colors['text']};
            margin-bottom: 1.2rem;
            font-weight: 600;
            font-size: 1.5rem;
        }}

        h4, h5, h6 {{
            font-family: 'Inter', sans-serif;
            color: {colors['text']};
            margin-bottom: 1rem;
            font-weight: 600;
        }}

        /* =================================
           4. SIDEBAR
           ================================= */
        .stSidebar {{
            background-color: {colors['card_background']};
            border-right: 1px solid {colors['border']};
            padding: 2rem 1.5rem;
            box-shadow: 5px 0 25px rgba(0, 0, 0, 0.25);
            width: 300px !important;
        }}

        .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3, .stSidebar .stMarkdown p {{
            color: {colors['text']};
            text-align: left;
        }}

        .stSidebar h2 {{
          position: relative;
          margin-top: 1rem;
          margin-bottom: 1.5rem;
          padding-bottom: 0.5rem;
          font-size: 1.5rem !important;
          color: {colors['accent']} !important;
        }}

        .stSidebar h2:after {{
          content: "";
          position: absolute;
          bottom: 0;
          left: 0;
          width: 3rem;
          height: 3px;
          background-color: {colors['accent']};
        }}
        
        .stSidebar .stSelectbox label, .stSidebar .stMultiselect label, .stSidebar .stNumberInput label, .stSidebar .stSlider label {{
            color: {colors['accent']} !important;
            font-weight: 600;
            font-size: 1.05rem;
        }}

        /* Selectbox/Multiselect widget improvements */
        div[data-baseweb="select"] > div:first-child {{
          background-color: {colors['background']} !important;
          color: {colors['text']} !important;
          border: 1px solid #555555 !important;
          border-radius: 10px !important;
        }}
        div[data-baseweb="select"] [role="listbox"] {{
          background-color: {colors['card_background']} !important;
          border: 1px solid #555555 !important;
          border-radius: 10px !important;
        }}
        div[data-baseweb="select"] [role="option"]:hover {{
          background-color: {colors['accent']} !important;
          color: {colors['background']} !important;
        }}

        /* =================================
           5. SPECIFIC WIDGETS & COMPONENTS
           ================================= */

        /* --- Header --- */
        .gradient-header {{
            background: linear-gradient(270deg, {colors['background']}, #1a2a3a, {colors['background']});
            background-size: 200% 200%;
            animation: gradientAnimation 12s ease infinite;
            padding: 2.5rem 1.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            text-align: center;
            border: 1px solid {colors['border']};
        }}

        /* --- Metric Cards --- */
        .stMetric {{
            background-color: {colors['card_background']};
            border: 1px solid {colors['border']};
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
        }}
        
        .stMetric > div:nth-child(1) {{ /* Label */
            font-size: 1rem;
            color: {colors['subtle_text']};
            font-weight: 500;
        }}
        
        .stMetric > div:nth-child(2) {{ /* Value */
            font-size: 2.5rem;
            font-weight: 700;
            color: {colors['accent']};
            margin: 0.5rem 0;
        }}

        /* --- Tabs --- */
        .stTabs [data-baseweb="tab-list"] {{
             border-bottom: 2px solid {colors['border']};
        }}
        .stTabs [data-baseweb="tab"] {{
            padding: 1rem 1.5rem;
            font-weight: 600;
            color: {colors['subtle_text']};
        }}
         .stTabs [aria-selected="true"] {{
            color: {colors['accent']} !important;
            border-bottom: 3px solid {colors['accent']};
         }}

        /* --- Buttons --- */
        .stDownloadButton button, .stButton button {{
            background: linear-gradient(90deg, {colors['accent']}, #00BFA5);
            color: {colors['background']};
            border: none;
            font-weight: 600;
            padding: 0.75rem 2rem;
            border-radius: 50px;
            transition: all 0.3s ease;
            font-size: 1rem;
            box-shadow: 0 4px 10px rgba(0, 188, 212, 0.3);
        }}
        .stDownloadButton button:hover, .stButton button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 6px 15px rgba(0, 188, 212, 0.4);
        }}

        /* --- Health & Info Cards --- */
        .health-card {{
            border-left: 4px solid;
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
            background-color: rgba(30, 30, 30, 0.7);
        }}
        
        /* =================================
           6. RESPONSIVE DESIGN
           ================================= */
        @media (max-width: 768px) {{
            .main .block-container {{
                padding: 1rem;
            }}
            h1 {{ font-size: 2.2rem; }}
            h2 {{ font-size: 1.6rem; }}
            
            /* Stack all columns */
            .col1, .col2, .col3,
            .status-col1, .status-col2, .status-col3,
            .health-col1, .health-col2,
            .map-col1, .map-col2,
            .forecast-col1, .forecast-col2,
            .poll-col1, .poll-col2 {{
                flex: 0 0 100% !important;
                max-width: 100% !important;
                margin-bottom: 1rem;
            }}
            
            .footer-info {{
                flex-direction: column;
                text-align: center;
                gap: 1.5rem;
            }}
        }}

        @media (max-width: 480px) {{
            .main .block-container {{ padding: 0.5rem; }}
            h1 {{ font-size: 1.8rem; }}
            h2 {{ font-size: 1.4rem; }}
            body {{ font-size: 14px; }}
            
            .stMetric > div:nth-child(2) {{ font-size: 2rem !important; }}
            
            .stTabs {{ overflow-x: auto; }}
            
            .stPlotlyChart {{ height: 350px !important; }}
        }}

        /* =================================
           7. ANIMATIONS & FOOTER
           ================================= */
        @keyframes gradientAnimation {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        
        .footer-container {{
          margin-top: 4rem;
          padding: 3rem 2rem;
          border-radius: 16px;
          background: linear-gradient(270deg, {colors['background']}, #1a2a3a, {colors['background']});
          background-size: 200% 200%;
          animation: gradientAnimation 8s ease infinite;
          border: 1px solid #2a3a4a;
          text-align: center;
        }}
        .footer-container h3 {{
          color: {colors['accent']};
          font-size: 1.8rem;
          margin-bottom: 1.5rem;
        }}
        .footer-info {{
          display: flex;
          justify-content: center;
          gap: 2.5rem;
          flex-wrap: wrap;
          margin-bottom: 2rem;
        }}
        .footer-info p {{ margin: 0; }}
        .footer-info .label {{ font-size: 0.9rem; color: #B0B0B0; }}
        .footer-info .value {{ font-weight: 500; color: {colors['text']}; }}
        .footer-links a {{
          color: {colors['accent']};
          font-weight: 600;
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
        }}
        .copyright {{
          font-size: 0.85rem;
          color: #707070;
          text-align: center;
          margin-top: 2rem;
        }}
        
        /* Custom animations */
        .fade-in {{
            animation: fadeIn 1s ease-in;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        .slide-up {{
            animation: slideUp 0.5s ease-out;
        }}
        
        @keyframes slideUp {{
            from {{ transform: translateY(20px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 10px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {colors['background']};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {colors['accent']};
            border-radius: 5px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: #00E5FF;
        }}
        
        /* Theme toggle button */
        .theme-toggle {{
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 999;
            background-color: {colors['card_background']};
            border: 1px solid {colors['border']};
            border-radius: 50%;
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }}
        
        .theme-toggle:hover {{
            transform: scale(1.1);
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
        }}
        
        /* Alert animations */
        .alert-animation {{
            animation: alertPulse 2s infinite;
        }}
        
        @keyframes alertPulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(255, 107, 107, 0.7); }}
            70% {{ box-shadow: 0 0 0 10px rgba(255, 107, 107, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(255, 107, 107, 0); }}
        }}
        
        /* Loading animation */
        .loading-animation {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100px;
        }}
        
        .loading-spinner {{
            width: 50px;
            height: 50px;
            border: 5px solid {colors['border']};
            border-top: 5px solid {colors['accent']};
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
    """
    
    return css

# ------------------- Custom CSS Styling -------------------
st.markdown(create_enhanced_css(), unsafe_allow_html=True)

# Add theme toggle button
theme_toggle_html = f"""
<div class="theme-toggle" onclick="toggleTheme()">
    <span style="font-size: 24px;">{'🌙' if st.session_state.theme == 'light' else '☀️'}</span>
</div>

<script>
    function toggleTheme() {{
        // This is a placeholder for theme toggle functionality
        // In a real implementation, you would use JavaScript to toggle between light and dark themes
        // and store the preference in localStorage or a cookie
        alert('Theme toggle functionality would be implemented here');
    }}
</script>
"""
st.markdown(theme_toggle_html, unsafe_allow_html=True)

# ------------------- Title Header -------------------
st.markdown(f"""
<div class="gradient-header fade-in">
    <h1>🌬️ IIT KGP AIR QUALITY DASHBOARD</h1>
    <p style="color: #B0B0B0; font-size: 1.1rem; max-width: 800px; margin: 0 auto;">
        Real-time Air Quality Monitoring and Predictive Analysis for Indian Cities
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------- Load Data -------------------
@st.cache_data(ttl=3600)
def load_data_and_metadata():
    """
    Tries to load today's CSV (named YY-MM-DD.csv). If not found, falls back to 'combined_air_quality.txt'.
    Returns: (df_loaded, load_msg, last_update_time)
    """
    today = pd.to_datetime("today").date()
    csv_path = f"data/{today}.csv"
    fallback_file = "combined_air_quality.txt"
    df_loaded = None
    is_today_data = False
    load_msg = ""
    last_update_time = None

    # 1) Attempt to load today's CSV
    if os.path.exists(csv_path):
        try:
            df_loaded = pd.read_csv(csv_path)
            if "date" in df_loaded.columns:
                df_loaded["date"] = pd.to_datetime(df_loaded["date"])
                is_today_data = True
                load_msg = f"Live data from: **{today}.csv**"
                last_update_time = pd.Timestamp(os.path.getmtime(csv_path), unit="s")
            else:
                load_msg = f"Warning: '{csv_path}' found but missing 'date' column. Using fallback."
        except Exception as e:
            load_msg = f"Error loading '{csv_path}': {e}. Using fallback."

    # 2) If today's CSV is missing or invalid, load fallback
    if df_loaded is None or not is_today_data:
        try:
            if not os.path.exists(fallback_file):
                st.error(f"FATAL: Main data file '{fallback_file}' not found.")
                return pd.DataFrame(), "Error: Main data file not found.", None
            df_loaded = pd.read_csv(fallback_file, sep="\t", parse_dates=["date"])
            base_load_msg = f"Displaying archive data from: **{fallback_file}**"
            load_msg = base_load_msg if not load_msg or is_today_data else load_msg + " " + base_load_msg
            last_update_time = pd.Timestamp(os.path.getmtime(fallback_file), unit="s")
        except Exception as e:
            st.error(f"FATAL: Error loading '{fallback_file}': {e}.")
            return pd.DataFrame(), f"Error loading fallback: {e}", None

    # Common post-processing
    for col, default_val in [("pollutant", np.nan), ("level", "Unknown")]:
        if col not in df_loaded.columns:
            df_loaded[col] = default_val

    df_loaded["pollutant"] = (
        df_loaded["pollutant"].astype(str)
        .str.split(",").str[0].str.strip()
        .replace(["nan", "NaN", "None", ""], np.nan)
    )
    df_loaded["level"] = df_loaded["level"].astype(str).fillna("Unknown")
    df_loaded["pollutant"] = df_loaded["pollutant"].fillna("Other")
    
    # Filter 2025 data to include only up to May
    if 2025 in df_loaded["date"].dt.year.unique():
        df_loaded = df_loaded[~((df_loaded["date"].dt.year == 2025) & (df_loaded["date"].dt.month > 5))]

    return df_loaded, load_msg, last_update_time

# Load data with a loading animation
with st.spinner("Loading air quality data..."):
    df, load_message, data_last_updated = load_data_and_metadata()
    st.session_state.data_loaded = True

if df.empty:
    st.error("Dashboard cannot operate without data. Please check data sources.")
    st.stop()

if data_last_updated:
    st.caption(
        f"<p style='text-align: center; color: {get_theme_colors()['subtle_text']}; font-size: 0.9rem;'>"
        f"📅 Last data update: {data_last_updated.strftime('%Y-%m-%d %H:%M:%S')} "
        f"</p>",
        unsafe_allow_html=True
    )

# ------------------- Sidebar Filters -------------------
with st.sidebar:
    st.header("🔭 Controls")
    st.info("Fetching real-time data from CPCB. Today's data available after 5:45 PM IST.", icon="ℹ️")

    unique_cities = sorted(df["city"].unique()) if "city" in df.columns else []
    default_city_val = ["Delhi"] if "Delhi" in unique_cities else (unique_cities[0:1] if unique_cities else [])
    selected_cities = st.multiselect("🏙️ Select Cities", unique_cities, default=default_city_val,
                                    help="Select one or more cities for detailed analysis")

    years = sorted(df["date"].dt.year.unique())
    # Default to 2024 if present, else last year
    default_year_val = 2024 if 2024 in years else (max(years) if years else None)
    if default_year_val:
        year_index = years.index(default_year_val)
    else:
        year_index = 0
    year = st.selectbox("🗓️ Select Year", years, index=year_index if years else 0,
                      help="Select the year for analysis")

    months_map_dict = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
    }
    
    # For 2025, only show months up to May
    if year == 2025:
        month_options_list = ["All Months"] + [months_map_dict[i] for i in range(1, 6)]
    else:
        month_options_list = ["All Months"] + list(months_map_dict.values())
        
    selected_month_name = st.selectbox("🌙 Select Month", month_options_list, index=0,
                                     help="Optionally select a specific month")

    month_number_filter = None
    if selected_month_name != "All Months":
        month_number_filter = [k for k, v in months_map_dict.items() if v == selected_month_name][0]

    # Filter data based on global selections
    df_period_filtered = df[df["date"].dt.year == year].copy()
    if month_number_filter:
        df_period_filtered = df_period_filtered[df_period_filtered["date"].dt.month == month_number_filter]
    
    # Advanced filters
    st.markdown("---")
    st.subheader("🔬 Advanced Filters")
    
    # AQI range filter
    aqi_range = st.slider(
        "AQI Range",
        min_value=0,
        max_value=500,
        value=(0, 500),
        help="Filter data by AQI range"
    )
    
    # Pollutant filter
    pollutants = sorted(df["pollutant"].unique()) if "pollutant" in df.columns else []
    selected_pollutants = st.multiselect(
        "Pollutants",
        pollutants,
        default=pollutants,
        help="Select pollutants to include in analysis"
    )
    
    # Analysis type
    analysis_type = st.selectbox(
        "Analysis Type",
        ["General", "Health Impact", "Pollution Sources", "Weather Correlation"],
        help="Select the type of analysis to focus on"
    )
    
    # Apply advanced filters
    if aqi_range != (0, 500):
        df_period_filtered = df_period_filtered[
            (df_period_filtered["index"] >= aqi_range[0]) & 
            (df_period_filtered["index"] <= aqi_range[1])
        ]
    
    if selected_pollutants:
        df_period_filtered = df_period_filtered[
            df_period_filtered["pollutant"].isin(selected_pollutants)
        ]
    
    st.markdown("---")
    st.markdown("""
    <div style="margin-top: 2rem; text-align: center;">
        <p style="font-size: 0.85rem; color: #B0B0B0;">
            Developed with ❤️ by IIT Kharagpur<br>Data Source: CPCB India
        </p>
    </div>
    """, unsafe_allow_html=True)

# ========================================================
# =========  NATIONAL KEY INSIGHTS (Enhanced)  ===========
# ========================================================
st.markdown("## 🇮🇳 National Air Quality Snapshot")

col1, col2, col3 = st.columns(3, gap="large")
with col1:
    with st.container():
        st.markdown(f"<div class='insight-card slide-up'><h3>🌆 Coverage</h3>", unsafe_allow_html=True)
        cities_count = df_period_filtered["city"].nunique()
        st.metric(label="Cities Monitored", value=cities_count)
        st.markdown(f"<p style='color:{get_theme_colors()['subtle_text']}; font-size:0.9rem;'>Across India in {year}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown(f"<div class='insight-card slide-up'><h3>📈 National Average</h3>", unsafe_allow_html=True)
        if not df_period_filtered.empty:
            avg_aqi_national = df_period_filtered["index"].mean()
            st.metric(label="Average AQI", value=f"{avg_aqi_national:.1f}")
            national_category = get_category(avg_aqi_national)
            st.markdown(f"<p style='color:{get_theme_colors()['category_colors'].get(national_category)}; font-weight:600;'>{national_category} Air Quality</p>", unsafe_allow_html=True)
        else:
            st.info("No data available")
        st.markdown("</div>", unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown(f"<div class='insight-card slide-up'><h3>📅 Time Period</h3>", unsafe_allow_html=True)
        period = f"{selected_month_name} {year}" if selected_month_name != "All Months" else f"Full Year {year}"
        st.markdown(f"<p style='font-size:1.8rem; text-align:center; margin:1rem 0; color:{get_theme_colors()['text']}; font-weight:600;'>{period}</p>", unsafe_allow_html=True)
        days_count = df_period_filtered["date"].nunique()
        st.markdown(f"<p style='text-align:center; color:{get_theme_colors()['subtle_text']};'>{days_count} days of data</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ========================================================
# ====== TOP & BOTTOM CITIES VISUALIZATION ======
# ========================================================
st.markdown(f"## 🏆 City Rankings for {year}")

if not df_period_filtered.empty:
    city_avg_aqi = df_period_filtered.groupby("city")["index"].mean().dropna().sort_values()

    if not city_avg_aqi.empty:
        max_cities = len(city_avg_aqi)
        num_to_show = st.slider(
            "Select Number of Cities to Rank",
            min_value=3,
            max_value=min(15, max_cities),
            value=min(5, max_cities),
            help="Adjust the slider to see more or fewer cities in the rankings."
        )

        top_cities = city_avg_aqi.head(num_to_show).reset_index()
        top_cities.columns = ["City", "Avg AQI"]
        top_cities["Category"] = top_cities["Avg AQI"].apply(get_category)

        bottom_cities = city_avg_aqi.tail(num_to_show).reset_index()
        bottom_cities.columns = ["City", "Avg AQI"]
        bottom_cities["Category"] = bottom_cities["Avg AQI"].apply(get_category)

        col_top, col_bottom = st.columns(2, gap="large")

        with col_top:
            st.markdown(f"<h5 style='color:{get_theme_colors()['text']}; text-align:center;'>🥇 Top {num_to_show} Cleanest Cities</h5>", unsafe_allow_html=True)
            fig_top = px.bar(
                top_cities.sort_values("Avg AQI", ascending=False),
                x="Avg AQI",
                y="City",
                color="Category",
                color_discrete_map=get_theme_colors()["category_colors"],
                orientation='h',
                text='Avg AQI'
            )
            fig_top.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_top.update_layout(
                yaxis_title=None, xaxis_title="Average AQI", showlegend=False,
                height=max(200, num_to_show * 50),
                **get_custom_plotly_layout_args()
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with col_bottom:
            st.markdown(f"<h5 style='color:{get_theme_colors()['text']}; text-align:center;'>⚠️ Top {num_to_show} Most Polluted Cities</h5>", unsafe_allow_html=True)
            fig_bottom = px.bar(
                bottom_cities.sort_values("Avg AQI", ascending=True),
                x="Avg AQI",
                y="City",
                color="Category",
                color_discrete_map=get_theme_colors()["category_colors"],
                orientation='h',
                text='Avg AQI'
            )
            fig_bottom.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_bottom.update_layout(
                yaxis_title=None, xaxis_title="Average AQI", showlegend=False,
                height=max(200, num_to_show * 50),
                **get_custom_plotly_layout_args()
            )
            st.plotly_chart(fig_bottom, use_container_width=True)

    else:
        st.info("No city averages available for the selected period.")
else:
    st.info("No data available for the selected period.")

# ========================================================
# =======   CITY-SPECIFIC ANALYSIS (Improved)   ==========
# ========================================================
export_data_list = []

if not selected_cities:
    st.info("✨ Select one or more cities from the sidebar to dive into detailed analysis.")
else:
    for city in selected_cities:
        st.markdown(f"## 🏙️ {city.upper()} DEEP DIVE – {year}")
        
        city_data_full = df_period_filtered[df_period_filtered["city"] == city].copy()
        if city_data_full.empty:
            st.warning(f"😔 No data available for {city} for {selected_month_name}, {year}. Try different filter settings.")
            continue
            
        latest_data = city_data_full.sort_values("date", ascending=False).iloc[0]
        current_aqi = latest_data["index"]
        current_level = latest_data["level"]
        current_pollutant = latest_data["pollutant"]
        health_msg = HEALTH_RECOMMENDATIONS.get(current_level, "No specific health recommendations available")
        
        status_col1, status_col2, status_col3 = st.columns([1,2,1], gap="large")
        with status_col1:
            st.markdown(f"<div style='text-align:center; padding:1rem; border-radius:16px; background:{get_theme_colors()['card_background']}; border:1px solid {get_theme_colors()['border']}; height:100%'>", unsafe_allow_html=True)
            st.markdown("<h6>Live Status</h6>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:3rem; font-weight:800; color:{get_theme_colors()['category_colors'].get(current_level, '#FFFFFF')}; line-height:1.2;'>{current_aqi:.0f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.2rem; font-weight:600;'>{current_level}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with status_col2:
            st.markdown(f"<div style='padding:1.5rem; border-radius:16px; background:{get_theme_colors()['card_background']}; border:1px solid {get_theme_colors()['border']}; height:100%;'>", unsafe_allow_html=True)
            st.markdown("<h6>Health Recommendation</h6>", unsafe_allow_html=True)
            st.markdown(f"<div class='health-card' style='border-left-color: {get_theme_colors()['category_colors'].get(current_level, '#FFFFFF')};'>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:1.1rem;'>{health_msg}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with status_col3:
            st.markdown(f"<div style='text-align:center; padding:1rem; border-radius:16px; background:{get_theme_colors()['card_background']}; border:1px solid {get_theme_colors()['border']}; height:100%;'>", unsafe_allow_html=True)
            st.markdown("<h6>Dominant Pollutant</h6>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:2.5rem; font-weight:700; color:{get_theme_colors()['pollutant_colors'].get(current_pollutant, '#FFFFFF')}; margin:1rem 0; line-height:1.2;'>{current_pollutant}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        city_data_full["day_of_year"] = city_data_full["date"].dt.dayofyear
        city_data_full["month_name"] = city_data_full["date"].dt.month_name()
        city_data_full["day_of_month"] = city_data_full["date"].dt.day
        export_data_list.append(city_data_full)

        # Enhanced tabs with more analysis options
        tab_trend, tab_dist, tab_heatmap_detail, tab_weekday, tab_health, tab_forecast, tab_comparison, tab_sources, tab_weather = st.tabs([
            "📊 TRENDS & CALENDAR", "📈 DISTRIBUTIONS", "🗓️ DETAILED HEATMAP", 
            "📅 WEEKDAY ANALYSIS", "❤️ HEALTH ANALYSIS", "🔮 HEALTH FORECAST",
            "🏙️ CITY COMPARISON", "🏭 POLLUTION SOURCES", "🌤️ WEATHER CORRELATION"
        ])

        with tab_trend:
            st.markdown("<h5>📅 Daily AQI Calendar</h5>", unsafe_allow_html=True)
            if city_data_full["index"].notna().any():
                fig_cal = create_pollution_calendar(city_data_full, year)
                st.plotly_chart(fig_cal, use_container_width=True)
            else:
                st.info("No AQI data available for calendar plot.")

            st.markdown("<h5>📈 AQI Trend with Anomaly Detection</h5>", unsafe_allow_html=True)
            if len(city_data_full) >= 2:
                fig_trend, trend_direction, trend_strength = create_aqi_trend_chart(city_data_full)
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Display trend information
                st.markdown(f"""
                <div style="background-color: {get_theme_colors()['card_background']}; border-radius: 16px; padding: 1.5rem; border: 1px solid {get_theme_colors()['border']}; margin-top: 1.5rem;">
                    <h6 style="color: {get_theme_colors()['accent']}; margin-bottom: 1rem;">Trend Analysis</h6>
                    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                        <b>Direction:</b> {trend_direction} ({trend_strength:.1f}%)
                    </p>
                    <p style="font-size: 1.1rem; margin-bottom: 0;">
                        <b>Interpretation:</b> {'Air quality is improving' if trend_direction == 'Improving' else 'Air quality is deteriorating' if trend_direction == 'Deteriorating' else 'Air quality is relatively stable'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Not enough data points to display trend.")

        with tab_dist:
            col_bar_dist, col_sun_dist = st.columns([2, 1], gap="large")

            if not city_data_full.empty:
                with col_bar_dist:
                    st.markdown("<h5>📊 AQI Category Distribution</h5>", unsafe_allow_html=True)
                    fig_dist_bar = create_enhanced_distribution_chart(city_data_full)
                    st.plotly_chart(fig_dist_bar, use_container_width=True)

                with col_sun_dist:
                    st.markdown("<h5>☀️ Category Proportions</h5>", unsafe_allow_html=True)
                    category_counts = city_data_full["level"].value_counts()
                    if not category_counts.empty:
                        fig_sunburst = px.sunburst(
                            names=category_counts.index,
                            values=category_counts.values,
                            color_discrete_map=get_theme_colors()["category_colors"],
                            height=400
                        )
                        fig_sunburst.update_layout(
                            margin=dict(t=20, l=20, r=20, b=20),
                            **get_custom_plotly_layout_args()
                        )
                        st.plotly_chart(fig_sunburst, use_container_width=True)
                    else:
                        st.caption("No data for sunburst chart.")

                st.markdown("<h5>🎻 Monthly AQI Distribution</h5>", unsafe_allow_html=True)
                months_map_number = {v: k for k, v in months_map_dict.items()}
                present_months = sorted(
                    city_data_full["month_name"].dropna().unique(),
                    key=lambda m: months_map_number.get(m, 13)
                )
                if present_months:
                    city_data_full["month_name_cat"] = pd.Categorical(city_data_full["month_name"], categories=present_months, ordered=True)
                    fig_violin = px.violin(
                        city_data_full.sort_values("month_name_cat"), x="month_name_cat", y="index",
                        color="month_name_cat", color_discrete_sequence=px.colors.qualitative.Vivid,
                        box=True, points="outliers", hover_data=["date", "level"],
                        labels={"index": "AQI Index", "month_name_cat": "Month"}
                    )
                    fig_violin.update_layout(
                        height=500, xaxis_title=None, showlegend=False,
                        **get_custom_plotly_layout_args()
                    )
                    st.plotly_chart(fig_violin, use_container_width=True)
                else:
                    st.info("No monthly data available to render violin plot.")
            else:
                st.info("No data available for distribution plots.")

        with tab_heatmap_detail:
            st.markdown("<h5>🔥 AQI Heatmap (Month vs. Day)</h5>", unsafe_allow_html=True)
            if not city_data_full.empty:
                present_months = sorted(
                    city_data_full["month_name"].dropna().unique(),
                    key=lambda m: months_map_number.get(m, 13)
                )
                if present_months:
                    heatmap_pivot = city_data_full.pivot_table(index="month_name", columns="day_of_month", values="index", observed=False)
                    heatmap_pivot = heatmap_pivot.reindex(present_months)

                    if not heatmap_pivot.dropna(how='all').empty:
                        fig_heat_detail = px.imshow(
                            heatmap_pivot, labels=dict(x="Day of Month", y="Month", color="AQI"),
                            aspect="auto", color_continuous_scale="Inferno", text_auto=".0f"
                        )
                        fig_heat_detail.update_layout(
                            height=550, xaxis_side="top",
                            **get_custom_plotly_layout_args()
                        )
                        st.plotly_chart(fig_heat_detail, use_container_width=True)
                    else:
                        st.info("No data available to render heatmap.")
                else:
                    st.info("No monthly data available for heatmap.")
            else:
                st.info("No data available for heatmap.")
        
        with tab_weekday:
            st.markdown("<h5>📅 AQI by Day of the Week</h5>", unsafe_allow_html=True)
            st.info("This chart shows the distribution of AQI values for each day of the week. It can help identify weekly patterns, like differences between weekdays and weekends.")
            
            if not city_data_full.empty:
                fig_weekday = create_weekday_pattern_chart(city_data_full)
                st.plotly_chart(fig_weekday, use_container_width=True)
                
                # Add weekday analysis
                weekday_df = city_data_full.copy()
                weekday_df['day_of_week'] = weekday_df['date'].dt.day_name()
                weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                weekday_df['day_of_week'] = pd.Categorical(weekday_df['day_of_week'], categories=weekday_order, ordered=True)
                
                weekday_avg = weekday_df.groupby('day_of_week')['index'].mean().reset_index()
                
                # Find weekday and weekend averages
                weekday_avg_value = weekday_avg[weekday_avg['day_of_week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]['index'].mean()
                weekend_avg_value = weekday_avg[weekday_avg['day_of_week'].isin(['Saturday', 'Sunday'])]['index'].mean()
                
                st.markdown(f"""
                <div style="background-color: {get_theme_colors()['card_background']}; border-radius: 16px; padding: 1.5rem; border: 1px solid {get_theme_colors()['border']}; margin-top: 1.5rem;">
                    <h6 style="color: {get_theme_colors()['accent']}; margin-bottom: 1rem;">Weekday vs Weekend Analysis</h6>
                    <div style="display: flex; justify-content: space-around;">
                        <div style="text-align: center;">
                            <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                                <b>Weekday Average</b>
                            </p>
                            <p style="font-size: 1.8rem; font-weight: 700; color: {get_theme_colors()['accent']}; margin: 0;">
                                {weekday_avg_value:.1f}
                            </p>
                        </div>
                        <div style="text-align: center;">
                            <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                                <b>Weekend Average</b>
                            </p>
                            <p style="font-size: 1.8rem; font-weight: 700; color: {get_theme_colors()['accent']}; margin: 0;">
                                {weekend_avg_value:.1f}
                            </p>
                        </div>
                    </div>
                    <p style="font-size: 1.1rem; margin-top: 1rem; margin-bottom: 0;">
                        <b>Interpretation:</b> {'Weekdays have higher pollution levels, likely due to increased traffic and industrial activity.' if weekday_avg_value > weekend_avg_value else 'Weekends have higher pollution levels, which might be due to recreational activities or specific local factors.'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No data available for weekday analysis.")

        with tab_health:
            st.markdown("<h5>❤️ Health Impact Analysis</h5>", unsafe_allow_html=True)
            
            # Create health impact scorecard
            fig_health, health_score, impact_level = create_health_impact_scorecard(city_data_full)
            st.plotly_chart(fig_health, use_container_width=True)
            
            # Display health recommendations
            health_card_html = create_health_recommendations_card(current_level)
            st.markdown(health_card_html, unsafe_allow_html=True)
            
            # Create AQI gauge chart
            fig_gauge = create_aqi_gauge_chart(current_aqi, current_level)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Health impact statistics
            col_health1, col_health2 = st.columns(2, gap="large")
            
            with col_health1:
                st.markdown(f"""
                <div style="background-color: {get_theme_colors()['card_background']}; border-radius: 16px; padding: 1.5rem; border: 1px solid {get_theme_colors()['border']}; margin-top: 1.5rem;">
                    <h6 style="color: {get_theme_colors()['accent']}; margin-bottom: 1rem;">Health Impact Statistics</h6>
                    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                        <b>Health Impact Score:</b> {health_score:.1f}/100
                    </p>
                    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                        <b>Impact Level:</b> {impact_level}
                    </p>
                    <p style="font-size: 1.1rem; margin-bottom: 0;">
                        <b>Current Risk Level:</b> {current_level}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_health2:
                # Calculate days in each category
                total_days = len(city_data_full)
                good_days = len(city_data_full[city_data_full['level'] == 'Good'])
                satisfactory_days = len(city_data_full[city_data_full['level'] == 'Satisfactory'])
                moderate_days = len(city_data_full[city_data_full['level'] == 'Moderate'])
                poor_days = len(city_data_full[city_data_full['level'] == 'Poor'])
                very_poor_days = len(city_data_full[city_data_full['level'] == 'Very Poor'])
                severe_days = len(city_data_full[city_data_full['level'] == 'Severe'])
                
                st.markdown(f"""
                <div style="background-color: {get_theme_colors()['card_background']}; border-radius: 16px; padding: 1.5rem; border: 1px solid {get_theme_colors()['border']}; margin-top: 1.5rem;">
                    <h6 style="color: {get_theme_colors()['accent']}; margin-bottom: 1rem;">Days by AQI Category</h6>
                    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                        <b>Good:</b> {good_days} days ({(good_days/total_days)*100:.1f}%)
                    </p>
                    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                        <b>Satisfactory:</b> {satisfactory_days} days ({(satisfactory_days/total_days)*100:.1f}%)
                    </p>
                    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                        <b>Moderate:</b> {moderate_days} days ({(moderate_days/total_days)*100:.1f}%)
                    </p>
                    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                        <b>Poor:</b> {poor_days} days ({(poor_days/total_days)*100:.1f}%)
                    </p>
                    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                        <b>Very Poor:</b> {very_poor_days} days ({(very_poor_days/total_days)*100:.1f}%)
                    </p>
                    <p style="font-size: 1.1rem; margin-bottom: 0;">
                        <b>Severe:</b> {severe_days} days ({(severe_days/total_days)*100:.1f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
        with tab_forecast:
            st.markdown("<h5>🔮 AQI Forecast</h5>", unsafe_allow_html=True)
            
            if len(city_data_full) >= 15:
                forecast_data, degree, mae = generate_forecast(city_data_full)
                
                if forecast_data is not None:
                    # Display forecast chart
                    fig_forecast = create_aqi_forecast_chart(city_data_full, forecast_data)
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Display forecast statistics
                    st.markdown(f"""
                    <div style="background-color: {get_theme_colors()['card_background']}; border-radius: 16px; padding: 1.5rem; border: 1px solid {get_theme_colors()['border']}; margin-top: 1.5rem;">
                        <h6 style="color: {get_theme_colors()['accent']}; margin-bottom: 1rem;">Forecast Model Statistics</h6>
                        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                            <b>Model Type:</b> Polynomial Regression (Degree {degree})
                        </p>
                        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                            <b>Mean Absolute Error:</b> {mae:.2f}
                        </p>
                        <p style="font-size: 1.1rem; margin-bottom: 0;">
                            <b>Forecast Period:</b> Next 7 days
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display forecast table
                    st.markdown("<h6>7-Day Forecast</h6>", unsafe_allow_html=True)
                    forecast_display = forecast_data.copy()
                    forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m-%d')
                    forecast_display['forecast'] = forecast_display['forecast'].round(1)
                    forecast_display['lower_bound'] = forecast_display['lower_bound'].round(1)
                    forecast_display['upper_bound'] = forecast_display['upper_bound'].round(1)
                    forecast_display['category'] = forecast_display['forecast'].apply(get_category)
                    
                    st.dataframe(
                        forecast_display[['date', 'forecast', 'lower_bound', 'upper_bound', 'category']].rename(columns={
                            'date': 'Date',
                            'forecast': 'Forecast AQI',
                            'lower_bound': 'Lower Bound',
                            'upper_bound': 'Upper Bound',
                            'category': 'AQI Category'
                        }),
                        use_container_width=True
                    )
                else:
                    st.error("Unable to generate forecast. Please try again with more data.")
            else:
                st.warning("Not enough data available for forecasting. Need at least 15 days of data.")
                
            # AQI prediction model
            st.markdown("<h5>🤖 Advanced AQI Prediction Model</h5>", unsafe_allow_html=True)
            
            if len(city_data_full) >= 50:
                model_result = create_aqi_prediction_model(city_data_full)
                
                if model_result is not None:
                    fig_model, model_mae = model_result
                    st.plotly_chart(fig_model, use_container_width=True)
                    
                    st.markdown(f"""
                    <div style="background-color: {get_theme_colors()['card_background']}; border-radius: 16px; padding: 1.5rem; border: 1px solid {get_theme_colors()['border']}; margin-top: 1.5rem;">
                        <h6 style="color: {get_theme_colors()['accent']}; margin-bottom: 1rem;">Model Performance</h6>
                        <p style="font-size: 1.1rem; margin-bottom: 0;">
                            <b>Model Mean Absolute Error:</b> {model_mae:.2f}
                        </p>
                        <p style="font-size: 1.1rem; margin-bottom: 0;">
                            <b>Interpretation:</b> On average, the model's predictions are off by {model_mae:.2f} AQI points.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Unable to create prediction model. Please try again with more data.")
            else:
                st.warning("Not enough data available for advanced prediction modeling. Need at least 50 days of data.")
                
        with tab_comparison:
            st.markdown("<h5>🏙️ City Comparison</h5>", unsafe_allow_html=True)
            
            if len(selected_cities) > 1:
                # Create comparison chart
                fig_comparison = create_comparison_chart(df_period_filtered, selected_cities)
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Create radar chart
                fig_radar = create_city_comparison_radar(df_period_filtered, selected_cities)
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Create monthly comparison chart
                fig_monthly = create_monthly_comparison_chart(df_period_filtered, selected_cities)
                st.plotly_chart(fig_monthly, use_container_width=True)
                
                # Create animated pollution trend chart
                fig_animation = create_pollution_trend_animation(df_period_filtered)
                st.plotly_chart(fig_animation, use_container_width=True)
            else:
                st.info("Please select at least two cities to enable comparison features.")
                
        with tab_sources:
            st.markdown("<h5>🏭 Pollution Sources Analysis</h5>", unsafe_allow_html=True)
            
            # Create pollutant breakdown chart
            fig_pollutant = create_pollutant_breakdown_chart(city_data_full)
            st.plotly_chart(fig_pollutant, use_container_width=True)
            
            # Create pollution sources Sankey diagram
            fig_sankey = create_pollution_sources_sankey(city_data_full)
            st.plotly_chart(fig_sankey, use_container_width=True)
            
            # Display pollutant statistics
            col_source1, col_source2 = st.columns(2, gap="large")
            
            with col_source1:
                # Calculate dominant pollutant
                dominant_pollutant = city_data_full['pollutant'].value_counts().index[0]
                dominant_count = city_data_full['pollutant'].value_counts().iloc[0]
                dominant_percentage = (dominant_count / len(city_data_full)) * 100
                
                st.markdown(f"""
                <div style="background-color: {get_theme_colors()['card_background']}; border-radius: 16px; padding: 1.5rem; border: 1px solid {get_theme_colors()['border']}; margin-top: 1.5rem;">
                    <h6 style="color: {get_theme_colors()['accent']}; margin-bottom: 1rem;">Dominant Pollutant</h6>
                    <p style="font-size: 1.8rem; font-weight: 700; color: {get_theme_colors()['pollutant_colors'].get(dominant_pollutant, '#FFFFFF')}; margin: 1rem 0;">
                        {dominant_pollutant}
                    </p>
                    <p style="font-size: 1.1rem; margin-bottom: 0;">
                        <b>Frequency:</b> {dominant_count} days ({dominant_percentage:.1f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_source2:
                # Calculate average AQI by pollutant
                pollutant_avg_aqi = city_data_full.groupby('pollutant')['index'].mean().sort_values(ascending=False)
                worst_pollutant = pollutant_avg_aqi.index[0]
                worst_aqi = pollutant_avg_aqi.iloc[0]
                
                st.markdown(f"""
                <div style="background-color: {get_theme_colors()['card_background']}; border-radius: 16px; padding: 1.5rem; border: 1px solid {get_theme_colors()['border']}; margin-top: 1.5rem;">
                    <h6 style="color: {get_theme_colors()['accent']}; margin-bottom: 1rem;">Most Harmful Pollutant</h6>
                    <p style="font-size: 1.8rem; font-weight: 700; color: {get_theme_colors()['pollutant_colors'].get(worst_pollutant, '#FFFFFF')}; margin: 1rem 0;">
                        {worst_pollutant}
                    </p>
                    <p style="font-size: 1.1rem; margin-bottom: 0;">
                        <b>Average AQI:</b> {worst_aqi:.1f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
        with tab_weather:
            st.markdown("<h5>🌤️ Weather Correlation Analysis</h5>", unsafe_allow_html=True)
            
            # Create weather correlation chart
            fig_weather = create_weather_correlation_chart(city_data_full)
            st.plotly_chart(fig_weather, use_container_width=True)
            
            # Create diurnal pattern chart
            fig_diurnal = create_diurnal_pattern_chart(city_data_full)
            st.plotly_chart(fig_diurnal, use_container_width=True)
            
            # Display weather correlation information
            st.markdown(f"""
            <div style="background-color: {get_theme_colors()['card_background']}; border-radius: 16px; padding: 1.5rem; border: 1px solid {get_theme_colors()['border']}; margin-top: 1.5rem;">
                <h6 style="color: {get_theme_colors()['accent']}; margin-bottom: 1rem;">Weather Impact Analysis</h6>
                <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                    <b>Temperature Correlation:</b> Higher temperatures can lead to increased ozone formation.
                </p>
                <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                    <b>Humidity Correlation:</b> High humidity can trap pollutants near the ground.
                </p>
                <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                    <b>Wind Speed Correlation:</b> Higher wind speeds can disperse pollutants.
                </p>
                <p style="font-size: 1.1rem; margin-bottom: 0;">
                    <b>Diurnal Pattern:</b> AQI typically peaks during morning and evening rush hours.
                </p>
            </div>
            """, unsafe_allow_html=True)

# ========================================================
# ================  FOOTER SECTION  ====================
# ========================================================
st.markdown(f"""
<div class="footer-container">
    <h3>📊 IIT KGP Air Quality Dashboard</h3>
    <div class="footer-info">
        <p>
            <span class="label">Data Source:</span>
            <span class="value">CPCB India</span>
        </p>
        <p>
            <span class="label">Last Updated:</span>
            <span class="value">{data_last_updated.strftime('%Y-%m-%d %H:%M:%S') if data_last_updated else 'Unknown'}</span>
        </p>
        <p>
            <span class="label">Total Records:</span>
            <span class="value">{format_number(len(df))}</span>
        </p>
    </div>
    <div class="footer-links">
        <a href="https://cpcb.nic.in/" target="_blank">
            <span>CPCB Official Website</span>
            <span>→</span>
        </a>
        <a href="https://www.iitkgp.ac.in/" target="_blank">
            <span>IIT Kharagpur</span>
            <span>→</span>
        </a>
    </div>
    <p class="copyright">
        © 2024 IIT Kharagpur. All rights reserved.
    </p>
</div>
""", unsafe_allow_html=True)

# Export functionality
if export_data_list:
    export_df = pd.concat(export_data_list)
    
    # Create a download button for the filtered data
    csv = export_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="air_quality_data.csv" style="text-decoration:none;"><button style="background: linear-gradient(90deg, {get_theme_colors()["accent"]}, #00BFA5); color: {get_theme_colors()["background"]}; border: none; font-weight: 600; padding: 0.75rem 2rem; border-radius: 50px; transition: all 0.3s ease; font-size: 1rem; box-shadow: 0 4px 10px rgba(0, 188, 212, 0.3); cursor: pointer;">Download Data as CSV</button></a>'
    st.markdown(href, unsafe_allow_html=True)

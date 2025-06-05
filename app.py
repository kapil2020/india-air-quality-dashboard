import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# --- Global Theme & Style Setup ---
pio.templates.default = "plotly_dark"

ACCENT_COLOR = "#00BCD4"                 # Vibrant Teal
TEXT_COLOR = "#EAEAEA"                   # Light gray text
SUBTLE_TEXT = "#B0B0B0"                  # Subtle gray
BG_COLOR = "#121212"                     # Very dark gray (almost black)
CARD_BG = "#1E1E1E"                      # Slightly lighter gray for cards
BORDER_COLOR = "#333333"                 # Dark border

CATEGORY_COLORS = {
    "Severe": "#F44336",
    "Very Poor": "#FF7043",
    "Poor": "#FFA726",
    "Moderate": "#FFEE58",
    "Satisfactory": "#9CCC65",
    "Good": "#4CAF50",
    "Unknown": "#444444"
}

POLLUTANT_COLORS = {
    "PM2.5": "#FF6B6B",
    "PM10": "#4ECDC4",
    "NO2": "#45B7D1",
    "SO2": "#F9C74F",
    "CO": "#F8961E",
    "O3": "#90BE6D",
    "Other": "#B0BEC5"
}

# ------------------- Page Config -------------------
st.set_page_config(
    layout="wide",
    page_title="IIT KGP AQI Dashboard",
    page_icon="🌬️"
)

# ------------------- Custom CSS Styling (Dark Theme) -------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Base background & text */
    body {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
        font-family: 'Inter', sans-serif;
    }}

    /* Container padding */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }}

    /* Card / Chart styling */
    .stPlotlyChart, .stDataFrame, .stAlert, .stMetric,
    .stDownloadButton > button, .stButton > button,
    div[data-testid="stExpander"], div[data-testid="stForm"] {{
        background-color: {CARD_BG} !important;
        border: 1px solid {BORDER_COLOR} !important;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }}

    .stpyplot {{
        background-color: {CARD_BG} !important;
        border: 1px solid {BORDER_COLOR} !important;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }}

    /* Tab bar styling */
    .stTabs [data-baseweb="tab-list"] {{
         background-color: transparent;
         border-bottom: 2px solid {BORDER_COLOR};
         box-shadow: none;
         padding-bottom: 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        color: {SUBTLE_TEXT} !important;
        font-weight: 600;
        padding: 0.8rem 1.5rem;
        border-radius: 8px 8px 0 0;
        margin-right: 0.5rem;
        transition: background-color 0.3s ease, color 0.3s ease;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {BG_COLOR};
        color: {ACCENT_COLOR} !important;
        border-bottom: 3px solid {ACCENT_COLOR};
    }}

    /* Headings */
    h1 {{
        color: {TEXT_COLOR};
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        letter-spacing: -1px;
    }}
    h2 {{
        color: {ACCENT_COLOR};
        border-bottom: 2px solid {ACCENT_COLOR};
        padding-bottom: 0.6rem;
        margin-top: 3rem;
        margin-bottom: 1.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    h3 {{
        color: {TEXT_COLOR};
        margin-top: 0rem;
        margin-bottom: 1.2rem;
        font-weight: 600;
    }}
    h4, h5 {{
        color: {TEXT_COLOR};
        margin-top: 0.2rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }}

    /* Sidebar styling */
    .stSidebar {{
        background-color: {CARD_BG};
        border-right: 1px solid {BORDER_COLOR};
        padding: 1.5rem;
    }}
    .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3, .stSidebar .stMarkdown p {{
        color: {TEXT_COLOR};
        text-align: left;
    }}
    .stSidebar .stSelectbox label, .stSidebar .stMultiselect label {{
        color: {ACCENT_COLOR} !important;
        font-weight: 600;
    }}

    /* Metric styling */
    .stMetric {{
        background-color: {BG_COLOR} !important;
        border: 1px solid {BORDER_COLOR} !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }}
    .stMetric > div:nth-child(1) {{
        font-size: 1rem;
        color: {SUBTLE_TEXT};
        font-weight: 500;
    }}
    .stMetric > div:nth-child(2) {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {ACCENT_COLOR};
    }}

    /* Expander styling */
    div[data-testid="stExpander"] summary {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {ACCENT_COLOR};
    }}
    div[data-testid="stExpander"] svg {{
        fill: {ACCENT_COLOR};
    }}

    /* Download button */
    .stDownloadButton button {{
        background-color: {ACCENT_COLOR} !important;
        color: {BG_COLOR} !important;
        border: none !important;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: background-color 0.3s ease;
    }}
    .stDownloadButton button:hover {{
        opacity: 0.85;
    }}

    /* Input widgets */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div:first-child,
    .stMultiselect div[data-baseweb="select"] > div:first-child {{
        background-color: {BG_COLOR} !important;
        color: {TEXT_COLOR} !important;
        border-color: {BORDER_COLOR} !important;
    }}
    .stDateInput input {{
        background-color: {BG_COLOR} !important;
        color: {TEXT_COLOR} !important;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: {BG_COLOR};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {BORDER_COLOR};
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {ACCENT_COLOR};
    }}
    </style>
    """, unsafe_allow_html=True
)

# --------------------------------------------
# Helper: Common Plotly layout arguments
# --------------------------------------------
def get_layout_args(height: int = None, title: str = None) -> dict:
    layout = {
        "font": {"family": "Inter", "color": TEXT_COLOR},
        "paper_bgcolor": CARD_BG,
        "plot_bgcolor": CARD_BG,
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        "margin": {"t": 50, "b": 40, "l": 40, "r": 40}
    }
    if height:
        layout["height"] = height
    if title:
        layout["title_text"] = title
        layout["title_font"] = {"color": ACCENT_COLOR, "size": 16, "family": "Inter"}
    return layout

# ------------------- Title & Subtitle -------------------
st.markdown("<h1>🌬️ IIT KGP AQI Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p style='text-align: center; color: {SUBTLE_TEXT}; font-size: 1.1rem; margin-bottom: 2.5rem;'>"
    f"Illuminating Air Quality Insights Across India"
    f"</p>",
    unsafe_allow_html=True
)

# ------------------- Load Data -------------------
@st.cache_data(ttl=3600)
def load_data():
    today = pd.to_datetime("today").date()
    csv_path = f"data/{today}.csv"
    fallback = "combined_air_quality.txt"

    df_loaded = None
    is_today = False
    msg = ""
    updated_time = None

    # Try today's CSV first
    if os.path.exists(csv_path):
        try:
            df_loaded = pd.read_csv(csv_path)
            if "date" in df_loaded.columns:
                df_loaded["date"] = pd.to_datetime(df_loaded["date"])
                is_today = True
                msg = f"Live data from: **{today}.csv**"
                updated_time = pd.Timestamp(os.path.getmtime(csv_path), unit="s")
            else:
                msg = f"Warning: '{today}.csv' found but missing 'date'. Using fallback."
        except Exception as e:
            msg = f"Error loading '{today}.csv': {e}. Using fallback."

    # If not valid, load fallback
    if df_loaded is None or not is_today:
        if not os.path.exists(fallback):
            st.error(f"FATAL: '{fallback}' not found.")
            return pd.DataFrame(), "Error: Main data file missing.", None
        try:
            df_loaded = pd.read_csv(fallback, sep="\t", parse_dates=["date"])
            base_msg = f"Displaying archive data from: **{fallback}**"
            msg = base_msg if not msg or is_today else msg + " " + base_msg
            updated_time = pd.Timestamp(os.path.getmtime(fallback), unit="s")
        except Exception as e:
            st.error(f"FATAL: Error loading '{fallback}': {e}.")
            return pd.DataFrame(), f"Error loading fallback: {e}", None

    # Post-process
    for col, default in [("pollutant", np.nan), ("level", "Unknown")]:
        if col not in df_loaded.columns:
            df_loaded[col] = default

    df_loaded["pollutant"] = (
        df_loaded["pollutant"].astype(str)
        .str.split(",").str[0].str.strip()
        .replace(["nan", "NaN", "None", ""], np.nan)
    ).fillna("Other")

    df_loaded["level"] = df_loaded["level"].astype(str).fillna("Unknown")

    return df_loaded, msg, updated_time

df, load_msg, last_update = load_data()

if df.empty:
    st.error("Dashboard cannot operate without data. Please check data sources.")
    st.stop()

if last_update:
    st.caption(
        f"<p style='text-align: center; color: {SUBTLE_TEXT}; font-size: 0.85rem;'>"
        f"Last data update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}"
        "</p>",
        unsafe_allow_html=True
    )

# ================= Sidebar Filters =================
st.sidebar.header("🔭 EXPLORATION CONTROLS")
st.sidebar.markdown("---")
st.sidebar.info("Fetching real-time data from CPCB. Today's data is available after 5:45 PM IST.")

unique_cities = sorted(df["city"].unique()) if "city" in df.columns else []
default_city = ["Delhi"] if "Delhi" in unique_cities else (unique_cities[:1] if unique_cities else [])
selected_cities = st.sidebar.multiselect("🏙️ Select Cities", unique_cities, default=default_city)

years = sorted(df["date"].dt.year.unique())
default_year = max(years) if years else None
year_idx = years.index(default_year) if default_year else 0
year = st.sidebar.selectbox("🗓️ Select Year", years, index=year_idx if years else 0)

months_map = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
}
month_list = ["All Months"] + list(months_map.values())
sel_month = st.sidebar.selectbox("🌙 Select Month (Optional)", month_list, index=0)

month_num = None
if sel_month != "All Months":
    month_num = [k for k, v in months_map.items() if v == sel_month][0]

# Filter based on year & month
df_filtered = df[df["date"].dt.year == year].copy()
if month_num:
    df_filtered = df_filtered[df_filtered["date"].dt.month == month_num]

# ================= National Snapshot =================
st.markdown("## 🇮🇳 National Snapshot")
with st.container():
    if year:
        # Key Metro Annual Average AQI
        st.markdown(f"##### Key Metro Annual Average AQI ({year})")
        major_cities = ["Delhi", "Mumbai", "Kolkata", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Ahmedabad"]
        df_major = df[df["date"].dt.year == year]
        df_major = df_major[df_major["city"].isin(major_cities)]

        if not df_major.empty:
            avg_aqi = df_major.groupby("city")["index"].mean().dropna()
            available = [c for c in major_cities if c in avg_aqi.index]
            if available:
                cols = st.columns(min(len(available), 4))
                for i, city_name in enumerate(available):
                    with cols[i % len(cols)]:
                        st.metric(label=city_name, value=f"{avg_aqi[city_name]:.1f}")
            else:
                st.info(f"No annual AQI data for the key metro cities in {year}.")
        else:
            st.info(f"No data available for the key metro cities in {year}.")

        # General Insights
        st.markdown(f"##### General Insights for {sel_month}, {year}")
        if not df_filtered.empty:
            avg_national = df_filtered["index"].mean()
            city_stats = df_filtered.groupby("city")["index"].mean().dropna()

            if not city_stats.empty:
                n_cities = df_filtered["city"].nunique()
                best_city, best_aqi = city_stats.idxmin(), city_stats.min()
                worst_city, worst_aqi = city_stats.idxmax(), city_stats.max()
                st.markdown(
                    f"""
                    <div style="font-size:1.05rem; line-height:1.7;">
                    Across <b>{n_cities}</b> observed cities, <b style="color:{ACCENT_COLOR};">{avg_national:.2f}</b> is the average AQI.
                    Best city: <b style="color:{CATEGORY_COLORS['Good']};">{best_city}</b> ({best_aqi:.2f}). 
                    Most challenged: <b style="color:{CATEGORY_COLORS['Severe']};">{worst_city}</b> ({worst_aqi:.2f}).
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("Could not compute AQI stats for the selected period.")
        else:
            st.info("No data available for the selected period.")
    else:
        st.warning("Please select a year to view insights.")
st.markdown("---")

# ================= City-Specific Deep Dive =================
export_list = []

if not selected_cities:
    st.info("✨ Select one or more cities from the sidebar to dive into detailed analysis.")
else:
    for city in selected_cities:
        st.markdown(f"## 🏙️ {city.upper()} DEEP DIVE – {year}")
        city_df = df_filtered[df_filtered["city"] == city].copy()
        curr_period = f"{sel_month}, {year}"

        if city_df.empty:
            st.warning(f"😔 No data for {city} in {curr_period}. Try different filters.")
            continue

        city_df["day_of_year"] = city_df["date"].dt.dayofyear
        city_df["month_name"] = city_df["date"].dt.month_name()
        city_df["day_of_month"] = city_df["date"].dt.day
        export_list.append(city_df)

        tab_trend, tab_dist, tab_heat = st.tabs(
            ["📊 TRENDS & CALENDAR", "📈 DISTRIBUTIONS", "🗓️ DETAILED HEATMAP"]
        )

        # ----------------- Trends & Calendar -----------------
        with tab_trend:
            st.markdown("##### 📅 Daily AQI Calendar")
            # Build a complete year calendar frame
            start_date = pd.to_datetime(f"{year}-01-01")
            end_date = pd.to_datetime(f"{year}-12-31")
            full_dates = pd.date_range(start_date, end_date)
            cal_df = pd.DataFrame({"date": full_dates})
            cal_df["week"] = cal_df["date"].dt.isocalendar().week
            cal_df["day_of_week"] = cal_df["date"].dt.dayofweek

            # Fix week numbering for Jan and Dec
            cal_df.loc[(cal_df["date"].dt.month == 1) & (cal_df["week"] > 50), "week"] = 0
            cal_df.loc[(cal_df["date"].dt.month == 12) & (cal_df["week"] == 1), "week"] = 53

            merged = pd.merge(
                cal_df,
                city_df[["date", "index", "level"]],
                on="date",
                how="left"
            )
            merged["level"] = merged["level"].fillna("Unknown")
            merged["aqi_text"] = merged["index"].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
            )

            # Identify first week of each month to annotate month labels exactly once
            month_first_weeks = (
                cal_df.groupby(cal_df["date"].dt.month)
                .agg(first_week=("week", "min"))
                .reset_index()
            )
            # Build a mapping: week number -> short month name (Jan, Feb, ...)
            month_annot = {
                row.first_week: pd.to_datetime(f"{year}-{int(row.date):02d}-01").strftime("%b")
                for _, row in month_first_weeks.rename(columns={"date": "month"}).iterrows()
            }

            # Plot heatmap
            fig_cal = go.Figure(
                data=go.Heatmap(
                    x=merged["week"],
                    y=merged["day_of_week"],
                    z=merged["level"].map({k: i for i, k in enumerate(CATEGORY_COLORS.keys())}),
                    customdata=pd.DataFrame({
                        "date": merged["date"].dt.strftime("%Y-%m-%d"),
                        "level": merged["level"],
                        "aqi": merged["aqi_text"]
                    }),
                    hovertemplate="<b>%{customdata[0]}</b><br>AQI: %{customdata[2]} (%{customdata[1]})<extra></extra>",
                    colorscale=[
                        [i/(len(CATEGORY_COLORS)-1), c] for i, c in enumerate(CATEGORY_COLORS.values())
                    ],
                    showscale=False,
                    xgap=2, ygap=2
                )
            )

            # Y-axis labels (Mon–Sun)
            day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

            # Build annotations: 12 month names and legend squares
            annotations = []
            # 1) Month labels at top (y=7)
            for wk, mname in month_annot.items():
                annotations.append(
                    go.layout.Annotation(
                        text=mname,
                        align="center",
                        showarrow=False,
                        xref="x", yref="y",
                        x=wk, y=7,
                        font=dict(color=SUBTLE_TEXT, size=12)
                    )
                )

            # 2) Legend keys at bottom (paper coords)
            for i, (lvl, clr) in enumerate(CATEGORY_COLORS.items()):
                # two rows of legend
                row_idx = i // 7
                col_idx = i % 7
                annotations.append(
                    go.layout.Annotation(
                        text=f"█ <span style='color:{TEXT_COLOR};'>{lvl}</span>",
                        align="left",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.05 + 0.12 * col_idx,
                        y=-0.1 - 0.1 * row_idx,
                        xanchor="left", yanchor="top",
                        font=dict(color=clr, size=12)
                    )
                )

            fig_cal.update_layout(
                yaxis=dict(
                    tickmode="array",
                    tickvals=list(range(7)),
                    ticktext=day_labels,
                    showgrid=False, zeroline=False
                ),
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    tickmode="array",
                    ticktext=[],
                    tickvals=[]
                ),
                height=300,
                margin=dict(t=40, b=80, l=40, r=40),
                plot_bgcolor=CARD_BG,
                paper_bgcolor=CARD_BG,
                font_color=TEXT_COLOR,
                annotations=annotations,
                showlegend=False
            )
            st.plotly_chart(fig_cal, use_container_width=True)

            # ---------------- AQI Trend & 7-Day Rolling ----------------
            st.markdown("##### 📈 AQI Trend & 7-Day Rolling Average")
            city_sorted = city_df.sort_values("date").copy()
            city_sorted["rolling_avg"] = city_sorted["index"].rolling(
                window=7, center=True, min_periods=1
            ).mean().round(2)

            fig_trend = go.Figure()
            fig_trend.add_trace(
                go.Scatter(
                    x=city_sorted["date"],
                    y=city_sorted["index"],
                    mode="lines+markers",
                    name="Daily AQI",
                    marker=dict(size=4, opacity=0.7, color=SUBTLE_TEXT),
                    line=dict(width=1.5, color=SUBTLE_TEXT),
                    customdata=city_sorted[["level"]],
                    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>AQI: %{y}<br>Category: %{customdata[0]}<extra></extra>"
                )
            )
            fig_trend.add_trace(
                go.Scatter(
                    x=city_sorted["date"],
                    y=city_sorted["rolling_avg"],
                    mode="lines",
                    name="7-Day Rolling Avg",
                    line=dict(color=ACCENT_COLOR, width=2.5, dash="dash"),
                    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>7-Day Avg AQI: %{y}<extra></extra>"
                )
            )
            fig_trend.update_layout(
                yaxis_title="AQI Index",
                xaxis_title="Date",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
                plot_bgcolor=BG_COLOR,
                paper_bgcolor=CARD_BG,
                font_color=TEXT_COLOR
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        # ------------- Distributions -----------------
        with tab_dist:
            col_bar, col_sun = st.columns(2)

            with col_bar:
                st.markdown("##### 📊 AQI Category (Days)")
                cat_cnt = (
                    city_df["level"]
                    .value_counts()
                    .reindex(CATEGORY_COLORS.keys(), fill_value=0)
                    .reset_index()
                )
                cat_cnt.columns = ["AQI Category", "Number of Days"]
                fig_bar = px.bar(
                    cat_cnt,
                    x="AQI Category",
                    y="Number of Days",
                    color="AQI Category",
                    color_discrete_map=CATEGORY_COLORS,
                    text_auto=True
                )
                fig_bar.update_layout(
                    height=400,
                    xaxis_title=None,
                    yaxis_title="Number of Days",
                    plot_bgcolor=BG_COLOR,
                    paper_bgcolor=CARD_BG,
                    font_color=TEXT_COLOR
                )
                fig_bar.update_traces(
                    textfont_size=12,
                    textposition="outside",
                    cliponaxis=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_sun:
                st.markdown("##### ☀️ AQI Category (Proportions)")
                if cat_cnt["Number of Days"].sum() > 0:
                    fig_sun = px.sunburst(
                        cat_cnt,
                        path=["AQI Category"],
                        values="Number of Days",
                        color="AQI Category",
                        color_discrete_map=CATEGORY_COLORS
                    )
                    fig_sun.update_layout(
                        height=400,
                        margin=dict(t=20, l=20, r=20, b=20),
                        plot_bgcolor=BG_COLOR,
                        paper_bgcolor=CARD_BG,
                        font_color=TEXT_COLOR
                    )
                    st.plotly_chart(fig_sun, use_container_width=True)
                else:
                    st.caption("No data for sunburst chart.")

            st.markdown("##### 🎻 Monthly AQI Distribution")
            month_order = list(months_map.values())
            city_df["month_name"] = pd.Categorical(
                city_df["month_name"], categories=month_order, ordered=True
            )
            fig_violin = px.violin(
                city_df.sort_values("month_name"),
                x="month_name",
                y="index",
                color="month_name",
                color_discrete_sequence=px.colors.qualitative.Vivid,
                box=True,
                points="outliers",
                labels={"index": "AQI Index", "month_name": "Month"},
                hover_data=["date", "level"]
            )
            fig_violin.update_layout(
                height=450,
                xaxis_title=None,
                showlegend=False,
                plot_bgcolor=BG_COLOR,
                paper_bgcolor=CARD_BG,
                font_color=TEXT_COLOR
            )
            fig_violin.update_traces(meanline_visible=True)
            st.plotly_chart(fig_violin, use_container_width=True)

        # ------------- Detailed Heatmap -----------------
        with tab_heat:
            st.markdown("##### 🔥 AQI Heatmap (Month vs. Day)")
            pivot = city_df.pivot_table(
                index="month_name",
                columns="day_of_month",
                values="index",
                observed=False
            )
            pivot = pivot.reindex(month_order)
            fig_heat = px.imshow(
                pivot,
                labels=dict(x="Day of Month", y="Month", color="AQI"),
                aspect="auto",
                color_continuous_scale="Inferno",
                text_auto=".0f"
            )
            fig_heat.update_layout(
                height=500,
                xaxis_side="top",
                plot_bgcolor=BG_COLOR,
                paper_bgcolor=CARD_BG,
                font_color=TEXT_COLOR
            )
            fig_heat.update_traces(
                hovertemplate="<b>Month:</b> %{y}<br><b>Day:</b> %{x}<br><b>AQI:</b> %{z}<extra></extra>"
            )
            st.plotly_chart(fig_heat, use_container_width=True)

# ================= City-Wise Comparison =================
if len(selected_cities) > 1:
    st.markdown("## 🆚 AQI COMPARISON ACROSS CITIES")
    comp_list = []
    for c in selected_cities:
        temp = df_filtered[df_filtered["city"] == c].copy()
        if not temp.empty:
            temp = temp.sort_values("date")
            temp["city_label"] = c
            comp_list.append(temp)

    if comp_list:
        df_comp = pd.concat(comp_list)
        fig_cmp = px.line(
            df_comp,
            x="date",
            y="index",
            color="city_label",
            labels={"index": "AQI Index", "date": "Date", "city_label": "City"},
            markers=False,
            line_shape="spline",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_cmp.update_layout(
            title_text=f"AQI Trends Comparison – {sel_month}, {year}",
            height=500,
            legend_title_text="City",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor=BG_COLOR,
            paper_bgcolor=CARD_BG,
            font_color=TEXT_COLOR
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.info("Not enough data or cities selected for comparison.")
st.markdown("---")

# ================= Prominent Pollutant Analysis =================
st.markdown("## 💨 PROMINENT POLLUTANT ANALYSIS")
with st.container():
    st.markdown("#### 🔍 Yearly Dominant Pollutant Trends")
    poll_city_A = st.selectbox(
        "Select City for Yearly Pollutant Trend:", unique_cities,
        key="pollutant_A", index=unique_cities.index(default_city[0]) if default_city and default_city[0] in unique_cities else 0
    )
    tempA = df[df["city"] == poll_city_A].copy()
    tempA["year_label"] = tempA["date"].dt.year

    if not tempA.empty:
        gpA = tempA.groupby(["year_label", "pollutant"]).size().unstack(fill_value=0)
        pctA = gpA.apply(lambda x: x / x.sum() * 100 if x.sum() > 0 else x, axis=1).fillna(0)
        pctA_long = pctA.reset_index().melt(
            id_vars="year_label", var_name="pollutant", value_name="percentage"
        )

        figA = px.bar(
            pctA_long,
            x="year_label",
            y="percentage",
            color="pollutant",
            title=f"Dominant Pollutants Over Years – {poll_city_A}",
            labels={"percentage": "Percentage of Days (%)", "year_label": "Year"},
            color_discrete_map=POLLUTANT_COLORS
        )
        figA.update_layout(
            xaxis_type="category",
            yaxis_ticksuffix="%",
            height=500,
            plot_bgcolor=BG_COLOR,
            paper_bgcolor=CARD_BG,
            font_color=TEXT_COLOR
        )
        figA.update_traces(texttemplate="%{value:.1f}%", textposition="auto")
        st.plotly_chart(figA, use_container_width=True)
    else:
        st.warning(f"No pollutant data for {poll_city_A} (Yearly).")

with st.container():
    st.markdown(f"#### ⛽ Dominant Pollutants for {sel_month}, {year}")
    poll_city_B = st.selectbox(
        "Select City for Filtered Pollutant View:", unique_cities,
        key="pollutant_B", index=unique_cities.index(default_city[0]) if default_city and default_city[0] in unique_cities else 0
    )
    tempB = df_filtered[df_filtered["city"] == poll_city_B].copy()
    if not tempB.empty and "pollutant" in tempB.columns:
        gpB = tempB.groupby("pollutant").size().reset_index(name="count")
        totalB = gpB["count"].sum()
        gpB["percentage"] = (gpB["count"] / totalB * 100).round(1) if totalB > 0 else 0

        figB = px.bar(
            gpB,
            x="pollutant",
            y="percentage",
            color="pollutant",
            title=f"Dominant Pollutants – {poll_city_B} ({sel_month}, {year})",
            labels={"percentage": "Percentage of Days (%)", "pollutant": "Pollutant"},
            color_discrete_map=POLLUTANT_COLORS
        )
        figB.update_layout(
            yaxis_ticksuffix="%",
            height=450,
            plot_bgcolor=BG_COLOR,
            paper_bgcolor=CARD_BG,
            font_color=TEXT_COLOR
        )
        figB.update_traces(texttemplate="%{value:.1f}%", textposition="auto")
        st.plotly_chart(figB, use_container_width=True)
    else:
        st.warning(f"No pollutant data for {poll_city_B} for the selected period.")

# ================= AQI Forecast =================
st.markdown("## 🔮 AQI FORECAST (LINEAR TREND)")
with st.container():
    forecast_city = st.selectbox(
        "Select City for AQI Forecast:", unique_cities,
        key="forecast_city", index=unique_cities.index(default_city[0]) if default_city and default_city[0] in unique_cities else 0
    )
    src = df_filtered[df_filtered["city"] == forecast_city].copy()
    if len(src) >= 15:
        fc_df = src.sort_values("date")[["date", "index"]].dropna()
        if len(fc_df) >= 2:
            fc_df["days_since_start"] = (fc_df["date"] - fc_df["date"].min()).dt.days
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(fc_df[["days_since_start"]], fc_df["index"])
            max_day = fc_df["days_since_start"].max()
            future_range = np.arange(0, max_day + 15 + 1)
            future_pred = model.predict(pd.DataFrame({"days_since_start": future_range}))

            base_date = fc_df["date"].min()
            future_dates = [base_date + pd.Timedelta(days=int(i)) for i in future_range]

            plot_obs = pd.DataFrame({"date": fc_df["date"], "AQI": fc_df["index"]})
            plot_fcst = pd.DataFrame({"date": future_dates, "AQI": np.maximum(0, future_pred)})

            fig_fcst = go.Figure()
            fig_fcst.add_trace(
                go.Scatter(
                    x=plot_obs["date"],
                    y=plot_obs["AQI"],
                    mode="lines+markers",
                    name="Observed AQI",
                    line=dict(color=ACCENT_COLOR)
                )
            )
            fig_fcst.add_trace(
                go.Scatter(
                    x=plot_fcst["date"],
                    y=plot_fcst["AQI"],
                    mode="lines",
                    name="Forecast",
                    line=dict(dash="dash", color="#FF6B6B")
                )
            )
            fig_fcst.update_layout(
                title=f"AQI Forecast – {forecast_city}",
                yaxis_title="AQI Index",
                xaxis_title="Date",
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor=BG_COLOR,
                paper_bgcolor=CARD_BG,
                font_color=TEXT_COLOR
            )
            st.plotly_chart(fig_fcst, use_container_width=True)
        else:
            st.warning(f"Not enough valid data points for {forecast_city} to forecast.")
    else:
        st.warning(
            f"Need at least 15 data points for {forecast_city} for forecasting; found {len(src)}."
        )

# ================= City AQI Hotspots =================
st.markdown("## 📍 City AQI Hotspots")
with st.container():
    coords = {}
    coords_path = "lat_long.txt"
    map_rendered = False

    try:
        if os.path.exists(coords_path):
            with open(coords_path, "r", encoding="utf-8") as f:
                content = f.read()
            local_vars = {}
            exec(content, {}, local_vars)
            if "city_coords" in local_vars and isinstance(local_vars["city_coords"], dict):
                coords = local_vars["city_coords"]
    except Exception as e_map:
        st.error(
            f"Map Error: Could not process '{coords_path}'. Scatter map unavailable. Error: {e_map}"
        )
        coords = {}

    if not df_filtered.empty:
        grp_data = df_filtered.groupby("city").agg(
            avg_aqi=("index", "mean"),
            dominant_pollutant=("pollutant", lambda x:
                x.mode().iloc[0]
                if not x.mode().empty and x.mode().iloc[0] not in ["Other", "nan"]
                else (x[~x.isin(["Other", "nan"])].mode().iloc[0]
                      if not x[~x.isin(["Other", "nan"])].mode().empty else "N/A")
            )
        ).reset_index().dropna(subset=["avg_aqi"])

        if coords and not grp_data.empty:
            latlon_list = []
            for c_name, coord_val in coords.items():
                if isinstance(coord_val, (list, tuple)) and len(coord_val) == 2:
                    try:
                        lat, lon = float(coord_val[0]), float(coord_val[1])
                        latlon_list.append({"city": c_name, "lat": lat, "lon": lon})
                    except (ValueError, TypeError):
                        pass

            latlon_df = pd.DataFrame(latlon_list)
            if not latlon_df.empty:
                merged_map = pd.merge(grp_data, latlon_df, on="city", how="inner")
                if not merged_map.empty:
                    # Determine AQI category by range
                    merged_map["AQI Category"] = merged_map["avg_aqi"].apply(
                        lambda val: next(
                            (k for k, rng in {
                                "Good": (0, 50),
                                "Satisfactory": (51, 100),
                                "Moderate": (101, 200),
                                "Poor": (201, 300),
                                "Very Poor": (301, 400),
                                "Severe": (401, float("inf"))
                            }.items() if rng[0] <= val <= rng[1]), "Unknown"
                        ) if pd.notna(val) else "Unknown"
                    )

                    fig_map = px.scatter_mapbox(
                        merged_map,
                        lat="lat",
                        lon="lon",
                        size=np.maximum(merged_map["avg_aqi"], 1),
                        size_max=25,
                        color="AQI Category",
                        color_discrete_map=CATEGORY_COLORS,
                        hover_name="city",
                        custom_data=["city", "avg_aqi", "dominant_pollutant", "AQI Category"],
                        text="city",
                        zoom=4.0,
                        center={"lat": 23.0, "lon": 82.5}
                    )

                    map_layout = get_layout_args(height=700, title=f"Average AQI Hotspots - {sel_month}, {year}")
                    map_layout["mapbox_style"] = "carto-darkmatter"
                    map_layout["margin"] = {"r": 10, "t": 60, "l": 10, "b": 10}
                    map_layout["legend"]["y"] = 0.98
                    map_layout["legend"]["x"] = 0.98
                    map_layout["legend"]["xanchor"] = "right"

                    fig_map.update_traces(
                        marker=dict(sizemin=5, opacity=0.75),
                        hovertemplate=(
                            "<b style='font-size:1.1em;'>%{customdata[0]}</b><br>"
                            "Avg. AQI: %{customdata[1]:.1f} (%{customdata[3]})<br>"
                            "Dominant Pollutant: %{customdata[2]}<extra></extra>"
                        )
                    )
                    fig_map.update_layout(**map_layout)
                    st.plotly_chart(fig_map, use_container_width=True)
                    map_rendered = True

        if not map_rendered:
            if not coords:
                st.warning(
                    f"Map Warning: '{coords_path}' missing or invalid. Scatter map unavailable."
                )
            elif "latlon_df" in locals() and latlon_df.empty:
                st.warning(f"Map Warning: Coordinates data loaded but appears empty/malformed.")
            elif "merged_map" in locals() and merged_map.empty:
                st.info("No matching city data for the scatter map. Showing fallback chart.")

            st.markdown("##### City AQI Overview (Map Unavailable/Incomplete)")
            if not grp_data.empty:
                sorted_cities = grp_data.sort_values(by="avg_aqi", ascending=True)
                fig_fb = px.bar(
                    sorted_cities.tail(20),
                    x="avg_aqi",
                    y="city",
                    orientation="h",
                    color="avg_aqi",
                    color_continuous_scale=px.colors.sequential.YlOrRd_r,
                    labels={"avg_aqi": "Average AQI", "city": "City"}
                )
                fb_layout = get_layout_args(
                    height=max(400, len(sorted_cities.tail(20)) * 25),
                    title=f"Top Cities by Average AQI - {sel_month}, {year} (Map Fallback)"
                )
                fb_layout["yaxis"] = {"categoryorder": "total ascending"}
                fig_fb.update_layout(**fb_layout)
                st.plotly_chart(fig_fb, use_container_width=True)
            else:
                st.warning("No city AQI data available for the selected period.")
    else:
        st.warning("No AQI data available for the selected filters.")

st.markdown("---")

# ================= Download Filtered Data =================
if export_list:
    st.markdown("## 📥 DOWNLOAD DATA")
    with st.container():
        combined_export = pd.concat(export_list)
        from io import StringIO
        buffer = StringIO()
        combined_export.to_csv(buffer, index=False)
        st.download_button(
            label="📤 Download Filtered City Data (CSV)",
            data=buffer.getvalue(),
            file_name=f"IITKGP_filtered_aqi_{year}_{sel_month.replace(' ', '')}.csv",
            mime="text/csv"
        )

# ------------------- Footer -------------------
st.markdown(
    f"""
    <div style="text-align: center; margin-top: 4rem; padding: 1.5rem; background-color: {CARD_BG}; border-radius: 12px; border: 1px solid {BORDER_COLOR};">
        <p style="margin:0.3em; color: {TEXT_COLOR}; font-size:0.9rem;">🌬️ IIT KGP AQI Dashboard</p>
        <p style="margin:0.3em; color: {SUBTLE_TEXT}; font-size:0.8rem;">Data Source: Central Pollution Control Board (CPCB), India. Coordinates approximate.</p>
        <p style="margin:0.5em 0; color: {TEXT_COLOR}; font-size:0.85rem;">Conceptualized by: Mr. Kapil Meena & Prof. Arkopal K. Goswami, IIT Kharagpur.</p>
        <p style="margin:0.3em; color: {SUBTLE_TEXT}; font-size:0.8rem;">Made with ❤️.</p>
        <p style="margin-top:0.8em;">
            <a href="https://github.com/kapil2020/india-air-quality-dashboard" target="_blank" style="color:{ACCENT_COLOR}; text-decoration:none; font-weight:600;">
                🔗 View on GitHub
            </a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

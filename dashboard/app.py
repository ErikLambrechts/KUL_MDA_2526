import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Flanders Cycling & Weather Dashboard",
    page_icon="🚲",
    layout="wide",
)

# --- 2. CONFIGURATION ---
# Same weather columns used in your model pipeline.
WEATHER_COLS = [
    "temp_max",
    "temp_min",
    "temp_avg",
    "precip_quantity",
    "humidity_relative",
    "pressure",
    "sun_duration",
    "short_wave_from_sky",
    "evapotrans_ref",
]

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

WEEKDAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent


def first_existing_path(candidates):
    """Return the first existing file path from a list of possible project locations."""
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


DATA_PATH = first_existing_path([
    Path("data/MDA_Final_Dataset.parquet"),
    APP_DIR / "data" / "MDA_Final_Dataset.parquet",
    PROJECT_ROOT / "data" / "MDA_Final_Dataset.parquet",
    Path("MDA_Final_Dataset.parquet"),
])

MODEL_METRICS_PATH = first_existing_path([
    Path("data/outputs/model_comparison_metrics.csv"),
    APP_DIR / "data" / "outputs" / "model_comparison_metrics.csv",
    PROJECT_ROOT / "data" / "outputs" / "model_comparison_metrics.csv",
    Path("outputs/model_comparison_metrics.csv"),
    Path("model_comparison_metrics.csv"),
])

ANALYZED_DATA_PATH = first_existing_path([
    Path("data/outputs/MDA_Analyzed_Dataset.parquet"),
    APP_DIR / "data" / "outputs" / "MDA_Analyzed_Dataset.parquet",
    PROJECT_ROOT / "data" / "outputs" / "MDA_Analyzed_Dataset.parquet",
    Path("outputs/MDA_Analyzed_Dataset.parquet"),
    Path("MDA_Analyzed_Dataset.parquet"),
])


# --- 3. LOAD DATA ---
@st.cache_data(show_spinner=False)
def load_data(path_as_string):
    df = pd.read_parquet(path_as_string)

    # Ensure date is a datetime object.
    df["date"] = pd.to_datetime(df["date"])

    # Fill any missing traffic counts with 0.
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    # Make configured weather fields numeric if they are present.
    for col in WEATHER_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Feature engineering copied from the model pipeline.
    df["month"] = df["date"].dt.month.astype(int)
    df["month_name"] = df["date"].dt.month_name()
    df["dow_num"] = df["date"].dt.dayofweek.astype(int)
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["dow_num"].isin([5, 6]).astype(int)
    df["day_type"] = df["is_weekend"].map({0: "Weekday", 1: "Weekend"})

    return df


@st.cache_data(show_spinner=False)
def load_model_metrics(path_as_string):
    if not path_as_string:
        return pd.DataFrame()
    return pd.read_csv(path_as_string)


@st.cache_data(show_spinner=False)
def load_analyzed_dataset(path_as_string):
    if not path_as_string:
        return pd.DataFrame()

    analyzed = pd.read_parquet(path_as_string)
    if "date" in analyzed.columns:
        analyzed["date"] = pd.to_datetime(analyzed["date"])
    if "amount" in analyzed.columns:
        analyzed["amount"] = pd.to_numeric(analyzed["amount"], errors="coerce")
    if "anomaly_flag" in analyzed.columns:
        analyzed["anomaly_label"] = analyzed["anomaly_flag"].map({1: "Normal", -1: "Anomaly"})
    return analyzed


if DATA_PATH is None:
    st.error(
        "Could not find `MDA_Final_Dataset.parquet`. Expected it in `data/`, "
        "next to the app, or in the project root."
    )
    st.stop()

with st.spinner("Loading golden dataset..."):
    df = load_data(str(DATA_PATH))

available_weather_cols = [col for col in WEATHER_COLS if col in df.columns]

# --- 4. SIDEBAR FILTERS ---
st.sidebar.header("Filter the Data")

# Filter by date range.
min_date = df["date"].min().date()
max_date = df["date"].max().date()
selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date, end_date = min_date, max_date

# Start from the full data and progressively filter.
filtered_df = df.copy()

# New: district filter, useful because the model pipeline trains by district.
selected_district = "All"
if "district" in filtered_df.columns:
    districts = sorted(filtered_df["district"].dropna().unique().tolist())
    selected_district = st.sidebar.selectbox("Select District", options=["All"] + districts)
    if selected_district != "All":
        filtered_df = filtered_df[filtered_df["district"] == selected_district]

# Existing: municipality/council filter.
selected_council = "All"
if "council" in filtered_df.columns:
    councils = sorted(filtered_df["council"].dropna().unique().tolist())
    selected_council = st.sidebar.selectbox("Select Municipality (Council)", options=["All"] + councils)
    if selected_council != "All":
        filtered_df = filtered_df[filtered_df["council"] == selected_council]

# Existing: station filter, now based on the selected district/council.
selected_station = "All"
if "name_site" in filtered_df.columns:
    stations = sorted(filtered_df["name_site"].dropna().unique().tolist())
    selected_station = st.sidebar.selectbox("Select Counting Station", options=["All"] + stations)
    if selected_station != "All":
        filtered_df = filtered_df[filtered_df["name_site"] == selected_station]

# New: weekday/weekend filter from the model pipeline feature engineering.
selected_day_type = st.sidebar.selectbox(
    "Select Day Type",
    options=["All", "Weekday", "Weekend"],
)
if selected_day_type != "All":
    filtered_df = filtered_df[filtered_df["day_type"] == selected_day_type]

# Apply date filter last.
filtered_df = filtered_df[
    (filtered_df["date"].dt.date >= start_date)
    & (filtered_df["date"].dt.date <= end_date)
]

# New: choose the weather variable shown in the weather scatter/correlation views.
selected_weather_col = None
if available_weather_cols:
    default_weather_index = (
        available_weather_cols.index("temp_max")
        if "temp_max" in available_weather_cols
        else 0
    )
    selected_weather_col = st.sidebar.selectbox(
        "Weather variable for scatter plot",
        options=available_weather_cols,
        index=default_weather_index,
    )

if filtered_df.empty:
    st.warning("No data matches the selected filters. Try widening the date range or selecting All.")
    st.stop()

# --- 5. MAIN DASHBOARD KPIS ---
st.title("Flanders Cycling Traffic & Weather Analysis")
st.markdown(
    "Explore how weather conditions impact cycling traffic across Flanders using official AWV and ClimateGrid data."
)

# Calculate KPIs.
total_cyclists = int(filtered_df["amount"].sum())
num_stations = filtered_df["name_site"].nunique() if "name_site" in filtered_df.columns else 0
filtered_days = filtered_df["date"].dt.date.nunique()

temp_col = "temp_max"
precip_col = "precip_quantity"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cyclists (Filtered)", f"{total_cyclists:,}")
col2.metric("Active Stations Selected", f"{num_stations}")

if temp_col in filtered_df.columns and not filtered_df[temp_col].isna().all():
    avg_temp = filtered_df[temp_col].mean()
    col3.metric("Average Max Temp", f"{avg_temp:.1f} °C")
else:
    col3.metric("Filtered Days", f"{filtered_days}")

if precip_col in filtered_df.columns and not filtered_df[precip_col].isna().all():
    avg_rain = filtered_df[precip_col].mean()
    col4.metric("Average Rainfall", f"{avg_rain:.1f} mm")
else:
    col4.metric("Filtered Days", f"{filtered_days}")

st.divider()

# --- 6. VISUALIZATIONS: ROW 1: TIME AND MAP ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Traffic Evolution Over Time")
    daily_traffic = filtered_df.groupby("date", as_index=False)["amount"].sum()

    fig_time = px.line(
        daily_traffic,
        x="date",
        y="amount",
        title="Total Daily Traffic",
        labels={"amount": "Number of Cyclists", "date": "Date"},
    )
    fig_time.update_traces(line_color="#1f77b4")
    st.plotly_chart(fig_time, use_container_width=True)

with col_right:
    st.subheader("Map of Counting Stations")
    required_map_cols = {"name_site", "latitude", "longitude", "amount"}
    if required_map_cols.issubset(filtered_df.columns):
        map_data = (
            filtered_df
            .dropna(subset=["latitude", "longitude"])
            .groupby(["name_site", "latitude", "longitude"], as_index=False)["amount"]
            .sum()
        )

        if map_data.empty:
            st.info("No valid latitude/longitude values for the selected filters.")
        else:
            fig_map = px.scatter_mapbox(
                map_data,
                lat="latitude",
                lon="longitude",
                hover_name="name_site",
                size="amount",
                color="amount",
                color_continuous_scale=px.colors.sequential.Viridis,
                zoom=7,
                height=400,
                title="Station Locations & Traffic Volume",
            )
            fig_map.update_layout(
                mapbox_style="carto-positron",
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
            )
            st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("The dataset does not contain the required map columns.")

# --- 7. VISUALIZATIONS: ROW 2: WEATHER AND DIRECTION ---
st.divider()
col_bottom_left, col_bottom_right = st.columns(2)

with col_bottom_left:
    st.subheader("Weather Impact on Cycling")

    if selected_weather_col:
        weather_impact = (
            filtered_df
            .dropna(subset=[selected_weather_col])
            .groupby("date", as_index=False)
            .agg(
                amount=("amount", "sum"),
                weather_value=(selected_weather_col, "mean"),
            )
        )

        if len(weather_impact) < 2:
            st.info("Not enough weather data for this filtered selection.")
        else:
            weather_label = selected_weather_col.replace("_", " ").title()
            fig_weather = px.scatter(
                weather_impact,
                x="weather_value",
                y="amount",
                trendline="lowess",  # Keep this: it now works because statsmodels is in requirements.txt.
                title=f"Daily Cyclists vs. {weather_label}",
                labels={"amount": "Total Cyclists", "weather_value": weather_label},
                opacity=0.6,
            )
            st.plotly_chart(fig_weather, use_container_width=True)
    else:
        st.info("No configured weather columns were found in the dataset.")

with col_bottom_right:
    st.subheader("Traffic by Direction")
    if "direction_name" in filtered_df.columns:
        direction_data = (
            filtered_df
            .groupby("direction_name", as_index=False)["amount"]
            .sum()
            .sort_values(by="amount")
        )

        if len(direction_data) > 10:
            direction_data = direction_data.tail(10)

        fig_dir = px.bar(
            direction_data,
            x="amount",
            y="direction_name",
            orientation="h",
            title="Top 10 Directions by Volume",
            labels={"amount": "Total Cyclists", "direction_name": "Direction"},
        )
        st.plotly_chart(fig_dir, use_container_width=True)
    else:
        st.info("No direction_name column found in the dataset.")

# --- 8. NEW: CALENDAR FEATURES FROM MODEL PIPELINE ---
st.divider()
st.subheader("Seasonality and Calendar Effects")
season_col1, season_col2, season_col3 = st.columns(3)

with season_col1:
    monthly = (
        filtered_df
        .groupby(["month", "month_name"], as_index=False)["amount"]
        .sum()
        .sort_values("month")
    )
    fig_month = px.bar(
        monthly,
        x="month_name",
        y="amount",
        title="Cyclists by Month",
        labels={"month_name": "Month", "amount": "Total Cyclists"},
        category_orders={"month_name": MONTH_ORDER},
    )
    st.plotly_chart(fig_month, use_container_width=True)

with season_col2:
    weekday = (
        filtered_df
        .groupby(["dow_num", "day_of_week"], as_index=False)["amount"]
        .sum()
        .sort_values("dow_num")
    )
    fig_weekday = px.bar(
        weekday,
        x="day_of_week",
        y="amount",
        title="Cyclists by Day of Week",
        labels={"day_of_week": "Day", "amount": "Total Cyclists"},
        category_orders={"day_of_week": WEEKDAY_ORDER},
    )
    st.plotly_chart(fig_weekday, use_container_width=True)

with season_col3:
    day_type = (
        filtered_df
        .groupby("day_type", as_index=False)["amount"]
        .mean()
        .sort_values("day_type")
    )
    fig_day_type = px.bar(
        day_type,
        x="day_type",
        y="amount",
        title="Average Row Traffic: Weekday vs Weekend",
        labels={"day_type": "Day Type", "amount": "Average Cyclists"},
        category_orders={"day_type": ["Weekday", "Weekend"]},
    )
    st.plotly_chart(fig_day_type, use_container_width=True)

# --- 9. NEW: WEATHER RELATIONSHIP SUMMARY ---
st.divider()
st.subheader("Weather Relationship Summary")
weather_col1, weather_col2 = st.columns(2)

with weather_col1:
    if available_weather_cols:
        aggregation = {"amount": "sum"}
        aggregation.update({col: "mean" for col in available_weather_cols})
        daily_weather = (
            filtered_df
            .groupby("date", as_index=False)
            .agg(aggregation)
            .dropna(subset=available_weather_cols, how="all")
        )

        if len(daily_weather) >= 2:
            corr = (
                daily_weather[["amount"] + available_weather_cols]
                .corr(numeric_only=True)["amount"]
                .drop(labels=["amount"], errors="ignore")
                .dropna()
                .sort_values()
                .reset_index()
            )
            corr.columns = ["weather_variable", "correlation_with_cyclists"]

            if corr.empty:
                st.info("Not enough variation to calculate weather correlations.")
            else:
                fig_corr = px.bar(
                    corr,
                    x="correlation_with_cyclists",
                    y="weather_variable",
                    orientation="h",
                    title="Daily Weather Correlation with Cyclist Volume",
                    labels={
                        "correlation_with_cyclists": "Correlation with Daily Cyclists",
                        "weather_variable": "Weather Variable",
                    },
                )
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough daily data to calculate correlations.")
    else:
        st.info("No configured weather columns were found in the dataset.")

with weather_col2:
    if precip_col in filtered_df.columns and not filtered_df[precip_col].isna().all():
        precip_daily = (
            filtered_df
            .groupby("date", as_index=False)
            .agg(amount=("amount", "sum"), precip_quantity=(precip_col, "mean"))
            .dropna(subset=["precip_quantity"])
        )
        precip_daily["rain_bin"] = pd.cut(
            precip_daily["precip_quantity"],
            bins=[-float("inf"), 0, 1, 5, 10, float("inf")],
            labels=["0 mm", "0-1 mm", "1-5 mm", "5-10 mm", "10+ mm"],
        )
        rain_summary = (
            precip_daily
            .groupby("rain_bin", observed=False)
            .agg(avg_daily_cyclists=("amount", "mean"), days=("amount", "size"))
            .reset_index()
        )

        fig_rain = px.bar(
            rain_summary,
            x="rain_bin",
            y="avg_daily_cyclists",
            hover_data=["days"],
            title="Average Daily Cyclists by Rainfall Bin",
            labels={
                "rain_bin": "Daily Precipitation",
                "avg_daily_cyclists": "Average Daily Cyclists",
                "days": "Days in Bin",
            },
        )
        st.plotly_chart(fig_rain, use_container_width=True)
    else:
        st.info("No precipitation column found for rainfall-bin analysis.")

# --- 10. NEW: MODEL PIPELINE OUTPUT VIEWER ---
st.divider()
st.subheader("Predictive Model Pipeline Results")
metrics_df = load_model_metrics(str(MODEL_METRICS_PATH) if MODEL_METRICS_PATH else "")

if metrics_df.empty:
    st.info(
        "No `model_comparison_metrics.csv` file found yet. Run `models_pipeline.py` offline, "
        "then place the output at `data/outputs/model_comparison_metrics.csv` to show R², RMSE and MAE here."
    )
else:
    st.caption(f"Loaded model metrics from: `{MODEL_METRICS_PATH}`")

    required_metric_cols = {"District", "Model", "R2_Score", "RMSE", "MAE"}
    if required_metric_cols.issubset(metrics_df.columns):
        metric_districts = sorted(metrics_df["District"].dropna().unique().tolist())
        selected_metric_districts = st.multiselect(
            "Districts to show in model comparison",
            options=metric_districts,
            default=metric_districts[: min(5, len(metric_districts))],
        )

        metric_plot_df = metrics_df.copy()
        if selected_metric_districts:
            metric_plot_df = metric_plot_df[metric_plot_df["District"].isin(selected_metric_districts)]

        fig_models = px.bar(
            metric_plot_df,
            x="Model",
            y="R2_Score",
            color="District",
            barmode="group",
            title="Model R² by District",
            labels={"R2_Score": "R² Score"},
        )
        st.plotly_chart(fig_models, use_container_width=True)

        st.dataframe(
            metric_plot_df.sort_values("R2_Score", ascending=False),
            use_container_width=True,
        )
    else:
        st.warning(
            "The metrics file was found, but it does not contain the expected columns: "
            "District, Model, R2_Score, RMSE, MAE."
        )
        st.dataframe(metrics_df, use_container_width=True)

# --- 11. NEW: ADVANCED ANALYSIS OUTPUT VIEWER ---
st.divider()
st.subheader("Advanced Analysis: Clusters and Anomalies")
analyzed_df = load_analyzed_dataset(str(ANALYZED_DATA_PATH) if ANALYZED_DATA_PATH else "")

if analyzed_df.empty:
    st.info(
        "No `MDA_Analyzed_Dataset.parquet` file found yet. Run the advanced analysis script offline, "
        "then place the output at `data/outputs/MDA_Analyzed_Dataset.parquet` to show PCA clusters and anomalies here."
    )
else:
    st.caption(f"Loaded analyzed dataset from: `{ANALYZED_DATA_PATH}`")

    advanced_filtered = analyzed_df.copy()
    if "date" in advanced_filtered.columns:
        advanced_filtered = advanced_filtered[
            (advanced_filtered["date"].dt.date >= start_date)
            & (advanced_filtered["date"].dt.date <= end_date)
        ]
    if selected_district != "All" and "district" in advanced_filtered.columns:
        advanced_filtered = advanced_filtered[advanced_filtered["district"] == selected_district]
    if selected_council != "All" and "council" in advanced_filtered.columns:
        advanced_filtered = advanced_filtered[advanced_filtered["council"] == selected_council]
    if selected_station != "All" and "name_site" in advanced_filtered.columns:
        advanced_filtered = advanced_filtered[advanced_filtered["name_site"] == selected_station]
    if selected_day_type != "All" and "date" in advanced_filtered.columns:
        if "day_type" not in advanced_filtered.columns:
            advanced_filtered["day_type"] = advanced_filtered["date"].dt.dayofweek.isin([5, 6]).map({False: "Weekday", True: "Weekend"})
        advanced_filtered = advanced_filtered[advanced_filtered["day_type"] == selected_day_type]

    if advanced_filtered.empty:
        st.info("The analyzed dataset exists, but no analyzed rows match the selected filters.")
    elif {"PCA1", "PCA2", "cluster_label"}.issubset(advanced_filtered.columns):
        plot_df = advanced_filtered.copy()
        plot_df["cluster_label"] = plot_df["cluster_label"].astype(str)
        if "anomaly_label" not in plot_df.columns and "anomaly_flag" in plot_df.columns:
            plot_df["anomaly_label"] = plot_df["anomaly_flag"].map({1: "Normal", -1: "Anomaly"})

        advanced_col1, advanced_col2 = st.columns([2, 1])

        with advanced_col1:
            hover_cols = [
                col for col in ["date", "name_site", "district", "council", "amount", selected_weather_col]
                if col and col in plot_df.columns
            ]
            fig_pca = px.scatter(
                plot_df,
                x="PCA1",
                y="PCA2",
                color="cluster_label",
                symbol="anomaly_label" if "anomaly_label" in plot_df.columns else None,
                hover_data=hover_cols,
                opacity=0.6,
                title="PCA View of Cycling / Weather Profiles",
                labels={"cluster_label": "Cluster", "anomaly_label": "Profile"},
            )
            st.plotly_chart(fig_pca, use_container_width=True)

        with advanced_col2:
            st.markdown("**Cluster averages**")
            cluster_summary = (
                plot_df
                .groupby("cluster_label", as_index=False)
                .agg(avg_cyclists=("amount", "mean"), rows=("amount", "size"))
                .sort_values("avg_cyclists", ascending=False)
            )
            st.dataframe(cluster_summary, use_container_width=True)

            if "anomaly_flag" in plot_df.columns:
                anomaly_count = int((plot_df["anomaly_flag"] == -1).sum())
                st.metric("Flagged Anomalies", f"{anomaly_count:,}")

        if "anomaly_flag" in plot_df.columns:
            anomalies = plot_df[plot_df["anomaly_flag"] == -1].copy()
            if not anomalies.empty:
                st.markdown("**Top flagged anomalies by cyclist count**")
                show_cols = [
                    col for col in [
                        "date", "name_site", "district", "council", "amount",
                        selected_weather_col, "cluster_label", "anomaly_label",
                    ]
                    if col and col in anomalies.columns
                ]
                st.dataframe(
                    anomalies.sort_values("amount", ascending=False)[show_cols].head(25),
                    use_container_width=True,
                )
    else:
        st.warning(
            "The analyzed dataset was found, but it does not contain `PCA1`, `PCA2`, and `cluster_label`. "
            "Check that the advanced analysis script exported the labeled dataset."
        )
        st.dataframe(advanced_filtered.head(50), use_container_width=True)


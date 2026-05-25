import streamlit as st
import pandas as pd
import plotly.express as px

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Flanders Cycling & Weather Dashboard", page_icon="🚲", layout="wide")

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_parquet('data/MDA_Final_Dataset.parquet')
    
    # Ensure date is a datetime object
    df['date'] = pd.to_datetime(df['date'])
    
    # Fill any missing amounts with 0 just in case
    df['amount'] = df['amount'].fillna(0)
    
    return df

with st.spinner("Loading golden dataset..."):
    df = load_data()

# --- 3. SIDEBAR FILTERS ---
st.sidebar.header("Filter the Data")

# Filter by Date Range
min_date = df['date'].min()
max_date = df['date'].max()
selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(selected_dates, (list, tuple)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date = end_date = selected_dates

# Filter by Municipality (Council)
# Using dropna() to avoid issues if some stations are missing council data
councils = sorted(df['council'].dropna().unique().tolist())
selected_council = st.sidebar.selectbox("Select Municipality (Council)", options=["All"] + councils)

# Apply Council Filter
if selected_council != "All":
    filtered_df = df[df['council'] == selected_council]
else:
    filtered_df = df

# Filter by Specific Station (Name Site) based on chosen council
stations = sorted(filtered_df['name_site'].dropna().unique().tolist())
selected_station = st.sidebar.selectbox("Select Counting Station", options=["All"] + stations)

# Apply Station Filter
if selected_station != "All":
    filtered_df = filtered_df[filtered_df['name_site'] == selected_station]

# Apply Date Filter
if len(selected_dates) == 2:
    start_date, end_date = selected_dates
    filtered_df = filtered_df[
    (filtered_df['date'].dt.date >= start_date) &
    (filtered_df['date'].dt.date <= end_date)
    ]

# --- 4. MAIN DASHBOARD KPIs ---
st.title(" Flanders Cycling Traffic & Weather Analysis")
st.markdown("Explore how weather conditions impact cycling traffic across Flanders using official AWV and ClimateGrid data.")

# Calculate KPIs
total_cyclists = int(filtered_df['amount'].sum())
num_stations = filtered_df['name_site'].nunique()

# Check for the temperature column (TX is standard for max temp in Belgian ClimateGrid)
temp_col = 'temp_max'

col1, col2, col3 = st.columns(3)
col1.metric("Total Cyclists (Filtered)", f"{total_cyclists:,}")
col2.metric("Active Stations Selected", f"{num_stations}")

if temp_col and not filtered_df[temp_col].isna().all():
    avg_temp = filtered_df[temp_col].mean()
    col3.metric("Average Max Temp", f"{avg_temp:.1f} °C")
else:
    # Safely compute date range days. start_date/end_date may not be defined
    # if the user didn't provide a 2-value date_input; fall back to full range.
    if 'start_date' in locals() and 'end_date' in locals():
        days = (end_date - start_date).days
    else:
        # min_date and max_date may be Timestamp; convert to date for subtraction
        days = (pd.to_datetime(max_date).date() - pd.to_datetime(min_date).date()).days
    col3.metric("Date Range", f"{days} days")

st.divider()

# --- 5. VISUALIZATIONS: ROW 1 (Time & Map) ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader(" Traffic Evolution Over Time")
    # Aggregate data by date so the chart isn't too messy
    daily_traffic = filtered_df.groupby('date')['amount'].sum().reset_index()
    
    fig_time = px.line(
        daily_traffic, 
        x='date', 
        y='amount', 
        title="Total Daily Traffic",
        labels={'amount': 'Number of Cyclists', 'date': 'Date'}
    )
    fig_time.update_traces(line_color='#1f77b4')
    st.plotly_chart(fig_time, use_container_width=True)

with col_right:
    st.subheader(" Map of Counting Stations")
    # Aggregate to get total counts per station for the map bubble sizes
    map_data = filtered_df.groupby(['name_site', 'latitude', 'longitude'])['amount'].sum().reset_index()
    
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
        title="Station Locations & Traffic Volume"
    )
    fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

# --- 6. VISUALIZATIONS: ROW 2 (Weather & Direction) ---
st.divider()
col_bottom_left, col_bottom_right = st.columns(2)

with col_bottom_left:
    st.subheader(" Weather Impact on Cycling")
    if temp_col:
        # Group by temperature to smooth the scatter plot
        weather_impact = filtered_df.groupby([temp_col, 'date'])['amount'].sum().reset_index()
        
        fig_weather = px.scatter(
            weather_impact, 
            x=temp_col, 
            y='amount', 
            trendline="lowess", # Adds a smoothed trendline
            title="Daily Cyclists vs. Max Temperature",
            labels={'amount': 'Total Cyclists', temp_col: 'Max Temperature (°C)'},
            opacity=0.6
        )
        st.plotly_chart(fig_weather, use_container_width=True)
    else:
        st.info("No temperature column ('TX') found in the dataset to display weather impact.")

with col_bottom_right:
    st.subheader("Traffic by Direction")
    if 'direction_name' in filtered_df.columns:
        direction_data = filtered_df.groupby('direction_name')['amount'].sum().reset_index().sort_values(by='amount')
        
        # Take top 10 directions if there are too many
        if len(direction_data) > 10:
            direction_data = direction_data.tail(10)
            
        fig_dir = px.bar(
            direction_data,
            x='amount',
            y='direction_name',
            orientation='h',
            title="Top 10 Directions by Volume",
            labels={'amount': 'Total Cyclists', 'direction_name': 'Direction'}
        )
        st.plotly_chart(fig_dir, use_container_width=True)
    else:
         st.info("No direction_name column found in the dataset.")
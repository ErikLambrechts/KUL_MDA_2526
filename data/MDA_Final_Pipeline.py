import os
import io
import glob
import requests
import pandas as pd
import geopandas as gpd
from datetime import datetime
from dateutil.relativedelta import relativedelta

#from models.models_pipeline import train_all_district_models_by_year

# ==========================================
# 1. CONFIGURATION (Set paths and dates here)
# ==========================================
CONFIG = {
    "START_DATE": datetime(2024, 1, 1), 
    "END_DATE": datetime(2025, 12, 31),
    "DATA_DIR": "./KUL_MDA_2526/data",
    "CLIMATE_GRID_DIR": "./KUL_MDA_2526/data",
    "CLIMATE_METADATA_FILE": "./KUL_MDA_2526/data/climategrid_pixel_metadata.csv"
}

# Ensure data directory exists
os.makedirs(CONFIG["DATA_DIR"], exist_ok=True)

# ==========================================
# 2. CYCLING DATA EXTRACTION
# ==========================================
def download_cycling_data():
    print("--- Step 1: Downloading Raw Cycling Data ---")
    base_url = 'https://opendata.apps.mow.vlaanderen.be/fietstellingen/'
    output_file = os.path.join(CONFIG["DATA_DIR"], 'raw_combined_data.csv')
    
    column_names = ['site_id', 'direction', 'type', 'from', 'to', 'amount']
    
    if os.path.exists(output_file):
        os.remove(output_file)

    current_date = CONFIG["START_DATE"]
    first_file = True
    downloaded_count = 0

    while current_date <= CONFIG["END_DATE"]:
        month_str = current_date.strftime('%Y-%m')
        file_name = f'data-{month_str}.csv'
        file_url = f"{base_url}{file_name}"

        try:
            response = requests.get(file_url, timeout=30)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text), header=None, names=column_names)
                if first_file:
                    df.to_csv(output_file, index=False, mode='w')
                    first_file = False
                else:
                    df.to_csv(output_file, index=False, mode='a', header=False)
                print(f"Successfully Downloaded: {file_name}")
                downloaded_count += 1
            else:
                print(f"Skipped {file_name}: Server returned status code {response.status_code}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

        current_date += relativedelta(months=1)

    if downloaded_count == 0:
        raise FileNotFoundError(f"\n[ERROR] No data files were successfully downloaded to '{output_file}'.")
        
    return output_file

# ==========================================
# 3. TRANSFORMATION (Aggregate & Enrich Metadata)
# ==========================================
def process_and_enrich_cycling_data(raw_csv_path):
    print("--- Step 2: Aggregating Raw Data to Daily Frequency (Memory Save Mode) ---")
    
    # Chunk reading prevents memory spikes. We group the raw data first.
   # Chunk reading prevents memory spikes. We group the raw data first.
    chunks = []
    for chunk in pd.read_csv(raw_csv_path, chunksize=500_000):
        # [NEW] Keep ONLY rows where the 'type' column is exactly 'FIETSERS'
        chunk = chunk[chunk['type'] == 'FIETSERS']
        
        # Skip processing this chunk if it becomes empty after filtering
        if chunk.empty:
            continue
            
        chunk['date'] = pd.to_datetime(chunk['from']).dt.normalize()
        
        # Initial chunk-level reduction
        grouped_chunk = chunk.groupby(['site_id', 'direction', 'date'], as_index=False)['amount'].sum()
        chunks.append(grouped_chunk)
        
    # Combine chunk groups and run a final group by to catch boundaries
    combined_raw = pd.concat(chunks, ignore_index=True)
    daily_df = combined_raw.groupby(['site_id', 'direction', 'date'], as_index=False)['amount'].sum()
    
    print("--- Step 3: Merging Daily Data with Metadata Tables ---")
    base_url = 'https://opendata.apps.mow.vlaanderen.be/fietstellingen/'
    
    # Load and merge directions
    directions = pd.read_csv(f'{base_url}richtingen.csv', header=None, names=['site_id', 'direction', 'direction_name'])
    daily_df = pd.merge(daily_df, directions, on=['site_id', 'direction'], how='left')

    # Load and merge site metadata
    site_columns = ['site_id', 'site_nr', 'longitude', 'latitude', 'name_site', 
                    'domain', 'path_nr', 'district', 'council', 'interval', 'installation_date']
    sites = pd.read_csv(f'{base_url}sites.csv', header=None, names=site_columns)
    daily_enriched_df = pd.merge(daily_df, sites, on='site_id', how='left')

    # Save checkpoint to parquet
    parquet_path = os.path.join(CONFIG["DATA_DIR"], 'daily_final_combined_data.parquet')
    daily_enriched_df.to_parquet(parquet_path, engine="fastparquet", index=False)
    print(f"Saved daily aggregated data to: {parquet_path}")
    
    return daily_enriched_df

# ==========================================
# 4. WEATHER INTEGRATION (Geospatial mapping)
# ==========================================
def merge_with_weather_data(cycling_df):
    print("--- Step 4: Mapping Weather Data Spatial Coordinates ---")
    
    pixels = pd.read_csv(CONFIG["CLIMATE_METADATA_FILE"], sep=";")
    
    # Drop rows where coordinates are completely missing to avoid GeoPandas errors
    cycling_clean = cycling_df.dropna(subset=['latitude', 'longitude'])
    sites = cycling_clean[["site_id", "latitude", "longitude"]].drop_duplicates()
    
    sites_gdf = gpd.GeoDataFrame(
        sites, geometry=gpd.points_from_xy(sites.longitude, sites.latitude), crs="EPSG:4326"
    ).to_crs("EPSG:3857")

    pixels_gdf = gpd.GeoDataFrame(
        pixels, geometry=gpd.points_from_xy(pixels["PIXEL_LON_CENTER"], pixels["PIXEL_LAT_CENTER"]), crs="EPSG:4326"
    ).to_crs("EPSG:3857")

    site_pixel_map = gpd.sjoin_nearest(sites_gdf, pixels_gdf, how="left")
    cycling_df = cycling_df.merge(site_pixel_map[["site_id", "PIXEL_ID"]], on="site_id", how="left")

    print("--- Step 5: Loading and Filtering Weather Files ---")
    files = glob.glob(f"{CONFIG['CLIMATE_GRID_DIR']}/*.csv")
    weather_list = []

    for f in files:
        df = pd.read_csv(f, sep=";")
        df["day"] = pd.to_datetime(df["day"])
        
        # Filter early per weather file to optimize memory
        df = df[
            (df["day"] >= CONFIG["START_DATE"]) & 
            (df["day"] <= CONFIG["END_DATE"])
        ]
        if not df.empty:
            weather_list.append(df)

    if not weather_list:
        raise ValueError("No weather files found matching the date range window.")

    weather = pd.concat(weather_list, ignore_index=True)

    print("--- Step 6: Final Datasets Merge ---")
    final_dataset = cycling_df.merge(
        weather,
        left_on=["PIXEL_ID", "date"],
        right_on=["pixel_id", "day"],
        how="left"
    )

    return final_dataset

# ==========================================
# 5. MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    print("Starting Data Pipeline...")
    
    # 1. Download raw data to file
    raw_csv_path = download_cycling_data()
    
    # 2. Process chunks, aggregate daily, and add metadata
    daily_cycling_data = process_and_enrich_cycling_data(raw_csv_path)
    
    # 3. Coordinate mapping and integration of weather
    final_output = merge_with_weather_data(daily_cycling_data)
    
    # 4. Save final output frame
    final_path = os.path.join(CONFIG["DATA_DIR"], 'MDA_Final_Dataset.parquet')
    final_output.to_parquet(final_path, engine="fastparquet", index=False)
    
    print(f"\nPipeline Complete Successfully! Final dataset saved to {final_path}")

    # ==========================================
    # DOWNSTREAM WORKSPACE MODEL ORCHESTRATION
    # ==========================================
    import subprocess
    import sys

    print("\n" + "="*55)
    print("      DATA PIPELINE COMPLETED SUCCESSFULLY!              ")
    print("      TRIGGERING DOWNSTREAM ANALYSIS SCRIPTS             ")
    print("="*55)

    # Define paths to your individual execution files
    models_script = r"D:\Ilonchyk\Masters\MDA\KUL_MDA_2526\models\models_pipeline.py"
    analysis_script = r"D:\Ilonchyk\Masters\MDA\KUL_MDA_2526\models\advanced_analysis.py"

    # 1. Trigger Supervised Regressions (Train 2024 -> Test 2025)
    if os.path.exists(models_script):
        print("\n>>> Launching: Supervised Predictive Engine...")
        try:
            subprocess.run([sys.executable, models_script], check=True)
            print(" -> Supervised predictions completed cleanly.")
        except subprocess.CalledProcessError as e:
            print(f" [CRITICAL ERROR] Supervised pipeline failed with code: {e}")
    else:
        print(f" [WARNING] Expected supervised file missing at: {models_script}")

    # 2. Trigger Unsupervised Clustering, PCA Plots & Anomalies
    if os.path.exists(analysis_script):
        print("\n>>> Launching: Exploratory & Unsupervised Mining Engine...")
        try:
            subprocess.run([sys.executable, analysis_script], check=True)
            print(" -> Unsupervised pattern discovery completed cleanly.")
        except subprocess.CalledProcessError as e:
            print(f" [CRITICAL ERROR] Unsupervised pipeline failed with code: {e}")
    else:
        print(f" [WARNING] Expected analysis file missing at: {analysis_script}")

    print("\n" + "="*55)
    print("     ALL ANALYTICAL WORKSPACE SEQUENCES FINALIZED        ")
    print("=======================================================\n")

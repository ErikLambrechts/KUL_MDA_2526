import pandas as pd
import numpy as np
import geopandas as gpd
import glob 

# loading the cycling parquet data
cycling = pd.read_parquet("D:/Ilonchyk/Masters/MDA/daily_final_combined_data.parquet", engine="fastparquet")
cycling["date"] = pd.to_datetime(cycling["date"])
cycling = cycling[
    (cycling["date"] >= "2024-01-01") &
    (cycling["date"] < "2026-01-01")
]

print(cycling.date.head())
print(cycling.info())
print(cycling.describe())

# loading the weather meta-data (has the exact coordinates of the grid cells)
pixels = pd.read_csv("D:/Ilonchyk/Masters/MDA/climateGrid/climategrid_pixel_metadata.csv", sep=";")
pixels.head()

# unique cycling sites
sites = cycling[
    ["site_id", "latitude", "longitude"]
].drop_duplicates()

# convert to GeoDataFrames
sites_gdf = gpd.GeoDataFrame(
    sites,
    geometry=gpd.points_from_xy(
        sites.longitude,
        sites.latitude
    ),
    crs="EPSG:4326"
)

pixels_gdf = gpd.GeoDataFrame(
    pixels,
    geometry=gpd.points_from_xy(
        pixels["PIXEL_LON_CENTER"],
        pixels["PIXEL_LAT_CENTER"]
    ),
    crs="EPSG:4326"
)

# convert to metric CRS
sites_gdf = sites_gdf.to_crs("EPSG:3857")
pixels_gdf = pixels_gdf.to_crs("EPSG:3857")

# nearest weather pixel (combining both cycling sites and weather pixels into a single GeoDataFrame)
site_pixel_map = gpd.sjoin_nearest(
    sites_gdf,
    pixels_gdf,
    how="left"
)

site_pixel_map.head()

cycling = cycling.merge(
    site_pixel_map[["site_id", "PIXEL_ID"]],
    on="site_id",
    how="left"
)

#combining all the weather data files 
files = glob.glob("D:/Ilonchyk/Masters/MDA/climateGrid/climateGrid/*.csv")

weather_list = []

for f in files:
    df = pd.read_csv(f, sep=";")
    weather_list.append(df)

weather = pd.concat(weather_list, ignore_index=True)

weather["day"] = pd.to_datetime(weather["day"])
weather = weather[
    (weather["day"] >= "2024-01-01") &
    (weather["day"] < "2026-01-01")
]

weather.head()

merged = cycling.merge(
    weather,
    left_on=["PIXEL_ID", "date"],
    right_on=["pixel_id", "day"],
    how="left"
)

merged["month"] = pd.to_datetime(
    merged["date"]).dt.month

merged["day_of_week"] = pd.to_datetime(
    merged["date"]).dt.dayofweek

merged["weekend"] = (
    merged["day_of_week"] >= 5).astype(int)

merged = merged.dropna()
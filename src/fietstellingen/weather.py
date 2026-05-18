import pandas as pd # type: ignore
from pathlib import Path
from .config import DATA_DIR

WEATHER_DIR = DATA_DIR / "weather"


def load_weather(site_id: int):
    file = WEATHER_DIR / f"weather_site_{site_id}.csv"
    df = pd.read_csv(file, parse_dates=["time"])
    return df


def load_all_weather():
    files = WEATHER_DIR.glob("weather_site_*.csv")

    dfs = []
    for f in files:
        site_id = int(f.stem.split("_")[-1])
        df = pd.read_csv(f, parse_dates=["time"])
        df["site ID"] = site_id
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
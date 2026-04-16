import requests
import pandas as pd
from pathlib import Path
from datetime import date
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from fietstellingen.loader import load_sites

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
WEATHER_DIR = DATA_DIR / "weather"

WEATHER_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def countdown(seconds: int):
    for i in range(seconds, 0, -1):
        print(f"Retrying in {i}s...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 30, end="\r")  # clear line


def fetch_weather_with_retry(lat, lon, retries=5, delay=5):
    for attempt in range(1, retries + 1):
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": "2015-01-01",
                "end_date": date.today().isoformat(),
                "hourly": [
                    "temperature_2m",
                    "precipitation",
                    "windspeed_10m",
                    "shortwave_radiation",
                ],
                "timezone": "Europe/Brussels",
            }

            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()

            data = response.json()
            df = pd.DataFrame(data["hourly"])
            df["time"] = pd.to_datetime(df["time"])

            return df

        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)

            print(f"[Attempt {attempt}/{retries}] Error: {e}")

            if status == 429 and attempt < retries:
                countdown(delay)
                continue

            if attempt >= retries:
                raise

            countdown(delay)

        except Exception as e:
            print(f"[Attempt {attempt}/{retries}] Unexpected error: {e}")

            if attempt >= retries:
                raise

            countdown(delay)


def download_weather_for_sites():
    sites = load_sites()

    for _, site in sites.iterrows():
        site_id = site["site ID"]
        lat = site["lat"]
        lon = site["long"]

        output_file = WEATHER_DIR / f"weather_site_{site_id}.csv"

        if output_file.exists():
            print(f"Skipping site {site_id}")
            continue

        print(f"\nDownloading weather for site {site_id}")

        try:
            df = fetch_weather_with_retry(lat, lon)
            df.to_csv(output_file, index=False)

            # small delay between sites (VERY important)
            time.sleep(1)

        except Exception as e:
            print(f"Failed for site {site_id}: {e}")


if __name__ == "__main__":
    download_weather_for_sites()
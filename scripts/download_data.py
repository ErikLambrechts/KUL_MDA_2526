import os
import requests
from pathlib import Path

BASE_URL = "https://opendata.apps.mow.vlaanderen.be/fietstellingen/"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

COUNTS_DIR = DATA_DIR / "counts"
META_DIR = DATA_DIR / "metadata"

COUNTS_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path):
    if dest.exists():
        print(f"Skipping {dest.name}")
        return

    print(f"Downloading {url}")
    r = requests.get(url)
    r.raise_for_status()

    with open(dest, "wb") as f:
        f.write(r.content)


def download_metadata():
    files = ["sites.csv", "richtingen.csv"]

    for file in files:
        url = BASE_URL + file
        download_file(url, META_DIR / file)


def download_counts(years=range(2015, 2026)):
    for year in years:
        for month in range(1, 13):
            filename = f"data-{year}-{month:02d}.csv"
            url = BASE_URL + filename
            dest = COUNTS_DIR / filename

            try:
                download_file(url, dest)
            except requests.HTTPError:
                print(f"Missing: {filename}")


if __name__ == "__main__":
    download_metadata()
    download_counts()

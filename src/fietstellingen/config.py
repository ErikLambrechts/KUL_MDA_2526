from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
COUNTS_DIR = DATA_DIR / "counts"
META_DIR = DATA_DIR / "metadata"
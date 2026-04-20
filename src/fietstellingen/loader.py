import pandas as pd
from .config import COUNTS_DIR, META_DIR


def load_sites():
    columns = [
        "site ID",
        "site nr",
        "long",
        "lat",
        "naam",
        "domein",
        "wegnr",
        "district",
        "gemeente",
        "interval",
        "datum_van",
    ]

    return pd.read_csv(
        META_DIR / "sites.csv",
        header=None,
        names=columns,
        parse_dates=["datum_van"]
    )

def load_directions():
    columns = [
        "site ID",
        "richting",
        "naam",
    ]

    return pd.read_csv(
        META_DIR / "richtingen.csv",
        header=None,
        names=columns,
    )


def load_counts(year: int = None):
    files = list(COUNTS_DIR.glob("data-*.csv"))

    if year:
        files = [f for f in files if f"data-{year}-" in f.name]

    columns = [
        "site ID",
        "richting",
        "type",
        "van",
        "tot",
        "aantal",
    ]

    dfs = []
    for f in files:
        df = pd.read_csv(
            f,
            header=None,
            names=columns,
            parse_dates=["van", "tot"]
        )
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def load_all():
    counts = load_counts()
    sites = load_sites()
    directions = load_directions()

    df = counts.merge(sites, on="site ID", how="left")
    df = df.merge(directions, on=["site ID", "richting"], how="left")

    return df

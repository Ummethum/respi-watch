"""
RespiWatch -- Recent air quality fetcher (last few weeks only)
=====================================================================
Fetches the last few weeks of air quality data per Kreis centroid from
Open-Meteo's Air Quality API, aggregating hourly -> weekly immediately
per batch (same "no raw hourly storage" pattern as the historical
fetch_air_quality.py, to avoid the >300MB checkpoint problem), then
merges into the existing air_quality_weekly.parquet.

Usage:
    python fetch_recent_air_quality.py
"""

import os
import time
import random
import numpy as np
import pandas as pd
import requests

# 1. CONFIGURATION

COORDS_PATH = "./data/city_coords/kreise_coords.csv"
OUTPUT_DIR = "./data/air_quality"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HOURLY_VARIABLES = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "european_aqi",
]

LOOKBACK_WEEKS = 6
BATCH_SIZE = 3
DELAY_BETWEEN_BATCHES = 0.25

# Retry settings for individual Open-Meteo requests -- read timeouts
# ("Read timed out. (read timeout=30)") happen intermittently, usually
# under server load, and are transient -- a retry with backoff nearly
# always succeeds on the 2nd or 3rd attempt rather than needing manual
# re-runs.
MAX_REQUEST_RETRIES = 3
REQUEST_TIMEOUT = 45          # was 30 -- a bit more headroom before
                                # even attempting a retry
RETRY_BACKOFF_BASE = 5        # seconds -- doubles each retry (5, 10, 20)

BASE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


# 2. LOAD COORDINATES

NUTS1_TO_BUNDESLAND = {
    "DE1": "Baden-Wuerttemberg", "DE2": "Bayern", "DE3": "Berlin",
    "DE4": "Brandenburg", "DE5": "Bremen", "DE6": "Hamburg",
    "DE7": "Hessen", "DE8": "Mecklenburg-Vorpommern", "DE9": "Niedersachsen",
    "DEA": "Nordrhein-Westfalen", "DEB": "Rheinland-Pfalz", "DEC": "Saarland",
    "DED": "Sachsen", "DEE": "Sachsen-Anhalt", "DEF": "Schleswig-Holstein",
    "DEG": "Thueringen",
}

coords = pd.read_csv(COORDS_PATH, dtype={"kreis_id": str})
print(f"Loaded {len(coords)} Kreis centroids")

end_date = pd.Timestamp.today().normalize()
start_date = end_date - pd.Timedelta(weeks=LOOKBACK_WEEKS)
print(f"Fetching {start_date.date()} -> {end_date.date()}")

# 3. FETCH + AGGREGATE ONE BATCH (hourly -> weekly immediately)

def _request_with_retries(url: str, params: dict) -> requests.Response:
    """
    Retries a GET request on timeout/transient errors with exponential
    backoff (5s, 10s, 20s...). Read timeouts against Open-Meteo happen
    intermittently under server load and are almost always transient
    -- a retry succeeds in the vast majority of cases without needing
    a manual re-run of the whole script.
    """
    last_exception = None
    for attempt in range(MAX_REQUEST_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            last_exception = e
            wait = RETRY_BACKOFF_BASE * (2 ** attempt)
            print(f"    Request failed (attempt {attempt+1}/{MAX_REQUEST_RETRIES}): "
                  f"{e} -- waiting {wait}s before retry")
            if attempt < MAX_REQUEST_RETRIES - 1:
                time.sleep(wait)

    raise last_exception


def fetch_and_aggregate_batch(batch_coords: pd.DataFrame) -> list[dict]:
    lats = ",".join(batch_coords["lat"].astype(str))
    lons = ",".join(batch_coords["lon"].astype(str))

    params = {
        "latitude": lats,
        "longitude": lons,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Europe/Berlin",
    }

    resp = _request_with_retries(BASE_URL, params)
    data = resp.json()

    if isinstance(data, dict):
        data = [data]

    weekly_rows = []
    for kreis_row, location_result in zip(batch_coords.itertuples(), data):
        hourly = location_result["hourly"]
        hourly_df = pd.DataFrame({"time": hourly["time"]})
        for var in HOURLY_VARIABLES:
            hourly_df[var] = hourly[var]

        hourly_df["time"] = pd.to_datetime(hourly_df["time"])
        hourly_df["year"] = hourly_df["time"].dt.isocalendar().year.astype(int)
        hourly_df["week"] = hourly_df["time"].dt.isocalendar().week.astype(int)

        for (year, week), group in hourly_df.groupby(["year", "week"]):
            weekly_rows.append({
                "kreis_id": kreis_row.kreis_id, "year": year, "week": week,
                "pm10_mean": group["pm10"].mean(),
                "pm10_max": group["pm10"].max(),
                "pm2_5_mean": group["pm2_5"].mean(),
                "pm2_5_max": group["pm2_5"].max(),
                "co_mean": group["carbon_monoxide"].mean(),
                "no2_mean": group["nitrogen_dioxide"].mean(),
                "no2_max": group["nitrogen_dioxide"].max(),
                "so2_mean": group["sulphur_dioxide"].mean(),
                "ozone_mean": group["ozone"].mean(),
                "ozone_max": group["ozone"].max(),
                "european_aqi_mean": group["european_aqi"].mean(),
                "european_aqi_max": group["european_aqi"].max(),
                "n_hours": len(group),
                "is_complete_week": len(group) >= 24 * 7 - 3,   # small tolerance
            })

    return weekly_rows

# 4. FETCH ALL KREISE IN BATCHES

def fetch_all_recent_air_quality() -> pd.DataFrame:
    all_rows = []
    n_batches = (len(coords) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(coords), BATCH_SIZE):
        batch = coords.iloc[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{n_batches} "
              f"({', '.join(batch['kreis_id'])})...", end=" ", flush=True)

        try:
            rows = fetch_and_aggregate_batch(batch)
            all_rows.extend(rows)
            print(f"OK ({len(rows)} weekly rows)")
        except requests.exceptions.RequestException as e:
            print(f"FAILED: {e}")

        time.sleep(DELAY_BETWEEN_BATCHES + random.uniform(0, 2))

    weekly = pd.DataFrame(all_rows)
    if weekly.empty:
        return weekly

    weekly = weekly.merge(coords[["kreis_id", "name"]], on="kreis_id", how="left")
    weekly["nuts1_code"] = weekly["kreis_id"].str[:3]
    weekly["bundesland_name"] = weekly["nuts1_code"].map(NUTS1_TO_BUNDESLAND)
    weekly["week_start"] = pd.to_datetime(
        weekly["year"].astype(str) + "-W" + weekly["week"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u",
    )
    return weekly

# 5. MERGE INTO HISTORICAL FILE

def merge_weekly(new_weekly: pd.DataFrame, path: str) -> pd.DataFrame:
    key_cols = ["kreis_id", "year", "week"]
    if os.path.exists(path):
        historical = pd.read_parquet(path)
        historical = historical.merge(
            new_weekly[key_cols], on=key_cols, how="left", indicator=True
        )
        historical = historical[historical["_merge"] == "left_only"].drop(columns="_merge")
        combined = pd.concat([historical, new_weekly], ignore_index=True)
    else:
        combined = new_weekly
    return combined.sort_values(["kreis_id", "year", "week"]).reset_index(drop=True)

# 6. RUN

if __name__ == "__main__":
    print(f"Fetching recent air quality for {len(coords)} Kreise...\n")

    weekly_df = fetch_all_recent_air_quality()
    if weekly_df.empty:
        raise SystemExit("No air quality data fetched -- aborting, not touching historical file.")

    path = os.path.join(OUTPUT_DIR, "air_quality_weekly.parquet")
    combined = merge_weekly(weekly_df, path)
    combined.to_parquet(path, index=False)

    print(f"\n\u2713 Updated: {path} ({len(combined):,} rows)")

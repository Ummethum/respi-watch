"""
RespiWatch -- Recent weather fetcher (last few weeks only)
=================================================================
Fetches the last few weeks of weather data per Kreis centroid from
Open-Meteo's ARCHIVE API (same source as the historical backfill, for
training/serving consistency -- using a different weather source for
live prediction than what the model was trained on would introduce a
subtle distribution shift), then merges into the existing
weather_daily.parquet / weather_weekly.parquet.

Why the ARCHIVE api, not the forecast api, for "recent" data:
    ERA5 reanalysis (the archive API's data source) has roughly a
    5-day processing lag -- but since your lag1/lag2/lag3 features
    only need data from 1+ weeks ago, that lag has always already
    passed by the time a week's data is actually used as a feature.
    Using the archive API keeps this script's data identical in
    methodology to the historical training data.

Usage:
    python fetch_recent_weather.py
"""

import os
import time
import random
import numpy as np
import pandas as pd
import requests

# 1. CONFIGURATION

COORDS_PATH = "./data/city_coords/kreise_coords.csv"
OUTPUT_DIR = "./data/weather"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DAILY_VARIABLES = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
    "shortwave_radiation_sum",
]

# How far back to fetch. Wider than the minimum needed (lag3 = 3 weeks)
# to comfortably cover ERA5's processing lag plus give a buffer for
# any missed weekly runs.
LOOKBACK_WEEKS = 2
BATCH_SIZE = 3       
DELAY_BETWEEN_BATCHES = 0.25
MAX_REQUEST_RETRIES = 3
REQUEST_TIMEOUT = 45         
RETRY_BACKOFF_BASE = 5        

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# 2. LOAD KREIS COORDINATES

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

# 3. FETCH ONE BATCH OF KREISE

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


def fetch_batch(batch_coords: pd.DataFrame) -> list[dict]:
    lats = ",".join(batch_coords["lat"].astype(str))
    lons = ",".join(batch_coords["lon"].astype(str))

    params = {
        "latitude": lats,
        "longitude": lons,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Europe/Berlin",
    }

    resp = _request_with_retries(BASE_URL, params)
    data = resp.json()

    # Multi-location requests return a LIST of per-location result
    # objects when >1 lat/lon pair is passed, a single dict otherwise
    if isinstance(data, dict):
        data = [data]

    rows = []
    for kreis_row, location_result in zip(batch_coords.itertuples(), data):
        daily = location_result["daily"]
        n_days = len(daily["time"])
        for i in range(n_days):
            row = {"kreis_id": kreis_row.kreis_id, "date": daily["time"][i]}
            for var in DAILY_VARIABLES:
                row[var] = daily[var][i]
            rows.append(row)

    return rows

# 4. FETCH ALL KREISE IN BATCHES

def fetch_all_recent_weather() -> pd.DataFrame:
    all_rows = []
    n_batches = (len(coords) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(coords), BATCH_SIZE):
        batch = coords.iloc[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{n_batches} "
              f"({', '.join(batch['kreis_id'])})...", end=" ", flush=True)

        try:
            rows = fetch_batch(batch)
            all_rows.extend(rows)
            print(f"OK ({len(rows)} rows)")
        except requests.exceptions.RequestException as e:
            print(f"FAILED: {e}")

        time.sleep(DELAY_BETWEEN_BATCHES + random.uniform(0, 2))

    result = pd.DataFrame(all_rows)
    if not result.empty:
        result["date"] = pd.to_datetime(result["date"])
    return result

# 5. AGGREGATE DAILY -> WEEKLY (matches historical column schema)

def aggregate_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    daily_df = daily_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df["year"] = daily_df["date"].dt.isocalendar().year.astype(int)
    daily_df["week"] = daily_df["date"].dt.isocalendar().week.astype(int)
    daily_df["iso_weekday"] = daily_df["date"].dt.isocalendar().day.astype(int)
    daily_df["is_weekend"] = daily_df["iso_weekday"].isin([6, 7])
    daily_df["rain_day"] = daily_df["precipitation_sum"] > 0.1

    weekly_rows = []
    for (kreis_id, year, week), group in daily_df.groupby(["kreis_id", "year", "week"]):
        weekend = group[group["is_weekend"]]

        weekly_rows.append({
            "kreis_id": kreis_id, "year": year, "week": week,
            "temp_max_mean": group["temperature_2m_max"].mean(),
            "temp_min_mean": group["temperature_2m_min"].mean(),
            "temp_mean_mean": group["temperature_2m_mean"].mean(),
            "temp_max_max": group["temperature_2m_max"].max(),
            "temp_min_min": group["temperature_2m_min"].min(),
            "precip_sum": group["precipitation_sum"].sum(),
            "rain_days": group["rain_day"].sum(),
            "wind_max": group["wind_gusts_10m_max"].max(),
            "radiation_sum": group["shortwave_radiation_sum"].sum(),
            "n_days": len(group),
            "weekend_temp_max_mean": weekend["temperature_2m_max"].mean() if len(weekend) else np.nan,
            "weekend_temp_min_mean": weekend["temperature_2m_min"].mean() if len(weekend) else np.nan,
            "weekend_temp_mean": weekend["temperature_2m_mean"].mean() if len(weekend) else np.nan,
            "weekend_precip_sum": weekend["precipitation_sum"].sum() if len(weekend) else np.nan,
            "weekend_rain_days": weekend["rain_day"].sum() if len(weekend) else 0,
            "weekend_wind_max": weekend["wind_gusts_10m_max"].max() if len(weekend) else np.nan,
            "weekend_radiation_sum": weekend["shortwave_radiation_sum"].sum() if len(weekend) else np.nan,
            "n_weekend_days": len(weekend),
            "is_complete_week": len(group) >= 7,
            "is_complete_weekend": len(weekend) >= 2,
        })

    weekly = pd.DataFrame(weekly_rows)
    weekly = weekly.merge(coords[["kreis_id", "name"]], on="kreis_id", how="left")
    weekly["nuts1_code"] = weekly["kreis_id"].str[:3]
    weekly["bundesland_name"] = weekly["nuts1_code"].map(NUTS1_TO_BUNDESLAND)   # matches historical convention
    weekly["week_start"] = pd.to_datetime(
        weekly["year"].astype(str) + "-W" + weekly["week"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u",
    )
    return weekly

# 6. MERGE INTO HISTORICAL FILES

def merge_daily(new_daily: pd.DataFrame, path: str) -> pd.DataFrame:
    key_cols = ["kreis_id", "date"]
    if os.path.exists(path):
        historical = pd.read_parquet(path)
        historical["date"] = pd.to_datetime(historical["date"])
        historical = historical.merge(
            new_daily[key_cols], on=key_cols, how="left", indicator=True
        )
        historical = historical[historical["_merge"] == "left_only"].drop(columns="_merge")
        combined = pd.concat([historical, new_daily], ignore_index=True)
    else:
        combined = new_daily
    return combined.sort_values(["kreis_id", "date"]).reset_index(drop=True)


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

# 7. RUN

if __name__ == "__main__":
    print(f"Fetching recent weather for {len(coords)} Kreise...\n")

    daily_df = fetch_all_recent_weather()
    if daily_df.empty:
        raise SystemExit("No weather data fetched -- aborting, not touching historical files.")

    weekly_df = aggregate_to_weekly(daily_df)

    daily_path = os.path.join(OUTPUT_DIR, "weather_daily.parquet")
    weekly_path = os.path.join(OUTPUT_DIR, "weather_weekly.parquet")

    combined_daily = merge_daily(daily_df, daily_path)
    combined_weekly = merge_weekly(weekly_df, weekly_path)

    combined_daily.to_parquet(daily_path, index=False)
    combined_weekly.to_parquet(weekly_path, index=False)

    print(f"\n\u2713 Updated: {daily_path} ({len(combined_daily):,} rows)")
    print(f"\u2713 Updated: {weekly_path} ({len(combined_weekly):,} rows)")

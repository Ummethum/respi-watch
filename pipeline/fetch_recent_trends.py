"""
RespiWatch -- Weekly Google Trends fetcher (recent window only)
=======================================================================
Fetches the LAST 12 MONTHS of Google Trends data for the 7 project
keywords via pytrends-modern's standard HTTP API mode. Google Trends
returns data at WEEKLY resolution directly for any timeframe beyond
~269 days (~8.8 months) -- "today 12-m" safely clears that threshold,
so no daily->weekly aggregation is needed, and each request pulls
fewer raw data points than an equivalent daily-resolution pull would.

Also rescales the freshly fetched values against the existing
historical data before merging (see rescale_to_historical_anchor())
-- Google Trends normalises each API call independently to its own
0-100 scale, so a 12-month fetch and the full 2004-2026 historical
build are NOT directly comparable without this correction.

Usage:
    python fetch_recent_trends.py
"""

import os
import time
import random
import pandas as pd
from pytrends_modern import TrendReq

# 1. CONFIGURATION

KEYWORDS = ["Grippe", "Influenza", "Grippeimpfung", "Fieber",
           "Husten", "Wadenwickel", "Sinupret"]

# Bundesland geo codes (Google Trends uses "DE-XX" ISO 3166-2 codes) --
# same convention as the historical Trends pipeline
GEO_CODES = {
    "DE-BW": "DE1", "DE-BY": "DE2", "DE-BE": "DE3", "DE-BB": "DE4",
    "DE-HB": "DE5", "DE-HH": "DE6", "DE-HE": "DE7", "DE-MV": "DE8",
    "DE-NI": "DE9", "DE-NW": "DEA", "DE-RP": "DEB", "DE-SL": "DEC",
    "DE-SN": "DED", "DE-ST": "DEE", "DE-SH": "DEF", "DE-TH": "DEG",
}

OUTPUT_DIR = "./data/google_trends"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIMEFRAME = "today 12-m" 

EMPTY_RESULT_RETRIES = 2

DELAY_BETWEEN_CALLS_MIN = 8
DELAY_BETWEEN_CALLS_MAX = 15

COOLDOWN_BEFORE_RETRY_PASS = 120   # seconds


# 2. BUILD THE CLIENT


def build_client() -> TrendReq:
    return TrendReq(
        hl="de-DE",
        tz=60,
        timeout=(10, 25),
        retries=2,
        backoff_factor=1.5,
        rotate_user_agent=True,
    )


# 3. FETCH ONE (geo, keyword-batch) COMBINATION

def fetch_trends_for_geo(pytrends: TrendReq, geo: str,
                         keywords: list[str]) -> pd.DataFrame | None:
    """
    pytrends allows up to 5 keywords per request -- our 7 keywords need
    two batches per geo, then an outer join on date to combine them
    (both batches share the same date index for the same geo/timeframe).
    """
    batches = [keywords[i:i+5] for i in range(0, len(keywords), 5)]
    batch_frames = []

    for batch in batches:
        for attempt in range(EMPTY_RESULT_RETRIES):
            try:
                pytrends.build_payload(batch, timeframe=TIMEFRAME, geo=geo)
                data = pytrends.interest_over_time()

                if data.empty:
                    wait = random.uniform(DELAY_BETWEEN_CALLS_MIN, DELAY_BETWEEN_CALLS_MAX) * (attempt + 1)
                    print(f"    (empty result for {geo}, batch {batch}, "
                          f"attempt {attempt+1}, waiting {wait:.0f}s)")
                    time.sleep(wait)
                    continue

                data = data.drop(columns=["isPartial"], errors="ignore")
                batch_frames.append(data)
                break

            except Exception as e:
                wait = random.uniform(DELAY_BETWEEN_CALLS_MIN, DELAY_BETWEEN_CALLS_MAX) * (attempt + 1)
                print(f"    Error for {geo}, batch {batch} "
                      f"(attempt {attempt+1}/{EMPTY_RESULT_RETRIES}): {e}")
                print(f"    Waiting {wait:.0f}s...")
                time.sleep(wait)
        else:
            print(f"    FAILED for {geo}, batch {batch} after "
                  f"{EMPTY_RESULT_RETRIES} attempts")
            return None

        time.sleep(random.uniform(DELAY_BETWEEN_CALLS_MIN, DELAY_BETWEEN_CALLS_MAX))

    if not batch_frames:
        return None

    combined = batch_frames[0]
    for frame in batch_frames[1:]:
        combined = combined.join(frame, how="outer")

    combined["geo"] = geo
    return combined.reset_index().rename(columns={"date": "date"})

# 4. FETCH ALL BUNDESLAENDER (two-pass: initial attempt, then a single
# retry pass for failures only, after a cooldown)

def _attempt_all_geos(pytrends: TrendReq, geo_codes: dict) -> tuple[list, list]:
    """One pass through the given geos. Returns (successes, failed_geos)."""
    successes = []
    failed = []

    for geo, nuts1_code in geo_codes.items():
        print(f"Fetching {geo} ({nuts1_code})...")
        result = fetch_trends_for_geo(pytrends, geo, KEYWORDS)

        if result is not None:
            result["nuts1_code"] = nuts1_code
            successes.append(result)
            print(f"  OK: {len(result)} days")
        else:
            print(f"  FAILED this pass")
            failed.append(geo)

        time.sleep(random.uniform(DELAY_BETWEEN_CALLS_MIN, DELAY_BETWEEN_CALLS_MAX))

    return successes, failed


def fetch_all_recent_trends() -> pd.DataFrame:
    pytrends = build_client()

    print(f"{'='*60}\nPass 1: all {len(GEO_CODES)} Bundeslaender\n{'='*60}")
    successes, failed_geos = _attempt_all_geos(pytrends, GEO_CODES)

    if failed_geos:
        print(f"\n{len(failed_geos)} Bundeslaender failed pass 1: {failed_geos}")
        print(f"Cooling down {COOLDOWN_BEFORE_RETRY_PASS}s before retry pass "
              f"(gives Google's rate limiter a real reset window)...")
        time.sleep(COOLDOWN_BEFORE_RETRY_PASS)

        retry_geo_codes = {g: GEO_CODES[g] for g in failed_geos}
        print(f"\n{'='*60}\nPass 2: retrying {len(retry_geo_codes)} failed Bundeslaender\n{'='*60}")
        retry_successes, still_failed = _attempt_all_geos(pytrends, retry_geo_codes)
        successes.extend(retry_successes)

        if still_failed:
            print(f"\n\u26a0\ufe0f  Still failed after retry pass: {still_failed}")
            print(f"These Bundeslaender's Trends data will be MISSING from "
                  f"this week's update -- their existing historical values "
                  f"stay untouched (not overwritten with anything), so this "
                  f"degrades gracefully rather than corrupting old data.")

    all_frames = successes

    if not all_frames:
        raise RuntimeError(
            "No Trends data fetched for any Bundesland via standard HTTP "
            "mode -- if this happens consistently (not just a one-off), "
            "Google may be rate-limiting the HTTP API specifically. See "
            "the Browser Mode fallback documented at the bottom of this "
            "file before assuming something else is broken."
        )

    return pd.concat(all_frames, ignore_index=True)


# 5. AGGREGATE DAILY -> WEEKLY

def normalize_to_weekly(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    With TIMEFRAME="today 12-m", Google Trends already returns
    weekly-resolution rows directly (see the TIMEFRAME comment above
    for why) -- this function no longer performs a real daily->weekly
    AGGREGATION. The groupby+mean below is effectively a no-op passthrough
    in that case (averaging a single value per nuts1_code/year/week
    group), kept anyway for robustness -- e.g. if TIMEFRAME is ever
    changed back to something under the ~269-day threshold, this still
    produces a correct result without needing code changes elsewhere.
    """
    raw_df = raw_df.copy()
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    raw_df["year"] = raw_df["date"].dt.isocalendar().year.astype(int)
    raw_df["week"] = raw_df["date"].dt.isocalendar().week.astype(int)

    keyword_cols = [k for k in KEYWORDS if k in raw_df.columns]

    weekly = (
        raw_df.groupby(["nuts1_code", "year", "week"])[keyword_cols]
        .mean()
        .reset_index()
    )
    weekly = weekly.rename(columns={k: f"trends_{k.lower()}" for k in keyword_cols})

    weekly["week_start"] = pd.to_datetime(
        weekly["year"].astype(str) + "-W" + weekly["week"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u",
    )
    return weekly

# 6. RESCALE FRESH FETCH AGAINST HISTORICAL ANCHOR

def rescale_to_historical_anchor(new_weekly: pd.DataFrame,
                                 historical_path: str) -> pd.DataFrame:
    """
    Google Trends normalises EACH API call independently to 0-100
    relative to the maximum WITHIN that call's own requested time
    range -- a "last 3 months" fetch and the full 2004-2026 historical
    build are on genuinely different scales for the same real-world
    search volume, even for weeks both cover. Without correcting for
    this, freshly fetched weeks would sit on a different scale than
    the rest of the series once merged, corrupting trends_* as a
    feature (a real bug found in production -- recent weeks looked
    artificially inflated/deflated relative to history).

    Fix: use the OVERLAP between this fresh fetch and the existing
    historical file (weeks present in both) as a self-referential
    anchor, the same ratio-based principle parse_trends_csv.py uses
    against its monthly full-range anchor export -- just calibrated
    against the already-correctly-scaled history instead of a fresh
    full-range pull, since re-fetching 22 years of history every week
    would defeat the point of a lightweight incremental update.

    Only uses overlap weeks OLDER than 2 weeks as calibration points --
    Google's own index for very recent weeks can still be revised, so
    anchoring on those specifically would make the scale factor itself
    noisy.
    """
    if not os.path.exists(historical_path):
        print("No existing historical file to calibrate against -- "
              "leaving this fetch unscaled (expected only on the very "
              "first run, before any historical data exists).")
        return new_weekly

    historical = pd.read_parquet(historical_path)

    trend_cols = [c for c in new_weekly.columns if c.startswith("trends_")]
    key_cols = ["nuts1_code", "year", "week"]

    calibration_cutoff = pd.Timestamp.today() - pd.Timedelta(weeks=2)
    stable_new = new_weekly[new_weekly["week_start"] <= calibration_cutoff]

    overlap = stable_new.merge(
        historical[key_cols + trend_cols], on=key_cols,
        suffixes=("_fresh", "_hist"),
    )

    if overlap.empty:
        print("\u26a0\ufe0f  No stable overlap weeks found with existing history -- "
              "cannot calibrate a rescale factor, leaving this fetch "
              "unscaled. This is expected on the first run, but worth "
              "investigating if it keeps happening on later runs.")
        return new_weekly

    scaled = new_weekly.copy()
    scale_factors = {}

    for col in trend_cols:
        fresh_col = f"{col}_fresh"
        hist_col = f"{col}_hist"
        if fresh_col not in overlap.columns or hist_col not in overlap.columns:
            continue

        valid = overlap[(overlap[fresh_col] > 0) & (overlap[hist_col] > 0)]
        if valid.empty:
            continue

        # Median ratio across overlap weeks -- robust to any single
        # noisy week
        ratios = valid[hist_col] / valid[fresh_col]
        scale = ratios.median()
        scale_factors[col] = scale
        scaled[col] = scaled[col] * scale

    print(f"Rescale factors calibrated from {len(overlap)} overlap week(s): "
          f"{ {k: round(v, 3) for k, v in scale_factors.items()} }")

    return scaled


# 7. MERGE INTO THE EXISTING HISTORICAL TRENDS FILE

def merge_into_historical(new_weekly: pd.DataFrame,
                          historical_path: str) -> pd.DataFrame:
    """
    Appends the freshly fetched weeks onto the historical Trends
    parquet, replacing any overlapping (nuts1_code, year, week) rows
    with the new values -- Trends data for a given week can shift
    slightly as Google's index recalculates, so newer data should win
    over what was fetched before for the same week.
    """
    if os.path.exists(historical_path):
        historical = pd.read_parquet(historical_path)
        key_cols = ["nuts1_code", "year", "week"]
        historical = historical.merge(
            new_weekly[key_cols], on=key_cols, how="left", indicator=True
        )
        historical = historical[historical["_merge"] == "left_only"].drop(columns="_merge")
        combined = pd.concat([historical, new_weekly], ignore_index=True)
    else:
        print(f"No existing historical file at {historical_path} -- "
              f"creating new one with just the recent data.")
        combined = new_weekly

    return combined.sort_values(["nuts1_code", "year", "week"]).reset_index(drop=True)

# 7. RUN

if __name__ == "__main__":
    print(f"Fetching last-month Google Trends for {len(GEO_CODES)} Bundeslaender, "
          f"{len(KEYWORDS)} keywords (pytrends-modern, standard HTTP mode)...\n")

    raw_df = fetch_all_recent_trends()
    weekly_df = normalize_to_weekly(raw_df)

    print(f"\nFetched {len(weekly_df)} Bundesland x week rows")

    historical_path = os.path.join(OUTPUT_DIR, "trends_wide_from_csv.parquet")

    weekly_df = rescale_to_historical_anchor(weekly_df, historical_path)

    combined_df = merge_into_historical(weekly_df, historical_path)

    combined_df.to_parquet(historical_path, index=False)
    print(f"\n\u2713 Updated: {historical_path} ({len(combined_df):,} total rows)")


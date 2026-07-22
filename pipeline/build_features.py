"""
RespiWatch — Feature engineering: lags, rolling windows, multi-step targets
===============================================================================
Transforms master_dataset_filled.parquet into a training-ready table:

    1. LAG FEATURES — every weather/pollen/air_quality/trends/AMELAG/ARE/
       GrippeWeb/Notaufnahme column gets shifted by 1, 2, 3 weeks (per
       Kreis). The original same-week columns are DROPPED afterward,
       since they represent information not available at prediction
       time

    2. ROLLING WINDOW FEATURES — 3-week rolling mean of key weather
       variables, smoothing single-week noise.

    3. AUTOREGRESSIVE TARGET LAGS — survstat_influenza/covid/rsv from
       1-3 weeks ago, added as columns here. NOTE: every model
       training/prediction script downstream EXCLUDES these from the
       actual feature set used (reporting-lag concern, see
       add_target_lags()'s own docstring below) -- they exist in the
       output but aren't currently used by any trained model. Maybe
       they will be tested in the future.

    4. MOMENTUM FEATURES — week-over-week change in the lagged target,
       added as a column here. Same caveat as point 3 above: excluded
       from the actual model feature set downstream, not currently used.

    5. CYCLICAL WEEK ENCODING — sin/cos of ISO week, for smoother
       seasonality representation than a raw 1-52 integer.

    6. MULTI-STEP TARGETS — target_survstat_influenza_t+1 / t+2 (and
       same for covid/rsv): the actual labels for 1-week-ahead and
       2-week-ahead forecasting models.

    7. HOLIDAY FEATURES ARE NOT LAGGED — school calendars and public
       holidays are known years in advance, so holiday_* columns are
       used as-is for the week being predicted (T, T+1, T+2), not
       shifted backward like the other sources.

    8. OPTIONAL "SOURCE AVAILABLE" FLAGS for sparse sources (pollen,
       AMELAG, ARE, GrippeWeb, Notaufnahme) — an explicit 0/1 flag
       alongside the NaN, which can help even NaN-native tree models
       distinguish "no data" from "value happened to be low".

IMPORTANT ASSUMPTION: lag/rolling shifts are done by ROW POSITION
within each Kreis's time-sorted series, not by literal week-number
arithmetic — this correctly handles the year-boundary (week 52 -> week
1) as long as the underlying spine has no gaps (true here, since it's
built from weather_weekly.parquet, which is complete for every week).

Usage:
    python build_features.py
"""

import os
import numpy as np
import pandas as pd

# 1. CONFIGURATION

MASTER_PATH = "./data/master/master_dataset_filled.parquet"
OUTPUT_DIR  = "./data/master"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGETS = ["survstat_influenza", "survstat_covid", "survstat_rsv"]

LAG_WEEKS = [1, 2, 3]           # how many weeks back to lag features
TARGET_LAG_WEEKS = [1, 2, 3]    # autoregressive lags of the target itself
ROLLING_WINDOW = 3              # weeks, for rolling mean features
FORECAST_HORIZONS = [1, 2]      # weeks ahead to build multi-step targets for

# Columns that must NEVER be lagged — IDs, metadata, targets (handled
# separately), and holiday features (known in advance, see docstring).
NEVER_LAG_PREFIXES = ["holiday_"]
NEVER_LAG_EXACT = [
    "kreis_id", "name", "nuts1_code", "bundesland_name",
    "year", "week", "week_start",
] + TARGETS

# QC/metadata columns to exclude entirely from feature engineering
EXCLUDE_COLS = [
    "weather_n_days", "weather_n_weekend_days",
    "weather_is_complete_week", "weather_is_complete_weekend",
    "air_quality_n_hours", "air_quality_is_complete_week",
    "pollen_n_hours", "pollen_is_complete_week",
    "amelag_sarscov2_n_sites", "amelag_influenza_n_sites", "amelag_rsv_n_sites",
]

# Sparse sources to generate an explicit "was data available" flag for
SOURCE_AVAILABILITY_FLAGS = {
    "pollen_available":       lambda df: df["pollen_birch_mean"].notna(),
    "amelag_available":       lambda df: df["amelag_sarscov2"].notna(),
    "are_available":          lambda df: df["are_konsultationsinzidenz"].notna(),
    "grippeweb_available":    lambda df: df["grippeweb_incidence"].notna(),
    "notaufnahme_available":  lambda df: df["notaufnahme_share_raw"].notna(),
}

# Weather columns to build rolling-mean features for (the ones most
# likely to matter for respiratory disease — extend as you like)
ROLLING_FEATURE_COLS = [
    "weather_temp_mean_mean", "weather_precip_sum",
    "air_quality_pm10_mean", "pollen_total_mean",
]

# 2. LOAD & PREP

def load_and_sort(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_values(["kreis_id", "week_start"]).reset_index(drop=True)
    return df

# 3. LAG FEATURES

def add_lag_features(df: pd.DataFrame, lag_weeks: list[int],
                     already_safe: list[str] = None) -> pd.DataFrame:
    """
    Shifts every eligible feature column by N weeks within each Kreis's
    time-sorted series, then drops the original same-week column —
    only the lagged versions remain as features.

    `already_safe` lists columns that must be left untouched — e.g.
    rolling-mean features that were already computed on lagged data
    (base_lag=1) and would otherwise get redundantly double-lagged.
    """
    df = df.copy()
    already_safe = already_safe or []

    lag_candidates = [
        c for c in df.columns
        if c not in NEVER_LAG_EXACT
        and c not in EXCLUDE_COLS
        and c not in already_safe
        and not any(c.startswith(p) for p in NEVER_LAG_PREFIXES)
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    print(f"Lagging {len(lag_candidates)} columns by {lag_weeks} week(s)...")

    grouped = df.groupby("kreis_id", group_keys=False)

    lagged_series = {}
    for col in lag_candidates:
        for lag in lag_weeks:
            lagged_series[f"{col}_lag{lag}"] = grouped[col].shift(lag)

    # Build all lag columns at once via concat, rather than repeated
    # single-column inserts — avoids the DataFrame fragmentation
    # performance warning on wide tables (many source columns × 3 lags).
    lagged_df = pd.DataFrame(lagged_series, index=df.index)
    df = pd.concat([df, lagged_df], axis=1)

    # Drop the original same-week columns — they're not available at
    # prediction time and would otherwise leak future information.
    df = df.drop(columns=lag_candidates)

    return df

# 4. ROLLING WINDOW FEATURES

def add_rolling_features(df: pd.DataFrame, cols: list[str],
                         window: int, base_lag: int = 1) -> pd.DataFrame:
    """
    Adds a rolling mean over `window` weeks, computed on data starting
    `base_lag` weeks back (so the rolling window itself never touches
    the current/future week either). E.g. with base_lag=1, window=3:
    rolling mean of weeks T-3, T-2, T-1 relative to the row's own week.

    Must be computed BEFORE add_lag_features() drops the raw columns —
    call this first, then lag features afterward.
    """
    df = df.copy()
    grouped = df.groupby("kreis_id", group_keys=False)

    for col in cols:
        if col not in df.columns:
            print(f"  ⚠️  {col} not found, skipping rolling feature.")
            continue
        shifted = grouped[col].shift(base_lag)
        df[f"{col}_rollmean{window}"] = (
            shifted.groupby(df["kreis_id"])
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return df

# 5. AUTOREGRESSIVE TARGET LAGS + MOMENTUM

def add_target_lags(df: pd.DataFrame, targets: list[str],
                    lag_weeks: list[int]) -> pd.DataFrame:
    """
    Adds lagged versions of the targets themselves as features (e.g.
    survstat_influenza_lag1) plus a momentum feature (change between
    the two most recent lags) — both are typically very strong
    predictors for epidemic curves.
    """
    df = df.copy()
    grouped = df.groupby("kreis_id", group_keys=False)

    for target in targets:
        for lag in lag_weeks:
            df[f"{target}_lag{lag}"] = grouped[target].shift(lag)

        # Momentum: change between lag1 and lag2 (is it rising or falling
        # going INTO the week we're predicting?)
        if 1 in lag_weeks and 2 in lag_weeks:
            df[f"{target}_momentum"] = (
                df[f"{target}_lag1"] - df[f"{target}_lag2"]
            )

    return df

# 6. CYCLICAL WEEK ENCODING

def add_cyclical_week(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)
    return df

# 7. SOURCE AVAILABILITY FLAGS

def add_availability_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computed BEFORE lagging, on the raw (pre-lag) columns — checking
    "was pollen data available THIS week" as it currently stands, not
    a lagged version of the flag.
    """
    df = df.copy()
    for flag_name, condition_fn in SOURCE_AVAILABILITY_FLAGS.items():
        try:
            df[flag_name] = condition_fn(df).astype(int)
        except KeyError as e:
            print(f"  ⚠️  Could not compute {flag_name}: missing column {e}")
    return df

# 8. MULTI-STEP TARGETS

def add_multistep_targets(df: pd.DataFrame, targets: list[str],
                          horizons: list[int]) -> pd.DataFrame:
    """
    Adds target_{name}_t_plus{h} columns: the target's value h weeks
    AHEAD of each row — these are the actual labels a 1-week-ahead or
    2-week-ahead forecasting model would be trained to predict, using
    everything else in this row (which is now entirely lagged/known-
    in-advance) as features.
    """
    df = df.copy()
    grouped = df.groupby("kreis_id", group_keys=False)

    for target in targets:
        for h in horizons:
            df[f"target_{target}_t_plus{h}"] = grouped[target].shift(-h)

    return df

# 9. RUN

if __name__ == "__main__":
    print(f"Loading {MASTER_PATH}...")
    df = load_and_sort(MASTER_PATH)
    print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns\n")

    # Drop QC/metadata columns entirely — not useful as features
    df = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])

    print("Step 1/6: Source availability flags...")
    df = add_availability_flags(df)

    print("Step 2/6: Rolling window features...")
    df = add_rolling_features(df, ROLLING_FEATURE_COLS, ROLLING_WINDOW, base_lag=1)
    rolling_cols = [f"{c}_rollmean{ROLLING_WINDOW}" for c in ROLLING_FEATURE_COLS
                   if f"{c}_rollmean{ROLLING_WINDOW}" in df.columns]

    # IMPORTANT: lag features must run BEFORE target lags / cyclical
    # week / multi-step targets below — those steps create new derived
    # columns (survstat_influenza_lag1, target_*_t_plus1, week_sin, ...)
    # that must NOT themselves get lagged again. Running this first
    # means only genuinely raw, same-week columns get shifted. Rolling
    # columns are excluded too (already safely lagged via base_lag=1).
    print("Step 3/6: Lag features (drops raw same-week feature columns)...")
    df = add_lag_features(df, LAG_WEEKS, already_safe=rolling_cols)

    print("Step 4/6: Autoregressive target lags + momentum...")
    df = add_target_lags(df, TARGETS, TARGET_LAG_WEEKS)

    print("Step 5/6: Cyclical week encoding...")
    df = add_cyclical_week(df)

    print("Step 6/6: Multi-step targets...")
    df = add_multistep_targets(df, TARGETS, FORECAST_HORIZONS)

    print(f"\nFinal shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")

    target_cols_t_plus = [c for c in df.columns if c.startswith("target_")]
    n_with_no_label_at_all = df[target_cols_t_plus].isna().all(axis=1).sum()
    print(f"{n_with_no_label_at_all:,} rows have no valid multi-step target for "
          f"ANY disease/horizon (end-of-series rows) -- KEPT, not dropped, "
          f"since generate_predictions.py needs the freshest of these for "
          f"live prediction. Training scripts filter per-target themselves.")

    out_path = os.path.join(OUTPUT_DIR, "master_dataset_features.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\n✓ Saved: {out_path}")

    print(f"\nSample columns: {sorted(df.columns)[:20]} ...")

# ── Notes for training ──────────────────────────────────────────────────
# - TIME-BASED (expanding window) cross-validation, never random
#   splits
# - Original targets (survstat_influenza/covid/rsv) are still present
#   in the output as columns, but should NOT be used as features
#   directly
# - NaN in lagged features (e.g. pollen_*_lag1 before 2021) is fine for
#   XGBoost/LightGBM, handled natively.
# - is_complete_week flags were dropped entirely (see EXCLUDE_COLS)

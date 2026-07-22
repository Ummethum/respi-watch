"""
RespiWatch — Prophet seasonal baseline, per Kreis
========================================================
Fits one Prophet model PER KREIS to capture the smooth yearly
seasonal shape + trend + holiday effects, then produces a
forecast/fitted-value table covering the FULL date range (train,
validation, test) — this becomes the "seasonal baseline" that
train_xgboost_residual.py subtracts from the actual target before
training XGBoost on what's left over.

CRITICAL for avoiding leakage: each Kreis's Prophet model is fit
ONLY on its training-period data (same chronological cutoff used
throughout this project). Values for the validation/test period come
from Prophet's own out-of-sample FORECAST, not from re-fitting on
data that includes them — exactly the same discipline as the
train/val/test splits used for XGBoost.

Usage:
    python fit_prophet_baseline.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore", module="prophet")
warnings.filterwarnings("ignore", module="cmdstanpy")

# 1. CONFIGURATION

MASTER_PATH = "./data/master/master_dataset_filled.parquet"   # raw targets, pre-feature-engineering
HOLIDAYS_PATH = "./data/prophet/prophet_holidays_national.csv"
OUTPUT_DIR = "./data/prophet/kreis_baselines"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_TARGET = "survstat_influenza"   # fit Prophet on the RAW weekly
                                      # series, not the shifted t_plus
                                      # labels — those get reconstructed
                                      # later via a join+shift
TEST_FRACTION = 0.15   # must match train_xgboost_final.py's split

# How far beyond the latest available date to forecast -- without this
# buffer, the baseline only covers dates that existed when this script
# last ran. Since master_dataset_filled.parquet grows weekly but this
# script doesn't, the gap between "latest feature week + horizon" and
# "baseline's forecast horizon" widens every week until predictions
# start missing a baseline entirely. Re-run this script periodically
# (e.g. yearly) to keep the buffer from running out, and rely on this
# setting as headroom between re-fits, not as a substitute for re-fitting.
FUTURE_BUFFER_WEEKS = 52

# Log-transform the target before fitting (Prophet fits on log1p(y),
# forecasts get back-transformed with expm1). Prophet's default
# additive model can produce a spurious upward trend that overshoots
# in the off-season -- e.g. predicting ~20 when the true off-season
# value is ~0. On the log scale, deviations near zero stay
# proportionally small after back-transforming, which keeps off-season
# forecasts anchored near the real floor instead of drifting upward.
USE_LOG_TRANSFORM = True

PROPHET_PARAMS: dict = {
    "yearly_seasonality": True,
    "weekly_seasonality": False,
    "daily_seasonality": False,
}

# 2. LOAD DATA

df: pd.DataFrame = pd.read_parquet(MASTER_PATH)
df = df.dropna(subset=[RAW_TARGET]).sort_values(["kreis_id", "week_start"]).reset_index(drop=True)

holidays: pd.DataFrame = pd.read_csv(HOLIDAYS_PATH, parse_dates=["ds"])

test_split_date = df["week_start"].quantile(1 - TEST_FRACTION)
print(f"Test split date: {test_split_date.date()} "
      f"(Prophet fit only on data before this)")

kreis_ids = df["kreis_id"].unique()
print(f"Fitting Prophet for {len(kreis_ids)} Kreise...\n")

# 3. FIT ONE PROPHET MODEL PER KREIS

all_baselines = []
failed_kreise = []

for i, kreis_id in enumerate(kreis_ids):
    kreis_df = df[df["kreis_id"] == kreis_id][["week_start", RAW_TARGET]].rename(
        columns={"week_start": "ds", RAW_TARGET: "y"}
    )

    kreis_df = kreis_df.drop_duplicates(subset="ds", keep="last")

    train_df = kreis_df[kreis_df["ds"] <= test_split_date].copy()

    if len(train_df) < 104:   # need at least ~2 years to fit yearly seasonality at all
        failed_kreise.append((kreis_id, f"only {len(train_df)} training weeks"))
        continue

    latest_date = kreis_df["ds"].max()
    future_dates = pd.date_range(
        start=latest_date + pd.Timedelta(weeks=1),
        periods=FUTURE_BUFFER_WEEKS, freq="W-MON",
    )
    full_range = pd.DataFrame({"ds": pd.concat([kreis_df["ds"], pd.Series(future_dates)], ignore_index=True)})

    try:
        fit_df = train_df.copy()
        if USE_LOG_TRANSFORM:
            fit_df["y"] = np.log1p(fit_df["y"])

        model = Prophet(holidays=holidays, **PROPHET_PARAMS)
        model.fit(fit_df)

        forecast = model.predict(full_range)
        yhat = np.expm1(forecast["yhat"]) if USE_LOG_TRANSFORM else forecast["yhat"]
        forecast["prophet_yhat"] = np.clip(yhat, 0, None)

        result = forecast[["ds", "prophet_yhat"]].copy()
        result["kreis_id"] = kreis_id
        all_baselines.append(result)

    except Exception as e:
        failed_kreise.append((kreis_id, str(e)[:100]))
        continue

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(kreis_ids)}] Kreise fitted...")

print(f"\nFitted {len(all_baselines)}/{len(kreis_ids)} Kreise successfully")
if failed_kreise:
    print(f"⚠️  {len(failed_kreise)} Kreise failed (usually too little history):")
    for kid, reason in failed_kreise[:10]:
        print(f"    {kid}: {reason}")

# 4. SAVE

baseline_df = pd.concat(all_baselines, ignore_index=True)
baseline_df = baseline_df.rename(columns={"ds": "week_start"})
baseline_df["year"] = baseline_df["week_start"].dt.isocalendar().year.astype(int)
baseline_df["week"] = baseline_df["week_start"].dt.isocalendar().week.astype(int)

out_path = os.path.join(OUTPUT_DIR, f"prophet_baseline_{RAW_TARGET}.parquet")
baseline_df.to_parquet(out_path, index=False)

print(f"\n✓ Saved: {out_path}  ({len(baseline_df):,} rows)")
print(f"Columns: {list(baseline_df.columns)}")

# Notes
# - Kreise with too little history (< 2 years of training data) are
#   simply skipped — train_xgboost_residual.py will have no baseline
#   for them and should fall back to raw-target prediction for those
#   specific Kreise
# - Fitting 400 separate Prophet models takes a while (roughly 1-3s
#   each) — this is a one-time cost, re-run only once per season,
#   not every prediction.
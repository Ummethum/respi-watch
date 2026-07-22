"""
Averages the last N weekly prediction snapshots (predictions_{date}.parquet,
already saved by generate_predictions.py) into one file -- shows how the
model's own next-week forecast has trended over the recent past, as a
smoothed view rather than a single week's possibly-noisy prediction.

Usage: python compute_recent_predictions_average.py
"""

import glob
import os
import re
from datetime import datetime
import pandas as pd

PREDICTIONS_DIR = "./data/predictions"
N_WEEKS = 4
DISEASE = "influenza"
HORIZON = 1   # average the "next week" forecast specifically, not t+2

OUTPUT_PATH = os.path.join(PREDICTIONS_DIR, "recent_avg_predictions.parquet")

# FIND THE LAST N DATED SNAPSHOTS

snapshot_files = glob.glob(os.path.join(PREDICTIONS_DIR, "predictions_*.parquet"))

def extract_date(path: str) -> datetime | None:
    match = re.search(r"predictions_(\d{4}-\d{2}-\d{2})\.parquet$", os.path.basename(path))
    return datetime.strptime(match.group(1), "%Y-%m-%d") if match else None

dated_files = sorted(
    [(extract_date(f), f) for f in snapshot_files if extract_date(f) is not None],
    key=lambda pair: pair[0],
)

if len(dated_files) < N_WEEKS:
    print(f"WARNING: only {len(dated_files)} dated snapshot(s) available, "
          f"need {N_WEEKS} for a full average -- using what's available. "
          f"This is expected for the first few weeks after starting the "
          f"weekly pipeline; it'll self-correct as more snapshots accumulate.")

recent_files = dated_files[-N_WEEKS:]
print(f"Averaging {len(recent_files)} snapshot(s): "
      f"{[d.strftime('%Y-%m-%d') for d, _ in recent_files]}")

# LOAD + AVERAGE

frames = []
for date, path in recent_files:
    df = pd.read_parquet(path)
    df = df[
        (df["disease"] == DISEASE) & (df["horizon"] == HORIZON)
        & (df.get("prediction_type", "forecast") == "forecast")
    ]
    frames.append(df[["kreis_id", "predicted_incidence"]])

combined = pd.concat(frames, ignore_index=True)
result = combined.groupby("kreis_id")["predicted_incidence"].mean().reset_index()
result["disease"] = DISEASE
result["horizon"] = HORIZON
result["n_weeks_averaged"] = len(recent_files)
result["generated_at"] = datetime.now()

result.to_parquet(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH} ({len(result)} Kreise)")

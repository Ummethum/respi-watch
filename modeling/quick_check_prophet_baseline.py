"""
Quick sanity-check plot: actual vs Prophet baseline for a handful of
Kreise, covering both history and the future buffer -- run this right
after fit_prophet_baseline.py to eyeball whether the off-season
overshoot fix and the future buffer actually look right, before
deciding whether to retrain XGBoost on top.

Usage: python quick_check_prophet_baseline.py
"""

import pandas as pd
import matplotlib.pyplot as plt

MASTER_PATH = "./data/master/master_dataset_filled.parquet"
BASELINE_PATH = "./data/prophet/kreis_baselines/prophet_baseline_survstat_influenza.parquet"
N_KREISE = 6
LAST_N_WEEKS_HISTORY = 104   # ~2 years of history, plus the full future buffer

master = pd.read_parquet(MASTER_PATH)
baseline = pd.read_parquet(BASELINE_PATH)

kreise = baseline["kreis_id"].unique()[:N_KREISE]

fig, axes = plt.subplots(len(kreise), 1, figsize=(12, 3 * len(kreise)), sharex=False)
if len(kreise) == 1:
    axes = [axes]

for ax, kreis_id in zip(axes, kreise):
    actual = master[master["kreis_id"] == kreis_id].sort_values("week_start").tail(LAST_N_WEEKS_HISTORY)
    base = baseline[baseline["kreis_id"] == kreis_id].sort_values("week_start")
    base = base[base["week_start"] >= actual["week_start"].min()]

    ax.plot(actual["week_start"], actual["survstat_influenza"], color="#2c3e50",
           marker="o", markersize=3, label="Actual")
    ax.plot(base["week_start"], base["prophet_yhat"], color="#c44e52",
           linestyle="--", label="Prophet baseline (incl. future buffer)")
    ax.axvline(actual["week_start"].max(), color="gray", linestyle=":", alpha=0.6,
              label="Last actual data point")
    ax.set_title(kreis_id, fontsize=10)
    ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig("prophet_baseline_check.png", dpi=150)
print("Saved: prophet_baseline_check.png")

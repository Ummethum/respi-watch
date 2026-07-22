"""
RespiWatch -- Generate weekly predictions for all 400 Kreise
====================================================================
Loads the (re)built master_dataset_features.parquet, takes the most
recent available week's feature row per Kreis, and runs it through
each trained model (per disease x horizon) to produce next-week and
week-after predictions -- the actual output your Streamlit app reads.

MODEL_REGISTRY below is the single place that decides which model file
to use for each (disease, horizon) combination, and whether it's a
plain XGBoost model or a Prophet-residual hybrid (needs the Prophet
baseline added back on top of the predicted residual). Update this
registry as you finalise which architecture you're actually shipping
per target -- you don't have to use the same approach for all 6.

Usage:
    python generate_predictions.py
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb

# 1. CONFIGURATION

FEATURES_PATH = "./data/master/master_dataset_features.parquet"
OUTPUT_DIR = "./data/predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_TARGETS = ["survstat_influenza", "survstat_covid", "survstat_rsv"]

MODEL_REGISTRY = [
    {
        "disease": "influenza", "horizon": 1,
        "model_path": "./data/models/xgboost_residual/model_residual_target_survstat_influenza_t_plus1.json",
        "residual": True,
        "prophet_baseline_path": "./data/prophet/kreis_baselines/prophet_baseline_survstat_influenza.parquet",
    },
    {
        "disease": "influenza", "horizon": 2,
        "model_path": "./data/models/xgboost_residual/model_residual_target_survstat_influenza_t_plus2.json",
        "residual": True,
        "prophet_baseline_path": "./data/prophet/kreis_baselines/prophet_baseline_survstat_influenza.parquet",
    },
    # Add covid/rsv entries here once those models exist, e.g.:
    # {
    #     "disease": "covid", "horizon": 1,
    #     "model_path": "./data/models/xgboost_final/model_target_survstat_covid_t_plus1.json",
    #     "residual": False,
    # },
]

CATEGORICAL_FEATURES = ["kreis_id", "nuts1_code"]
ALWAYS_EXCLUDE = ["name", "bundesland_name", "week_start", "year", "week"] + RAW_TARGETS

# 2. LOAD FEATURES, ISOLATE THE MOST RECENT WEEK PER KREIS

df: pd.DataFrame = pd.read_parquet(FEATURES_PATH)

latest_week_start = df["week_start"].max()
latest_df = df[df["week_start"] == latest_week_start].copy()

print(f"Most recent available week: {latest_week_start.date()}")
print(f"Kreise with data for this week: {latest_df['kreis_id'].nunique()} / "
      f"{df['kreis_id'].nunique()}")

missing_kreise = set(df["kreis_id"].unique()) - set(latest_df["kreis_id"].unique())
if missing_kreise:
    print(f"\u26a0\ufe0f  {len(missing_kreise)} Kreise have NO data for the most "
          f"recent week (likely a missed upstream fetch for that Kreis) -- "
          f"they will have no prediction this run: {sorted(missing_kreise)[:10]}")

all_label_cols = [c for c in df.columns if c.startswith("target_")]
survstat_feature_cols = [c for c in df.columns if c.startswith("survstat_")]

feature_cols = [
    c for c in df.columns
    if c not in ALWAYS_EXCLUDE
    and c not in all_label_cols
    and c not in survstat_feature_cols
]

X_latest: pd.DataFrame = latest_df[feature_cols].copy()
for col in CATEGORICAL_FEATURES:
    X_latest[col] = X_latest[col].astype("category")

# 3. PREDICT WITH EACH REGISTERED MODEL

def predict_with_registry_entry(entry: dict, X: pd.DataFrame,
                                latest_df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(entry["model_path"]):
        print(f"  \u26a0\ufe0f  Model file not found, skipping: {entry['model_path']}")
        return pd.DataFrame()

    model = xgb.XGBRegressor()
    model.load_model(entry["model_path"])

    raw_pred = model.predict(X)

    if entry["residual"]:
        baseline = pd.read_parquet(entry["prophet_baseline_path"])
        baseline_sorted = (
        baseline.sort_values(["kreis_id", "week_start"])
        .drop_duplicates(subset=["kreis_id", "week_start"], keep="last")
        .copy()
    )
        baseline_sorted["prophet_yhat_future"] = (
            baseline_sorted.groupby("kreis_id")["prophet_yhat"].shift(-entry["horizon"])
        )
        # Match each Kreis's baseline value for the SAME base week
        # used to build X -- i.e. the baseline forecast for
        # (latest_week_start + horizon), aligned via the shift above.
        merged = latest_df[["kreis_id", "week_start"]].merge(
            baseline_sorted[["kreis_id", "week_start", "prophet_yhat_future"]],
            on=["kreis_id", "week_start"], how="left",
        )
        baseline_component = merged["prophet_yhat_future"].values

        n_missing = pd.isna(baseline_component).sum()
        if n_missing > 0:
            print(f"  \u26a0\ufe0f  {n_missing} Kreise missing a Prophet baseline for "
                  f"this week -- their prediction falls back to the raw "
                  f"residual prediction alone (likely an underestimate).")
            baseline_component = np.nan_to_num(baseline_component, nan=0.0)

        final_pred = np.clip(baseline_component + raw_pred, 0, None)
    else:
        final_pred = np.clip(raw_pred, 0, None)

    target_week = latest_df["week_start"] + pd.Timedelta(weeks=entry["horizon"])

    return pd.DataFrame({
        "kreis_id": latest_df["kreis_id"].values,
        "disease": entry["disease"],
        "horizon": entry["horizon"],
        "base_week": latest_df["week_start"].values,
        "target_week": target_week.values,
        "predicted_incidence": final_pred,
    })


if __name__ == "__main__":
    print(f"\nGenerating predictions for {len(MODEL_REGISTRY)} "
          f"(disease, horizon) combinations...\n")

    all_predictions = []
    for entry in MODEL_REGISTRY:
        print(f"{entry['disease']} t+{entry['horizon']}...")
        pred_df = predict_with_registry_entry(entry, X_latest, latest_df)
        if not pred_df.empty:
            all_predictions.append(pred_df)
            print(f"  OK: {len(pred_df)} Kreis predictions "
                  f"(target week: {pred_df['target_week'].iloc[0].date()})")

    if not all_predictions:
        raise SystemExit("No predictions generated -- check MODEL_REGISTRY "
                        "paths before investigating further.")

    combined = pd.concat(all_predictions, ignore_index=True)
    combined["generated_at"] = datetime.now()

    # "Latest" file -- what the Streamlit app should actually read
    latest_path = os.path.join(OUTPUT_DIR, "latest_predictions.parquet")
    combined.to_parquet(latest_path, index=False)

    # Dated snapshot -- keeps a history of what was predicted when,
    # useful later for checking your own forecast accuracy over time
    snapshot_path = os.path.join(
        OUTPUT_DIR, f"predictions_{datetime.now():%Y-%m-%d}.parquet"
    )
    combined.to_parquet(snapshot_path, index=False)

    print(f"\n\u2713 Saved: {latest_path} ({len(combined):,} rows)")
    print(f"\u2713 Saved snapshot: {snapshot_path}")

    print(f"\nSample predictions:")
    print(combined.head(10).to_string(index=False))

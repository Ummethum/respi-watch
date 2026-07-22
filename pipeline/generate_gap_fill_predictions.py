"""
RespiWatch -- Gap-fill ("nowcast") predictions between last known
SurvStat data and the latest available feature week
========================================================================
generate_predictions.py produces genuine FUTURE forecasts (t+1/t+2 from
the latest available feature week). This script fills the separate
gap that exists BEFORE that point: SurvStat's own reporting lag means
the last 2-3 weeks of REAL incidence data aren't available yet, even
though weather/trends/AMELAG/etc. feature data for those same weeks
already exists (those sources don't share SurvStat's lag).

Since the models use NO autoregressive survstat_* features (a
deliberate earlier decision, due to SurvStat's own reporting-lag
issue), each week's prediction is fully independent of any other --
there's no recursive "prediction feeds into next prediction" problem.
This means the t+1 model can simply be applied to EVERY anchor week's
already-available feature row between the last known SurvStat week and
the latest feature week, one week at a time, walking forward:

    anchor week W   (last known real SurvStat value)
      -> predict target week W+1 using the t+1 model on week W's features
    anchor week W+1 (features already available, real SurvStat not yet)
      -> predict target week W+2
    anchor week W+2
      -> predict target week W+3 (== latest available feature week)

Combined with generate_predictions.py's own t+1/t+2 forecasts from the
latest feature week, this gives a CONTINUOUS predicted line with no
gap, from right after the last real data point through to the actual
2-week-ahead forecast horizon.

Usage:
    python generate_gap_fill_predictions.py
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb

# -----------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------

FEATURES_PATH = "./data/master/master_dataset_features.parquet"
OUTPUT_DIR = "./data/predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_TARGETS = ["survstat_influenza", "survstat_covid", "survstat_rsv"]

# Only t_plus1 models needed here -- applied repeatedly, one anchor
# week at a time, to walk across the whole gap. Uses the SAME registry
# format/semantics as generate_predictions.py's MODEL_REGISTRY.
GAP_FILL_MODEL_REGISTRY = [
    {
        "disease": "influenza",
        "model_path": "./data/models/xgboost_residual/model_residual_target_survstat_influenza_t_plus1.json",
        "residual": True,
        "prophet_baseline_path": "./data/prophet/kreis_baselines/prophet_baseline_survstat_influenza.parquet",
    },
    # Add covid/rsv entries here once those t_plus1 models exist, e.g.:
    # {
    #     "disease": "covid",
    #     "model_path": "./data/models/xgboost_final/model_target_survstat_covid_t_plus1.json",
    #     "residual": False,
    # },
]

CATEGORICAL_FEATURES = ["kreis_id", "nuts1_code"]
ALWAYS_EXCLUDE = ["name", "bundesland_name", "week_start", "year", "week"] + RAW_TARGETS

MAX_GAP_WEEKS = 6   # safety cap -- if the gap between last known
                     # SurvStat and the latest feature week is somehow
                     # larger than this (e.g. a broken pipeline run),
                     # stop rather than silently generating a huge
                     # chain of nowcasts that's probably a symptom of
                     # something else being wrong upstream

# 2. LOAD FEATURES

df: pd.DataFrame = pd.read_parquet(FEATURES_PATH)
df = df.sort_values(["kreis_id", "week_start"]).reset_index(drop=True)

all_label_cols = [c for c in df.columns if c.startswith("target_")]
survstat_feature_cols = [c for c in df.columns if c.startswith("survstat_")]

feature_cols = [
    c for c in df.columns
    if c not in ALWAYS_EXCLUDE
    and c not in all_label_cols
    and c not in survstat_feature_cols
]

# 3. FIND EACH KREIS'S GAP: (last known SurvStat week, latest feature week]

def get_last_known_survstat_week(kreis_df: pd.DataFrame, raw_target_col: str) -> pd.Timestamp | None:
    valid = kreis_df[kreis_df[raw_target_col].notna()]
    if valid.empty:
        return None
    return valid["week_start"].max()


def build_gap_weeks(kreis_df: pd.DataFrame, raw_target_col: str,
                    latest_feature_week: pd.Timestamp) -> list[pd.Timestamp]:
    """
    Returns the list of ANCHOR weeks (not target weeks) to walk
    through for one Kreis -- i.e. every week from the last known real
    SurvStat week through the week just before the latest feature
    week. Each anchor week's t+1 prediction fills in the target week
    right after it.
    """
    last_known = get_last_known_survstat_week(kreis_df, raw_target_col)
    if last_known is None:
        return []

    anchor_weeks = []
    current = last_known
    while current < latest_feature_week:
        anchor_weeks.append(current)
        current += pd.Timedelta(weeks=1)
        if len(anchor_weeks) > MAX_GAP_WEEKS:
            print(f"  \u26a0\ufe0f  Gap exceeds MAX_GAP_WEEKS={MAX_GAP_WEEKS} for this "
                  f"Kreis -- stopping early, check upstream data freshness.")
            break

    return anchor_weeks

# 4. PREDICT ONE ANCHOR WEEK -> ONE TARGET WEEK (t+1)

def predict_single_week(model: xgb.XGBRegressor, entry: dict,
                        anchor_row: pd.DataFrame) -> float:
    X = anchor_row[feature_cols].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    raw_pred = model.predict(X)[0]

    if entry["residual"]:
        baseline = pd.read_parquet(entry["prophet_baseline_path"])
        baseline_row = baseline[
            (baseline["kreis_id"] == anchor_row["kreis_id"].iloc[0]) &
            (baseline["week_start"] == anchor_row["week_start"].iloc[0] + pd.Timedelta(weeks=1))
        ]
        if baseline_row.empty:
            return np.nan   # no Prophet baseline for this target week -- caller handles
        baseline_value = baseline_row["prophet_yhat"].iloc[0]
        return float(np.clip(baseline_value + raw_pred, 0, None))

    return float(np.clip(raw_pred, 0, None))

# 5. RUN

if __name__ == "__main__":
    latest_feature_week = df["week_start"].max()
    print(f"Latest available feature week: {latest_feature_week.date()}")

    all_nowcasts = []

    for entry in GAP_FILL_MODEL_REGISTRY:
        disease = entry["disease"]
        raw_target_col = f"survstat_{disease}"
        print(f"\n{'='*60}\n{disease}\n{'='*60}")

        if not os.path.exists(entry["model_path"]):
            print(f"  \u26a0\ufe0f  Model not found, skipping: {entry['model_path']}")
            continue

        model = xgb.XGBRegressor()
        model.load_model(entry["model_path"])

        for kreis_id, kreis_df in df.groupby("kreis_id"):
            gap_anchor_weeks = build_gap_weeks(kreis_df, raw_target_col, latest_feature_week)

            if not gap_anchor_weeks:
                continue   # no gap (either no SurvStat history at all, or
                            # already fully up to date -- both fine, just
                            # nothing to fill for this Kreis/disease

            for anchor_week in gap_anchor_weeks:
                anchor_row = kreis_df[kreis_df["week_start"] == anchor_week]
                if anchor_row.empty:
                    continue   # feature row missing for this specific week
                                # (e.g. a one-off upstream fetch failure) --
                                # skip this single week rather than crash

                predicted = predict_single_week(model, entry, anchor_row)
                if pd.isna(predicted):
                    continue

                all_nowcasts.append({
                    "kreis_id": kreis_id,
                    "disease": disease,
                    "horizon": 1,
                    "base_week": anchor_week,
                    "target_week": anchor_week + pd.Timedelta(weeks=1),
                    "predicted_incidence": predicted,
                    "prediction_type": "nowcast",   # distinguishes this from
                                                      # generate_predictions.py's
                                                      # "forecast" rows
                })

        n_kreise_with_gap = df.groupby("kreis_id").apply(
            lambda g: len(build_gap_weeks(g, raw_target_col, latest_feature_week)) > 0,
            include_groups=False,
        ).sum()
        print(f"  {n_kreise_with_gap} Kreise had a gap to fill")

    if not all_nowcasts:
        print("\nNo gap-fill predictions generated (no gap found anywhere, or "
              "no models available).")
    else:
        nowcast_df = pd.DataFrame(all_nowcasts)
        nowcast_df["generated_at"] = datetime.now()

        out_path = os.path.join(OUTPUT_DIR, "gap_fill_predictions.parquet")
        nowcast_df.to_parquet(out_path, index=False)

        print(f"\n\u2713 Saved: {out_path} ({len(nowcast_df):,} rows)")
        print(f"\nSample:")
        print(nowcast_df.head(10).to_string(index=False))


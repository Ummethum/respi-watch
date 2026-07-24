"""
Builds a prediction-history file for the app's "history of predictions"
chart -- entirely self-contained, does NOT read predictions_{date}.parquet
snapshots or latest_predictions.parquet. Instead, reconstructs what the
CURRENT model would have predicted for each of the past N weeks by
walking backward and re-running it at each historical anchor week
(same technique as bootstrap_recent_avg_predictions.py), then extends
two weeks further with the model's genuine current forecast.

Why reconstruct rather than accumulate real past snapshots: this stays
consistent with whatever model is CURRENTLY live -- an accumulated
history mixes predictions from whatever model happened to be live each
particular week, which drifts out of sync with itself every time the
model gets retrained. A freshly reconstructed history is always
internally consistent, at the cost of not being a genuine record of
"what we actually told users at the time".

Output rows, by target_week:
    - N_WEEKS_HISTORY rows of reconstructed one-week-ahead predictions,
      the last of which IS the genuine "next week" forecast (horizon=1
      from the latest available anchor)
    - one additional row: the genuine "in 2 weeks" forecast (horizon=2
      from the latest available anchor)
These last two rows are the only ones beyond what's currently known --
the app is expected to plot them only on the PREDICTION chart, never
merged into the actual-incidence chart, since no real data exists yet
for those weeks.

Usage: python build_prediction_history.py
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb

FEATURES_PATH = "./data/master/master_dataset_features.parquet"
PROPHET_BASELINE_PATH = "./data/prophet/kreis_baselines/prophet_baseline_survstat_influenza.parquet"
MODEL_PATHS = {
    1: "./data/models/xgboost_residual/model_residual_target_survstat_influenza_t_plus1.json",
    2: "./data/models/xgboost_residual/model_residual_target_survstat_influenza_t_plus2.json",
}
OUTPUT_PATH = "./data/predictions/prediction_history.parquet"

DISEASE = "influenza"
N_WEEKS_HISTORY = 26   # matches the ~6 months shown on the actual-incidence chart
RAW_TARGETS = ["survstat_influenza", "survstat_covid", "survstat_rsv"]
ALWAYS_EXCLUDE = ["name", "bundesland_name", "week_start", "year", "week"] + RAW_TARGETS
CATEGORICAL_FEATURES = ["kreis_id", "nuts1_code"]


def predict_at_anchor(df: pd.DataFrame, anchor: pd.Timestamp, horizon: int,
                      model: xgb.XGBRegressor, baseline: pd.DataFrame) -> pd.DataFrame:
    anchor_df = df[df["week_start"] == anchor].copy()
    anchor_df = anchor_df.drop_duplicates(subset=["kreis_id", "week_start"], keep="last")
    if anchor_df.empty:
        return pd.DataFrame()

    label_cols = [c for c in df.columns if c.startswith("target_")]
    survstat_cols = [c for c in df.columns if c.startswith("survstat_")]
    feature_cols = [c for c in df.columns if c not in ALWAYS_EXCLUDE + label_cols + survstat_cols]

    X = anchor_df[feature_cols].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    residual_pred = model.predict(X)

    baseline_sorted = baseline.sort_values(["kreis_id", "week_start"]).drop_duplicates(
        subset=["kreis_id", "week_start"], keep="last"
    )
    target_week = anchor + pd.Timedelta(weeks=horizon)
    baseline_at_target = baseline_sorted[baseline_sorted["week_start"] == target_week][
        ["kreis_id", "prophet_yhat"]
    ]

    merged = anchor_df[["kreis_id"]].merge(baseline_at_target, on="kreis_id", how="left")
    predicted = np.clip(merged["prophet_yhat"].values + residual_pred, 0, None)

    return pd.DataFrame({
        "kreis_id": anchor_df["kreis_id"].values,
        "target_week": target_week,
        "predicted_incidence": predicted,
    })


if __name__ == "__main__":
    df = pd.read_parquet(FEATURES_PATH)
    baseline = pd.read_parquet(PROPHET_BASELINE_PATH)

    model_t1 = xgb.XGBRegressor()
    model_t1.load_model(MODEL_PATHS[1])
    model_t2 = xgb.XGBRegressor()
    model_t2.load_model(MODEL_PATHS[2])

    latest_feature_week = df["week_start"].max()
    print(f"Latest available feature week: {latest_feature_week.date()}")
    print(f"Reconstructing {N_WEEKS_HISTORY} weeks of one-week-ahead predictions...")

    history_frames = []
    for weeks_back in range(N_WEEKS_HISTORY - 1, -1, -1):   # oldest first
        anchor = latest_feature_week - pd.Timedelta(weeks=weeks_back)
        pred = predict_at_anchor(df, anchor, 1, model_t1, baseline)
        if not pred.empty:
            history_frames.append(pred)

    if not history_frames:
        raise SystemExit("No weeks could be reconstructed -- check FEATURES_PATH "
                        "has enough recent data.")

    # Extra point: the genuine "in 2 weeks" forecast, extending one
    # week further than the history loop's last point (which already
    # covers "next week" via horizon=1 from the latest anchor).
    future_t2 = predict_at_anchor(df, latest_feature_week, 2, model_t2, baseline)

    history = pd.concat(history_frames + [future_t2], ignore_index=True)
    history = history.drop_duplicates(subset=["kreis_id", "target_week"], keep="last")
    history = history.sort_values(["kreis_id", "target_week"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    history.to_parquet(OUTPUT_PATH, index=False)

    n_weeks = history["target_week"].nunique()
    print(f"\nSaved: {OUTPUT_PATH} ({len(history):,} rows, "
          f"{history['kreis_id'].nunique()} Kreise, {n_weeks} distinct weeks)")
    print(f"Target week range: {history['target_week'].min().date()} to "
          f"{history['target_week'].max().date()} "
          f"(last 2 of these are genuine future forecasts, not history)")
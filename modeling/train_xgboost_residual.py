"""
RespiWatch — XGBoost on Prophet residuals
==============================================
Instead of predicting the raw incidence directly, XGBoost here learns
to predict the RESIDUAL between the actual value and Prophet's smooth
seasonal baseline (fit_prophet_baseline.py) — the part of the signal
Prophet's yearly-seasonality curve can't explain, which is exactly
what the exogenous features (weather, trends, AMELAG, ...) should be
informative about.

Final prediction = Prophet's baseline forecast for that future week
                    + XGBoost's predicted residual for that week

Usage:
    python train_xgboost_residual.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. CONFIGURATION

FEATURES_PATH = "./data/master/master_dataset_features.parquet"
PROPHET_BASELINE_PATH = "./data/prophet/kreis_baselines/prophet_baseline_survstat_influenza.parquet"
OUTPUT_DIR = "./data/models/xgboost_residual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_LABEL = "target_survstat_influenza_t_plus1"
HORIZON = 1   # must match the "_t_plus{H}" in TARGET_LABEL above —
              # used to shift the Prophet baseline to the right future week
RAW_TARGET_COL = "survstat_influenza"   # the un-shifted raw column,
                                          # needed to align with Prophet's
                                          # own week_start-indexed output

TEST_FRACTION = 0.15
VALIDATION_FRACTION = 0.15
EARLY_STOPPING_ROUNDS = 20
N_ESTIMATORS_MAX = 1000

EXAMPLE_KREIS_ID = "DE21H"
BEST_PARAMS: dict = {}

# 2. LOAD FEATURES + PROPHET BASELINE

df: pd.DataFrame = pd.read_parquet(FEATURES_PATH)
df = df.dropna(subset=[TARGET_LABEL]).reset_index(drop=True)
df = df.sort_values(["kreis_id", "week_start"]).reset_index(drop=True)

baseline: pd.DataFrame = pd.read_parquet(PROPHET_BASELINE_PATH)
print(f"Loaded {len(df):,} feature rows, {len(baseline):,} Prophet baseline rows "
      f"({baseline['kreis_id'].nunique()} Kreise)")

# 3. ALIGN PROPHET BASELINE TO THE FORECAST HORIZON

baseline_sorted = baseline.sort_values(["kreis_id", "week_start"]).copy()
baseline_sorted["prophet_yhat_future"] = (
    baseline_sorted.groupby("kreis_id")["prophet_yhat"].shift(-HORIZON)
)

df = df.merge(
    baseline_sorted[["kreis_id", "week_start", "prophet_yhat_future"]],
    on=["kreis_id", "week_start"], how="left",
)

n_missing_baseline = df["prophet_yhat_future"].isna().sum()
print(f"Rows without a Prophet baseline: {n_missing_baseline:,} "
      f"({n_missing_baseline/len(df)*100:.1f}%) — likely Kreise Prophet "
      f"couldn't fit (too little history) or edge-of-series rows")

# Only keep rows where we actually have a baseline to compute a residual against
df = df.dropna(subset=["prophet_yhat_future"]).reset_index(drop=True)

# 4. COMPUTE RESIDUAL TARGET

df["residual_target"] = df[TARGET_LABEL] - df["prophet_yhat_future"]

print(f"\nResidual target summary:")
print(df["residual_target"].describe().to_string())
print(f"\n(Compare to raw target range: {df[TARGET_LABEL].min():.1f} to "
      f"{df[TARGET_LABEL].max():.1f} — residuals should be a much "
      f"narrower range centered near 0, since Prophet already explains "
      f"the bulk of the seasonal swing)")


# 5. FEATURE SELECTION

RAW_TARGETS = ["survstat_influenza", "survstat_covid", "survstat_rsv"]
ALWAYS_EXCLUDE = [
    "name", "bundesland_name", "week_start", "year", "week",
    "prophet_yhat_future", "residual_target",
] + RAW_TARGETS
all_label_cols = [c for c in df.columns if c.startswith("target_")]
other_labels_to_exclude = [c for c in all_label_cols if c != TARGET_LABEL]
survstat_feature_cols = [c for c in df.columns if c.startswith("survstat_")]
CATEGORICAL_FEATURES = ["kreis_id", "nuts1_code"]

feature_cols = [
    c for c in df.columns
    if c not in ALWAYS_EXCLUDE
    and c not in other_labels_to_exclude
    and c not in survstat_feature_cols
    and c != TARGET_LABEL
]

X: pd.DataFrame = df[feature_cols].copy()
y_residual: pd.Series = df["residual_target"].copy()
for col in CATEGORICAL_FEATURES:
    X[col] = X[col].astype("category")

print(f"\nFeatures: {len(feature_cols)}  |  Kreise: {df['kreis_id'].nunique()}")

# 6. TRAIN / VALIDATION / TEST SPLIT

test_split_date = df["week_start"].quantile(1 - TEST_FRACTION)
val_split_date = df["week_start"].quantile(1 - TEST_FRACTION - VALIDATION_FRACTION)

train_mask = df["week_start"] <= val_split_date
val_mask = (df["week_start"] > val_split_date) & (df["week_start"] <= test_split_date)
test_mask = df["week_start"] > test_split_date

X_train, y_train = X[train_mask], y_residual[train_mask]
X_val, y_val = X[val_mask], y_residual[val_mask]
X_test, y_test_residual = X[test_mask], y_residual[test_mask]

print(f"\nTrain: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")


# 7. FIT XGBOOST ON RESIDUALS

# Note: residuals can be negative (actual below Prophet's baseline),

model = xgb.XGBRegressor(
    tree_method="hist",
    enable_categorical=True,
    random_state=42,
    n_estimators=N_ESTIMATORS_MAX,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    eval_metric="rmse",
    **BEST_PARAMS,
)
print("\nFitting XGBoost on residuals...")
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print(f"Stopped at {model.best_iteration} trees")


# 8. RECONSTRUCT FINAL PREDICTION & EVALUATE

# The business-relevant evaluation is on the RECONSTRUCTED absolute
# prediction (Prophet baseline + predicted residual), not on how well
# XGBoost predicted the residual in isolation.

residual_pred_test = model.predict(X_test)
prophet_baseline_test = df.loc[test_mask, "prophet_yhat_future"].values
final_pred_test = np.clip(prophet_baseline_test + residual_pred_test, 0, None)

actual_test = df.loc[test_mask, TARGET_LABEL].values

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom == 0, np.nan, denom)
    return np.nanmean(100 * np.abs(y_true - y_pred) / denom)

hybrid_rmse = mean_squared_error(actual_test, final_pred_test) ** 0.5
hybrid_mae = mean_absolute_error(actual_test, final_pred_test)
hybrid_smape = smape(actual_test, final_pred_test)

# Reference: Prophet's baseline ALONE, with no XGBoost correction at all
prophet_only_rmse = mean_squared_error(actual_test, prophet_baseline_test) ** 0.5

print(f"\n{'='*60}")
print(f"Reconstructed prediction — held-out test set")
print("=" * 60)
print(f"Prophet baseline alone (no XGBoost) RMSE : {prophet_only_rmse:.3f}")
print(f"Hybrid (Prophet + XGBoost residual) RMSE : {hybrid_rmse:.3f}")
print(f"Hybrid MAE                               : {hybrid_mae:.3f}")
print(f"Hybrid SMAPE                             : {hybrid_smape:.2f}%")
print(f"\nImprovement over Prophet alone: "
      f"{(1 - hybrid_rmse/prophet_only_rmse)*100:.1f}%")
print("(If this is small or negative, XGBoost isn't adding much beyond "
      "the seasonal baseline — check feature importance below to see "
      "if any exogenous feature is contributing meaningfully at all.)")

# 9. PLOTS

# Feature importance (on the RESIDUAL model)
importance = pd.Series(model.feature_importances_, index=feature_cols)
importance = importance.sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(importance.index[::-1], importance.values[::-1], color="#55a868")
ax.set_title(f"Top 20 feature importances — residual model, {TARGET_LABEL}")
ax.set_xlabel("Importance (gain)")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "residual_feature_importance.png"), dpi=150)
plt.close(fig)
print(f"\n✓ Saved: residual_feature_importance.png")

# Example time series: actual vs Prophet-only vs hybrid, one Kreis
example_mask = test_mask & (df["kreis_id"] == EXAMPLE_KREIS_ID)
example_df = df[example_mask].sort_values("week_start").copy()
example_X = X[example_mask]
example_residual_pred = model.predict(example_X)
example_df["hybrid_pred"] = np.clip(
    example_df["prophet_yhat_future"] + example_residual_pred, 0, None
)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(example_df["week_start"], example_df[TARGET_LABEL],
       color="#2c3e50", linewidth=2, marker="o", markersize=3, label="Actual")
ax.plot(example_df["week_start"], example_df["prophet_yhat_future"],
       color="#8c8c8c", linewidth=1.8, linestyle=":", label="Prophet baseline only")
ax.plot(example_df["week_start"], example_df["hybrid_pred"],
       color="#c44e52", linewidth=1.8, linestyle="--", label="Hybrid (Prophet + XGBoost)")
ax.set_title(f"{TARGET_LABEL} — {EXAMPLE_KREIS_ID} (test period)")
ax.set_xlabel("Week")
ax.set_ylabel("Incidence per 100,000")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, f"hybrid_timeseries_{EXAMPLE_KREIS_ID}.png"), dpi=150)
plt.close(fig)
print(f"✓ Saved: hybrid_timeseries_{EXAMPLE_KREIS_ID}.png")

model.save_model(os.path.join(OUTPUT_DIR, f"model_residual_{TARGET_LABEL}.json"))
print(f"\n✓ Saved model: model_residual_{TARGET_LABEL}.json")

# ── Notes ────────────────────────────────────────────────────────────────
# - Kreise Prophet couldn't fit (too little history) are silently
#   dropped from this whole pipeline

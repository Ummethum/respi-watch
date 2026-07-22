"""
National-level line chart: predicted vs. actual influenza incidence,
averaged across all Kreise, for a chosen date range

National average here is an unweighted mean across Kreise (each
Kreis's incidence is already per-100,000, so this is "the average
Kreis's incidence", not a population-weighted true national rate

Usage: python plot_national_predicted_vs_actual.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

FEATURES_PATH = "./data/master/master_dataset_features.parquet"
PROPHET_BASELINE_PATH = "./data/prophet/kreis_baselines/prophet_baseline_survstat_influenza.parquet"
MODEL_PATH = "./data/models/xgboost_residual/model_residual_target_survstat_influenza_t_plus1.json"

TARGET_LABEL = "target_survstat_influenza_t_plus1"
HORIZON = 1
CATEGORICAL_FEATURES = ["kreis_id", "nuts1_code"]

START_DATE = pd.Timestamp("2024-12-16")
END_DATE = pd.Timestamp("2025-01-20")

DARK_MODE = True

FONT_SIZE = 16

OUTPUT_PATH = "national_predicted_vs_actual.png"

# RECONSTRUCT PREDICTIONS (same pattern as evaluate_residual_model.py)

df: pd.DataFrame = pd.read_parquet(FEATURES_PATH)
df = df.dropna(subset=[TARGET_LABEL]).reset_index(drop=True)
df = df.sort_values(["kreis_id", "week_start"]).reset_index(drop=True)

baseline: pd.DataFrame = pd.read_parquet(PROPHET_BASELINE_PATH)
baseline_sorted = baseline.sort_values(["kreis_id", "week_start"]).copy()
baseline_sorted["prophet_yhat_future"] = (
    baseline_sorted.groupby("kreis_id")["prophet_yhat"].shift(-HORIZON)
)
df = df.merge(
    baseline_sorted[["kreis_id", "week_start", "prophet_yhat_future"]],
    on=["kreis_id", "week_start"], how="left",
)
df = df.dropna(subset=["prophet_yhat_future"]).reset_index(drop=True)

RAW_TARGETS = ["survstat_influenza", "survstat_covid", "survstat_rsv"]
ALWAYS_EXCLUDE = [
    "name", "bundesland_name", "week_start", "year", "week",
    "prophet_yhat_future",
] + RAW_TARGETS
all_label_cols = [c for c in df.columns if c.startswith("target_")]
other_labels_to_exclude = [c for c in all_label_cols if c != TARGET_LABEL]
survstat_feature_cols = [c for c in df.columns if c.startswith("survstat_")]

feature_cols = [
    c for c in df.columns
    if c not in ALWAYS_EXCLUDE
    and c not in other_labels_to_exclude
    and c not in survstat_feature_cols
    and c != TARGET_LABEL
]

X = df[feature_cols].copy()
for col in CATEGORICAL_FEATURES:
    X[col] = X[col].astype("category")

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)
residual_pred = model.predict(X)
df["predicted"] = np.clip(df["prophet_yhat_future"] + residual_pred, 0, None)
df["actual"] = df[TARGET_LABEL]

# FILTER TO THE CHOSEN WINDOW, AGGREGATE TO NATIONAL MEAN

# TARGET_LABEL (t+1) means df["actual"]/["predicted"] at row week_start
# represent the value ONE WEEK LATER -- align to that target week so
# the x-axis reflects the week the number is actually FOR, not the
# anchor week the prediction was made from.
df["target_week"] = df["week_start"] + pd.Timedelta(weeks=HORIZON)

window = df[(df["target_week"] >= START_DATE) & (df["target_week"] <= END_DATE)]

if window.empty:
    raise SystemExit(f"No data found for {START_DATE.date()} to {END_DATE.date()} -- "
                    f"check the date range falls within your dataset's coverage.")

national = window.groupby("target_week")[["actual", "predicted"]].mean().reset_index()
print(f"{len(national)} week(s) in range, {window['kreis_id'].nunique()} Kreise averaged per week")

# PLOT

if DARK_MODE:
    plt.style.use("dark_background")
    bg_color = "#0e1117"       
    actual_color = "#f0f2f6"
    predicted_color = "#e07b7b"
else:
    bg_color = "white"
    actual_color = "#2c3e50"
    predicted_color = "#c44e52"

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE * 1.2,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE * 0.85,
    "ytick.labelsize": FONT_SIZE * 0.85,
    "legend.fontsize": FONT_SIZE * 0.85,
})

fig, ax = plt.subplots(figsize=(10, 5))
if DARK_MODE:
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

ax.plot(national["target_week"], national["actual"], color=actual_color,
       marker="o", markersize=5, label="Actual (national average)", linewidth=2)
ax.plot(national["target_week"], national["predicted"], color=predicted_color,
       marker="o", markersize=5, linestyle="--", label="Predicted (national average)", linewidth=2)

ax.set_xlabel("Week")
ax.set_ylabel("Influenza incidence (avg. across Kreise)")
ax.set_title(f"National influenza incidence: predicted vs. actual\n"
            f"{START_DATE.strftime('%b %d, %Y')} -- {END_DATE.strftime('%b %d, %Y')}")
ax.legend()
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=200, facecolor=bg_color)
print(f"Saved: {OUTPUT_PATH} ({'dark' if DARK_MODE else 'light'} mode)")

mae = (national["actual"] - national["predicted"]).abs().mean()
print(f"Mean absolute error over this window: {mae:.2f}")

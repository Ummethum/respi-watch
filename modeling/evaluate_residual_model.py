"""
RespiWatch -- Evaluation of the Prophet-residual hybrid model
====================================================================

    1. best/median/worst example Kreise.

    2. FALSE ALARM ANALYSIS -- how many Kreise does the model
       "invent" a wave for (predicts a clear rise) when actual
       incidence stays low/flat?

Usage:
    python evaluate_residual_model.py
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
MODEL_PATH = "./data/models/xgboost_residual/model_residual_target_survstat_influenza_t_plus1.json"

TARGET_LABEL = "target_survstat_influenza_t_plus1"
HORIZON = 1
TEST_FRACTION = 0.15

OUTPUT_DIR = "./data/models/evaluation_residual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# How many example Kreise to show per category (best/median/worst) --
N_EXAMPLES_PER_CATEGORY = 5

# False-alarm definition: a week counts as a "false alarm" if the
# ACTUAL value stayed low (<= LOW_PERCENTILE of all actual test
# values -- "near 0", in the user's words) but the model PREDICTED a
# clear rise (>= WAVE_PERCENTILE of all actual test values -- i.e. a
# level that would normally only occur during a real wave).
LOW_PERCENTILE = 25     # "stayed low" threshold
WAVE_PERCENTILE = 75    # "predicted a wave" threshold

# 2. LOAD DATA & MODEL, RECONSTRUCT PREDICTIONS

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
CATEGORICAL_FEATURES = ["kreis_id", "nuts1_code"]

feature_cols = [
    c for c in df.columns
    if c not in ALWAYS_EXCLUDE
    and c not in other_labels_to_exclude
    and c not in survstat_feature_cols
    and c != TARGET_LABEL
]

X: pd.DataFrame = df[feature_cols].copy()
for col in CATEGORICAL_FEATURES:
    X[col] = X[col].astype("category")

test_split_date = df["week_start"].quantile(1 - TEST_FRACTION)
test_mask = df["week_start"] > test_split_date

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

residual_pred = model.predict(X)
df["predicted"] = np.clip(df["prophet_yhat_future"] + residual_pred, 0, None)
df["actual"] = df[TARGET_LABEL]
df["residual"] = df["actual"] - df["predicted"]

eval_df = df[test_mask].copy()
print(f"Test set: {len(eval_df):,} rows, {eval_df['kreis_id'].nunique()} Kreise")

# 3. PER-KREIS METRICS

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom == 0, np.nan, denom)
    return np.nanmean(100 * np.abs(y_true - y_pred) / denom)

per_kreis = (
    eval_df.groupby("kreis_id")
    .apply(lambda g: pd.Series({
        "rmse": mean_squared_error(g["actual"], g["predicted"]) ** 0.5,
        "mae": mean_absolute_error(g["actual"], g["predicted"]),
        "smape": smape(g["actual"].values, g["predicted"].values),
        "mean_actual": g["actual"].mean(),
        "n_weeks": len(g),
    }), include_groups=False)
    .reset_index()
)

per_kreis.to_csv(os.path.join(OUTPUT_DIR, "per_kreis_metrics_hybrid.csv"), index=False)
print(f"\nPer-Kreis RMSE: median={per_kreis['rmse'].median():.2f}, "
      f"min={per_kreis['rmse'].min():.2f}, max={per_kreis['rmse'].max():.2f}")

# 4. BEST/MEDIAN/WORST EXAMPLES

sorted_kreise = per_kreis.sort_values("rmse")
n = len(sorted_kreise)
n_ex = min(N_EXAMPLES_PER_CATEGORY, n // 3)

best_n = sorted_kreise.iloc[:n_ex]["kreis_id"].tolist()
mid_start = n // 2 - n_ex // 2
median_n = sorted_kreise.iloc[mid_start: mid_start + n_ex]["kreis_id"].tolist()
worst_n = sorted_kreise.iloc[-n_ex:]["kreis_id"].tolist()

selected = [("Best", k) for k in best_n] + [("Median", k) for k in median_n] + \
           [("Worst", k) for k in worst_n]

n_rows = 3
fig, axes = plt.subplots(n_rows, n_ex, figsize=(4 * n_ex, 3.2 * n_rows), sharex=True)
if n_ex == 1:
    axes = axes.reshape(n_rows, 1)

for idx, (category, kreis_id) in enumerate(selected):
    row = idx // n_ex
    col = idx % n_ex
    ax = axes[row, col]

    sub = eval_df[eval_df["kreis_id"] == kreis_id].sort_values("week_start")
    rmse_val = per_kreis.loc[per_kreis["kreis_id"] == kreis_id, "rmse"].values[0]

    ax.plot(sub["week_start"], sub["actual"], color="#2c3e50",
           marker="o", markersize=2, label="Actual")
    ax.plot(sub["week_start"], sub["predicted"], color="#c44e52",
           linestyle="--", marker="o", markersize=2, label="Predicted")
    ax.set_title(f"{category}: {kreis_id}\nRMSE={rmse_val:.2f}", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=6)

axes[0, 0].legend(fontsize=7)
fig.suptitle(f"Best / median / worst Kreise ({n_ex} each) -- hybrid model", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "best_median_worst_expanded.png"), dpi=150)
plt.close(fig)
print(f"\n✓ Saved: best_median_worst_expanded.png ({n_ex} examples per category)")

# 5. FALSE ALARM ANALYSIS

low_threshold = eval_df["actual"].quantile(LOW_PERCENTILE / 100)
wave_threshold = eval_df["actual"].quantile(WAVE_PERCENTILE / 100)

print(f"\n{'='*60}")
print("False alarm analysis")
print("=" * 60)
print(f"'Stayed low' threshold (actual, {LOW_PERCENTILE}th pct): {low_threshold:.2f}")
print(f"'Predicted wave' threshold ({WAVE_PERCENTILE}th pct)   : {wave_threshold:.2f}")

eval_df["is_false_alarm"] = (
    (eval_df["actual"] <= low_threshold) & (eval_df["predicted"] >= wave_threshold)
)
eval_df["actual_was_low"] = eval_df["actual"] <= low_threshold

false_alarm_per_kreis = (
    eval_df.groupby("kreis_id")
    .agg(
        n_low_weeks=("actual_was_low", "sum"),
        n_false_alarms=("is_false_alarm", "sum"),
    )
    .reset_index()
)
false_alarm_per_kreis["false_alarm_rate_pct"] = np.where(
    false_alarm_per_kreis["n_low_weeks"] > 0,
    false_alarm_per_kreis["n_false_alarms"] / false_alarm_per_kreis["n_low_weeks"] * 100,
    0.0,
)

n_kreise_total = false_alarm_per_kreis["kreis_id"].nunique()
n_kreise_with_any_false_alarm = (false_alarm_per_kreis["n_false_alarms"] > 0).sum()
total_false_alarm_weeks = false_alarm_per_kreis["n_false_alarms"].sum()
total_low_weeks = false_alarm_per_kreis["n_low_weeks"].sum()

print(f"\nKreise with >= 1 false alarm week : {n_kreise_with_any_false_alarm} "
      f"/ {n_kreise_total} ({n_kreise_with_any_false_alarm/n_kreise_total*100:.1f}%)")
print(f"Total false alarm weeks           : {total_false_alarm_weeks} "
      f"/ {total_low_weeks} low-incidence weeks "
      f"({total_false_alarm_weeks/max(total_low_weeks,1)*100:.1f}%)")

false_alarm_per_kreis = false_alarm_per_kreis.sort_values(
    "false_alarm_rate_pct", ascending=False
)
false_alarm_per_kreis.to_csv(
    os.path.join(OUTPUT_DIR, "false_alarm_per_kreis.csv"), index=False
)
print(f"\n✓ Saved: false_alarm_per_kreis.csv")

print(f"\nTop 10 Kreise by false alarm rate:")
print(false_alarm_per_kreis.head(10).to_string(index=False))

# 6. PLOT: FALSE ALARM RATE DISTRIBUTION

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(
    false_alarm_per_kreis[false_alarm_per_kreis["n_low_weeks"] > 0]["false_alarm_rate_pct"],
    bins=20, color="#c44e52", edgecolor="white",
)
axes[0].set_xlabel("False alarm rate (% of low-incidence weeks)")
axes[0].set_ylabel("Number of Kreise")
axes[0].set_title("Distribution of per-Kreis false alarm rate")

kreise_with_alarms = false_alarm_per_kreis[false_alarm_per_kreis["n_false_alarms"] > 0]
kreise_without = n_kreise_total - len(kreise_with_alarms)
axes[1].bar(
    ["No false alarms", "\u2265 1 false alarm"],
    [kreise_without, len(kreise_with_alarms)],
    color=["#55a868", "#c44e52"],
)
axes[1].set_ylabel("Number of Kreise")
axes[1].set_title(f"Kreise with any false alarm: {len(kreise_with_alarms)}/{n_kreise_total}")
for i, v in enumerate([kreise_without, len(kreise_with_alarms)]):
    axes[1].text(i, v, str(v), ha="center", va="bottom")

fig.suptitle(f"False alarms -- model predicts a wave, actual stays low")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "false_alarm_distribution.png"), dpi=150)
plt.close(fig)
print(f"✓ Saved: false_alarm_distribution.png")

# 7. PLOT: EXAMPLE FALSE-ALARM KREISE (worst offenders)

worst_false_alarm_kreise = false_alarm_per_kreis[
    false_alarm_per_kreis["n_false_alarms"] > 0
].head(4)["kreis_id"].tolist()

if worst_false_alarm_kreise:
    fig, axes = plt.subplots(
        len(worst_false_alarm_kreise), 1,
        figsize=(14, 3 * len(worst_false_alarm_kreise)), sharex=True,
    )
    if len(worst_false_alarm_kreise) == 1:
        axes = [axes]

    for ax, kreis_id in zip(axes, worst_false_alarm_kreise):
        sub = eval_df[eval_df["kreis_id"] == kreis_id].sort_values("week_start")
        n_alarms = false_alarm_per_kreis.loc[
            false_alarm_per_kreis["kreis_id"] == kreis_id, "n_false_alarms"
        ].values[0]

        ax.plot(sub["week_start"], sub["actual"], color="#2c3e50",
               marker="o", markersize=3, label="Actual")
        ax.plot(sub["week_start"], sub["predicted"], color="#c44e52",
               linestyle="--", marker="o", markersize=3, label="Predicted")
        ax.axhline(low_threshold, color="gray", linestyle=":", alpha=0.6,
                  label="'Low' threshold")
        ax.axhline(wave_threshold, color="orange", linestyle=":", alpha=0.6,
                  label="'Wave' threshold")

        alarm_weeks = sub[sub["is_false_alarm"]]
        ax.scatter(alarm_weeks["week_start"], alarm_weeks["predicted"],
                  color="red", s=60, zorder=5, label="False alarm week")

        ax.set_title(f"{kreis_id} -- {n_alarms} false alarm week(s)", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Worst false-alarm Kreise -- model 'invents' a wave", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "worst_false_alarm_examples.png"), dpi=150)
    plt.close(fig)
    print(f"✓ Saved: worst_false_alarm_examples.png")
else:
    print("No false alarms found -- nothing to plot for worst offenders.")

# Notes
# - LOW_PERCENTILE / WAVE_PERCENTILE define what counts as "low" and
#   "wave" using the TEST SET's own actual-value distribution -- these
#   are relative, not absolute incidence numbers. Adjust if you want a
#   stricter/looser false-alarm definition.
# - false_alarm_per_kreis.csv has one row per Kreis -- sort/filter it
#   yourself to dig into specific problem Kreise.

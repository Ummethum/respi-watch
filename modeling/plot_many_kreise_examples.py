"""
Line plots (actual vs. predicted) for N example Kreise, using the same
Prophet-baseline + XGBoost-residual reconstruction as
evaluate_residual_model.py. Split across multiple pages for
readability instead of one huge dense grid.

Usage: python plot_many_kreise_examples.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

FEATURES_PATH = "./data/master/master_dataset_features.parquet"
PROPHET_BASELINE_PATH = "./data/prophet/kreis_baselines/prophet_baseline_survstat_influenza.parquet"
MODEL_PATH = "./data/models/xgboost_residual/model_residual_target_survstat_influenza_t_plus1.json"

TARGET_LABEL = "target_survstat_influenza_t_plus1"
HORIZON = 1
TEST_FRACTION = 0.15

OUTPUT_DIR = "./data/models/evaluation_residual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_KREISE = 50
KREISE_PER_PAGE = 10   # 5x2 grid per image -- readable, several pages total
RANDOM_SEED = 42        # for reproducible Kreis sampling across re-runs

# LOAD DATA & MODEL, RECONSTRUCT PREDICTIONS (same as evaluate_residual_model.py)

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

eval_df = df[test_mask].copy()
print(f"Test set: {len(eval_df):,} rows, {eval_df['kreis_id'].nunique()} Kreise")

# SAMPLE N KREISE AND PLOT, PAGED

all_kreise = eval_df["kreis_id"].unique()
n_kreise = min(N_KREISE, len(all_kreise))
rng = np.random.RandomState(RANDOM_SEED)
selected_kreise = rng.choice(all_kreise, size=n_kreise, replace=False)
print(f"Plotting {n_kreise} randomly sampled Kreise (seed={RANDOM_SEED})")

n_pages = (n_kreise + KREISE_PER_PAGE - 1) // KREISE_PER_PAGE

for page in range(n_pages):
    page_kreise = selected_kreise[page * KREISE_PER_PAGE : (page + 1) * KREISE_PER_PAGE]
    n_cols = 5
    n_rows = (len(page_kreise) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=False)
    axes = np.atleast_2d(axes)

    for idx, kreis_id in enumerate(page_kreise):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        sub = eval_df[eval_df["kreis_id"] == kreis_id].sort_values("week_start")
        ax.plot(sub["week_start"], sub["actual"], color="#2c3e50",
               marker="o", markersize=2, label="Actual")
        ax.plot(sub["week_start"], sub["predicted"], color="#c44e52",
               linestyle="--", marker="o", markersize=2, label="Predicted")
        ax.set_title(kreis_id, fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=6)

    # Hide unused panels on the last page
    for idx in range(len(page_kreise), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis("off")

    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Actual vs. predicted -- {n_kreise} Kreise (page {page+1}/{n_pages})", fontsize=13)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"many_kreise_examples_page{page+1}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

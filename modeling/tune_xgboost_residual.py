"""
RespiWatch — Hyperparameter tuning for the Prophet-residual XGBoost model
================================================================================
Same rolling-window CV / multi-objective / quantile-regression infrastructure
as tune_xgboost_hyperparameters.py, adapted to tune the RESIDUAL model from
train_xgboost_residual.py — i.e. XGBoost predicts (actual - Prophet baseline),
not the raw incidence.

Usage:
    python tune_xgboost_residual.py
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. CONFIGURATION

FEATURES_PATH = "./data/master/master_dataset_features.parquet"
PROPHET_BASELINE_PATH = "./data/prophet/kreis_baselines/prophet_baseline_survstat_influenza.parquet"
OUTPUT_DIR = "./data/models/xgboost_residual_tuning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_LABEL = "target_survstat_influenza_t_plus1"
HORIZON = 1   # must match "_t_plus{H}" in TARGET_LABEL

TEST_FRACTION = 0.15   # must match train_xgboost_residual.py

MAX_TRAIN_WINDOW_YEARS = 10
MAX_CV_REGION_YEARS = 13
TEST_WINDOW_WEEKS = 52
STEP_WEEKS = 52
MIN_YEARS_FOR_ROLLING_CV = 4

SUBSAMPLE_N_KREISE = None
N_ESTIMATORS_MAX = 1000
EARLY_STOPPING_ROUNDS = 20
N_TRIALS = 50

# Quantile regression toggle
USE_QUANTILE_OBJECTIVE = False
QUANTILE_ALPHA_FOR_TUNING = 0.75

# 2. LOAD FEATURES + PROPHET BASELINE, COMPUTE RESIDUAL TARGET

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
df["residual_target"] = df[TARGET_LABEL] - df["prophet_yhat_future"]

print(f"Loaded {len(df):,} rows with a valid residual target")

if SUBSAMPLE_N_KREISE is not None:
    rng_kreise = np.random.RandomState(42)
    all_kreise = df["kreis_id"].unique()
    n_sample = min(SUBSAMPLE_N_KREISE, len(all_kreise))
    sampled_kreise = rng_kreise.choice(all_kreise, size=n_sample, replace=False)
    df = df[df["kreis_id"].isin(sampled_kreise)].reset_index(drop=True)
    print(f"Subsampled to {n_sample}/{len(all_kreise)} Kreise")

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
y: pd.Series = df["residual_target"].copy()

for col in CATEGORICAL_FEATURES:
    X[col] = X[col].astype("category")

print(f"Features: {len(feature_cols)}")

# 3. ROLLING-WINDOW CV FOLDS

test_split_date = df["week_start"].quantile(1 - TEST_FRACTION)
earliest_date = df["week_start"].min()
available_years = (test_split_date - earliest_date).days / 365.25

CV_REGION_YEARS = min(MAX_CV_REGION_YEARS, available_years * 0.6)
TRAIN_WINDOW_YEARS = min(MAX_TRAIN_WINDOW_YEARS, available_years * 0.45)
use_rolling_cv = available_years >= MIN_YEARS_FOR_ROLLING_CV

print(f"\n{TARGET_LABEL}: {available_years:.1f} years of history available")

cv_region_start = test_split_date - pd.Timedelta(days=CV_REGION_YEARS * 365.25)


def generate_rolling_folds(df, region_start, region_end,
                           train_window_years, test_window_weeks, step_weeks):
    folds = []
    test_start = region_start
    while test_start + pd.Timedelta(weeks=test_window_weeks) <= region_end:
        test_end = test_start + pd.Timedelta(weeks=test_window_weeks)
        train_start = max(
            df["week_start"].min(),
            test_start - pd.Timedelta(days=train_window_years * 365.25),
        )
        train_mask = (df["week_start"] >= train_start) & (df["week_start"] < test_start)
        test_mask = (df["week_start"] >= test_start) & (df["week_start"] < test_end)
        if train_mask.sum() > 0 and test_mask.sum() > 0:
            folds.append({
                "train_idx": df.index[train_mask].to_numpy(),
                "test_idx": df.index[test_mask].to_numpy(),
                "train_start": train_start, "test_start": test_start, "test_end": test_end,
            })
        test_start += pd.Timedelta(weeks=step_weeks)
    return folds


if use_rolling_cv:
    cv_folds = generate_rolling_folds(
        df, cv_region_start, test_split_date,
        TRAIN_WINDOW_YEARS, TEST_WINDOW_WEEKS, STEP_WEEKS,
    )
else:
    val_window_years = min(1.0, available_years * 0.3)
    val_start = test_split_date - pd.Timedelta(days=val_window_years * 365.25)
    train_mask = df["week_start"] < val_start
    val_mask = (df["week_start"] >= val_start) & (df["week_start"] < test_split_date)
    cv_folds = [{
        "train_idx": df.index[train_mask].to_numpy(),
        "test_idx": df.index[val_mask].to_numpy(),
        "train_start": df.loc[train_mask, "week_start"].min(),
        "test_start": val_start, "test_end": test_split_date,
    }]

print(f"Generated {len(cv_folds)} fold(s) for tuning\n")

# 4. OPTUNA OBJECTIVE - multi-objective (val_rmse, |overfitting_gap|)

def objective(trial: optuna.Trial) -> tuple[float, float]:
    """
    Same multi-objective setup as tune_xgboost_hyperparameters.py:
    minimizes mean validation RMSE AND the magnitude of the train/val
    RMSE gap - both computed on the RESIDUAL predictions here (not
    reconstructed absolute predictions; that reconstruction happens in
    the evaluation script, not during hyperparameter search).
    """
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    fold_val_rmses, fold_overfit_gaps, fold_n_trees = [], [], []
    fold_pbar = tqdm(cv_folds, desc=f"Trial {trial.number}", leave=False, unit="fold")

    for fold in fold_pbar:
        X_fold_train = X.loc[fold["train_idx"]]
        y_fold_train = y.loc[fold["train_idx"]]
        X_fold_test = X.loc[fold["test_idx"]]
        y_fold_test = y.loc[fold["test_idx"]]

        if USE_QUANTILE_OBJECTIVE:
            objective_kwargs = {
                "objective": "reg:quantileerror",
                "quantile_alpha": QUANTILE_ALPHA_FOR_TUNING,
                "eval_metric": "quantile",
            }
        else:
            objective_kwargs = {"eval_metric": "rmse"}

        model = xgb.XGBRegressor(
            tree_method="hist",
            enable_categorical=True,
            random_state=42,
            n_estimators=N_ESTIMATORS_MAX,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **objective_kwargs,
            **params,
        )
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_test, y_fold_test)],
            verbose=False,
        )

        y_pred_train = model.predict(X_fold_train)
        y_pred_test = model.predict(X_fold_test)

        train_rmse = mean_squared_error(y_fold_train, y_pred_train) ** 0.5
        val_rmse = mean_squared_error(y_fold_test, y_pred_test) ** 0.5

        fold_val_rmses.append(val_rmse)
        fold_overfit_gaps.append(val_rmse - train_rmse)
        fold_n_trees.append(model.best_iteration)

        fold_pbar.set_postfix({
            "mean_val_rmse": f"{np.mean(fold_val_rmses):.3f}",
            "mean_gap": f"{np.mean(fold_overfit_gaps):.3f}",
        })

    fold_pbar.close()

    mean_val_rmse = float(np.mean(fold_val_rmses))
    mean_overfit_gap = float(np.mean(fold_overfit_gaps))

    trial.set_user_attr("n_trees_mean", float(np.mean(fold_n_trees)))
    trial.set_user_attr("overfitting_gap_signed", mean_overfit_gap)
    trial.set_user_attr("n_folds", len(cv_folds))

    return mean_val_rmse, abs(mean_overfit_gap)

# 5. RUN

if __name__ == "__main__":
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    pareto_trials = study.best_trials
    pareto_df = pd.DataFrame([
        {
            **t.params,
            "val_rmse": t.values[0],
            "overfitting_gap_abs": t.values[1],
            "overfitting_gap_signed": t.user_attrs.get("overfitting_gap_signed"),
            "n_trees_mean": t.user_attrs.get("n_trees_mean"),
            "trial_number": t.number,
        }
        for t in pareto_trials
    ]).sort_values("val_rmse")

    print(f"\nPareto front: {len(pareto_df)} trials")
    print(pareto_df[["trial_number", "val_rmse", "overfitting_gap_abs",
                     "overfitting_gap_signed", "n_trees_mean"]].to_string(index=False))

    TOLERANCE = 0.05
    best_val_rmse = pareto_df["val_rmse"].min()
    within_tolerance = pareto_df[pareto_df["val_rmse"] <= best_val_rmse * (1 + TOLERANCE)]
    recommended = within_tolerance.sort_values("overfitting_gap_abs").iloc[0]

    INT_PARAMS = {"max_depth", "min_child_weight"}
    recommended_params = {
        k: (int(recommended[k]) if k in INT_PARAMS else float(recommended[k]))
        for k in ["max_depth", "learning_rate", "subsample", "colsample_bytree",
                  "min_child_weight", "reg_alpha", "reg_lambda"]
    }
    print(f"\nRecommended params: {recommended_params}")

    pareto_df.to_csv(os.path.join(OUTPUT_DIR, f"pareto_front_residual_{TARGET_LABEL}.csv"), index=False)
    print(f"\n✓ Saved pareto front CSV")

# -- Notes -------------------------------------------------------------
# - Paste recommended_params into train_xgboost_residual.py's BEST_PARAMS.
# - USE_QUANTILE_OBJECTIVE here biases the RESIDUAL prediction upward,
#   not the raw incidence

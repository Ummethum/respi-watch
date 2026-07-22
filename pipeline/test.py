"""
RespiWatch -- Push trained models + historical base data to a SEPARATE
Hugging Face Dataset repo (the "pipeline" repo, not the "app" repo).
============================================================================
Two separate Hugging Face repos, two separate purposes:

    - App repo (push_data_to_hub.py): small, changes every week
      automatically, only what app.py needs at runtime.
    - Pipeline repo (THIS script): larger, changes only occasionally
      (when models get retrained, or historical base data changes) --
      trained models + the historical base files that
      fetch_recent_*.py scripts merge new weeks onto every week.

This is NOT part of run_weekly_update.py's automated weekly steps --
run it manually whenever you retrain a model or update historical base
data, not every week.

One-time setup (in addition to the app repo's own setup):
    1. Create a SECOND Hugging Face Dataset repo, e.g.
       "yourname/respiwatch-pipeline-data"
    2. Add a second line to your .env file:
           HF_REPO_ID_PIPELINE=yourname/respiwatch-pipeline-data
       (HF_TOKEN is shared with push_data_to_hub.py -- same account,
       same token works for both repos as long as it has write access)

Usage:
    python push_pipeline_data_to_hub.py
"""

import os
from dotenv import load_dotenv

# Same Xet workaround as push_data_to_hub.py -- see that script's own
# docstring for the full explanation of why this needs to be set
# before importing huggingface_hub, not after.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi

load_dotenv()

# -----------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------

HF_REPO_ID_PIPELINE = os.environ.get("HF_REPO_ID_PIPELINE", "HenningU/respiwatch-pipeline-data")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Trained models -- the actual model artifacts generate_predictions.py
# loads at inference time.
FILES_TO_UPLOAD = [
    "./data/prophet/kreis_baselines/prophet_baseline_survstat_influenza.parquet",
    "./data/models/xgboost_residual/model_residual_target_survstat_influenza_t_plus1.json",
    "./data/models/xgboost_residual/model_residual_target_survstat_influenza_t_plus2.json",
]

# Historical base data -- what fetch_recent_*.py scripts merge new
# weeks onto every week. Anyone cloning the repo fresh needs these to
# run the pipeline at all, rather than starting from zero history.
FOLDERS_TO_UPLOAD = [
    {
        "folder_path": "./data/weather",
        "path_in_repo": "historical/weather",
        "allow_patterns": ["weather_weekly.parquet"],
    },
    {
        "folder_path": "./data/air_quality",
        "path_in_repo": "historical/air_quality",
        "allow_patterns": ["air_quality_weekly.parquet"],
    },
    {
        "folder_path": "./data/pollen",
        "path_in_repo": "historical/pollen",
        "allow_patterns": ["pollen_weekly.parquet"],
    },
    {
        "folder_path": "./data/google_trends",
        "path_in_repo": "historical/google_trends",
        "allow_patterns": ["trends_wide_from_csv.parquet"],
    },
    {
        "folder_path": "./data/rki/processed",
        "path_in_repo": "historical/rki_processed",
        "allow_patterns": ["rki_incidence_wide_berlin_aggregated.parquet",
                          "kreis_name_crosswalk.csv"],
    },
    {
        "folder_path": "./data/rki_github/processed",
        "path_in_repo": "historical/rki_github_processed",
        "allow_patterns": ["are_weekly.parquet", "grippeweb_weekly.parquet",
                          "notaufnahme_weekly.parquet"],
    },
    {
        "folder_path": "./data/amelag/processed",
        "path_in_repo": "historical/amelag",
        "allow_patterns": ["amelag_weekly_bundesland.parquet"],
    },
    {
        "folder_path": "./data/holidays",
        "path_in_repo": "historical/holidays",
        "allow_patterns": ["holidays_weekly.parquet"],
    },
    {
        "folder_path": "./data/rki",
        "path_in_repo": "historical/rki_raw",
        "allow_patterns": ["berlin_population.csv"],
    },
]

# Not uploaded here, and deliberately so: master_dataset_filled.parquet /
# master_dataset_features.parquet -- these get REBUILT fresh from the
# historical base files above every week by build_master_dataset.py /
# build_features.py, so shipping a copy here would just be a second,
# quickly-stale copy of something the app repo already has (see
# push_data_to_hub.py).

# -----------------------------------------------
# 2. UPLOAD (identical logic to push_data_to_hub.py)
# -----------------------------------------------

def _diagnose_and_reraise_xet_error(e: Exception):
    error_text = str(e)
    is_xet_bug = (
        isinstance(e, TypeError)
        and ("XetProgressReporter" in error_text or "xet" in error_text.lower())
    )
    if is_xet_bug:
        print(
            "\n\u2717 Upload failed with what looks like the known hf_xet "
            "progress-reporter bug (HF_HUB_DISABLE_XET=1 didn't fully "
            "suppress it in this huggingface_hub version).\n"
            "  Most reliable fix: uninstall hf_xet entirely:\n\n"
            "      pip uninstall hf_xet -y\n\n"
            "  Then re-run this script."
        )
    raise e


def push_pipeline_data():
    if not HF_TOKEN:
        raise SystemExit(
            "HF_TOKEN environment variable not set -- see .env setup notes "
            "at the top of this file."
        )
    if HF_REPO_ID_PIPELINE == "YOUR_USERNAME/respiwatch-pipeline-data":
        raise SystemExit(
            "HF_REPO_ID_PIPELINE not set -- add it to your .env file. "
            "This is a SEPARATE repo from HF_REPO_ID (the app data repo), "
            "see this script's docstring."
        )

    api = HfApi(token=HF_TOKEN)

    print(f"Uploading {len(FILES_TO_UPLOAD)} model file(s) + "
          f"{len(FOLDERS_TO_UPLOAD)} historical-data folder(s) "
          f"to {HF_REPO_ID_PIPELINE}...\n")

    for local_path in FILES_TO_UPLOAD:
        if not os.path.exists(local_path):
            print(f"  \u26a0\ufe0f  Not found, skipping: {local_path}")
            continue

        filename = os.path.basename(local_path)
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        # Keep models under their own "models/" prefix in the repo, so
        # they don't collide with historical/ and are easy to find.
        path_in_repo = f"models/{filename}" if local_path.endswith((".json",)) \
            else f"prophet/{filename}"

        print(f"  {filename} ({size_mb:.1f} MB) -> {path_in_repo}...", end=" ", flush=True)

        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=HF_REPO_ID_PIPELINE,
                repo_type="dataset",
            )
        except Exception as e:
            _diagnose_and_reraise_xet_error(e)
        print("OK")

    for folder_entry in FOLDERS_TO_UPLOAD:
        folder_path = folder_entry["folder_path"]

        if not os.path.isdir(folder_path):
            print(f"  \u26a0\ufe0f  Folder not found, skipping: {folder_path}")
            continue

        print(f"  {folder_path}/ -> {folder_entry['path_in_repo']}...", end=" ", flush=True)

        try:
            api.upload_folder(
                folder_path=folder_path,
                path_in_repo=folder_entry["path_in_repo"],
                repo_id=HF_REPO_ID_PIPELINE,
                repo_type="dataset",
                allow_patterns=folder_entry.get("allow_patterns"),
                ignore_patterns=folder_entry.get("ignore_patterns"),
            )
        except Exception as e:
            _diagnose_and_reraise_xet_error(e)
        print("OK")

    print(f"\n\u2713 Upload complete: "
          f"https://huggingface.co/datasets/{HF_REPO_ID_PIPELINE}")


if __name__ == "__main__":
    push_pipeline_data()

# -- Notes ------------------------------------------------------------------
# - Run this manually whenever you retrain a model (fit_prophet_baseline.py /
#   train_xgboost_residual.py) or refresh historical base data -- NOT
#   part of the automated weekly run_weekly_update.py chain.
# - To restore a fresh clone's pipeline to a working state, download
#   these files back into the matching local paths BEFORE running
#   run_weekly_update.py for the first time -- otherwise
#   generate_predictions.py has no model to load, and the
#   fetch_recent_*.py scripts have no history to merge new weeks onto.
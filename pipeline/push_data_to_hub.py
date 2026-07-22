"""
RespiWatch -- Push weekly-updated data files to a Hugging Face Dataset
============================================================================
The final step of the weekly pipeline: uploads the freshly rebuilt data
files to a private (or public) Hugging Face Dataset repo, which the
DEPLOYED Streamlit app then reads from at runtime -- your GitHub repo
never contains any data, only code.

Why Hugging Face Datasets instead of Git LFS:
    - Genuinely zero data in the GitHub repo (matches "I don't want to
      store any data in the repository" exactly -- LFS still leaves
      pointer files + a growing LFS object store tied to the repo).
    - No cumulative storage growth problem -- re-uploading a file
      OVERWRITES it in the dataset repo, it doesn't add another
      historical copy the way an LFS push does.
    - Free tier is generous for this project's data sizes.
    - The Streamlit app just does a normal authenticated HTTP download
      at startup (cached) -- no LFS-specific tooling needed on the
      Streamlit Cloud side at all.

One-time setup (do this once, not part of the weekly automation):
    1. Create a free account at https://huggingface.co
    2. Create a new Dataset repo (can be private): e.g. "yourname/respiwatch-data"
    3. Create an access token with WRITE permission:
       https://huggingface.co/settings/tokens
    4. Create a file named ".env" in the project root (same folder as
       this script) with the line:
           HF_TOKEN=hf_xxxxxxxxxxxx
       This file is already excluded via .gitignore -- it will never
       get committed. load_dotenv() below reads it automatically.

Usage:
    python push_data_to_hub.py
"""

import os
from dotenv import load_dotenv

# as of mid-2025 through at least mid-2026, hf_xet has
# a recurring bug where its progress-reporting callback signature
# doesn't match what huggingface_hub calls it with, raising
# "XetProgressReporter.close() takes 1 positional argument but 2 were
# given" or similar TypeErrors that abort the upload entirely (tracked
# across multiple open GitHub issues in huggingface_hub/xet-core).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi
load_dotenv()

# 1. CONFIGURATION

HF_REPO_ID = "HenningU/respiwatch-data"   # <- change this
HF_TOKEN = os.environ.get("HF_TOKEN")

# Individual files the deployed app needs at runtime
FILES_TO_UPLOAD = [
    "./data/predictions/latest_predictions.parquet",
    "./data/predictions/gap_fill_predictions.parquet",
    "./data/predictions/recent_avg_predictions.parquet",
    "./data/master/master_dataset_filled.parquet",
]

# Whole FOLDERS to upload
FOLDERS_TO_UPLOAD = [
    {
        "folder_path": "./data/city_coords",
        "path_in_repo": "city_coords",   # -> appears as city_coords/ in the dataset repo
    },
]

# -----------------------------------------------
# 2. UPLOAD
# -----------------------------------------------

def _diagnose_and_reraise_xet_error(e: Exception):
    """
    HF_HUB_DISABLE_XET=1 (set above) is the documented fix, but has a
    reported reliability issue in some huggingface_hub versions where
    it's silently ignored. If an upload still fails with the
    characteristic Xet progress-reporter TypeError, give a clear,
    actionable message pointing to the more reliable fix (uninstalling
    hf_xet entirely) instead of letting the user debug a confusing
    "close() takes 1 positional argument" traceback from scratch.
    """
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
            "  Most reliable fix: uninstall hf_xet entirely so this code "
            "path can't run at all:\n\n"
            "      pip uninstall hf_xet -y\n\n"
            "  Then re-run this script -- huggingface_hub automatically "
            "falls back to its older (still fully functional, just not "
            "chunk-accelerated) upload method when hf_xet isn't installed."
        )
    raise e


def push_data():
    if not HF_TOKEN:
        raise SystemExit(
            "HF_TOKEN environment variable not set -- see the setup notes "
            "at the top of this file. Refusing to proceed without it "
            "rather than failing with a confusing auth error deep inside "
            "huggingface_hub."
        )

    api = HfApi(token=HF_TOKEN)

    print(f"Uploading {len(FILES_TO_UPLOAD)} files + {len(FOLDERS_TO_UPLOAD)} "
          f"folder(s) to {HF_REPO_ID}...\n")

    for local_path in FILES_TO_UPLOAD:
        if not os.path.exists(local_path):
            print(f"  \u26a0\ufe0f  Not found, skipping: {local_path}")
            continue

        filename = os.path.basename(local_path)
        size_mb = os.path.getsize(local_path) / (1024 * 1024)

        print(f"  {filename} ({size_mb:.1f} MB)...", end=" ", flush=True)

        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=filename,
                repo_id=HF_REPO_ID,
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

        n_files = sum(len(files) for _, _, files in os.walk(folder_path))
        print(f"  Folder {folder_path}/ ({n_files} file(s)) -> "
              f"{folder_entry.get('path_in_repo', '(repo root)')}...", end=" ", flush=True)

        try:
            api.upload_folder(
                folder_path=folder_path,
                path_in_repo=folder_entry.get("path_in_repo"),
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                allow_patterns=folder_entry.get("allow_patterns"),
                ignore_patterns=folder_entry.get("ignore_patterns"),
            )
        except Exception as e:
            _diagnose_and_reraise_xet_error(e)
        print("OK")

    print(f"\n\u2713 Upload complete: https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    push_data()
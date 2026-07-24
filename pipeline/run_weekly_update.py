"""
RespiWatch -- Weekly update orchestrator (the actual cron-job entrypoint)
================================================================================
Runs the full weekly pipeline in the correct order, stopping early (and
NOT touching the master dataset or predictions) if any upstream step
fails

Order matters here:
    1. Download GitHub sources (ARE, GrippeWeb, Notaufnahme, AMELAG)
    2. Fetch recent weather / air quality / pollen (existing scripts,
       re-run for just the last few weeks -- "easy", per your note)
    3. Fetch recent Google Trends (last month, pytrends)
    4. Re-parse all the above into their processed weekly parquets
    5. Rebuild the master dataset (build_master_dataset.py)
    6. Rebuild features (build_features.py)
    7. Run predictions with the saved models

Holidays are DELIBERATELY NOT part of this weekly job -- see chat,
they're known years in advance and only need a yearly refresh.

Usage:
    python run_weekly_update.py
"""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from datetime import datetime

# 1. CONFIGURATION -- the pipeline steps, in order

STEPS = [
    ("Download GitHub sources (ARE/GrippeWeb/Notaufnahme/AMELAG)",
     "download_github_sources.py"),
    ("Fetch recent weather",
     "fetch_recent_weather.py"),
    ("Fetch recent air quality",
     "fetch_recent_air_quality.py"),
    ("Fetch recent pollen",
     "fetch_recent_pollen.py"),
    ("Fetch recent Google Trends",
     "fetch_recent_trends.py"),
    ("Fetch recent SurvStat (RKI incidence, via SOAP API)",
     "fetch_recent_survstat.py"),
    ("Parse RKI GitHub sources (ARE/GrippeWeb/Notaufnahme)",
     "parse_rki_github_sources.py"),
    ("Parse AMELAG",
     "parse_amelag.py"),
    ("Parse RKI incidence (from fetch_recent_survstat.py's TSVs)",
     "parse_rki_incidence.py"),
    ("Aggregate Berlin boroughs into one NUTS-3 unit",
     "aggregate_berlin_boroughs.py"),
    ("Rebuild master dataset",
     "build_master_dataset.py"),
    ("Fill SurvStat zeros (coverage-aware)",
     "fill_survstat_zeros.py"),
    ("Rebuild features",
     "build_features.py"),
    ("Generate predictions for all Kreise",
     "generate_predictions.py"),
    ("Generate gap-fill (nowcast) predictions",
     "generate_gap_fill_predictions.py"),
    ("Compute recent (last 4 weeks) predictions average",
     "compute_recent_predictions_average.py"),
    ("Build prediction history",
     "build_prediction_history.py"),
    ("Push data to Hugging Face",
     "push_data_to_hub.py"),
]

LOG_FILE = "./weekly_update.log"

# 2. RUN EACH STEP, STOP ON FIRST FAILURE

def run_step(description: str, script: str) -> bool:
    print(f"\n{'='*70}")
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {description}")
    print(f"  -> {script}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, script)],
        capture_output=True, text=True,
    )

    print(result.stdout[-3000:])   # last part of output, avoid flooding logs
    if result.returncode != 0:
        print(f"\n\u2717 FAILED (exit code {result.returncode})")
        print(result.stderr[-3000:])
        return False

    print(f"\n\u2713 OK")
    return True


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Weekly update started: {start_time:%Y-%m-%d %H:%M:%S}")

    with open(LOG_FILE, "a") as log:
        log.write(f"\n\n{'#'*70}\n# Run started {start_time:%Y-%m-%d %H:%M:%S}\n{'#'*70}\n")

    for description, script in STEPS:
        success = run_step(description, script)

        with open(LOG_FILE, "a") as log:
            status = "OK" if success else "FAILED"
            log.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {description}: {status}\n")

        if not success:
            print(f"\n{'!'*70}")
            print(f"Pipeline stopped at: {description}")
            print(f"Master dataset / features / predictions were NOT rebuilt --")
            print(f"the last successful run's outputs remain in place.")
            print(f"Check {LOG_FILE} and fix the failing step before re-running.")
            print("!" * 70)
            sys.exit(1)

    elapsed = datetime.now() - start_time
    print(f"\n{'='*70}")
    print(f"Weekly update completed successfully in {elapsed}")
    print("=" * 70)

# -- Notes ------------------------------------------------------------------
# - Cron schedule suggestion: Friday mornings, e.g.
#     0 6 * * 5  cd /path/to/project && python run_weekly_update.py
#   (Friday, after AMELAG/Notaufnahme's Wednesday and ARE/GrippeWeb's
#   Thursday updates have both landed)
#
# - fetch_recent_holidays.py is NOT in STEPS -- run it
#   separately on a yearly cadence, see that script's own
#   docstring for why.
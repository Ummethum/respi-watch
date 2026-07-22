"""
RespiWatch -- Holiday coverage check + conditional yearly refresh
========================================================================
UNLIKE the other three "recent fetch" scripts, this one does NOT need
to run every week -- school vacations and public holidays are known
YEARS in advance (school calendars get published long before the
school year starts), so there's nothing new to "catch up on" weekly
the way there is for weather/air_quality/pollen/trends.

What this script actually does: checks whether the existing
holidays_weekly.parquet still covers far enough into the FUTURE (not
the past) to support your 1-2 week forecast horizon plus some buffer.
If it does, it does nothing. If coverage is running low (e.g. you're
approaching the end of what was fetched last time), it re-runs the
full fetch_holidays.py for an extended year range.

Suggested cadence: run this check monthly (or even just manually every
few months), NOT as part of the weekly cron job -- include it there
only as a low-cost safety check if you want, since it's a no-op most
of the time anyway.

Usage:
    python fetch_recent_holidays.py
"""

import os
import subprocess
import sys
import pandas as pd

# 1. CONFIGURATION

HOLIDAYS_PATH = "./data/holidays/holidays_weekly.parquet"
FETCH_HOLIDAYS_SCRIPT = "fetch_holidays.py"   # the original yearly-range fetcher

# Minimum months of FUTURE coverage required beyond today before
# triggering a refresh. Comfortably more than your 2-week forecast
# horizon -- the real constraint is "don't let coverage silently run
# out", not the forecast horizon itself.
MIN_FUTURE_MONTHS_REQUIRED = 6

# When a refresh IS needed, extend coverage this many years past the
# currently-covered end date (matches fetch_holidays.py's END_YEAR
# convention -- you'll need to bump that constant in fetch_holidays.py
# to this new target before it runs, see step 3 below).
EXTEND_YEARS = 2

# 2. CHECK CURRENT COVERAGE

def check_coverage() -> tuple[bool, pd.Timestamp | None]:
    if not os.path.exists(HOLIDAYS_PATH):
        print(f"No existing holidays file at {HOLIDAYS_PATH} -- refresh needed.")
        return True, None

    df = pd.read_parquet(HOLIDAYS_PATH)
    max_covered_date = df["week_start"].max()

    today = pd.Timestamp.today().normalize()
    months_of_future_coverage = (max_covered_date - today).days / 30.44

    print(f"Current holiday data covers up to: {max_covered_date.date()}")
    print(f"That's {months_of_future_coverage:.1f} months of future coverage "
          f"from today ({today.date()})")

    needs_refresh = months_of_future_coverage < MIN_FUTURE_MONTHS_REQUIRED
    return needs_refresh, max_covered_date

# 3. TRIGGER A REFRESH IF NEEDED

def trigger_refresh(current_max_date: pd.Timestamp | None):
    target_year = (current_max_date.year if current_max_date is not None
                   else pd.Timestamp.today().year) + EXTEND_YEARS

    print(f"\nCoverage running low -- triggering a refresh of "
          f"{FETCH_HOLIDAYS_SCRIPT} extended through {target_year}.")
    print(f"NOTE: you need to update END_YEAR = {target_year} at the top "
          f"of {FETCH_HOLIDAYS_SCRIPT} before/when this runs -- this "
          f"script does not edit that file automatically, to avoid "
          f"silently rewriting your own configured constants.")

    response = input(f"\nProceed with running {FETCH_HOLIDAYS_SCRIPT} now? [y/N] ")
    if response.strip().lower() != "y":
        print("Skipped -- update END_YEAR yourself and re-run "
              f"{FETCH_HOLIDAYS_SCRIPT} manually when ready.")
        return

    result = subprocess.run([sys.executable, FETCH_HOLIDAYS_SCRIPT])
    if result.returncode != 0:
        print(f"\n\u2717 {FETCH_HOLIDAYS_SCRIPT} failed (exit code "
              f"{result.returncode}) -- holidays_weekly.parquet unchanged.")
        sys.exit(1)

    print(f"\n\u2713 Holiday coverage refreshed.")

# 4. RUN

if __name__ == "__main__":
    needs_refresh, current_max_date = check_coverage()

    if needs_refresh:
        trigger_refresh(current_max_date)
    else:
        print(f"\n\u2713 Coverage sufficient (>= {MIN_FUTURE_MONTHS_REQUIRED} months "
              f"future buffer) -- nothing to do.")

# Notes
# - This is DELIBERATELY interactive (asks before running the full
#   scrape+API fetch) rather than fully automatic, since
#   fetch_holidays.py scrapes schulferien.org across many years and
#   hits feiertage-api.de repeatedly

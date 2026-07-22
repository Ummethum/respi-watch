"""
RespiWatch -- Weekly downloader for RKI GitHub data sources
=================================================================
Downloads the four RKI GitHub-hosted TSV files that get updated on a
weekly schedule. These are stable "raw" URLs on the main branch --
RKI overwrites the file in place each update, so there's no versioning
to track, just re-download and overwrite the local copy.

Update schedule (informational, for when you schedule the cron job):
    AMELAG        -- Wednesdays
    ARE           -- Thursdays
    GrippeWeb     -- Thursdays
    Notaufnahme   -- Wednesdays

Given this, running the cron job on Friday morning guarantees all four
have their latest weekly update available.

Usage:
    python download_github_sources.py
"""

import os
import time
import requests

# 1. CONFIGURATION

# (local_path, raw_github_url)
SOURCES = [
    (
        "./data/amelag/amelag_einzelstandorte.tsv",
        "https://raw.githubusercontent.com/robert-koch-institut/"
        "Abwassersurveillance_AMELAG/main/amelag_einzelstandorte.tsv",
    ),
    (
        "./data/rki_github/ARE-Konsultationsinzidenz.tsv",
        "https://raw.githubusercontent.com/robert-koch-institut/"
        "ARE-Konsultationsinzidenz/main/ARE-Konsultationsinzidenz.tsv",
    ),
    (
        "./data/rki_github/GrippeWeb_Daten_des_Wochenberichts.tsv",
        "https://raw.githubusercontent.com/robert-koch-institut/"
        "GrippeWeb_Daten_des_Wochenberichts/main/"
        "GrippeWeb_Daten_des_Wochenberichts.tsv",
    ),
    (
        "./data/rki_github/Grippeweb_Zuordnung_Regionen.tsv",
        "https://raw.githubusercontent.com/robert-koch-institut/"
        "GrippeWeb_Daten_des_Wochenberichts/main/"
        "Kontextmaterialien/GrippeWeb_Zuordung_Regionen.tsv",
    ),
    (
        "./data/rki_github/Notaufnahmesurveillance_Zeitreihen_Syndrome.tsv",
        "https://raw.githubusercontent.com/robert-koch-institut/"
        "Daten_der_Notaufnahmesurveillance/main/"
        "Notaufnahmesurveillance_Zeitreihen_Syndrome.tsv",
    ),
]

MAX_RETRIES = 3
RETRY_DELAY = 10   # seconds

# 2. DOWNLOAD WITH RETRY

def download_file(url: str, local_path: str) -> bool:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            content = resp.text
            first_line = content.split("\n", 1)[0]
            if "\t" not in first_line or content.count("\n") < 2:
                raise ValueError(
                    f"Downloaded content doesn't look like a valid TSV "
                    f"(first line: {first_line[:80]!r})"
                )

            with open(local_path, "w", encoding="utf-8") as f:
                f.write(content)

            size_kb = len(content.encode("utf-8")) / 1024
            n_lines = content.count("\n")
            print(f"  OK: {local_path} ({size_kb:.0f} KB, {n_lines:,} lines)")
            return True

        except (requests.exceptions.RequestException, ValueError) as e:
            wait = RETRY_DELAY * (attempt + 1)
            print(f"  Attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"  Waiting {wait}s before retry...")
                time.sleep(wait)

    print(f"  FAILED after {MAX_RETRIES} attempts: {local_path}")
    return False


if __name__ == "__main__":
    print(f"Downloading {len(SOURCES)} RKI GitHub sources...\n")

    results = {}
    for local_path, url in SOURCES:
        print(f"{os.path.basename(local_path)}:")
        results[local_path] = download_file(url, local_path)
        print()

    n_success = sum(results.values())
    print(f"{'='*50}")
    print(f"Downloaded {n_success}/{len(SOURCES)} sources successfully")

    failed = [path for path, ok in results.items() if not ok]
    if failed:
        print(f"\nFAILED (kept old local copy, if any):")
        for path in failed:
            print(f"  {path}")
        print("\nDo NOT proceed to re-run the parsers/master build for a "
              "failed source without checking why first -- a stale local "
              "copy is safer than silently building on missing data, but "
              "you should know it happened.")
        raise SystemExit(1)
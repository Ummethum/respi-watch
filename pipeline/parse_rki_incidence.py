"""
RespiWatch — RKI SurvStat incidence data parser
===================================================
Parses yearly SurvStat exports (COVID, Influenza, RSV) from
data/rki/{disease}/*.csv into a unified long-format table.

Expected input format (same as the original Data_2026.csv structure):
    - Encoding: UTF-16, tab-separated, CRLF line endings
    - Row 1 (and possibly row 2): header — first cell "Kreis" (or similar),
      remaining cells are week numbers (01-53)
    - Data rows: Kreis name (e.g. "SK München", "LK Weimarer Land",
      "Unbekannt"), followed by weekly incidence values with German
      comma-decimal formatting (e.g. "31,84"), empty cells for missing
      weeks represented as "" 
    - One file per year, per disease folder

Folder layout expected:
    data/rki/covid/*.csv
    data/rki/influenza/*.csv
    data/rki/rsv/*.csv

Only Influenza is expected to cover the full 2004-2026 range; COVID and
RSV will have fewer years available (COVID from ~2020, RSV from ~2023),
and the script does not assume fixed year ranges — it just processes
whatever files it finds per folder and reports what it found.

IMPORTANT — Kreis identification:
    Rows are identified by Kreis NAME (e.g. "SK München"), not by
    AGS/NUTS code. This script keeps the raw name as-is; mapping names
    to NUTS-3 codes (to join with weather/trends/pollen data) is a
    separate step — see the note at the bottom of this file.

Usage:
    python parse_rki_incidence.py
"""

import os
import re
import glob
import pandas as pd
import numpy as np


# 1. CONFIGURATION

RKI_BASE_DIR = "./data/rki"
OUTPUT_DIR   = "./data/rki/processed"

# Folder name -> disease label used in output. Adjust folder names here
# if yours differ (e.g. if the folders are named "COVID-19" instead of
# "covid").
DISEASE_FOLDERS = {
    "covid":      "covid",
    "influenza":  "influenza",
    "rsv":        "rsv",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Regex to find a 4-digit year (2000-2029) anywhere in the filename
YEAR_PATTERN = re.compile(r"(20[0-2]\d)")

# 2. ROBUST FILE READING

def find_header_line(lines: list[str]) -> tuple[int | None, list[str] | None]:
    """
    Finds the actual week-number header row.

    Structure clarified from the source file:
        Row 1: A1="Kreis", B1="Meldewoche" (label row, not the real header)
        Row 2: A2="" (empty), B2:BB2 = week numbers 1-53  <- this is what we want
        Row 3+: A3=Kreis name, B3:BB3 = weekly values

    So the week-number row's FIRST cell is empty, not "Kreis" — detection
    must rely purely on the remaining cells matching the week-number
    pattern (mostly integers 1-53), not on any text in the first cell.
    Requiring at least 40 valid week numbers (out of up to 53) avoids
    false positives from other lines.
    """
    for i, line in enumerate(lines):
        cells = [c.strip().strip('"') for c in line.rstrip("\n\r").split("\t")]
        if len(cells) < 10:
            continue

        week_values = []
        valid = True
        for c in cells[1:]:
            if c == "":
                continue
            try:
                week_num = int(c)
            except ValueError:
                valid = False
                break
            if not (1 <= week_num <= 53):
                valid = False
                break
            week_values.append(str(week_num))

        if valid and len(week_values) >= 40:
            return i, week_values

    return None, None


def read_survstat_csv(path: str) -> pd.DataFrame | None:
    """
    Reads one SurvStat yearly export and returns a long-format
    DataFrame: kreis_name, week, incidence.
    Returns None if the file couldn't be parsed.
    """
    try:
        with open(path, encoding="utf-16") as f:
            lines = f.readlines()
    except (UnicodeError, UnicodeDecodeError):
        # Fall back to utf-8 in case a file was saved differently
        try:
            with open(path, encoding="utf-8-sig") as f:
                lines = f.readlines()
        except Exception:
            return None

    header_idx, week_cols = find_header_line(lines)
    if header_idx is None:
        return None

    # Parse data rows
    records = []
    for line in lines[header_idx + 1:]:
        cells = [c.strip().strip('"') for c in line.rstrip("\n\r").split("\t")]
        if not cells or cells[0] == "":
            continue

        kreis_name = cells[0]
        values = cells[1:]

        for week_str, value_str in zip(week_cols, values):
            if value_str == "":
                continue
            # German decimal format: comma -> dot
            value_str_clean = value_str.replace(",", ".")
            try:
                value = float(value_str_clean)
            except ValueError:
                continue

            records.append({
                "kreis_name": kreis_name,
                "week": int(week_str),
                "incidence": value,
            })

    if not records:
        return None

    return pd.DataFrame(records)

# 3. YEAR EXTRACTION FROM FILENAME

def extract_year(filename: str) -> int | None:
    match = YEAR_PATTERN.search(filename)
    if match:
        return int(match.group(1))
    return None

# 4. PROCESS ONE DISEASE FOLDER

def process_disease_folder(disease: str, folder_path: str) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    print(f"\n{'='*60}")
    print(f"{disease.upper()}: {len(csv_files)} file(s) in {folder_path}")
    print(f"{'='*60}")

    all_records = []
    years_found = []
    year_to_file = {}   # tracks which file were already used for each year,
                          # to detect and handle duplicate-year files below

    for path in sorted(csv_files, key=os.path.getmtime):   # oldest first,
                                                              # so the NEWEST
                                                              # file for a
                                                              # given year
                                                              # wins below
        fname = os.path.basename(path)
        year = extract_year(fname)

        if year is None:
            print(f"  ⚠️  Could not extract year from filename: {fname} — skipping")
            continue

        df = read_survstat_csv(path)
        if df is None or df.empty:
            print(f"  ⚠️  Could not parse: {fname} — skipping")
            continue

        if year in year_to_file:
            # Two files both claim this year -- almost always a manually-
            # exported historical file PLUS a freshly re-fetched one
            # (e.g. from fetch_recent_survstat.py) for the same year,
            # with revised SurvStat numbers. Silently concatenating BOTH
            # here previously caused duplicate (kreis_name, year, week)
            # rows with DIFFERENT values downstream, which corrupted
            # build_features.py's shift()-based lag/target computation
            # for that Kreis's entire series -- keep only the newer file
            # (by file modification time) and say so loudly.
            print(f"  ⚠️  DUPLICATE YEAR {year}: both {year_to_file[year]} and "
                  f"{fname} claim this year. Keeping {fname} (newer file), "
                  f"discarding {year_to_file[year]}. Delete the stale file "
                  f"to silence this warning.")
            all_records = [r for r, y in zip(all_records, years_found) if y != year]
            years_found = [y for y in years_found if y != year]

        df["year"] = year
        all_records.append(df)
        years_found.append(year)
        year_to_file[year] = fname

        n_kreise = df["kreis_name"].nunique()
        n_weeks = df["week"].nunique()
        print(f"  ✓ {fname}: year={year}, {n_kreise} Kreise, "
              f"{n_weeks} weeks, {len(df)} data points")

    if not all_records:
        print(f"  No usable data found for {disease}.")
        return pd.DataFrame()

    combined = pd.concat(all_records, ignore_index=True)

    # Final safety net: even without duplicate files, verify no
    # (kreis_name, year, week) combination ended up duplicated --
    # fail loudly rather than silently passing corrupted data downstream.
    dupe_mask = combined.duplicated(subset=["kreis_name", "year", "week"], keep=False)
    if dupe_mask.any():
        n_dupes = dupe_mask.sum()
        raise ValueError(
            f"{n_dupes} duplicate (kreis_name, year, week) rows remain in "
            f"{disease} after per-year deduplication -- investigate before "
            f"proceeding, this will corrupt build_features.py's lag/target "
            f"computation if not fixed here."
        )
    combined["disease"] = disease

    print(f"\n  {disease}: years covered = {sorted(set(years_found))}")

    return combined

# 5. ADD ISO WEEK_START (matching other scripts' convention)

def add_week_start(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a week_start column (Monday of the ISO week), matching the
    convention used in fetch_weather.py / fetch_air_quality.py /
    fetch_pollen.py / parse_trends_csv.py.

    Assumption: RKI's "Meldewoche" numbering follows the same
    ISO 8601 / WHO epidemiological week definition (Monday-start).
    This is the standard convention RKI uses, but if your data seems
    off by a few days after joining with other sources, this is the
    first place to double-check.
    """
    df = df.copy()
    df["week_start"] = pd.to_datetime(
        df["year"].astype(str) + "-W" +
        df["week"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )
    return df

# 6. RUN

if __name__ == "__main__":
    all_diseases = []

    for disease, folder_name in DISEASE_FOLDERS.items():
        folder_path = os.path.join(RKI_BASE_DIR, folder_name)
        if not os.path.isdir(folder_path):
            print(f"\n⚠️  Folder not found, skipping: {folder_path}")
            continue

        df_disease = process_disease_folder(disease, folder_path)
        if not df_disease.empty:
            all_diseases.append(df_disease)

    if not all_diseases:
        print("\nNo data processed at all — check RKI_BASE_DIR and folder names.")
    else:
        df_long = pd.concat(all_diseases, ignore_index=True)
        df_long = add_week_start(df_long)

        # Flag the "Unbekannt" (unassigned) row separately so it can
        # easily be excluded from per-Kreis modeling later
        df_long["is_unassigned"] = df_long["kreis_name"].str.strip().eq("Unbekannt")

        long_path = os.path.join(OUTPUT_DIR, "rki_incidence_long.parquet")
        df_long.to_parquet(long_path, index=False)
        print(f"\n{'='*60}")
        print(f"✓ Saved long format: {long_path}  ({df_long.shape[0]:,} rows)")

        # Wide format: one column per disease
        df_wide = df_long.pivot_table(
            index=["kreis_name", "year", "week", "week_start"],
            columns="disease",
            values="incidence",
            aggfunc="mean",
        ).reset_index()
        df_wide.columns.name = None

        wide_path = os.path.join(OUTPUT_DIR, "rki_incidence_wide.parquet")
        df_wide.to_parquet(wide_path, index=False)
        print(f"✓ Saved wide format: {wide_path}  ({df_wide.shape})")

        # Summary
        print(f"\nSummary per disease:")
        for disease in df_long["disease"].unique():
            sub = df_long[df_long["disease"] == disease]
            print(f"  {disease:12s}: {sub['year'].min()}-{sub['year'].max()}, "
                  f"{sub['kreis_name'].nunique()} Kreise, {len(sub):,} rows")

        print(f"\nTotal unique Kreis names across all diseases: "
              f"{df_long['kreis_name'].nunique()}")
        print("\nDone ✓")
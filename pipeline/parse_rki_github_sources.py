"""
RespiWatch — Additional RKI GitHub sources parser
======================================================
Parses ARE-Konsultationsinzidenz, GrippeWeb, and Notaufnahmesurveillance
into a format joinable with your existing weekly Kreis-level pipeline.

IMPORTANT: run inspect_rki_github_sources.py FIRST and compare its
column output against the COLUMN_PATTERNS dicts below. Column detection
here is pattern-based (substring matching on lowercased column names)
rather than hardcoded exact names, since RKI's own docs describe
columns in prose rather than as an exact header dump. If detection
picks the wrong column for something, check the printed "Detected
columns" output at the top of each parser's run and adjust the
relevant pattern list.

Geographic granularity reminder (see prior discussion):
    - ARE-Konsultationsinzidenz: national from 2012/13, Bundesland-level
      only from season 2022/23 onward. BundeslandId: 0 = national,
      1-16 = Bundesland (AGS-based numbering).
    - GrippeWeb: coarse regions, not Bundesland or Kreis — needs the
      separate Grippeweb_Zuordnung_Regionen.tsv crosswalk (not yet
      integrated here — see note at bottom).
    - Notaufnahmesurveillance: NATIONAL ONLY, daily resolution, no
      regional breakdown at all in the main time series file.

Usage:
    python parse_rki_github_sources.py
"""

import os
import re
import pandas as pd
import numpy as np

# 1. CONFIGURATION

RKI_GITHUB_DIR = "./data/rki_github"
OUTPUT_DIR     = "./data/rki_github/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATHS = {
    "are":         os.path.join(RKI_GITHUB_DIR, "ARE-Konsultationsinzidenz.tsv"),
    "grippeweb":   os.path.join(RKI_GITHUB_DIR, "GrippeWeb_Daten_des_Wochenberichts.tsv"),
    "notaufnahme": os.path.join(RKI_GITHUB_DIR, "Notaufnahmesurveillance_Zeitreihen_Syndrome.tsv"),
    "grippeweb_regions": os.path.join(RKI_GITHUB_DIR, "Grippeweb_Zuordnung_Regionen.tsv"),
}

# Official AGS-based Bundesland numbering used across RKI datasets
# (0 = national/"Deutschland gesamt")
BUNDESLAND_ID_TO_NUTS1 = {
    0:  "DE0",  # national sentinel, matches parse_trends_csv.py convention
    1:  "DEF",  # Schleswig-Holstein
    2:  "DE6",  # Hamburg
    3:  "DE9",  # Niedersachsen
    4:  "DE5",  # Bremen
    5:  "DEA",  # Nordrhein-Westfalen
    6:  "DE7",  # Hessen
    7:  "DEB",  # Rheinland-Pfalz
    8:  "DE1",  # Baden-Württemberg
    9:  "DE2",  # Bayern
    10: "DEC",  # Saarland
    11: "DE3",  # Berlin
    12: "DE4",  # Brandenburg
    13: "DE8",  # Mecklenburg-Vorpommern
    14: "DED",  # Sachsen
    15: "DEE",  # Sachsen-Anhalt
    16: "DEG",  # Thüringen
}

# Bundesland name (as spelled in Grippeweb_Zuordnung_Regionen.tsv, ASCII
# umlaut style: "ue"/"oe"/"ae") -> nuts1_code. Needed specifically for
# the GrippeWeb region crosswalk, since that file uses full names
# rather than the numeric Bundesland_ID used elsewhere.
BUNDESLAND_NAME_TO_NUTS1 = {
    "Baden-Wuerttemberg":      "DE1",
    "Bayern":                  "DE2",
    "Berlin":                  "DE3",
    "Brandenburg":             "DE4",
    "Bremen":                  "DE5",
    "Hamburg":                 "DE6",
    "Hessen":                  "DE7",
    "Mecklenburg-Vorpommern":  "DE8",
    "Niedersachsen":           "DE9",
    "Nordrhein-Westfalen":     "DEA",
    "Rheinland-Pfalz":         "DEB",
    "Saarland":                "DEC",
    "Sachsen":                 "DED",
    "Sachsen-Anhalt":          "DEE",
    "Schleswig-Holstein":      "DEF",
    "Thueringen":              "DEG",
}


def load_grippeweb_region_crosswalk(path: str) -> pd.DataFrame:
    """
    Loads the Bundesland -> coarse Region mapping.
    Returns a DataFrame with columns: bundesland_name, region, nuts1_code
    (excludes "Bundesweit" rows — that's the national aggregate already
    present as its own Region value in the main GrippeWeb file, not a
    real region to expand).
    """
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    df = df[df["Region"] != "Bundesweit"].copy()
    df["nuts1_code"] = df["Bundesland"].map(BUNDESLAND_NAME_TO_NUTS1)

    missing = df[df["nuts1_code"].isna()]["Bundesland"].unique()
    if len(missing) > 0:
        print(f"⚠️  Bundesland names not found in BUNDESLAND_NAME_TO_NUTS1: "
              f"{list(missing)} — check spelling (umlaut style) and add "
              f"them to the dict above.")

    return df[["Bundesland", "Region", "nuts1_code"]].rename(
        columns={"Bundesland": "bundesland_name", "Region": "region"}
    )


def expand_grippeweb_to_bundeslaender(grippeweb_df: pd.DataFrame,
                                       crosswalk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands GrippeWeb's coarse-region rows into one row per Bundesland
    within that region (each Bundesland in a region shares the SAME
    incidence value, since GrippeWeb only estimates at region
    resolution — this is an honest broadcast, not a finer estimate).

    "Bundesweit" rows are kept as-is with nuts1_code="DE0" (national),
    matching the sentinel convention used for ARE-Konsultationsinzidenz.
    """
    national = grippeweb_df[grippeweb_df["region"] == "Bundesweit"].copy()
    national["nuts1_code"] = "DE0"

    regional = grippeweb_df[grippeweb_df["region"] != "Bundesweit"].copy()
    expanded = regional.merge(
        crosswalk_df[["region", "nuts1_code"]],
        on="region", how="left",
    )

    unmatched = expanded["nuts1_code"].isna().sum()
    if unmatched > 0:
        unmatched_regions = expanded[expanded["nuts1_code"].isna()]["region"].unique()
        print(f"⚠️  {unmatched} rows had no matching region in the crosswalk: "
              f"{list(unmatched_regions)}")

    return pd.concat([national, expanded], ignore_index=True)

# 2. FLEXIBLE COLUMN DETECTION

def find_column(columns: list[str], patterns: list[str]) -> str | None:
    """
    Returns the first column whose lowercased name contains any of the
    given substrings, checked in priority order (first pattern that
    matches ANY column wins, so put more specific patterns first).
    """
    cols_lower = {c: c.lower() for c in columns}
    for pattern in patterns:
        for col, col_lower in cols_lower.items():
            if pattern in col_lower:
                return col
    return None


def parse_season_or_week(df: pd.DataFrame, week_col: str) -> pd.DataFrame:
    """
    RKI's week columns commonly appear as either:
      - separate Jahr + Woche integer columns, or
      - a single "2023-W40" / "2023W40" style string column
    Normalises whatever is found into explicit year/week integer columns.
    """
    sample = str(df[week_col].dropna().iloc[0])

    if re.match(r"^\d{4}-?W\d{1,2}$", sample):
        # e.g. "2023-W40" or "2023W40"
        extracted = df[week_col].astype(str).str.extract(r"(\d{4})-?W(\d{1,2})")
        df["year"] = extracted[0].astype(int)
        df["week"] = extracted[1].astype(int)
    else:
        # Already looks like a plain integer week; year must come from
        # elsewhere (handled by caller via a separate year column)
        df["week"] = df[week_col].astype(int)

    return df

# 3. PARSER: ARE-KONSULTATIONSINZIDENZ

def parse_are(path: str) -> pd.DataFrame:
    print(f"\n{'='*60}\nARE-Konsultationsinzidenz\n{'='*60}")
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")

    cols = list(df.columns)
    col_year    = find_column(cols, ["jahr"])
    col_week    = find_column(cols, ["kalenderwoche", "woche"])
    col_season  = find_column(cols, ["saison"])
    # "bundesland_id" must be checked BEFORE plain "bundesland", since
    # "bundesland" is a substring of "Bundesland_ID" too and would
    # otherwise match the wrong (text) column, e.g. "Bundesweit"
    # instead of the numeric ID 0.
    col_bl      = find_column(cols, ["bundesland_id", "bundeslandid", "bundesland"])
    col_age     = find_column(cols, ["altersgruppe"])
    col_value   = find_column(cols, ["inzidenz"])

    print("Detected columns:")
    print(f"  year={col_year}  week={col_week}  season={col_season}  "
          f"bundesland={col_bl}  age={col_age}  value={col_value}")

    if col_value is None:
        raise ValueError("Could not detect the incidence value column — "
                          "check inspect_rki_github_sources.py output and "
                          "adjust find_column patterns above.")

    out = pd.DataFrame()

    if col_year and col_week:
        out["year"] = df[col_year].astype(int)
        out["week"] = df[col_week].astype(int)
    elif col_week:
        df = parse_season_or_week(df, col_week)
        out["year"] = df["year"] if "year" in df.columns else np.nan
        out["week"] = df["week"]
    else:
        raise ValueError("Could not detect year/week columns.")

    out["bundesland_id"] = df[col_bl].astype(int) if col_bl else 0
    out["nuts1_code"] = out["bundesland_id"].map(BUNDESLAND_ID_TO_NUTS1)
    out["age_group"] = df[col_age] if col_age else "00+"
    out["are_incidence"] = pd.to_numeric(df[col_value], errors="coerce")

    print(f"Rows: {len(out)}  |  Years: {out['year'].min()}-{out['year'].max()}  "
          f"|  Bundesland IDs present: {sorted(out['bundesland_id'].unique())}")

    return out

# 4. PARSER: GRIPPEWEB

def parse_grippeweb(path: str) -> pd.DataFrame:
    print(f"\n{'='*60}\nGrippeWeb\n{'='*60}")
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")

    cols = list(df.columns)
    col_year    = find_column(cols, ["jahr"])
    col_week    = find_column(cols, ["kalenderwoche", "woche"])
    col_region  = find_column(cols, ["region"])
    col_age     = find_column(cols, ["altersgruppe"])
    col_disease = find_column(cols, ["erkrankung", "syndrom"])
    col_value   = find_column(cols, ["inzidenz"])

    print("Detected columns:")
    print(f"  year={col_year}  week={col_week}  region={col_region}  "
          f"age={col_age}  disease={col_disease}  value={col_value}")

    if col_value is None:
        raise ValueError("Could not detect the incidence value column — "
                          "check inspect_rki_github_sources.py output.")

    out = pd.DataFrame()

    if col_year and col_week:
        out["year"] = df[col_year].astype(int)
        out["week"] = df[col_week].astype(int)
    elif col_week:
        df = parse_season_or_week(df, col_week)
        out["year"] = df["year"] if "year" in df.columns else np.nan
        out["week"] = df["week"]
    else:
        raise ValueError("Could not detect year/week columns.")

    out["region"] = df[col_region] if col_region else "unknown"
    out["age_group"] = df[col_age] if col_age else "00+"
    out["disease"] = df[col_disease] if col_disease else "ARE"
    out["grippeweb_incidence"] = pd.to_numeric(df[col_value], errors="coerce")

    print(f"Rows: {len(out)}  |  Years: {out['year'].min()}-{out['year'].max()}  "
          f"|  Regions: {sorted(out['region'].unique())}")

    print("\nNote: 'region' is GrippeWeb's own coarse region system "
          "(Sueden/Osten/Mitte (West)/Norden (West)/Bundesweit), not "
          "Bundesland or Kreis directly — it gets expanded to nuts1_code "
          "via Grippeweb_Zuordnung_Regionen.tsv in the RUN section below.")

    return out

# 5. PARSER: NOTAUFNAHMESURVEILLANCE

def parse_notaufnahme(path: str) -> pd.DataFrame:
    print(f"\n{'='*60}\nNotaufnahmesurveillance\n{'='*60}")
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")

    cols = list(df.columns)
    col_date   = find_column(cols, ["datum", "date"])
    col_type   = find_column(cols, ["notaufnahmetyp", "ed_type", "edtype"])
    col_age    = find_column(cols, ["altersgruppe", "age_group", "agegroup"])
    col_syndr  = find_column(cols, ["syndrom", "syndrome"])

    col_value_raw     = find_column(cols, ["anteil", "relative_cases"])
    col_value_smooth  = find_column(cols, ["7-tage", "7tage", "gleitend", "7day"])
    col_expected      = find_column(cols, ["erwartungswert", "expected_value"])
    col_n_included    = find_column(cols, ["notaufnahmen", "ed_count"])

    print("Detected columns:")
    print(f"  date={col_date}  type={col_type}  age={col_age}  "
          f"syndrome={col_syndr}  smoothed_value={col_value_smooth}  "
          f"raw_value={col_value_raw}  expected_value={col_expected}  "
          f"n_included_eds={col_n_included}")

    if col_date is None:
        raise ValueError("Could not detect the date column — check "
                          "inspect_rki_github_sources.py output.")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[col_date], errors="coerce")
    out["year"] = out["date"].dt.isocalendar().year.astype(int)
    out["week"] = out["date"].dt.isocalendar().week.astype(int)

    out["notaufnahmetyp"] = df[col_type] if col_type else "all"
    out["age_group"] = df[col_age] if col_age else "00+"
    out["syndrome"] = df[col_syndr] if col_syndr else "unknown"

    if col_value_raw:
        out["notaufnahme_share_raw"] = pd.to_numeric(df[col_value_raw], errors="coerce")
    if col_value_smooth:
        out["notaufnahme_share_smoothed"] = pd.to_numeric(df[col_value_smooth], errors="coerce")
    if col_expected:
        out["notaufnahme_expected_value"] = pd.to_numeric(df[col_expected], errors="coerce")
    if col_n_included:
        out["n_eds_included"] = pd.to_numeric(df[col_n_included], errors="coerce")

    print(f"Rows: {len(out)}  |  Date range: {out['date'].min()} → {out['date'].max()}  "
          f"|  Syndromes: {sorted(out['syndrome'].dropna().unique())}")
    print("\n⚠️  This dataset is NATIONAL ONLY — no Bundesland/Kreis column "
          "exists here. Aggregate to weekly (already daily -> we keep both "
          "year/week and date) and treat as a national-context feature, "
          "broadcast to all Kreise, or use for nowcasting validation "
          "rather than as a per-Kreis training feature.")

    # Aggregate daily -> weekly mean per syndrome/type/age, since your
    # modeling granularity is weekly. Built dynamically so it never
    # references a column that wasn't actually detected/created above.
    value_cols = [c for c in [
        "notaufnahme_share_raw", "notaufnahme_share_smoothed",
        "notaufnahme_expected_value", "n_eds_included",
    ] if c in out.columns]

    weekly = (
        out.groupby(["year", "week", "notaufnahmetyp", "age_group", "syndrome"])[value_cols]
        .mean()
        .reset_index()
    )

    return weekly

# 6. RUN

if __name__ == "__main__":
    results = {}

    for key, path in PATHS.items():
        if key == "grippeweb_regions":
            continue 

        if not os.path.exists(path):
            print(f"⚠️  File not found, skipping: {path}")
            continue

        try:
            if key == "are":
                results["are"] = parse_are(path)
            elif key == "grippeweb":
                results["grippeweb"] = parse_grippeweb(path)
            elif key == "notaufnahme":
                results["notaufnahme"] = parse_notaufnahme(path)
        except Exception as e:
            print(f"\n✗ Failed to parse {key}: {e}")
            print("  Run inspect_rki_github_sources.py and check the actual "
                  "column names against the patterns in this script.\n")

    # Expand GrippeWeb from coarse regions to nuts1_code
    if "grippeweb" in results and os.path.exists(PATHS["grippeweb_regions"]):
        print(f"\n{'='*60}\nExpanding GrippeWeb regions -> nuts1_code\n{'='*60}")
        crosswalk_df = load_grippeweb_region_crosswalk(PATHS["grippeweb_regions"])
        print(f"Crosswalk loaded: {len(crosswalk_df)} Bundesland-Region pairs")

        results["grippeweb"] = expand_grippeweb_to_bundeslaender(
            results["grippeweb"], crosswalk_df
        )
        n_nuts1 = results["grippeweb"]["nuts1_code"].nunique()
        print(f"✓ Expanded to {n_nuts1} distinct nuts1_code values "
              f"(16 Bundesländer + 1 national sentinel 'DE0' = 17 expected)")
    elif "grippeweb" in results:
        print(f"\n⚠️  {PATHS['grippeweb_regions']} not found — GrippeWeb "
              f"output will keep coarse 'region' names instead of "
              f"nuts1_code, and won't join directly with your other data.")

    print(f"\n{'='*60}\nSaving outputs\n{'='*60}")
    for key, df in results.items():
        out_path = os.path.join(OUTPUT_DIR, f"{key}_weekly.parquet")
        df.to_parquet(out_path, index=False)
        print(f"✓ Saved: {out_path}  ({df.shape[0]:,} rows)")
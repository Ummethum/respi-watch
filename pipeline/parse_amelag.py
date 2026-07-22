"""
RespiWatch — AMELAG wastewater surveillance parser (RKI methodology)
========================================================================
Implements RKI's own AMELAG aggregation methodology (as described in
the AMELAG Technischer Leitfaden, Abschnitt 5.4 "Statistische Analysen"),
applied per Bundesland instead of nationwide.

RKI's method, step by step:
    1. Compute each Standort's weekly mean measured value.
    2. For each week, compute the cross-site mean over all Standorte
       reporting that week ("Wochenmittelwert").
    3. For each Standort, compute its deviation from that week's
       cross-site mean, for every week it reported.
    4. For each Standort, average these deviations ACROSS ALL WEEKS —
       this gives one constant "bias correction" term per Standort,
       capturing systematic differences between sites/labs (e.g.
       different quantification baselines).
    5. Subtract each Standort's constant bias-correction term from its
       own weekly values -> "adjusted" values.
    6. For weeks with enough reporting Standorte, compute the
       population-weighted mean (by connected "einwohner") over the
       adjusted values.

Deviations from RKI's exact method, due to data/scope limitations:
    - RKI groups by "Standort-Labor-Kombination" (site+lab pair), since
      a lab change can shift the systematic bias too. Our data only has
      a "laborwechsel" yes/no flag, no actual lab identifier — so this
      script groups by Standort alone.
    - RKI requires >=20 reporting Standorte nationally per week before
      including it. Applied per Bundesland, 20 is too strict (most
      Bundesländer never have that many sites)
    - Uses "viruslast" (raw), not "viruslast_normalisiert", matching
      current AMELAG practice — flow-normalisation was discontinued
      01.08.2025 per the technical guide (see AMELAG-Leitfaden.pdf,
      Abschnitt 5.1), since it didn't improve data quality.

Usage:
    python parse_amelag.py
"""

import os
import pandas as pd
import numpy as np

# 1. CONFIGURATION

AMELAG_PATH = "./data/amelag/amelag_einzelstandorte.tsv"   # adjust to your actual filename/extension
OUTPUT_DIR  = "./data/amelag/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Current AMELAG practice uses raw viruslast, not normalised, as of
# the 01.08.2025 discontinuation of flow-normalisation for SARS-CoV-2
# (and Influenza/RSV were never normalised to begin with, per the
# technical guide).
VALUE_COL = "viruslast"

# Minimum number of reporting Kläranlagen within a Bundesland for a
# given week to be included at all — mirrors RKI's own >=20 threshold
# (applied nationally), scaled down since individual Bundesländer have
# far fewer sites.
MIN_SITES_PER_WEEK = 2

# German Bundesland abbreviation -> nuts1_code, matching the convention
# used throughout this project.
BUNDESLAND_TO_NUTS1 = {
    "BW": "DE1", "BY": "DE2", "BE": "DE3", "BB": "DE4",
    "HB": "DE5", "HH": "DE6", "HE": "DE7", "MV": "DE8",
    "NI": "DE9", "NW": "DEA", "RP": "DEB", "SL": "DEC",
    "SN": "DED", "ST": "DEE", "SH": "DEF", "TH": "DEG",
}

VIRUS_FAMILY_SYNONYMS = {
    "amelag_sarscov2":  ["SARS-CoV-2"],
    "amelag_influenza": ["Influenza A+B"],
    "amelag_rsv":       ["RSV A+B", "RSV A/B"],
}

# 2. LOAD & PREP

def load_amelag(path: str) -> pd.DataFrame:
    sep = "\t" if path.endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")

    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    df = df.dropna(subset=["datum"])

    df["nuts1_code"] = df["bundesland"].map(BUNDESLAND_TO_NUTS1)
    missing_bl = df[df["nuts1_code"].isna()]["bundesland"].unique()
    if len(missing_bl) > 0:
        print(f"⚠️  Unmapped Bundesland codes found: {list(missing_bl)} — "
              f"check BUNDESLAND_TO_NUTS1.")

    df["year"] = df["datum"].dt.isocalendar().year.astype(int)
    df["week"] = df["datum"].dt.isocalendar().week.astype(int)

    return df


def print_site_counts_per_bundesland(df: pd.DataFrame) -> None:
    """Diagnostic: how many distinct Kläranlagen exist per Bundesland,
    to help pick a sensible MIN_SITES_PER_WEEK."""
    counts = df.groupby("nuts1_code")["standort"].nunique().sort_values(ascending=False)
    print("Distinct Kläranlagen per Bundesland (for MIN_SITES_PER_WEEK sanity check):")
    print(counts.to_string())
    print(f"\nCurrent MIN_SITES_PER_WEEK = {MIN_SITES_PER_WEEK}")
    below_threshold = counts[counts < MIN_SITES_PER_WEEK]
    if not below_threshold.empty:
        print(f"⚠️  These Bundesländer NEVER reach the threshold even with "
              f"ALL their sites reporting in the same week — they will end "
              f"up with zero usable weeks:")
        print(below_threshold.to_string())

# 3. RKI METHODOLOGY: BIAS-CORRECTED, POPULATION-WEIGHTED AGGREGATION

def aggregate_rki_method(df: pd.DataFrame, family_col: str,
                         accepted_typen: list[str]) -> pd.DataFrame:
    """
    Implements the RKI AMELAG aggregation method (see module docstring),
    applied independently within each Bundesland (nuts1_code).
    """
    sub = df[df["typ"].isin(accepted_typen)].copy()
    sub = sub.dropna(subset=[VALUE_COL, "einwohner"])
    sub = sub[sub["einwohner"] > 0]

    if sub.empty:
        print(f"  ⚠️  {family_col}: no usable rows found.")
        return pd.DataFrame(columns=["nuts1_code", "year", "week", family_col])

    # Step 1: weekly mean per Standort
    site_week = (
        sub.groupby(["nuts1_code", "standort", "year", "week"])
        .agg(site_week_value=(VALUE_COL, "mean"),
             einwohner=("einwohner", "mean"))   # stable per site, mean handles minor row variation
        .reset_index()
    )

    all_family_results = []

    for nuts1_code, group in site_week.groupby("nuts1_code"):
        group = group.copy()

        # Step 2: cross-site weekly mean, within this Bundesland
        weekly_mean = group.groupby(["year", "week"])["site_week_value"].transform("mean")

        # Step 3: each site's deviation from that week's cross-site mean
        group["deviation"] = group["site_week_value"] - weekly_mean

        # Step 4: each site's CONSTANT bias correction (mean deviation
        # across all its reporting weeks)
        site_bias = group.groupby("standort")["deviation"].transform("mean")

        # Step 5: adjusted values = original weekly value minus the
        # site's constant bias term
        group["adjusted_value"] = group["site_week_value"] - site_bias

        # Step 6: population-weighted mean of adjusted values per week,
        # only for weeks meeting the minimum site count
        weekly_n_sites = group.groupby(["year", "week"])["standort"].transform("nunique")
        group = group[weekly_n_sites >= MIN_SITES_PER_WEEK]

        if group.empty:
            continue

        def weighted_mean(g: pd.DataFrame) -> float:
            return np.average(g["adjusted_value"], weights=g["einwohner"])

        agg = (
            group.groupby(["year", "week"])
            .apply(weighted_mean, include_groups=False)
            .reset_index(name=family_col)
        )
        agg["nuts1_code"] = nuts1_code

        n_sites_report = (
            group.groupby(["year", "week"])["standort"].nunique()
            .reset_index(name=f"{family_col}_n_sites")
        )
        agg = agg.merge(n_sites_report, on=["year", "week"], how="left")

        all_family_results.append(agg)

    if not all_family_results:
        print(f"  ⚠️  {family_col}: no Bundesland reached MIN_SITES_PER_WEEK "
              f"in any week.")
        return pd.DataFrame(columns=["nuts1_code", "year", "week", family_col])

    result = pd.concat(all_family_results, ignore_index=True)
    print(f"  ✓ {family_col}: {len(result)} Bundesland×week rows across "
          f"{result['nuts1_code'].nunique()} Bundesländer")
    return result

# 4. RUN

if __name__ == "__main__":
    print(f"Loading {AMELAG_PATH}...")
    df = load_amelag(AMELAG_PATH)
    print(f"Loaded {len(df):,} rows, {df['standort'].nunique()} Kläranlagen, "
          f"date range {df['datum'].min().date()} → {df['datum'].max().date()}\n")

    print("Available 'typ' values in data:")
    print(sorted(df["typ"].dropna().unique()))
    print()

    print_site_counts_per_bundesland(df)
    print()

    results = []
    for family_col, synonyms in VIRUS_FAMILY_SYNONYMS.items():
        print(f"\nAggregating {family_col} (typ in {synonyms})...")
        agg = aggregate_rki_method(df, family_col, synonyms)
        if not agg.empty:
            results.append(agg)

    if not results:
        print("\nNo data aggregated — check VALUE_COL, MIN_SITES_PER_WEEK, "
              "and VIRUS_FAMILY_SYNONYMS against the printed diagnostics above.")
    else:
        merged = results[0]
        for r in results[1:]:
            merged = merged.merge(r, on=["nuts1_code", "year", "week"], how="outer")

        merged["week_start"] = pd.to_datetime(
            merged["year"].astype(str) + "-W" +
            merged["week"].astype(str).str.zfill(2) + "-1",
            format="%G-W%V-%u",
        )
        merged = merged.sort_values(["nuts1_code", "year", "week"]).reset_index(drop=True)

        out_path = os.path.join(OUTPUT_DIR, "amelag_weekly_bundesland.parquet")
        merged.to_parquet(out_path, index=False)
        print(f"\n✓ Saved: {out_path}  ({merged.shape[0]:,} rows)")
        print(f"Columns: {list(merged.columns)}")
        print(f"\nDate range: {merged['week_start'].min().date()} → "
              f"{merged['week_start'].max().date()}")
        print(f"Bundesländer covered: {sorted(merged['nuts1_code'].unique())}")

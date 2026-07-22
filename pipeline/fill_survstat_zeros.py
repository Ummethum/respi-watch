"""
RespiWatch -- SurvStat zero-filling (coverage-aware)
=======================================================
Distinguishes two different meanings of NaN in the survstat_* target
columns of master_dataset.parquet:

    1. Within a disease's actual surveillance period, a gap most
       likely means 0 reported cases that week (surrounding weeks in
       the same Kreis/year DO have values, proving reporting was
       active) -- this gets filled with 0.

    2. Before a disease's surveillance existed at all (RSV pre-2023,
       COVID pre-2020), NaN means "no reporting system yet", not zero
       cases -- this is left as NaN, since filling it with 0 would
       falsely claim "zero RSV cases in 2015" when no data was ever
       collected.

Coverage start per disease is determined empirically: the first
year/week with ANY non-null value across ALL Kreise. This assumes
national IfSG reporting obligations start uniformly (a reasonable
simplification -- individual Kreise are not expected to have wildly
different reporting start dates for the same nationally mandated
disease).

Usage:
    python fill_survstat_zeros.py

Output:
    data/master/master_dataset_filled.parquet   -- updated dataset
    data/master/survstat_fill_report.csv         -- per-disease,
                                                    per-year fill stats
                                                    for sanity-checking
"""

import os
import pandas as pd
import numpy as np

# 1. CONFIGURATION

MASTER_PATH = "./data/master/master_dataset.parquet"
OUTPUT_DIR  = "./data/master"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. COVERAGE DETECTION

def find_coverage_start(df: pd.DataFrame, disease_col: str) -> tuple[int, int]:
    """
    Returns (year, week) of the first row with a non-null value for
    this disease, across all Kreise -- i.e. when national surveillance
    for this disease effectively begins in the dataset.
    """
    valid = df[df[disease_col].notna()]
    if valid.empty:
        return None, None
    first = valid.sort_values(["year", "week"]).iloc[0]
    return int(first["year"]), int(first["week"])


def within_coverage(df: pd.DataFrame, start_year: int, start_week: int) -> pd.Series:
    """Boolean mask: True for rows at or after the coverage start."""
    return (
        (df["year"] > start_year) |
        ((df["year"] == start_year) & (df["week"] >= start_week))
    )

# 3. FILL LOGIC

def fill_disease_column(df: pd.DataFrame, disease_col: str) -> tuple[pd.DataFrame, dict]:
    """
    Fills NaN with 0 only within the disease's coverage period.
    Returns the updated DataFrame and a stats dict for reporting.
    """
    start_year, start_week = find_coverage_start(df, disease_col)

    if start_year is None:
        print(f"  \u26a0\ufe0f  {disease_col}: no non-null values found at all -- skipping.")
        return df, {}

    mask_coverage = within_coverage(df, start_year, start_week)
    mask_missing = df[disease_col].isna()

    mask_fill = mask_coverage & mask_missing
    mask_true_missing = (~mask_coverage) & mask_missing
    mask_original = ~mask_missing

    n_fill = mask_fill.sum()
    n_true_missing = mask_true_missing.sum()
    n_original = mask_original.sum()
    n_total = len(df)

    df.loc[mask_fill, disease_col] = 0.0

    stats = {
        "disease": disease_col,
        "coverage_start": f"{start_year}-W{start_week:02d}",
        "n_total_rows": n_total,
        "n_original_values": n_original,
        "n_filled_with_zero": n_fill,
        "n_true_missing_pre_coverage": n_true_missing,
        "pct_filled": round(100 * n_fill / n_total, 2),
        "pct_true_missing": round(100 * n_true_missing / n_total, 2),
    }
    return df, stats

# 4. SUSPICIOUS-WEEK DETECTION

def flag_suspicious_weeks(df: pd.DataFrame, disease_col: str,
                          start_year: int, start_week: int) -> pd.DataFrame:
    """
    Within the coverage period, flags (year, week) combinations where
    ALL Kreise got filled with 0 -- this could be genuine (a very quiet
    week for a rare disease) or could indicate a systemic reporting
    gap (e.g. delayed submission around a holiday) rather than true
    zero incidence everywhere. Worth a manual glance, not necessarily
    a bug.
    """
    mask_coverage = within_coverage(df, start_year, start_week)
    coverage_df = df[mask_coverage]

    weekly_stats = coverage_df.groupby(["year", "week"])[disease_col].agg(
        n_zero=lambda s: (s == 0).sum(),
        n_total="count",
    )
    weekly_stats["pct_zero"] = 100 * weekly_stats["n_zero"] / weekly_stats["n_total"]

    suspicious = weekly_stats[weekly_stats["pct_zero"] >= 99.0].reset_index()
    return suspicious

# 5. RUN

if __name__ == "__main__":
    print(f"Loading {MASTER_PATH}...")
    df = pd.read_parquet(MASTER_PATH)

    disease_cols = [c for c in df.columns if c.startswith("survstat_")]
    print(f"Found target columns: {disease_cols}\n")

    all_stats = []
    suspicious_reports = []

    for col in disease_cols:
        print(f"{'='*60}\n{col}\n{'='*60}")

        start_year, start_week = find_coverage_start(df, col)
        if start_year is None:
            continue

        df, stats = fill_disease_column(df, col)
        all_stats.append(stats)

        print(f"  Coverage starts: {stats['coverage_start']}")
        print(f"  Original values      : {stats['n_original_values']:,} "
              f"({100*stats['n_original_values']/stats['n_total_rows']:.1f}%)")
        print(f"  Filled with 0         : {stats['n_filled_with_zero']:,} "
              f"({stats['pct_filled']:.1f}%)")
        print(f"  True missing (no data): {stats['n_true_missing_pre_coverage']:,} "
              f"({stats['pct_true_missing']:.1f}%)")

        suspicious = flag_suspicious_weeks(df, col, start_year, start_week)
        if not suspicious.empty:
            suspicious["disease"] = col
            suspicious_reports.append(suspicious)
            print(f"\n  \u26a0\ufe0f  {len(suspicious)} week(s) where ~100% of Kreise show 0 "
                  f"-- worth a quick sanity check (could be genuine, or a "
                  f"systemic reporting gap):")
            print(suspicious.head(10).to_string(index=False))
        print()

    # Save updated dataset
    output_path = os.path.join(OUTPUT_DIR, "master_dataset_filled.parquet")
    df.to_parquet(output_path, index=False)
    print(f"\u2713 Saved: {output_path}")

    # Save fill report
    report_df = pd.DataFrame(all_stats)
    report_path = os.path.join(OUTPUT_DIR, "survstat_fill_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"\u2713 Saved fill report: {report_path}")
    print("\n" + report_df.to_string(index=False))

    if suspicious_reports:
        suspicious_all = pd.concat(suspicious_reports, ignore_index=True)
        suspicious_path = os.path.join(OUTPUT_DIR, "survstat_suspicious_weeks.csv")
        suspicious_all.to_csv(suspicious_path, index=False)
        print(f"\n\u2713 Saved suspicious-weeks report: {suspicious_path}")
        print("  Review these -- near-100%-zero weeks across all Kreise are")
        print("  usually fine for rare diseases, but worth a glance in case")
        print("  they cluster around holidays or system changes.")

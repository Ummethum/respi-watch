"""
RespiWatch — Berlin borough aggregation (population-weighted)
=================================================================
Aggregates the 12 individual Berlin boroughs (SK Berlin Mitte, SK
Berlin Pankow, ...) in the RKI incidence data into a single "Berlin"
row, weighted by each borough's population — needed because weather/
trends/pollen data only has ONE Berlin entry at NUTS-3 level.

Population data source: Amt für Statistik Berlin-Brandenburg,
Einwohnerregisterstatistik (https://www.statistik-berlin-brandenburg.de),
covering 2014-2025. For years outside this range (2004-2013, 2026+),
the nearest available year's population shares are used as a
reasonable approximation — population *shares* between boroughs shift
slowly, so borrowing the closest year's weights introduces only minor
error, especially far from the data's edges.

Usage:
    python aggregate_berlin_boroughs.py
"""

import os
import re
import pandas as pd
import numpy as np

# 1. CONFIGURATION

POPULATION_CSV = "./data/rki/berlin_population.csv"
RKI_LONG_PARQUET = "./data/rki/processed/rki_incidence_long.parquet"
OUTPUT_PARQUET = "./data/rki/processed/rki_incidence_long_berlin_aggregated.parquet"

# Maps population CSV column names -> RKI-style borough names
COLUMN_TO_RKI_NAME = {
    "Mitte":                    "SK Berlin Mitte",
    "Friedrichs-\nhain-\nKreuz-\nberg": "SK Berlin Friedrichshain-Kreuzberg",
    "Pankow":                   "SK Berlin Pankow",
    "Charlotten-\nburg-\nWilmers-\ndorf": "SK Berlin Charlottenburg-Wilmersdorf",
    "Spandau":                  "SK Berlin Spandau",
    "Steglitz-\nZehlen-\ndorf": "SK Berlin Steglitz-Zehlendorf",
    "Tempel-\nhof-\nSchöne-\nberg": "SK Berlin Tempelhof-Schöneberg",
    "Neukölln":                 "SK Berlin Neukölln",
    "Treptow-\nKöpenick":       "SK Berlin Treptow-Köpenick",
    "Marzahn-\nHellers-\ndorf": "SK Berlin Marzahn-Hellersdorf",
    "Lichten-\nberg":           "SK Berlin Lichtenberg",
    "Reinicken-\ndorf":         "SK Berlin Reinickendorf",
}

BERLIN_BOROUGHS = list(COLUMN_TO_RKI_NAME.values())

# 2. LOAD & CLEAN POPULATION DATA

def clean_number(value: str) -> int:
    """Removes any whitespace (regular or non-breaking space) used as
    thousands separator, e.g. '356 506' -> 356506."""
    return int(re.sub(r"\s", "", str(value)))


def load_population_weights(path: str) -> pd.DataFrame:
    """
    Returns a long-format DataFrame: year, rki_name, population,
    population_share (share of total Berlin population that year).
    """
    df = pd.read_csv(path)

    # Extract year from "Stichtag" (format DD/MM/YYYY, always 31/12)
    df["year"] = pd.to_datetime(df["Stichtag"], format="%d/%m/%Y").dt.year

    # Clean numeric columns (all except Stichtag/year)
    value_cols = [c for c in df.columns if c not in ("Stichtag", "year")]
    for col in value_cols:
        df[col] = df[col].apply(clean_number)

    # Melt boroughs (excluding the "Berlin" total column) to long format
    borough_cols = [c for c in COLUMN_TO_RKI_NAME if c in df.columns]
    missing = set(COLUMN_TO_RKI_NAME) - set(borough_cols)
    if missing:
        print(f"⚠️  Warning: expected borough columns not found: {missing}")

    long_df = df.melt(
        id_vars=["year"], value_vars=borough_cols,
        var_name="column_name", value_name="population",
    )
    long_df["rki_name"] = long_df["column_name"].map(COLUMN_TO_RKI_NAME)

    # Population share within Berlin for that year
    totals = long_df.groupby("year")["population"].transform("sum")
    long_df["population_share"] = long_df["population"] / totals

    return long_df[["year", "rki_name", "population", "population_share"]]


def extend_weights_to_full_range(weights: pd.DataFrame, target_years: list[int]) -> pd.DataFrame:
    """
    For years outside the population data's range (2014-2025 here),
    borrows the nearest available year's population_share as a
    reasonable approximation. Prints which years were extrapolated.
    """
    available_years = sorted(weights["year"].unique())
    min_year, max_year = min(available_years), max(available_years)

    extended_rows = []
    borrowed_years = []

    for year in target_years:
        if year in available_years:
            continue  # already covered

        nearest = min_year if year < min_year else max_year
        borrowed_years.append((year, nearest))

        borrowed = weights[weights["year"] == nearest].copy()
        borrowed["year"] = year
        extended_rows.append(borrowed)

    if borrowed_years:
        print(f"\n⚠️  Population data covers {min_year}-{max_year}. "
              f"Borrowing nearest-year shares for:")
        for year, nearest in borrowed_years:
            print(f"    {year} -> using {nearest} population shares")

    if extended_rows:
        return pd.concat([weights] + extended_rows, ignore_index=True)
    return weights

# 3. AGGREGATE BERLIN BOROUGHS IN RKI DATA

def aggregate_berlin(rki_df: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces the 12 Berlin borough rows with a single population-
    weighted "Berlin" row per year/week/disease, leaving all other
    Kreise untouched.
    """
    is_berlin = rki_df["kreis_name"].isin(BERLIN_BOROUGHS)
    berlin_df = rki_df[is_berlin].copy()
    other_df = rki_df[~is_berlin].copy()

    if berlin_df.empty:
        print("⚠️  No Berlin borough rows found in RKI data — nothing to aggregate.")
        return rki_df

    # Extend population weights to cover every year present in the data
    target_years = sorted(berlin_df["year"].unique())
    weights = extend_weights_to_full_range(weights, target_years)

    # Join weights onto the Berlin rows
    berlin_df = berlin_df.merge(
        weights[["year", "rki_name", "population_share"]],
        left_on=["year", "kreis_name"], right_on=["year", "rki_name"],
        how="left",
    )

    missing_weights = berlin_df["population_share"].isna().sum()
    if missing_weights > 0:
        print(f"⚠️  {missing_weights} rows have no population weight "
              f"(unmatched borough name?) — falling back to equal "
              f"weighting for those rows.")
        berlin_df["population_share"] = berlin_df["population_share"].fillna(
            1 / len(BERLIN_BOROUGHS)
        )

    # Population-weighted average incidence per year/week/disease
    def weighted_mean(group: pd.DataFrame) -> float:
        return np.average(group["incidence"], weights=group["population_share"])

    group_cols = ["year", "week", "disease"] if "disease" in berlin_df.columns else ["year", "week"]

    berlin_agg = (
        berlin_df.groupby(group_cols)
        .apply(weighted_mean, include_groups=False)
        .reset_index(name="incidence")
    )
    berlin_agg["kreis_name"] = "Berlin"

    # Carry over week_start / is_unassigned if present in the original data
    for col in ["week_start", "is_unassigned"]:
        if col in rki_df.columns and col not in berlin_agg.columns:
            if col == "is_unassigned":
                berlin_agg[col] = False
            else:
                # Recompute week_start for consistency
                berlin_agg["week_start"] = pd.to_datetime(
                    berlin_agg["year"].astype(str) + "-W" +
                    berlin_agg["week"].astype(str).str.zfill(2) + "-1",
                    format="%G-W%V-%u",
                )

    result = pd.concat([other_df, berlin_agg], ignore_index=True)
    return result

# 4. RUN

if __name__ == "__main__":
    print("Loading population weights...")
    weights = load_population_weights(POPULATION_CSV)
    print(f"  {weights['year'].nunique()} years, "
          f"{weights['rki_name'].nunique()} boroughs")
    print(weights.groupby("year")["population_share"].sum().round(3).to_string())

    print("\nLoading RKI incidence data...")
    rki_df = pd.read_parquet(RKI_LONG_PARQUET)
    n_before = len(rki_df)
    n_berlin_before = rki_df["kreis_name"].isin(BERLIN_BOROUGHS).sum()
    print(f"  {n_before:,} total rows, {n_berlin_before:,} Berlin borough rows")

    print("\nAggregating Berlin boroughs...")
    result = aggregate_berlin(rki_df, weights)

    n_after = len(result)
    n_berlin_after = (result["kreis_name"] == "Berlin").sum()
    print(f"\n  Rows before: {n_before:,}")
    print(f"  Rows after : {n_after:,}")
    print(f"  Berlin (aggregated) rows: {n_berlin_after:,}")

    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    result.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\n✓ Saved long format: {OUTPUT_PARQUET}")

    index_cols = ["kreis_name", "year", "week"]
    if "week_start" in result.columns:
        index_cols.append("week_start")

    df_wide = result.pivot_table(
        index=index_cols,
        columns="disease",
        values="incidence",
        aggfunc="mean",
    ).reset_index()
    df_wide.columns.name = None

    wide_path = OUTPUT_PARQUET.replace("_long_", "_wide_")
    df_wide.to_parquet(wide_path, index=False)
    print(f"✓ Saved wide format: {wide_path}  ({df_wide.shape})")

    print("\nSample of aggregated Berlin rows (wide format):")
    print(df_wide[df_wide["kreis_name"] == "Berlin"].head(10).to_string())

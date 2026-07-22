"""
Joins all processed data sources into one modeling-ready DataFrame:
one row per Kreis x ISO week, all targets and features as columns.

Join levels:
    - Kreis-level (weather, air quality, pollen, RKI incidence): join on kreis_id/year/week
    - Bundesland-level (Trends, ARE, GrippeWeb, AMELAG, holidays): join on nuts1_code/year/week
    - National-level (Notaufnahme): join on year/week

All joins are LEFT joins onto a weather-based spine, so every Kreis x
week row is kept even where a source has no data yet (e.g. pollen
before 2021). NaN is left as-is for tree models to handle natively.

Usage: python build_master_dataset.py
Output: data/master/master_dataset.parquet
"""

import os
import pandas as pd

ARE_AGE_GROUP_FILTER = "00+"
GRIPPEWEB_DISEASE_FILTER = "ARE"
GRIPPEWEB_AGE_GROUP_FILTER = "00+"
NOTAUFNAHME_SYNDROME_FILTER = "ARI"
NOTAUFNAHME_TYPE_FILTER = "all"
NOTAUFNAHME_AGE_FILTER = "00+"

PATHS = {
    "coords": "./data/city_coords/kreise_coords.csv",
    "weather": "./data/weather/weather_weekly.parquet",
    "air_quality": "./data/air_quality/air_quality_weekly.parquet",
    "pollen": "./data/pollen/pollen_weekly.parquet",
    "trends": "./data/google_trends/trends_wide_from_csv.parquet",
    "rki_incidence": "./data/rki/processed/rki_incidence_wide_berlin_aggregated.parquet",
    "kreis_crosswalk": "./data/rki/processed/kreis_name_crosswalk.csv",
    "are": "./data/rki_github/processed/are_weekly.parquet",
    "grippeweb": "./data/rki_github/processed/grippeweb_weekly.parquet",
    "notaufnahme": "./data/rki_github/processed/notaufnahme_weekly.parquet",
    "amelag": "./data/amelag/processed/amelag_weekly_bundesland.parquet",
    "holidays": "./data/holidays/holidays_weekly.parquet",
}

OUTPUT_DIR = "./data/master"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_load(path: str, label: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"WARNING: {label} not found, skipping -- {path}")
        return None
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    print(f"Loaded {label}: {len(df):,} rows")
    return df


def assert_unique_columns(df: pd.DataFrame, context: str):
    """All fetch_*.py scripts write unprefixed columns; prefixing
    happens only here. A duplicate means that convention was broken
    somewhere upstream -- fix it there, not here."""
    if not df.columns.is_unique:
        dupes = df.columns[df.columns.duplicated(keep=False)].unique().tolist()
        raise ValueError(f"Duplicate columns in '{context}': {dupes}")


def prefix_columns(df: pd.DataFrame, prefix: str, exclude: list[str]) -> pd.DataFrame:
    """Prepends `prefix` to every column not in `exclude`, skipping
    columns that already have it (avoids double-prefixing)."""
    rename_map = {
        c: f"{prefix}{c}" for c in df.columns
        if c not in exclude and not c.startswith(prefix)
    }
    return df.rename(columns=rename_map)


def build_spine() -> pd.DataFrame:
    weather = safe_load(PATHS["weather"], "weather")
    if weather is None:
        raise FileNotFoundError("weather_weekly.parquet is required.")

    spine_cols = ["kreis_id", "name", "nuts1_code", "bundesland_name", "year", "week", "week_start"]
    spine_cols = [c for c in spine_cols if c in weather.columns]
    spine = weather[spine_cols].drop_duplicates().reset_index(drop=True)
    print(f"Spine: {len(spine):,} rows ({spine['kreis_id'].nunique()} Kreise)")
    return spine


def join_kreis_level(spine: pd.DataFrame) -> pd.DataFrame:
    df = spine.copy()
    keys = ["kreis_id", "year", "week"]

    weather = safe_load(PATHS["weather"], "weather")
    if weather is not None:
        cols = [c for c in weather.columns if c not in ["name", "nuts1_code", "bundesland_name", "week_start"]]
        weather_slim = prefix_columns(weather[cols], "weather_", exclude=keys)
        assert_unique_columns(weather_slim, "weather")
        df = df.merge(weather_slim, on=keys, how="left")

    air_quality = safe_load(PATHS["air_quality"], "air_quality")
    if air_quality is not None:
        cols = [c for c in air_quality.columns if c not in ["name", "nuts1_code", "bundesland_name", "week_start"]]
        aq_slim = prefix_columns(air_quality[cols], "air_quality_", exclude=keys)
        assert_unique_columns(aq_slim, "air_quality")
        df = df.merge(aq_slim, on=keys, how="left")

    pollen = safe_load(PATHS["pollen"], "pollen")
    if pollen is not None:
        cols = [c for c in pollen.columns if c not in ["name", "nuts1_code", "bundesland_name", "week_start"]]
        pollen_slim = prefix_columns(pollen[cols], "pollen_", exclude=keys)
        pollen_slim = pollen_slim.rename(columns={"pollen_total_pollen_mean": "pollen_total_mean"})
        assert_unique_columns(pollen_slim, "pollen")
        df = df.merge(pollen_slim, on=keys, how="left")

    return df


def join_rki_incidence(df: pd.DataFrame) -> pd.DataFrame:
    """RKI incidence is keyed by kreis_name -- route through the crosswalk first."""
    incidence = safe_load(PATHS["rki_incidence"], "rki_incidence")
    crosswalk = safe_load(PATHS["kreis_crosswalk"], "kreis_crosswalk")
    if incidence is None or crosswalk is None:
        return df

    incidence = incidence.merge(crosswalk[["rki_name", "kreis_id"]],
                                left_on="kreis_name", right_on="rki_name", how="left")
    unmatched = incidence["kreis_id"].isna().sum()
    if unmatched > 0:
        names = incidence[incidence["kreis_id"].isna()]["kreis_name"].unique()
        print(f"WARNING: {unmatched} incidence rows unmatched to a kreis_id: {list(names)[:10]}")

    keys = ["kreis_id", "year", "week"]
    target_cols = [c for c in incidence.columns if c not in ["kreis_name", "rki_name", "week_start"]]
    incidence_slim = prefix_columns(incidence[target_cols], "survstat_", exclude=keys)
    assert_unique_columns(incidence_slim, "rki_incidence")
    return df.merge(incidence_slim, on=keys, how="left")


def join_bundesland_level(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["nuts1_code", "year", "week"]

    trends = safe_load(PATHS["trends"], "trends")
    if trends is not None:
        cols = [c for c in trends.columns if c.startswith("trends_")]
        trends_slim = trends[keys + cols].drop_duplicates(subset=keys)
        df = df.merge(trends_slim, on=keys, how="left")

    are = safe_load(PATHS["are"], "are")
    if are is not None:
        are_slim = (are[are["age_group"] == ARE_AGE_GROUP_FILTER][keys + ["are_incidence"]]
                    .drop_duplicates(subset=keys)
                    .rename(columns={"are_incidence": "are_konsultationsinzidenz"}))
        df = df.merge(are_slim, on=keys, how="left")

    grippeweb = safe_load(PATHS["grippeweb"], "grippeweb")
    if grippeweb is not None:
        gw_slim = grippeweb[
            (grippeweb["disease"] == GRIPPEWEB_DISEASE_FILTER) &
            (grippeweb["age_group"] == GRIPPEWEB_AGE_GROUP_FILTER)
        ][keys + ["grippeweb_incidence"]].drop_duplicates(subset=keys)
        df = df.merge(gw_slim, on=keys, how="left")

    amelag = safe_load(PATHS["amelag"], "amelag")
    if amelag is not None:
        cols = [c for c in amelag.columns if c.startswith("amelag_")]
        amelag_slim = amelag[keys + cols].drop_duplicates(subset=keys)
        df = df.merge(amelag_slim, on=keys, how="left")

    holidays = safe_load(PATHS["holidays"], "holidays")
    if holidays is not None:
        cols = [c for c in holidays.columns if c.startswith("holiday_")]
        holidays_slim = holidays[keys + cols].drop_duplicates(subset=keys)
        df = df.merge(holidays_slim, on=keys, how="left")

    return df


def join_national_level(df: pd.DataFrame) -> pd.DataFrame:
    notaufnahme = safe_load(PATHS["notaufnahme"], "notaufnahme")
    if notaufnahme is None:
        return df

    na_slim = notaufnahme[
        (notaufnahme["notaufnahmetyp"] == NOTAUFNAHME_TYPE_FILTER) &
        (notaufnahme["syndrome"] == NOTAUFNAHME_SYNDROME_FILTER) &
        (notaufnahme["age_group"] == NOTAUFNAHME_AGE_FILTER)
    ][["year", "week", "notaufnahme_share_raw"]].drop_duplicates(subset=["year", "week"])

    return df.merge(na_slim, on=["year", "week"], how="left")


if __name__ == "__main__":
    spine = build_spine()
    df = join_kreis_level(spine)
    df = join_rki_incidence(df)
    df = join_bundesland_level(df)
    df = join_national_level(df)
    assert_unique_columns(df, "final dataset")

    print(f"Final shape: {df.shape}")
    print(f"Kreise: {df['kreis_id'].nunique()}, "
          f"{df['year'].min()}-W{df['week'].min():02d} to {df['year'].max()}-W{df['week'].max():02d}")

    out_path = os.path.join(OUTPUT_DIR, "master_dataset.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

    coverage = (df.notna().sum() / len(df) * 100).round(1).sort_values()
    coverage.reset_index().rename(columns={"index": "column", 0: "pct_non_null"}).to_csv(
        os.path.join(OUTPUT_DIR, "master_dataset_coverage_report.csv"), index=False
    )

# Notes:
# - NaN in survstat_* columns means no reporting yet (COVID pre-2020,
#   RSV pre-2023), not zero cases -- don't fillna(0) here.
# - NaN elsewhere (pollen pre-2021, AMELAG pre-2022) is fine for
#   XGBoost/LightGBM, which handle missing values natively.

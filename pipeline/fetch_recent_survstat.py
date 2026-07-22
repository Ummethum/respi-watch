"""
RespiWatch -- Recent SurvStat fetcher via the official SOAP web service
================================================================================
Uses SurvStat's own SOAP API (zeep client) to pull the CURRENT year's
Kreis x week incidence data directly -- no manual export, no browser
automation. Based on a confirmed-working query template for Influenza;
generalised here to loop over all three project diseases and to handle
the year-boundary edge case (early January still needs the tail end of
the PREVIOUS year's weeks).

Output format matches a manual SurvStat TSV export

Usage:
    python fetch_recent_survstat.py
"""

import os
from datetime import datetime
import pandas as pd
from zeep import Client

# 1. CONFIGURATION

WSDL_URL = "https://tools.rki.de/SurvStat/SurvStatWebService.svc?wsdl"
OUTPUT_DIR = "./data/rki/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DISEASE_FILTERS = {
    "influenza": "[PathogenOut].[KategorieNz].[Krankheit DE].&[Influenza, saisonal]",
    "covid":     "[PathogenOut].[KategorieNz].[Krankheit DE].&[COVID-19]",              # PLACEHOLDER -- verify
    "rsv":       "[PathogenOut].[KategorieNz].[Krankheit DE].&[RSV (Meldepflicht gemäß IfSG)]",        # PLACEHOLDER -- verify
}

REFERENZ_DEFINITION_ID = 1

# How many weeks back "recent" needs to reach -- used only to decide
# whether the previous year also needs fetching (early January edge
# case), not to trim the output -- the full current year's weeks are
# always fetched and merged, letting the downstream merge logic handle
# overlap the same way the other fetch_recent_*.py scripts do.
LOOKBACK_WEEKS = 6

# 2. DETERMINE WHICH YEAR(S) TO FETCH

def years_to_fetch() -> list[int]:
    today = datetime.today()
    current_year = today.isocalendar()[0]
    current_week = today.isocalendar()[1]

    years = [current_year]
    if current_week <= LOOKBACK_WEEKS:
        # Early in the year -- the lookback window still reaches into
        # the tail end of the previous year's weeks (52/53)
        years.append(current_year - 1)
    return years

# 3. FETCH ONE (disease, year) COMBINATION

def fetch_disease_year(client: Client, disease_filter_value: str, year: int) -> pd.DataFrame:
    factory = client.type_factory("ns2")

    res = client.service.GetOlapData({
        "Language": "German",
        "Measures": {"Count": 0},
        "Cube": "SurvStat",
        "IncludeTotalColumn": False,
        "IncludeTotalRow": False,
        "IncludeNullRows": True,
        "IncludeNullColumns": True,
        "HierarchyFilters": factory.FilterCollection([
            {
                "Key": {
                    "DimensionId": "[PathogenOut].[KategorieNz]",
                    "HierarchyId": "[PathogenOut].[KategorieNz].[Krankheit DE]",
                },
                "Value": factory.FilterMemberCollection([disease_filter_value]),
            },
            {
                "Key": {
                    "DimensionId": "[ReferenzDefinition]",
                    "HierarchyId": "[ReferenzDefinition].[ID]",
                },
                "Value": factory.FilterMemberCollection(
                    [f"[ReferenzDefinition].[ID].&[{REFERENZ_DEFINITION_ID}]"]
                ),
            },
            {
                "Key": {
                    "DimensionId": "[ReportingDate]",
                    "HierarchyId": "[ReportingDate].[WeekYear]",
                },
                "Value": factory.FilterMemberCollection(
                    [f"[ReportingDate].[WeekYear].&[{year}]"]
                ),
            },
        ]),
        "RowHierarchy": "[DeutschlandNodes].[Kreise71Web].[CountyKey71]",
        "ColumnHierarchy": "[ReportingDate].[Week].[Week]",
    })

    columns = [c["Caption"] for c in res.Columns.QueryResultColumn]

    rows = []
    for r in res.QueryResults.QueryResultRow:
        rows.append(
            [r["Caption"]] +
            [int(v) if v is not None else None for v in r["Values"]["string"]]
        )

    df = pd.DataFrame(rows, columns=["Kreis"] + columns)
    return df

# 4. WRITE IN THE SAME TWO-HEADER-ROW STRUCTURE AS A MANUAL EXPORT

def write_survstat_tsv(df: pd.DataFrame, path: str):
    """
    Best-effort replica of the manual export structure described:
        Row 1: [empty]  01   02   03  ...   (week numbers)
        Row 2: Kreis    [blank per week column]
        Row 3+: <Kreis name>   <value>  <value>  ...

    Written as UTF-16 TSV, matching the historical manual-export
    encoding convention parse_rki_incidence.py was built around.
    """
    week_cols = [c for c in df.columns if c != "Kreis"]

    header_row_1 = [""] + list(week_cols)
    header_row_2 = ["Kreis"] + [""] * len(week_cols)

    with open(path, "w", encoding="utf-16", newline="") as f:
        f.write("\t".join(header_row_1) + "\n")
        f.write("\t".join(header_row_2) + "\n")
        for _, row in df.iterrows():
            values = [str(row["Kreis"])] + [
                "" if pd.isna(row[c]) else str(int(row[c])) for c in week_cols
            ]
            f.write("\t".join(values) + "\n")


# 5. RUN

if __name__ == "__main__":
    print(f"Connecting to SurvStat SOAP service...")
    client = Client(WSDL_URL)

    years = years_to_fetch()
    print(f"Fetching year(s): {years}\n")

    for disease, filter_value in DISEASE_FILTERS.items():
        for year in years:
            print(f"{disease} {year}...", end=" ", flush=True)
            try:
                df = fetch_disease_year(client, filter_value, year)
            except Exception as e:
                print(f"FAILED: {e}")
                continue

            n_non_null = df.drop(columns="Kreis").notna().sum().sum()
            print(f"{len(df)} rows, {n_non_null} non-null values")

            if n_non_null == 0:
                print(f"  \u26a0\ufe0f  ALL VALUES EMPTY for {disease} {year} -- this usually "
                      f"means DISEASE_FILTERS['{disease}'] doesn't match SurvStat's "
                      f"actual internal disease name. Verify before trusting this "
                      f"output (see the placeholder warning at the top of this file).")

            out_path = os.path.join(OUTPUT_DIR, f"survstat_{disease}_{year}.tsv")
            write_survstat_tsv(df, out_path)
            print(f"  \u2713 Saved: {out_path}")

    print(f"\nDone. Point parse_rki_incidence.py at these files (or wherever "
          f"it expects manual exports) to continue the existing pipeline "
          f"unchanged: parse_rki_incidence.py -> aggregate_berlin_boroughs.py "
          f"-> match_kreis_names.py.")

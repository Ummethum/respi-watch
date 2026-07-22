"""
Presentation graphic: three Germany maps side by side for a chosen
forecast -- predicted incidence, actual incidence, and their
difference. Uses the same demo predictions as generate_demo_snapshot.py
plus the FULL (untruncated) master data for the real actual values at
the target week, since the demo snapshot's own master data is cut off
right before that week by design.

Usage: python plot_prediction_vs_actual_maps.py
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

DEMO_CUTOFF_DATE = pd.Timestamp("2025-11-24")
HORIZON = 1   # which prediction (t+1 or t+2) to compare against actuals

DEMO_PREDICTIONS_PATH = "./data/demo/latest_predictions.parquet"
FULL_MASTER_PATH = "./data/master/master_dataset_filled.parquet"
SHAPEFILE_PATH = "./data/city_coords/NUTS5000_N3.shp"
SHAPEFILE_ID_FIELD = "NUTS_CODE"

DISEASE = "influenza"
OUTPUT_PATH = "prediction_vs_actual_maps.png"

# LOAD

predictions = pd.read_parquet(DEMO_PREDICTIONS_PATH)
predictions = predictions[
    (predictions["disease"] == DISEASE) & (predictions["horizon"] == HORIZON)
]
target_week = predictions["target_week"].iloc[0]
print(f"Target week: {target_week.date()} (cutoff {DEMO_CUTOFF_DATE.date()} + {HORIZON} week(s))")

master = pd.read_parquet(FULL_MASTER_PATH)
actual = master[master["week_start"] == target_week][["kreis_id", f"survstat_{DISEASE}"]]
actual = actual.rename(columns={f"survstat_{DISEASE}": "actual_incidence"})

merged = predictions[["kreis_id", "predicted_incidence"]].merge(actual, on="kreis_id", how="left")
merged["difference"] = (merged["predicted_incidence"] - merged["actual_incidence"]).abs()

n_missing_actual = merged["actual_incidence"].isna().sum()
if n_missing_actual > 0:
    print(f"WARNING: {n_missing_actual} Kreise have no actual value for "
          f"{target_week.date()} yet (reporting lag?) -- shown as gray on the maps.")

shapefile = gpd.read_file(SHAPEFILE_PATH)
if shapefile.crs is None:
    shapefile = shapefile.set_crs(epsg=25832)
shapefile = shapefile.to_crs(epsg=4326)

geo = shapefile.merge(merged, left_on=SHAPEFILE_ID_FIELD, right_on="kreis_id", how="left")

# PLOT

vmax = max(geo["predicted_incidence"].max(), geo["actual_incidence"].max())

fig, axes = plt.subplots(1, 3, figsize=(21, 8))

geo.plot(column="predicted_incidence", cmap="Reds", vmin=0, vmax=vmax,
        legend=True, ax=axes[0], edgecolor="white", linewidth=0.1,
        missing_kwds={"color": "lightgray"})
axes[0].set_title(f"Predicted incidence -- {target_week.date()}", fontsize=13)

geo.plot(column="actual_incidence", cmap="Reds", vmin=0, vmax=vmax,
        legend=True, ax=axes[1], edgecolor="white", linewidth=0.1,
        missing_kwds={"color": "lightgray"})
axes[1].set_title(f"Actual incidence -- {target_week.date()}", fontsize=13)

geo.plot(column="difference", cmap="Reds", vmin=0,
        legend=True, ax=axes[2], edgecolor="white", linewidth=0.1,
        missing_kwds={"color": "lightgray"})
axes[2].set_title("Absolute difference (predicted - actual)", fontsize=13)

for ax in axes:
    ax.axis("off")

fig.suptitle(f"RespiWatch -- Influenza forecast vs. actual "
            f"(base week {DEMO_CUTOFF_DATE.date()}, +{HORIZON} week)", fontsize=15)
fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved: {OUTPUT_PATH}")


mae = merged["difference"].mean()
print(f"\nMean absolute error across {merged['difference'].notna().sum()} Kreise: {mae:.2f}")

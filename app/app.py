"""
RespiWatch -- Streamlit dashboard
========================================
Interactive map + Kreis-level dashboard for respiratory disease
forecasts across all 400 German Kreise.

Data sources read (all produced by the weekly pipeline):
    - data/predictions/latest_predictions.parquet        (next week / in 2 weeks forecasts)
    - data/predictions/recent_avg_predictions.parquet     (average of the last 4 weeks' forecasts)
    - data/master/master_dataset_filled.parquet           (raw sources: SurvStat,
                                                             AMELAG, ARE, GrippeWeb,
                                                             Google Trends...)
    - data/city_coords/kreise_coords.csv                  (Kreis names + centroids,
                                                             only used for search)
    - data/city_coords/NUTS5000_N3.shp                    (REQUIRED -- true polygon
                                                             choropleth)

Run with:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 1. CONFIGURATION

def _get_secret(key: str) -> str | None:
    """
    Reads a secret from st.secrets 
    """
    try:
        return st.secrets.get(key)
    except Exception:
        return None


# hugging face repo
HF_REPO_ID = _get_secret("HF_REPO_ID")   # e.g. "yourname/respiwatch-data"
HF_TOKEN = _get_secret("HF_TOKEN")       # only needed if the dataset repo is private


SHAPEFILE_BASENAME = "NUTS5000_N3"
SHAPEFILE_HF_FOLDER = "city_coords"
SHAPEFILE_EXTENSIONS = [".shp", ".shx", ".dbf", ".prj", ".cpg"]
SHAPEFILE_ID_FIELD = "NUTS_CODE"

SELECTED_DISEASE = "influenza"   
DISEASE_LABELS = {"influenza": "Influenza", "covid": "COVID-19", "rsv": "RSV"}

# Plain-language labels for the two forecast horizons -- "t+1"/"t+2"
HORIZON_LABELS = {
    1: "Prediction for next week",
    2: "Prediction for in 2 weeks",
}

# Maps show the actual predicted incidence value directly, colour-coded
# on a fixed 0-50 scale (white/pale = low, dark red = high)
INCIDENCE_COLOR_MIN = 0
INCIDENCE_COLOR_MAX = 50

st.set_page_config(page_title="RespiWatch", layout="wide")

# 2. DATA LOADING (cached -- data only changes on the weekly run)

def _resolve_data_path(filename: str) -> str:
    """
    Downloads the given file fresh from the Hugging Face Dataset repo
    (cached by huggingface_hub itself between reruns).
    """
    if not HF_REPO_ID:
        st.error("HF_REPO_ID is not configured")
        st.stop()

    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=HF_REPO_ID, filename=filename,
        repo_type="dataset", token=HF_TOKEN,
    )


@st.cache_data(ttl=3600)
def load_forecast_predictions() -> pd.DataFrame:
    """Genuine future forecasts (next week / in 2 weeks), produced by
    generate_predictions.py."""
    path = _resolve_data_path("latest_predictions.parquet")
    df = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
    if not df.empty and "prediction_type" not in df.columns:
        df["prediction_type"] = "forecast"
    return df


@st.cache_data(ttl=3600)
def load_gap_fill_predictions() -> pd.DataFrame:
    """
    Nowcasts bridging the SurvStat reporting-lag gap, produced by
    generate_gap_fill_predictions.py. Unlike the other data files,
    this one is LEGITIMATELY ABSENT sometimes. generate_gap_fill_
    predictions.py only writes it when there's an actual gap to fill;
    in a week where SurvStat is fully caught up, there's nothing to
    nowcast and no file gets produced or uploaded. A 404 from Hugging
    Face here is therefore an expected, normal case, not an error.
    """
    try:
        path = _resolve_data_path("gap_fill_predictions.parquet")
    except Exception as e:
        if "EntryNotFound" in type(e).__name__ or "404" in str(e):
            return pd.DataFrame()
        raise

    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=3600)
def load_recent_avg_predictions() -> pd.DataFrame:
    """
    Average of the model's own last-4-weeks forecasts, produced by
    compute_recent_predictions_average.py.
    """
    try:
        path = _resolve_data_path("recent_avg_predictions.parquet")
    except Exception as e:
        if "EntryNotFound" in type(e).__name__ or "404" in str(e):
            return pd.DataFrame()
        raise

    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=3600)
def load_prediction_history() -> pd.DataFrame:
    """
    History of past next-week forecasts, one row per Kreis per week,
    produced by build_prediction_history.py -- what the model predicted
    for each past week, at the time it was predicted (not a backtest,
    the actual live forecasts as they were made). LEGITIMATELY ABSENT
    for a fresh deployment until enough weekly runs have accumulated
    at least one snapshot to build from.
    """
    try:
        path = _resolve_data_path("prediction_history.parquet")
    except Exception as e:
        if "EntryNotFound" in type(e).__name__ or "404" in str(e):
            return pd.DataFrame()
        raise

    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=3600)
def load_master() -> pd.DataFrame:
    path = _resolve_data_path("master_dataset_filled.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=3600)
def load_coords() -> pd.DataFrame:
    path = _resolve_data_path("city_coords/kreise_coords.csv")
    return pd.read_csv(path, dtype={"kreis_id": str})


@st.cache_resource(ttl=3600)
def load_shapefile():
    """
    Downloads all shapefile sidecar files (.shp/.shx/.dbf/.prj/.cpg)
    from the city_coords/ folder in the Hugging Face dataset repo --
    hf_hub_download's cache places all files from the same repo+
    revision under one shared local snapshot directory, so downloading
    each one explicitly (even though only the .shp path is used
    directly below) is what makes this work.
    """
    import geopandas as gpd

    shp_local_path = None
    for ext in SHAPEFILE_EXTENSIONS:
        filename = f"{SHAPEFILE_HF_FOLDER}/{SHAPEFILE_BASENAME}{ext}"
        local_path = _resolve_data_path(filename)
        if ext == ".shp":
            shp_local_path = local_path

    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    gdf = gpd.read_file(shp_local_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=25832)
    return gdf.to_crs(epsg=4326)

# 3. INCIDENCE TABLE

def get_last_known_actual(master_df: pd.DataFrame, kreis_id: str,
                          disease: str) -> tuple[float | None, pd.Timestamp | None]:
    """
    Returns the most recent NON-NULL actual value for this Kreis and
    disease, plus the date it's from. SurvStat's own reporting lag
    means the most recent 1-2 weeks are typically still incomplete/null
    by the time this app displays them.
    """
    col = f"survstat_{disease}"
    sub = master_df[(master_df["kreis_id"] == kreis_id) & master_df[col].notna()]
    if sub.empty:
        return None, None
    latest = sub.sort_values("week_start").iloc[-1]
    return latest[col], latest["week_start"]


def build_incidence_table(pred_slice: pd.DataFrame, master: pd.DataFrame,
                          disease: str) -> pd.DataFrame:
    """
    Builds one row per Kreis: predicted value and last known actual.
    Takes an ALREADY-FILTERED prediction slice (one row per Kreis,
    whatever that slice represents: next week's forecast, in-2-weeks forecast,
    or the recent-4-week average) so this single function serves all three
    maps without needing to know which one it's building.
    """
    if pred_slice.empty:
        return pd.DataFrame(columns=[
            "kreis_id", "predicted_incidence", "last_actual", "last_actual_week",
        ])

    rows = []
    for _, row in pred_slice.iterrows():
        last_actual, last_actual_date = get_last_known_actual(
            master, row["kreis_id"], disease
        )
        rows.append({
            "kreis_id": row["kreis_id"],
            "predicted_incidence": row["predicted_incidence"],
            "last_actual": last_actual,
            "last_actual_week": last_actual_date,
        })
    return pd.DataFrame(rows)

# 4. MAP

def render_map(incidence_df: pd.DataFrame, shapefile, coords: pd.DataFrame,
               map_key: str, selected_kreis_id: str = None) -> go.Figure:
    """`map_key` must be unique per map instance (Streamlit needs a
    distinct key for each of the 3 side-by-side maps' click events).

    Colors by the actual predicted incidence value on a fixed 0-50
    scale (not a dynamic per-map min/max) -- fixed so the three maps
    stay visually comparable to each other and to the same map in a
    future week, rather than each map silently rescaling itself.
    """
    geo_merged = shapefile.merge(
        incidence_df, left_on=SHAPEFILE_ID_FIELD, right_on="kreis_id", how="left"
    )
    geo_merged = geo_merged.merge(coords[["kreis_id", "name"]], on="kreis_id", how="left")
    geo_merged = geo_merged.rename(columns={"predicted_incidence": "Predicted incidence"})

    fig = px.choropleth_mapbox(
        geo_merged,
        geojson=geo_merged.geometry.__geo_interface__,
        locations=geo_merged.index,
        color="Predicted incidence",
        color_continuous_scale="Reds",
        range_color=(0, 50),
        hover_name="name",
        hover_data={"Predicted incidence": ":.1f"},
        custom_data=["kreis_id"],   # <- this is what makes click selection
                                     #    actually work: the event payload's
                                     #    "customdata" field will directly
                                     #    contain the kreis_id
        mapbox_style="carto-positron",
        center={"lat": 51.2, "lon": 10.4},
        zoom=4.3,
        opacity=0.8,
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=500,
                     coloraxis_colorbar=dict(title="Incidence"))

    if selected_kreis_id:
        highlight = geo_merged[geo_merged["kreis_id"] == selected_kreis_id].reset_index(drop=True)
        if not highlight.empty:
            fig.add_trace(go.Choroplethmapbox(
                geojson=highlight.geometry.__geo_interface__,
                locations=highlight.index,
                z=[1],
                showscale=False,
                marker=dict(line=dict(width=4, color="#1a1aff"), opacity=0.0),
                hoverinfo="skip",
            ))
    return fig


def render_map_with_summary(incidence_df: pd.DataFrame, shapefile, coords: pd.DataFrame,
                            title: str, help_text: str, map_key: str,
                            selected_kreis_id: str) -> str | None:
    """
    Renders one map with a plain-language title, a short numeric
    summary underneath, and returns the kreis_id if this map's click-
    selection changed.
    """
    st.markdown(f"**{title}**")
    st.caption(help_text)

    fig = render_map(incidence_df, shapefile, coords, map_key, selected_kreis_id)
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                           selection_mode="points", key=map_key)

    valid = incidence_df["predicted_incidence"].dropna()
    if not valid.empty:
        st.caption(f"Average: {valid.mean():.1f} -- lowest: {valid.min():.1f} -- "
                  f"highest: {valid.max():.1f}")

    if event and event.get("selection") and event["selection"].get("points"):
        point = event["selection"]["points"][0]
        clicked_kreis_id = point.get("customdata", [None])[0]
        if clicked_kreis_id:
            return clicked_kreis_id
    return None

# 5. KREIS DASHBOARD

def render_kreis_dashboard(kreis_id: str, disease: str,
                           forecast_predictions: pd.DataFrame,
                           gap_fill_predictions: pd.DataFrame,
                           prediction_history: pd.DataFrame,
                           master: pd.DataFrame, coords: pd.DataFrame):
    kreis_name = coords.loc[coords["kreis_id"] == kreis_id, "name"].values
    kreis_name = kreis_name[0] if len(kreis_name) else kreis_id

    st.header(f"{kreis_name}")

    # ---- Prediction summary ----
    st.subheader(f"Forecast -- {DISEASE_LABELS[disease]}")

    kreis_preds_all = forecast_predictions[
        (forecast_predictions["kreis_id"] == kreis_id)
        & (forecast_predictions["disease"] == disease)
    ].sort_values("target_week")

    # Chart line: forecast + gap-fill nowcasts combined, so the line is
    # continuous from right after the last known actual through to the
    # real forecast horizon.
    kreis_gap_fill = gap_fill_predictions[
        (gap_fill_predictions["kreis_id"] == kreis_id)
        & (gap_fill_predictions["disease"] == disease)
    ] if not gap_fill_predictions.empty else gap_fill_predictions

    kreis_preds_all = pd.concat(
        [kreis_gap_fill, kreis_preds_all], ignore_index=True
    ).sort_values("target_week")

    kreis_preds = kreis_preds_all[
        kreis_preds_all["prediction_type"] == "forecast"
    ].sort_values("horizon") if not kreis_preds_all.empty else kreis_preds_all

    if kreis_preds.empty:
        st.warning("No forecast available for this Kreis (missing data in the "
                  "most recent week -- check the weekly update log).")
    else:
        cols = st.columns(len(kreis_preds))
        for col, (_, row) in zip(cols, kreis_preds.iterrows()):
            label = HORIZON_LABELS.get(row["horizon"], f"Prediction (+{row['horizon']} weeks)")
            col.metric(
                label=f"{label} ({row['target_week'].strftime('%Y-%m-%d')})",
                value=f"{row['predicted_incidence']:.1f}",
            )

    # SurvStat: last known actual (with reporting-lag caveat)
    st.subheader("Most recently reported cases (official RKI data)")
    last_actual, last_actual_date = get_last_known_actual(master, kreis_id, disease)
    if last_actual is not None:
        weeks_old = (pd.Timestamp.today() - last_actual_date).days // 7
        st.metric(
            label=f"Week of {last_actual_date.strftime('%Y-%m-%d')} "
                  f"({weeks_old} week(s) old)",
            value=f"{last_actual:.1f}",
        )
        st.caption("Note: the most recently reported 1-2 weeks are often still "
                  "incomplete due to normal reporting delays -- the value shown "
                  "here is the newest one considered fully reported. The "
                  "forecast itself can be more up to date than this number, "
                  "since it also uses weather and search-trend data that "
                  "isn't affected by the same reporting delay.")
    else:
        st.info("No officially reported data available for this Kreis/disease.")

    # Two separate history charts: actual reported values, and what
    # the model has been predicting over roughly the same time window
    kreis_history = master[
        (master["kreis_id"] == kreis_id)
    ].sort_values("week_start").tail(26)   # last ~6 months

    st.subheader("History: model's next-week predictions")
    kreis_pred_history = prediction_history[
        prediction_history["kreis_id"] == kreis_id
    ].sort_values("target_week") if not prediction_history.empty else prediction_history

    if not kreis_pred_history.empty:
        st.caption("What the model predicts for each week, reconstructed fresh with "
                  "the current model -- the red segment marks the two upcoming weeks: "
                  "genuine forecasts, not yet confirmed by official data.")
        fig = go.Figure()

        # Historical portion: same style as the actual-incidence chart below,
        # so the two charts read as a matched pair.
        past = kreis_pred_history.iloc[:-2] if len(kreis_pred_history) > 2 else kreis_pred_history.iloc[:0]
        fig.add_trace(go.Scatter(
            x=past["target_week"], y=past["predicted_incidence"],
            mode="lines+markers", name="Predicted (past)",
            line=dict(color="#2c3e50"),
        ))

        # Future portion: exactly the last 2 weeks -- genuine forecasts,
        # highlighted in red. No overlap with the "past" trace above, so
        # there's a small visual break where the two segments meet rather
        # than a continuously-colored line.
        future = kreis_pred_history.iloc[-2:] if len(kreis_pred_history) >= 2 else kreis_pred_history
        fig.add_trace(go.Scatter(
            x=future["target_week"], y=future["predicted_incidence"],
            mode="lines+markers", name="Predicted (upcoming)",
            line=dict(color="#c44e52"),
        ))

        fig.update_layout(height=300, margin={"t": 20, "b": 20}, showlegend=False,
                         xaxis_title="Week", yaxis_title="Predicted incidence per 100,000 people")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No prediction history available -- check that "
               "build_prediction_history.py has run successfully.")

    st.subheader("History: officially reported incidence")
    if not kreis_history.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=kreis_history["week_start"], y=kreis_history[f"survstat_{disease}"],
            mode="lines+markers", name="Officially reported",
            line=dict(color="#2c3e50"),
            connectgaps=False,   # explicit: NaN weeks (incomplete reporting)
                                  # show as a genuine gap, never interpolated
        ))
        fig.update_layout(height=300, margin={"t": 20, "b": 20},
                         xaxis_title="Week", yaxis_title="Incidence per 100,000 people")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No reported history available for this Kreis.")

    # Other RKI sources (Bundesland-level, broadcast to this Kreis)
    st.subheader("Other official data sources (state-level, same for every Kreis in that state)")
    source_cols = {
        "Wastewater monitoring (AMELAG)": f"amelag_{disease}" if disease != "influenza" else "amelag_influenza",
        "Doctor visits for respiratory illness (ARE)": "are_konsultationsinzidenz",
        "GrippeWeb (self-reported symptoms survey)": "grippeweb_incidence",
    }

    tab_labels = list(source_cols.keys())
    tabs = st.tabs(tab_labels)
    for tab, (label, col_name) in zip(tabs, source_cols.items()):
        with tab:
            if col_name not in master.columns:
                st.info(f"{label}: data not available.")
                continue
            source_history = kreis_history[["week_start", col_name]].dropna()
            if source_history.empty:
                st.info(f"{label}: no data for this period/Kreis.")
            else:
                fig = px.line(source_history, x="week_start", y=col_name, markers=True,
                             labels={"week_start": "Week", col_name: label})
                fig.update_layout(height=250, margin={"t": 10, "b": 10})
                st.plotly_chart(fig, use_container_width=True)

    # ---- Google Trends ----
    st.subheader("Related Google searches (state-level)")
    trend_keywords = ["trends_grippe", "trends_influenza", "trends_grippeimpfung",
                     "trends_fieber", "trends_husten"]
    keyword_labels = {
        "trends_grippe": "Grippe (flu)", "trends_influenza": "Influenza",
        "trends_grippeimpfung": "Grippeimpfung (flu vaccine)",
        "trends_fieber": "Fieber (fever)", "trends_husten": "Husten (cough)",
    }
    available_trends = [c for c in trend_keywords if c in kreis_history.columns]
    if available_trends:
        trends_long = kreis_history.melt(
            id_vars="week_start", value_vars=available_trends,
            var_name="keyword", value_name="value",
        )
        trends_long["keyword"] = trends_long["keyword"].map(keyword_labels)
        fig = px.line(trends_long, x="week_start", y="value", color="keyword", markers=True,
                     labels={"week_start": "Week", "value": "Search interest (0-100 scale)",
                            "keyword": "Search term"})
        fig.update_layout(height=300, margin={"t": 10, "b": 10})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Google Trends data found.")

# 6. MAIN APP

def main():
    st.title("RespiWatch")
    st.caption("Forecasting respiratory disease incidence for all 400 German Kreise")

    forecast_predictions = load_forecast_predictions()
    recent_avg_predictions = load_recent_avg_predictions()
    gap_fill_predictions = load_gap_fill_predictions()
    prediction_history = load_prediction_history()
    master = load_master()
    coords = load_coords()

    if forecast_predictions.empty:
        st.error("No forecast data found -- has the weekly pipeline been run yet?")
        st.stop()

    try:
        shapefile = load_shapefile()
    except Exception as e:
        st.error(f"Could not load the map data from Hugging Face -- this app "
                f"deliberately only shows the true Kreis polygon map, no "
                f"fallback. Error: {e}")
        st.stop()

    disease = SELECTED_DISEASE

    # Sidebar: search
    with st.sidebar:
        st.header("Settings")
        st.caption(f"Disease shown: {DISEASE_LABELS[disease]}")

        st.divider()
        st.subheader("Search Kreis")
        kreis_options = dict(zip(coords["name"], coords["kreis_id"]))
        selected_name = st.selectbox(
            "Kreis", options=[""] + sorted(kreis_options.keys()),
            index=0, placeholder="Type a Kreis name...",
        )

    # Determine selected Kreis: search bar takes priority, else
    # whichever map was most recently clicked
    selected_kreis_id = None
    if selected_name:
        selected_kreis_id = kreis_options[selected_name]
        st.session_state["map_selected_kreis"] = selected_kreis_id
    elif "map_selected_kreis" in st.session_state:
        selected_kreis_id = st.session_state["map_selected_kreis"]

    # Build the three incidence tables (no trend categorisation)
    recent_avg_slice = recent_avg_predictions[
        recent_avg_predictions["disease"] == disease
    ] if not recent_avg_predictions.empty else recent_avg_predictions
    incidence_df_recent_avg = build_incidence_table(recent_avg_slice, master, disease)

    next_week_slice = forecast_predictions[
        (forecast_predictions["disease"] == disease)
        & (forecast_predictions["horizon"] == 1)
        & (forecast_predictions["prediction_type"] == "forecast")
    ]
    incidence_df_next_week = build_incidence_table(next_week_slice, master, disease)

    in_2_weeks_slice = forecast_predictions[
        (forecast_predictions["disease"] == disease)
        & (forecast_predictions["horizon"] == 2)
        & (forecast_predictions["prediction_type"] == "forecast")
    ]
    incidence_df_in_2_weeks = build_incidence_table(in_2_weeks_slice, master, disease)

    if incidence_df_next_week.empty and incidence_df_in_2_weeks.empty:
        st.warning("No forecast data available yet -- check that "
                  "generate_predictions.py has run successfully.")

    # Three maps, side by side
    st.subheader(f"Germany -- {DISEASE_LABELS[disease]} forecast")
    st.caption("Click a Kreis on any map to see its full details below. "
              "Color shows the predicted incidence per 100,000 people "
              "(white = 0, dark red = 50 or higher).")

    col1, col2, col3 = st.columns(3)

    with col1:
        clicked = render_map_with_summary(
            incidence_df_recent_avg, shapefile, coords,
            title="Average forecast (last 4 weeks)",
            help_text="Smoothed view of what the model recently predicted.",
            map_key="map_recent_avg", selected_kreis_id=selected_kreis_id,
        )
        if clicked:
            selected_kreis_id = clicked
            st.session_state["map_selected_kreis"] = clicked

    with col2:
        clicked = render_map_with_summary(
            incidence_df_next_week, shapefile, coords,
            title=HORIZON_LABELS[1],
            help_text="The model's forecast for the coming week.",
            map_key="map_next_week", selected_kreis_id=selected_kreis_id,
        )
        if clicked:
            selected_kreis_id = clicked
            st.session_state["map_selected_kreis"] = clicked

    with col3:
        clicked = render_map_with_summary(
            incidence_df_in_2_weeks, shapefile, coords,
            title=HORIZON_LABELS[2],
            help_text="The model's forecast for two weeks from now.",
            map_key="map_in_2_weeks", selected_kreis_id=selected_kreis_id,
        )
        if clicked:
            selected_kreis_id = clicked
            st.session_state["map_selected_kreis"] = clicked

    st.divider()

    # Kreis dashboard
    if selected_kreis_id:
        render_kreis_dashboard(selected_kreis_id, disease, forecast_predictions,
                              gap_fill_predictions, prediction_history, master, coords)
    else:
        st.info("Select a Kreis using the search box in the sidebar, or click "
               "any of the three maps above, to see details.")


if __name__ == "__main__":
    main()
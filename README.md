# RespiWatch

**Objective**

Given roughly 20 years of German influenza surveillance data combined with weather, pollen, and search-trend signals, the goal is to forecast weekly influenza incidence 1 and 2 weeks into the future for all ~400 German Kreise (NUTS-3 districts) individually, and serve those forecasts through a public dashboard.

Official surveillance data has a real reporting lag — the most recent 1-3 weeks are typically still incomplete by the time they're published. RespiWatch combines lagged official sources with real-time signals (weather, pollen, Google Trends) that aren't subject to that same delay, so a forecast can be more current than the raw official numbers it's built from.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.x-lightgrey)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)
![Prophet](https://img.shields.io/badge/Prophet-1.x-black)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![Optuna](https://img.shields.io/badge/Optuna-3.x-6f42c1)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Datasets-yellow)

---

## Repository Structure

| Folder | Contents |
|---|---|
| `pipeline/` | Weekly cron pipeline — downloading, parsing, joining all sources, generating and publishing forecasts |
| `modeling/` | XGBoost + Prophet training, hyperparameter tuning, evaluation, and standalone plotting scripts |
| `app/` | The Streamlit dashboard (`app.py`) |

---

## Data
 
**Data is not stored in this repository.** It lives across two separate Hugging Face Dataset repos, each with a different purpose and update cadence
 
| Repo | Purpose | Updated | Contents |
|---|---|---|---|
| **App data repo** | What the deployed Streamlit app reads at runtime | Automatically, every week (`push_data_to_hub.py`, the last step of `run_weekly_update.py`) | `latest_predictions.parquet`, `gap_fill_predictions.parquet`, `recent_avg_predictions.parquet`, `master_dataset_filled.parquet`, `city_coords/` (Kreis names + shapefile) |
| [**Pipeline data repo**](https://huggingface.co/datasets/HenningU/respiwatch-pipeline-data) | What `run_weekly_update.py` itself needs to run at all, on a fresh clone | Manually, only when a model is retrained or historical base data changes | Trained models (Prophet baselines, XGBoost residual models) + historical base data per source (weather, air quality, pollen, Google Trends, RKI incidence + Kreis-name crosswalk, ARE/GrippeWeb/Notaufnahme, AMELAG, holidays, Berlin population weights) |

To reproduce the pipeline from scratch: download the pipeline data repo's contents into the matching local `data/` paths first (so `fetch_recent_*.py` scripts have history to merge onto and `generate_predictions.py` has a model to load), then run `run_weekly_update.py`, which rebuilds the master dataset and forecasts locally and pushes the *app* repo's files automatically as its final step.

### Data sources & attribution

This project would not exist without the following open datasets. All are used under their respective licenses — attribution requirements are noted below

| Source | Provider | Granularity | License / citation |
|---|---|---|---|
| SurvStat@RKI 2.0 (Influenza, COVID-19, RSV incidence) | [Robert Koch-Institut](https://survstat.rki.de) | Kreis | Free to use with attribution. Suggested citation: *"Robert Koch-Institut: SurvStat@RKI 2.0, https://survstat.rki.de, query date: \<date\>"* |
| [AMELAG](https://github.com/robert-koch-institut/Abwassersurveillance_AMELAG) (wastewater surveillance) | Robert Koch-Institut | Bundesland | CC BY 4.0 |
| [ARE-Konsultationsinzidenz](https://github.com/robert-koch-institut/ARE-Konsultationsinzidenz) | Robert Koch-Institut | Bundesland / national | CC BY 4.0 |
| [GrippeWeb](https://github.com/robert-koch-institut/GrippeWeb_Daten_des_Wochenberichts) (self-reported symptom survey) | Robert Koch-Institut | Bundesland | CC BY 4.0 |
| [Notaufnahmesurveillance](https://github.com/robert-koch-institut/Daten_der_Notaufnahmesurveillance) | Robert Koch-Institut | National | CC BY 4.0 |
| Weather, air quality, pollen | [Open-Meteo](https://open-meteo.com) | Kreis | CC BY 4.0 |
| Search interest | [Google Trends](https://trends.google.com) | Bundesland | Public interface; no formal redistribution license — used here as an aggregated, non-identifying signal only |
| Kreis boundaries | [BKG VG5000](https://www.bkg.bund.de) shapefile | Kreis | Official German federal survey agency data |
| School holidays | [schulferien.org](https://www.schulferien.org) | Bundesland | Public reference site; no formal redistribution license found
| Public holidays | [feiertage-api.de](https://feiertage-api.de) | Bundesland | Free public API, no attribution requirement stated |

**Robert Koch-Institut (RKI)** is the source of the great majority of the epidemiological signal this project relies on — SurvStat, AMELAG, ARE, GrippeWeb, and Notaufnahmesurveillance are all RKI datasets. All RKI GitHub-hosted datasets used here are published under CC BY 4.0; SurvStat itself (accessed via its SOAP query interface, not a GitHub repo) requires citation as shown above rather than a formal open license. This project gratefully acknowledges RKI's ongoing publication of this data as open, machine-readable public health infrastructure.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env               # fill in HF_REPO_ID / HF_TOKEN for CLI scripts
```

For the app specifically, also create these secrets:
```toml
HF_REPO_ID = "yourname/respiwatch-data"
HF_TOKEN = "hf_your_token"
```

### Running the weekly pipeline

```bash
python pipeline/run_weekly_update.py
```

Always run scripts from the repository root (not from inside `pipeline/`) — every script's data paths are relative to the current working directory, and `data/` lives at the repo root. See `run_weekly_cron.sh` for scheduling this via cron / Windows Task Scheduler.


### Training / retraining models

```bash
python modeling/tune_xgboost_residual.py           # Optuna search
python modeling/fit_prophet_baseline.py            # per-Kreis seasonal baselines
python modeling/train_xgboost_residual.py          # residual-correction model
python modeling/evaluate_residual_model.py         # per-Kreis diagnostics
```

---

## Approach

### 1️⃣ Data Collection & Preparation

- Weekly automated ingestion: RKI sources (via SOAP API and GitHub-hosted open data), Open-Meteo (weather/air quality/pollen), Google Trends
- Berlin's 12 SurvStat boroughs are population-weighted into a single NUTS-3-consistent "Berlin" entry
- Coverage-aware zero-filling: a gap in the *middle* of a disease's active surveillance period means zero cases; a gap *before* that disease's surveillance existed at all is left as a true missing value, not zero

### 2️⃣ Feature Engineering

- All features lagged 1-3 weeks (no same-week information leakage into the target)

### 3️⃣ Modeling

- **Baseline**: Prophet, fit per-Kreis on the seasonal curve, log-transformed to prevent off-season overshoot
- **Residual correction**: XGBoost trained on the gap between Prophet's baseline and reality, using weather/trends/AMELAG/holiday features, deliberately *not* using raw recent SurvStat values as an input feature, since those are subject to the same reporting lag the whole project is trying to work around

### 4️⃣ Deployment

- Weekly cron pipeline rebuilds the dataset, regenerates forecasts, and pushes everything to a Hugging Face Dataset repo
- Streamlit dashboard reads exclusively from Hugging Face at runtime
---

## Results

Evaluated against a genuinely held-out final 15% (2023-2026) of the historical data, never used in training or hyperparameter tuning.

**What worked**
- The model reliably detects the *onset* of a seasonal wave. The direction and timing of a rise is caught even when the exact magnitude isn't.
- Log-transforming the Prophet baseline eliminated the off-season "phantom incidence" overshoot it was showing before

**What didn't**
- Tree-ensemble regression systematically underestimates the exact height of sharp peaks, a known limitation of this model family, only partially mitigated here, not solved

---

## Key Learnings

**Data correctness compounds silently**
- A single overly-aggressive name-normalization step (stripping `LK`/`SK` prefixes) silently merged 22 pairs of real, distinct Kreise (e.g. Stadt vs. Landkreis Heilbronn) into one across the *entire* multi-decade dataset — invisible until specifically diagnosed, since both "merged" values looked individually plausible
- A missing crosswalk file (`Grippeweb_Zuordnung_Regionen.tsv`, needed to map GrippeWeb's regions to Bundesland codes) failed *silently* with only a console warning, not an error — the pipeline kept running and producing data with an entire source quietly missing
- Lesson taken from both: prefer loud, explicit failure (`assert`-style checks) over silent gaps whenever a join or transformation could plausibly produce corrupted-but-plausible-looking output

**Regional performance isn't uniform**
- Model error is visibly higher in parts of eastern Germany. Smaller population counts mean noisier per-100,000 incidence figures independent of model quality, though this wasn't exhaustively separated from potential data-coverage differences (e.g. AMELAG station density) during this project

**Reporting lag has to be designed around, not patched over**
- Using raw recent SurvStat values as a model input looks appealing (recent case counts are informative!) but silently trains the model on an idealized, fully-reported version of data that doesn't exist yet at actual prediction time — a genuine train/serve skew risk, addressed here by deliberately excluding recent raw SurvStat lags from the feature set entirely

---

## Known Limitations

- Reported case counts (SurvStat, 2004–2023) are used as the training target, but they're an imperfect proxy for actual flu activity. Underreporting, changes in testing/reporting behavior over time, and healthcare-seeking behavior all shift the relationship between "true" incidence and what gets officially recorded, independent of anything the model can learn.
- Peak magnitude is systematically underestimated by the tree-ensemble residual model, direction and timing are more reliable than exact height
- The national/Bundesland-level sources (AMELAG, ARE, GrippeWeb, Notaufnahme) apply the same value to every Kreis within a state, they cannot capture intra-state variation
- Google Trends data is rescaled against overlapping historical weeks to correct for the API's own re-normalization behavior between calls, an approximation, not an exact calibration

---

## Author

**Henning Ummethum** · [LinkedIn](https://www.linkedin.com/in/henning-ummethum/) · [GitHub](https://github.com/Ummethum)
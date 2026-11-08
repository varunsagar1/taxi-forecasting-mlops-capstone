# Taxi Forecasting MLOps — BITS Pilani Digital AI Engineering Capstone

**Varun Arikatla, BITS ID 2026a1c100024 — Course C1-26_AIEOPC151, C1 AIEMLOps Capstone**
**Submitted Nov 8, 2026 (extended deadline)**

## Decision

Can we predict next-hour taxi pickup demand per zone accurately enough to
pre-position drivers, without systematically under-serving outer-borough zones?

The forecast target is pickups in the next hour, per zone, in New York City.

## Results — run Nov 8, 2026 (smoke scope)

- **Data:** NYC TLC yellow taxi, **January 2024 only, 2,964,624 trips**, 265 zones
  in the lookup; smoke scope = **top-20 zones, 11,436 zone-hours**
  (2024-01-07 23:00 to 2024-02-01 00:00), split train 7,997 / val 1,728 / test 1,711.
- **Best model: LightGBM** — val RMSE **31.00**, MAE **19.91**; test RMSE **31.45**,
  MAE **20.45**. XGBoost was effectively tied (val RMSE 31.00, MAE 19.81);
  Random Forest val RMSE 32.13, MAE 20.64.
- **MLflow:** experiment `taxi-demand-capstone`, run ID
  `e38677e286a64196b5eca00998c86197`.
- **SHAP top features:** lag_1 (63.69), hour (17.03), lag_24 (14.61),
  lag_168 (6.32), day-of-week (3.98). See `shap_summary.png` /
  `shap_importance.csv`.
- **Fairness (borough, test):** Manhattan RMSE 29.90, Queens 43.06 —
  gap **13.16**. Inverse-frequency mitigation retrain: gap **13.32 (+0.17)**,
  overall RMSE 31.44 — the mitigation did **not** reduce the gap; reported
  honestly in `fairness.json` and `MONITORING.md`. Only Manhattan and Queens
  appear in the top-20-zone test set.
- **Serving (verified Nov 8):** `GET /health` → `{"status": "ok", "model": "lgbm",
  "history_through": "2024-02-01 00:00:00"}`; `POST /predict`
  `{"zone_id": 161, "timestamp": "2024-01-15T18:00:00"}` →
  `{"predicted_pickups_next_hour": 303.8}`; unknown zone 9999 → 400.
  Docker was **not** verified — Docker is not installed on the machine used
  for this run; `Dockerfile` is included untested here.
- **Scope limitation:** this is the January-2024 smoke run, not a full-year
  run. Full-year (12 months, ~1.2 GB) was not attempted for the deadline.

## How to reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python 01_download.py --start 2024-01 --end 2024-01
.venv/bin/python 02_train.py --smoke
.venv/bin/uvicorn app:app --port 8000
# curl http://127.0.0.1:8000/health
# python batch_predict.py sample_inputs.csv
```

## Project structure

- `01_download.py` — download NYC TLC yellow-taxi data + zone lookup
- `02_train.py` — hourly demand features, train RF / XGBoost / LightGBM, MLflow logging, SHAP, fairness check
- `app.py` — FastAPI serving (`GET /health`, `POST /predict`)
- `batch_predict.py` — batch inference over the API
- `Dockerfile` — containerised API (included, not verified on the build machine)
- `.github/workflows/ci.yml` — GitHub Actions CI
- `metrics.json`, `fairness.json`, `shap_summary.png`, `shap_importance.csv` — run outputs
- `MONITORING.md` — monitoring and drift plan
- `serving/` — champion model (`best_model.joblib`), feature columns, history

## Data notes

- Source: NYC TLC trip record data, https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Monthly parquet files: `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet`
- Zone lookup: https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv (263 zones + Unknown/Outside-NYC)

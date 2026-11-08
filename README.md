# Taxi Forecasting MLOps — BITS Pilani Digital AI Engineering Capstone

## Results — run Nov 8, 2026 (smoke scope, submitted for the extended deadline)

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

---

# Capstone starter scaffold — Taxi Forecasting MLOps

Head-start kit for the **C1 AIEMLOps Capstone** (BITS Pilani Digital, course C1-26_AIEOPC151).
Submission due **Oct 25, 2026, 11:59 PM**. The assignment brief unlocks **Oct 18** —
a scheduled check will read it tomorrow morning and finalize the report plan.

## What this covers (and what it doesn't)

This scaffold handles the **brief-agnostic 80%**: data → features → models → MLflow →
SHAP → subgroup fairness check → FastAPI + Docker. It does **not** presume the brief's
exact deliverable wording. Two things stay provisional until the brief lands:

1. **Data slice** — defaults to NYC TLC **yellow taxi, all of 2024** (stable schema;
   2025+ added a `cbd_congestion_fee` column). If the brief names different months,
   rerun `01_download.py` with `--start/--end`.
2. **Fairness subgroups** — the scaffold evaluates per-**borough** error gaps
   (Manhattan vs outer boroughs) with a measurable mitigation. If the brief defines
   subgroups differently (e.g. hour-of-day, zone type), swap the grouping key in
   `02_train.py` — the machinery is identical.

You still do the real work: model iteration, the fairness analysis, and the
"durable report" itself. This just removes the plumbing.

## The 8-day arc

| When | Do |
|---|---|
| **Today (Oct 17)** | `01_download.py` + `02_train.py` → baseline numbers tonight, no brief needed |
| **Oct 18** | Brief unlocks → lock the 7-part report structure; adjust subgroup definition |
| **Oct 19–20** | Iterate models, SHAP deep-dive, measure the fairness mitigation |
| **Oct 21** | Daughter's birthday — light day, let MLflow runs sit |
| **Oct 22–23** | `app.py` + Docker, CI sketch, drift/monitoring section |
| **Oct 24** | Draft the durable report (map below) |
| **Oct 25** | Review, submit by 11:59 PM |

## How the scaffold maps to the 7-part durable report

Known required parts (from the course read-through; two more unlock with the brief):

| Report part | Comes from |
|---|---|
| **Decision** (the operational question) | `DECISION.md` sketch below — demand forecast → dispatch readiness |
| **Evidence** | `metrics.json` + per-borough RMSE table from `02_train.py` |
| **Source** | `data/` manifest printed by `01_download.py` (exact months, row counts) |
| **Scope** | Time window + zones covered, printed in the training log |
| **Model identity** | MLflow run ID + params logged for the chosen model |

**The operational thread** the report must trace: *service health → demand forecast →
SHAP context → subgroup evidence*. In scaffold terms: data-quality checks in step 1 →
hourly zone demand predictions → SHAP summary → per-borough error gaps.

## Setup (on your laptop)

```bash
python --version        # 3.10+ needed
pip install -r requirements.txt

python 01_download.py --start 2024-01 --end 2024-12   # ~1.2 GB, 12 parquet files
python 02_train.py                                    # trains RF/XGB/LGBM, logs to MLflow

mlflow ui                                             # browse runs at localhost:5000
```

Then serving:

```bash
uvicorn app:app --reload          # POST /predict {"zone_id": 161, "timestamp": "2024-12-01T18:00:00"}
docker build -t taxi-demand .     # containerized API
docker run -p 8000:8000 taxi-demand
```

## CI/CD sketch (GitHub Actions)

```yaml
# .github/workflows/ci.yml
name: capstone-ci
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: python -c "import app; print('import ok')"
      - run: docker build -t taxi-demand .
```

Extend with a retraining job once the brief defines the retrain trigger
(this is the Week 36 drift/monitoring story).

## Data notes (verified Oct 17, 2026)

- TLC trip record page is live: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Monthly parquet files live on the TLC CloudFront bucket; zone lookup CSV verified
  at https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv
  (263 zones + Unknown/Outside-NYC).
- Canonical file pattern used by `01_download.py`:
  `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet`
  (green: `green_tripdata_YYYY-MM.parquet`). If a download 404s, copy the exact
  link from the TLC page — the pattern has been stable for years.

## Suggested decision statement (edit to match the brief)

> *"Can we predict next-hour taxi pickup demand per zone accurately enough to
> pre-position drivers, without systematically under-serving outer-borough zones?"*

That one sentence carries the forecast task, the operational use, and the
fairness question — the three things the report must connect.

## Smoke-tested end-to-end (Oct 18)

Ran `01_download.py` + `02_train.py --smoke` (Jan 2024, top-20 zones) and the
FastAPI `/predict` endpoint — all green: download URLs live, RF/XGB/LGBM train
(best: LGBM), SHAP + fairness/mitigation write valid JSON/PNG, and
`POST /predict` returns sane forecasts (bad zone_id → 400). Two bugs fixed:

- `02_train.py` used `groupby(...).apply(..., include_groups=False)`, which only
  exists on pandas ≥ 2.2 — replaced with a version-portable per-group RMSE helper.
- `fairness.json` wrote numpy floats, which crash `json.dump` — converted to
  plain floats.

Note: MLflow logging was stubbed in the test (venv had no mlflow); the API calls
used are standard. On his laptop, `pip install -r requirements.txt` covers it.
Full-year run (all 265 zones, 12 months) is ~200x the smoke data — RF's 200 trees
will be the slow part; consider training the full comparison overnight, or
trimming to `--smoke` zones first and scaling up.

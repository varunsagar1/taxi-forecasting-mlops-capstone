# VIDEO SCRIPT — Taxi Forecasting MLOps Capstone
# BITS Pilani Digital, AI Engineering — Varun Arikatla, BITS ID 2026a1c100024
# Target: 11 minutes total. Hard limit: 12 minutes. Do NOT go over.
# Rules: webcam intro FIRST, full-screen capture, show repo + commit ID on
# screen, no secrets / passwords / tokens on screen at any point.
# Before recording: push the repo (see PUSH_COMMANDS.md), open the repo in
# the browser, open a terminal in the capstone-starter folder, start the API
# (uvicorn app:app --port 8000) in a second terminal tab, and have
# metrics.json, fairness.json, shap_summary.png and shap_importance.csv open.
# Record in ONE take if you can. If you stumble, pause, re-say the line, keep going.

## 0:00–0:30 — Webcam intro (Phase 0 — identity, DO THIS FIRST, camera on)
[On camera, full face visible, say exactly:]
"Hi, I am Varun Arikatla, BITS ID 2026a1c100024, BITS Pilani Digital,
AI Engineering programme. This is my capstone project: Taxi Forecasting
MLOps — predicting next-hour taxi pickup demand per zone in New York City,
using NYC Taxi and Limousine Commission yellow-taxi data from January 2024."
[Then switch to full-screen capture. Keep your webcam picture-in-picture on
if your recorder supports it; if not, the intro above is enough.]

## 0:30–1:30 — Phase 1: Decision and scope (1 min)
[On screen: README.md, Results section at the top.]
"The operational question is: can we predict next-hour pickup demand per
zone accurately enough to pre-position drivers, without systematically
under-serving outer-borough zones? That is the decision, the forecast task,
and the fairness question in one sentence.
Scope: yellow taxi, January 2024 only, top 20 zones by volume. This is a
smoke-scope run — I am stating that plainly. The data is 2,964,624 trips
in January 2024, from the NYC TLC public trip-record files, with a 265-zone
lookup. The forecast target is pickups in the next hour, per zone."

## 1:30–2:45 — Phase 2: Data and features (1 min 15 sec)
[On screen: terminal, run `ls data/` and show `metrics.json` scope block.
Then open 01_download.py briefly.]
"Phase 2 is data and features. 01_download.py pulls the January 2024
parquet file — 48 megabytes — plus the zone lookup, and prints a manifest:
2,964,624 trips, one file, 265 zones.
02_train.py aggregates those trips into hourly demand per zone — 11,436
zone-hours after features. Features are calendar — hour, day of week,
month, weekend, US holiday — plus lag 1 hour, lag 24 hours, lag 168 hours,
and a 24-hour rolling mean, plus a borough one-hot. The split is
chronological, 70 / 15 / 15 — train 7,997, validation 1,728, test 1,711 —
so the model never sees the future."

## 2:45–4:30 — Phase 3: Training and MLflow (1 min 45 sec)
[On screen: metrics.json, then `mlflow ui` in the browser if it loads,
otherwise the run ID in metrics.json.]
"Phase 3 is training. Three models, same features, same split: Random
Forest with 200 trees, XGBoost with 500 trees, and LightGBM with 500 trees.
All logged to MLflow, experiment taxi-demand-capstone.
Validation RMSE: Random Forest 32.13, XGBoost 31.00, LightGBM 31.00.
LightGBM wins on validation RMSE by a hair — 30.997 against XGBoost's
30.999 — so LightGBM is the champion. On the held-out test set it scores
RMSE 31.45, MAE 20.45. The MLflow run ID is
e38677e286a64196b5eca00998c86197 — that is the model identity, in
metrics.json, so anyone can find this exact run."

## 4:30–5:45 — Phase 4: Explainability — SHAP (1 min 15 sec)
[On screen: shap_summary.png, then shap_importance.csv.]
"Phase 4 is explainability. SHAP on 2,000 test rows. The top driver by far
is lag 1 hour — mean absolute SHAP 63.69. Next are hour of day at 17.03,
lag 24 hours at 14.61, lag 168 hours at 6.32, and day of week at 3.98.
In plain words: what happened in this zone one hour ago dominates the
forecast, then the time of day, then yesterday at the same hour. That is
what the SHAP summary plot shows, and the numbers are in
shap_importance.csv."

## 5:45–7:15 — Phase 5: Serving — FastAPI, batch, Docker, CI (1 min 30 sec)
[On screen: terminal — `curl http://127.0.0.1:8000/health`, then the
POST predict, then `python batch_predict.py sample_inputs.csv`.]
"Phase 5 is serving. app.py is a FastAPI service. GET /health returns
status ok, model lgbm, history through 2024-02-01.
POST /predict with zone 161, timestamp 2024-01-15T18:00:00 returns
303.8 predicted pickups for the next hour. An unknown zone, 9999,
returns a 400 error — input validation works.
Batch inference is a loop over the endpoint — batch_predict.py reads
sample_inputs.csv and writes batch_predictions.csv. Four rows:
zone 161 at 18:00 gives 303.8, zone 161 next morning at 08:00 gives 108.1,
zone 237 at 18:00 gives 235.0, zone 132 at noon gives 150.6.
Docker: a Dockerfile is in the repo, python 3.11 slim, uvicorn on port
8000. I am being honest here — Docker was not verified on the machine I
ran this on, because Docker is not installed there. The Dockerfile is
included, not claimed as tested. CI is a GitHub Actions workflow that
installs requirements and imports the app."

## 7:15–8:30 — Phase 6: Monitoring and drift (1 min 15 sec)
[On screen: MONITORING.md.]
"Phase 6 is monitoring and drift. This is a plan, not a deployed system —
there is no AWS deployment and no live alert in this repo, and I am not
claiming one. The plan in MONITORING.md is: track input distribution —
lag features, zero-demand share, new zone IDs — and prediction
distribution weekly; compute PSI on the key features; alert if PSI is
above 0.25 for two weeks, or if weekly RMSE rises more than 15 percent
above the 31.45 baseline for two weeks. The retrain trigger is: retrain
on the latest four weeks of TLC data when an alert fires, or monthly,
and only promote the new model if test RMSE improves, logged in the same
MLflow experiment."

## 8:30–10:00 — Phase 7: Fairness and Responsible AI (1 min 30 sec)
[On screen: fairness.json, then shap_summary.png again briefly.]
"Phase 7 is fairness and Responsible AI. Subgroup is borough, on the test
set. In the top-20 zones only Manhattan and Queens appear. Manhattan RMSE
29.90, Queens 43.06 — a gap of 13.16. Queens is the worst-served borough
here, and that matters for the dispatch question I started with.
The mitigation I tried was inverse-frequency sample weighting — upweight
low-volume zones — and retraining LightGBM. The result: gap 13.32,
up 0.17, overall RMSE 31.44, essentially unchanged. So the mitigation did
not work in this run, and I am reporting that, not hiding it. The honest
conclusion is that reweighting alone does not fix a borough gap driven by
volume and demand patterns, and the next step would be borough-aware
features or a separate calibration — that is future work, not a claim I
have already done it."

## 10:00–11:00 — Phase 8: Golden-path rerun and repo (1 min)
[On screen: browser — GitHub repo page, show the repo name, the commit ID,
and the file list: 01_download.py, 02_train.py, app.py, Dockerfile,
metrics.json, fairness.json, shap_summary.png, shap_importance.csv,
MONITORING.md, README.md.]
"Phase 8 is reproducibility — the golden path. From a clean checkout:
pip install -r requirements.txt; python 01_download.py --start 2024-01
--end 2024-01; python 02_train.py --smoke; uvicorn app:app --port 8000.
That reproduces every number I just showed.
[READ THE ACTUAL REPO URL AND COMMIT ID OFF THE SCREEN — do not guess them.]
This is the public repo at [SAY THE URL], commit [SAY THE COMMIT ID].
Thank you — that is the Taxi Forecasting MLOps capstone."
[Stop recording immediately. Check the video is 10–12 minutes, not over 12.]

## After recording — submit (see SUBMIT_CHECKLIST.md)
- Upload the mp4 and paste the repo link as a .txt link in Lumen, before
  12:29 PM CST today (11:59 PM IST assumed).
- Do a 30-second spot check of the video: your name + BITS ID are audible
  in the intro, the repo + commit ID are visible, no password/token on screen.

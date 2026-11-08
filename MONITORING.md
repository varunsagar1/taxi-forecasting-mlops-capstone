# Monitoring and drift plan

This is a plan, not a deployed system. There is no AWS deployment and no live
drift alert in this repo.

## What to monitor

- Input distribution: hourly pickups per zone, lag_1 / lag_24 means, share of
  zero-demand zone-hours, new/missing zone IDs.
- Prediction distribution: mean predicted pickups per zone per day, share of
  predictions above the 99th percentile of training predictions.
- Performance: weekly test-window RMSE/MAE on the most recent week, per
  borough, compared with the test baseline in `metrics.json`
  (LightGBM test RMSE 31.45, MAE 20.45).

## Drift check (weekly)

- Population Stability Index (PSI) on lag_1, lag_24, hour-of-day mix.
- Alert if PSI > 0.25 on any key feature for two consecutive weeks, or if
  weekly RMSE rises > 15% above baseline for two consecutive weeks.

## Retrain trigger

- Retrain on the latest 4 weeks of NYC TLC data when either alert above
  fires, or monthly, whichever is first. Re-run `01_download.py` +
  `02_train.py`, compare the new champion on the held-out week, and only
  promote it if test RMSE improves. Log the new run in MLflow under the
  same experiment (`taxi-demand-capstone`).

## Evidence

- The evidence for this submission is the set of artifacts in this repo:
  `shap_summary.png` + `shap_importance.csv` (SHAP),
  `fairness.json` (per-borough RMSE before/after mitigation), and
  `metrics.json` (model identity + scores). There is no live web dashboard.
- Fairness result: the inverse-frequency mitigation did **not** reduce the
  borough gap in this smoke run (gap 13.16 → 13.32, +0.17;
  overall RMSE 31.45 → 31.44). The gap is reported, not hidden.

"""Step 2: hourly zone demand forecasting -- features, RF/XGB/LGBM, MLflow, SHAP,
per-borough subgroup evaluation + a measurable fairness mitigation.

Usage:
    python 02_train.py                 # full run on ./data
    python 02_train.py --smoke          # top-20 zones only, quick iteration
    python 02_train.py --data ./data --taxi yellow

Outputs:
    metrics.json, fairness.json        # -> the "Evidence" part of the report
    shap_summary.png, shap_importance.csv
    serving/                           # model + history for app.py / Docker
    MLflow experiment "taxi-demand-capstone" (run `mlflow ui` to browse)
"""
import argparse
import glob
import json
import os

import holidays
import joblib
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb

PICKUP_COL = {"yellow": "tpep_pickup_datetime", "green": "lpep_pickup_datetime"}


def load_trips(data_dir, taxi):
    col = PICKUP_COL[taxi]
    files = sorted(glob.glob(os.path.join(data_dir, f"{taxi}_tripdata_*.parquet")))
    assert files, f"no parquet files in {data_dir} -- run 01_download.py first"
    df = pd.concat(
        (pd.read_parquet(f, columns=[col, "PULocationID"]) for f in files),
        ignore_index=True,
    )
    df[col] = pd.to_datetime(df[col])
    df = df.dropna(subset=[col, "PULocationID"])
    df = df[(df["PULocationID"] >= 1) & (df["PULocationID"] <= 265)]
    return df.rename(columns={col: "pickup_dt", "PULocationID": "zone_id"})


def build_demand(df):
    df["pickup_hour"] = df["pickup_dt"].dt.floor("h")
    demand = (
        df.groupby(["pickup_hour", "zone_id"])
        .size()
        .reset_index(name="pickups")
        .sort_values(["zone_id", "pickup_hour"])
        .reset_index(drop=True)
    )
    return demand


def add_features(demand, zone_lookup):
    us_holidays = holidays.US(years=sorted(demand["pickup_hour"].dt.year.unique()))
    d = demand.copy()
    d["hour"] = d["pickup_hour"].dt.hour
    d["dow"] = d["pickup_hour"].dt.dayofweek
    d["month"] = d["pickup_hour"].dt.month
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    d["is_holiday"] = d["pickup_hour"].dt.date.isin(us_holidays).astype(int)

    grp = d.groupby("zone_id")["pickups"]
    d["lag_1"] = grp.transform(lambda s: s.shift(1))
    d["lag_24"] = grp.transform(lambda s: s.shift(24))
    d["lag_168"] = grp.transform(lambda s: s.shift(168))
    d["roll_24"] = grp.transform(lambda s: s.shift(1).rolling(24, min_periods=1).mean())

    d = d.merge(zone_lookup[["LocationID", "Borough"]],
                left_on="zone_id", right_on="LocationID", how="left")
    d = pd.get_dummies(d, columns=["Borough"], prefix="boro", dummy_na=False)
    return d.dropna().reset_index(drop=True)


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--taxi", default="yellow", choices=["yellow", "green"])
    ap.add_argument("--smoke", action="store_true", help="top-20 zones only")
    args = ap.parse_args()

    print("Loading trips...")
    trips = load_trips(args.data, args.taxi)
    print(f"  {len(trips):,} trips")
    demand = build_demand(trips)
    print(f"  {len(demand):,} zone-hours before features")

    zones = pd.read_csv(os.path.join(args.data, "taxi_zone_lookup.csv"))
    if args.smoke:
        top = demand.groupby("zone_id")["pickups"].sum().nlargest(20).index
        demand = demand[demand["zone_id"].isin(top)]
        print("  smoke mode: top-20 zones only")

    data = add_features(demand, zones)
    feature_cols = [c for c in data.columns
                    if c not in ("pickup_hour", "pickups", "zone_id", "LocationID")]
    print(f"  {len(data):,} rows x {len(feature_cols)} features "
          f"({data['pickup_hour'].min()} .. {data['pickup_hour'].max()})")

    # --- chronological split: no peeking into the future ---
    times = np.sort(data["pickup_hour"].unique())
    t1, t2 = times[int(len(times) * 0.70)], times[int(len(times) * 0.85)]
    train = data[data["pickup_hour"] < t1]
    val = data[(data["pickup_hour"] >= t1) & (data["pickup_hour"] < t2)]
    test = data[data["pickup_hour"] >= t2]
    X_train, y_train = train[feature_cols], train["pickups"]
    X_val, y_val = val[feature_cols], val["pickups"]
    X_test, y_test = test[feature_cols], test["pickups"]
    print(f"  split: train {len(train):,} / val {len(val):,} / test {len(test):,}")

    models = {
        "rf": RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                   n_jobs=-1, random_state=42),
        "xgb": xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8,
                               subsample=0.8, tree_method="hist", random_state=42),
        "lgbm": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                 num_leaves=63, random_state=42, verbose=-1),
    }

    mlflow.set_experiment("taxi-demand-capstone")
    results = {}
    with mlflow.start_run(run_name="model-comparison") as run:
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            r = rmse(y_val, model.predict(X_val))
            m = float(mean_absolute_error(y_val, model.predict(X_val)))
            results[name] = {"val_rmse": r, "val_mae": m}
            mlflow.log_params({f"{name}_{k}": v for k, v in model.get_params().items()
                               if isinstance(v, (int, float, str, bool))})
            mlflow.log_metrics({f"{name}_val_rmse": r, f"{name}_val_mae": m})
            print(f"  {name}: val RMSE={r:.2f} MAE={m:.2f}")

    best_name = min(results, key=lambda n: results[n]["val_rmse"])
    best = models[best_name]
    test_rmse, test_mae = rmse(y_test, best.predict(X_test)), \
        float(mean_absolute_error(y_test, best.predict(X_test)))
    print(f"\nBest: {best_name}  test RMSE={test_rmse:.2f} MAE={test_mae:.2f}")

    metrics = {"best_model": best_name,
               "val": results,
               "test": {"rmse": test_rmse, "mae": test_mae},
               "mlflow_run_id": run.info.run_id,   # -> "Model identity" in the report
               "scope": {"taxi": args.taxi,
                         "from": str(data["pickup_hour"].min()),
                         "to": str(data["pickup_hour"].max()),
                         "zones": int(data["zone_id"].nunique())}}
    json.dump(metrics, open("metrics.json", "w"), indent=2)

    # --- SHAP: which features drive the forecast ("SHAP context" for the report) ---
    print("Computing SHAP values (sample of 2000 test rows)...")
    sample = X_test.sample(min(2000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(best)
    sv = explainer.shap_values(sample)
    shap.summary_plot(sv, sample, show=False, max_display=15)
    import matplotlib
    matplotlib.pyplot.savefig("shap_summary.png", bbox_inches="tight", dpi=120)
    matplotlib.pyplot.close()
    imp = pd.DataFrame({"feature": sample.columns,
                        "mean_abs_shap": np.abs(sv).mean(axis=0)}
                       ).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv("shap_importance.csv", index=False)
    print("  shap_summary.png, shap_importance.csv written")

    # --- subgroup fairness: per-borough error gaps (provisional grouping!) ---
    # If the brief defines subgroups differently, change this group key.
    print("Subgroup evaluation (per borough)...")
    test_df = test.copy()
    test_df["pred"] = best.predict(X_test)
    boro_cols = [c for c in feature_cols if c.startswith("boro_")]
    test_df["borough"] = test_df[boro_cols].idxmax(axis=1).str.replace("boro_", "")


    def per_group_rmse(df, pred_col):
        """Portable per-group RMSE (works on pandas < 2.2 and >= 2.2)."""
        sq = (df["pickups"].to_numpy() - df[pred_col].to_numpy()) ** 2
        return np.sqrt(pd.Series(sq, index=df["borough"].to_numpy())
                       .groupby(level=0).mean())

    per_boro = per_group_rmse(test_df, "pred")
    gap_before = float(per_boro.max() - per_boro.min())
    print(per_boro.round(2).to_string())
    print(f"  fairness gap (max-min borough RMSE): {gap_before:.2f}")

    # --- measurable mitigation: upweight low-volume zones, retrain, re-measure ---
    print("Mitigation: inverse-frequency sample weighting, retraining...")
    zone_freq = train["zone_id"].value_counts(normalize=True)
    w = 1.0 / np.sqrt(zone_freq.reindex(train["zone_id"]).values)
    mitigated = models[best_name].__class__(**best.get_params())
    mitigated.fit(X_train, y_train, sample_weight=w)  # RF, XGB and LGBM all accept this
    test_df["pred_m"] = mitigated.predict(X_test)
    per_boro_m = per_group_rmse(test_df, "pred_m")
    gap_after = float(per_boro_m.max() - per_boro_m.min())
    overall_after = rmse(y_test, test_df["pred_m"])
    print(per_boro_m.round(2).to_string())
    print(f"  gap after mitigation: {gap_after:.2f} "
          f"(delta {gap_after - gap_before:+.2f}); overall RMSE {overall_after:.2f}")

    def ser(s):  # plain floats: numpy scalars are not JSON-serializable
        return {k: float(v) for k, v in s.round(3).items()}

    fairness = {
        "subgroup": "borough (PROVISIONAL -- confirm against the Oct 18 brief)",
        "per_borough_rmse_before": ser(per_boro),
        "gap_before": round(gap_before, 3),
        "mitigation": "inverse-sqrt-frequency sample weighting, retrained best model",
        "per_borough_rmse_after": ser(per_boro_m),
        "gap_after": round(gap_after, 3),
        "gap_delta": round(gap_after - gap_before, 3),
        "overall_rmse_after": round(overall_after, 3),
    }
    json.dump(fairness, open("fairness.json", "w"), indent=2)

    # --- serving artifacts ---
    os.makedirs("serving", exist_ok=True)
    joblib.dump(best, "serving/best_model.joblib")
    json.dump({"features": feature_cols, "model": best_name,
               "pickup_col": PICKUP_COL[args.taxi]},
              open("serving/feature_columns.json", "w"))
    demand.to_parquet("serving/history.parquet")
    print("\nDone. Serving artifacts in ./serving/. Run `mlflow ui` to browse the run.")


if __name__ == "__main__":
    main()

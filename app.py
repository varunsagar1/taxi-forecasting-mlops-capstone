"""Step 3: serve the demand model. Run artifacts from 02_train.py must exist in ./serving/.

    uvicorn app:app --reload
    POST /predict  {"zone_id": 161, "timestamp": "2024-12-15T18:00:00"}
"""
import json
from datetime import datetime

import holidays
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(title="taxi-demand")

MODEL = joblib.load("serving/best_model.joblib")
META = json.load(open("serving/feature_columns.json"))
FEATURES = META["features"]
HISTORY = pd.read_parquet("serving/history.parquet")  # pickup_hour, zone_id, pickups
ZONES = pd.read_csv("data/taxi_zone_lookup.csv")[["LocationID", "Borough"]]
US_HOLIDAYS = holidays.US()


def build_features(zone_id: int, ts: datetime) -> pd.DataFrame:
    hour = ts.replace(minute=0, second=0, microsecond=0)
    z = ZONES[ZONES["LocationID"] == zone_id]
    if z.empty:
        raise HTTPException(400, f"unknown zone_id {zone_id}")
    borough = z.iloc[0]["Borough"]

    hist = HISTORY[HISTORY["zone_id"] == zone_id].set_index("pickup_hour")["pickups"]
    if hist.empty:
        raise HTTPException(400, f"no history for zone_id {zone_id}")

    def at(h):
        try:
            return float(hist.loc[h])
        except KeyError:
            return float(hist.mean())  # cold start fallback

    row = {
        "hour": hour.hour,
        "dow": hour.weekday(),
        "month": hour.month,
        "is_weekend": int(hour.weekday() >= 5),
        "is_holiday": int(hour.date() in US_HOLIDAYS),
        "lag_1": at(hour - pd.Timedelta(hours=1)),
        "lag_24": at(hour - pd.Timedelta(hours=24)),
        "lag_168": at(hour - pd.Timedelta(hours=168)),
        "roll_24": float(hist[hist.index < hour].tail(24).mean() or hist.mean()),
    }
    for col in FEATURES:
        if col.startswith("boro_"):
            row[col] = int(col == f"boro_{borough}")
    return pd.DataFrame([row])[FEATURES]


@app.get("/health")
def health():
    return {"status": "ok", "model": META["model"],
            "history_through": str(HISTORY["pickup_hour"].max())}


@app.post("/predict")
def predict(body: dict):
    try:
        zone_id = int(body["zone_id"])
        ts = datetime.fromisoformat(body["timestamp"])
    except (KeyError, ValueError, TypeError):
        raise HTTPException(400, 'body must be {"zone_id": int, "timestamp": ISO-8601}')
    X = build_features(zone_id, ts)
    pred = float(MODEL.predict(X)[0])
    return {"zone_id": zone_id, "timestamp": body["timestamp"],
            "predicted_pickups_next_hour": round(pred, 1)}

"""Batch inference: loop a CSV of zone_id,timestamp rows through POST /predict.

Usage (with the API running on localhost:8000):
    python batch_predict.py sample_inputs.csv
    # writes batch_predictions.csv
"""
import csv
import sys

import requests

API = "http://127.0.0.1:8000/predict"


def main(path):
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        resp = requests.post(API, json={"zone_id": int(r["zone_id"]),
                                        "timestamp": r["timestamp"]}, timeout=30)
        resp.raise_for_status()
        out.append(resp.json())
        print(out[-1])
    with open("batch_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["zone_id", "timestamp",
                                          "predicted_pickups_next_hour"])
        w.writeheader()
        w.writerows(out)
    print(f"Wrote batch_predictions.csv ({len(out)} rows)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sample_inputs.csv")

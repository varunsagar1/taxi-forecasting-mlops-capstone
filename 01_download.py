"""Step 1: download NYC TLC trip data + zone lookup.

Usage:
    python 01_download.py --start 2024-01 --end 2024-12
    python 01_download.py --start 2024-07 --end 2024-12 --taxi green --out ./data

Defaults to yellow taxi (the course's dataset). 2024 is the default window
because its schema is stable -- 2025+ files add a cbd_congestion_fee column.
Each yellow-taxi month is ~100-200 MB; a full year is ~1.2 GB.
"""
import argparse
import os
import sys

import requests

BASE = "https://d37ci6vzurychx.cloudfront.net"
ZONE_LOOKUP_URL = f"{BASE}/misc/taxi_zone_lookup.csv"


def download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.exists(dest):
        print(f"  skip (exists): {dest}")
        return
    print(f"  GET {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        if r.status_code == 404:
            raise SystemExit(
                f"404 for {url}\n"
                "The file pattern may have changed -- copy the exact link from "
                "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
            )
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r    {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="")
        print()


def month_range(start: str, end: str):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01", help="YYYY-MM, default 2024-01")
    ap.add_argument("--end", default="2024-12", help="YYYY-MM, default 2024-12")
    ap.add_argument("--taxi", default="yellow", choices=["yellow", "green"])
    ap.add_argument("--out", default="./data")
    args = ap.parse_args()

    months = list(month_range(args.start, args.end))
    print(f"Downloading {args.taxi} taxi data for {len(months)} month(s) -> {args.out}")
    for ym in months:
        url = f"{BASE}/trip-data/{args.taxi}_tripdata_{ym}.parquet"
        download(url, os.path.join(args.out, f"{args.taxi}_tripdata_{ym}.parquet"))

    print("Downloading zone lookup...")
    download(ZONE_LOOKUP_URL, os.path.join(args.out, "taxi_zone_lookup.csv"))

    # --- manifest: the "Source" part of the durable report ---
    import pandas as pd

    files = sorted(f for f in os.listdir(args.out) if f.endswith(".parquet"))
    rows = 0
    print("\nManifest (paste into report -> Source):")
    for f in files:
        n = pd.read_parquet(os.path.join(args.out, f), columns=["tpep_pickup_datetime"]).shape[0] \
            if args.taxi == "yellow" else \
            pd.read_parquet(os.path.join(args.out, f), columns=["lpep_pickup_datetime"]).shape[0]
        rows += n
        print(f"  {f}: {n:,} trips")
    print(f"  TOTAL: {rows:,} trips across {len(files)} files")
    print(f"  Zone lookup: {pd.read_csv(os.path.join(args.out, 'taxi_zone_lookup.csv')).shape[0]} zones")


if __name__ == "__main__":
    main()

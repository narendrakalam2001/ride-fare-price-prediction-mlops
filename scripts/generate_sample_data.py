# ============================================================
# GENERATE SAMPLE DATA — Ride Fare Price Prediction ML System
# ============================================================
# IMPORTANT — READ THIS:
#
# The real dataset for this project is Kaggle's "New York City Taxi
# Fare Prediction" (~55M rows):
#   https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction
#
# This training/serving sandbox has no network access to kaggle.com,
# so the real CSV could not be downloaded here. This script generates
# a REALISTIC SYNTHETIC PROXY with the exact same schema
# (pickup_datetime, pickup/dropoff lat-lon, passenger_count, fare_amount)
# so that the full pipeline (EDA → training → champion/challenger →
# API → dashboard → simulation) can be run and demonstrated end-to-end.
#
# TO USE THE REAL DATA:
#   1. Download train.csv from the Kaggle competition page above
#      (requires a free Kaggle account + accepting competition rules).
#   2. Place it at data/train.csv (or point RIDE_FARE_DATA_PATH to it).
#   3. For the full 55M rows, train with sample_frac (e.g. 0.05) or
#      nrows to fit memory — see src/data_loader.load_raw_data().
#   4. Re-run scripts/train_model.py. No other code changes needed.
#
# The synthetic fare formula below approximates the real NYC taxi
# fare structure: base fare + per-km rate + per-minute rate (via a
# distance-derived duration proxy) + rush-hour/airport surcharges +
# realistic noise — close enough to sanity-check the whole pipeline,
# but metrics from this data are NOT a substitute for training on the
# real competition data before treating this as a portfolio result.
# ============================================================

import numpy as np
import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import haversine_km
from src.config import JFK_COORDS, LGA_COORDS, EWR_COORDS, NYC_BOUNDS

RNG = np.random.RandomState(42)

MANHATTAN_CENTER = (40.7580, -73.9855)


def _random_nyc_point(rng, center=MANHATTAN_CENTER, spread=0.06):
    lat = np.clip(rng.normal(center[0], spread), NYC_BOUNDS["min_lat"], NYC_BOUNDS["max_lat"])
    lon = np.clip(rng.normal(center[1], spread), NYC_BOUNDS["min_lon"], NYC_BOUNDS["max_lon"])
    return lat, lon


def generate_synthetic_taxi_data(n_rows: int = 60_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    pickup_lat, pickup_lon   = zip(*[_random_nyc_point(rng) for _ in range(n_rows)])
    dropoff_lat, dropoff_lon = zip(*[_random_nyc_point(rng) for _ in range(n_rows)])

    pickup_lat, pickup_lon   = np.array(pickup_lat),  np.array(pickup_lon)
    dropoff_lat, dropoff_lon = np.array(dropoff_lat), np.array(dropoff_lon)

    # ── Random pickup timestamps over a 3-year window ─────────
    start = pd.Timestamp("2013-01-01", tz="UTC")
    end   = pd.Timestamp("2016-06-30", tz="UTC")
    seconds_range = int((end - start).total_seconds())
    pickup_datetime = start + pd.to_timedelta(
        rng.randint(0, seconds_range, size=n_rows), unit="s"
    )

    passenger_count = rng.choice([1, 1, 1, 2, 2, 3, 4, 5, 6], size=n_rows)

    # ── Distance-driven fare formula (approximates NYC taxi meter) ─
    distance_km = haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    distance_km = np.clip(distance_km, 0.2, 60)

    hour = pd.DatetimeIndex(pickup_datetime).hour
    dow  = pd.DatetimeIndex(pickup_datetime).dayofweek

    is_rush   = np.isin(hour, [7, 8, 9, 16, 17, 18, 19]).astype(float)
    is_night  = np.isin(hour, [23, 0, 1, 2, 3, 4]).astype(float)
    is_weekend= (dow >= 5).astype(float)

    near_jfk = (haversine_km(pickup_lat, pickup_lon, *JFK_COORDS) < 2) | \
               (haversine_km(dropoff_lat, dropoff_lon, *JFK_COORDS) < 2)
    near_lga = (haversine_km(pickup_lat, pickup_lon, *LGA_COORDS) < 2) | \
               (haversine_km(dropoff_lat, dropoff_lon, *LGA_COORDS) < 2)
    near_ewr = (haversine_km(pickup_lat, pickup_lon, *EWR_COORDS) < 2) | \
               (haversine_km(dropoff_lat, dropoff_lon, *EWR_COORDS) < 2)
    airport  = (near_jfk | near_lga | near_ewr).astype(float)

    base_fare      = 2.50
    per_km_rate    = 2.20 + 0.6 * is_rush            # rush hour costs more per km (congestion)
    duration_proxy_min = distance_km / (28 - 10 * is_rush) * 60   # slower avg speed in rush hour
    per_min_rate   = 0.35

    fare = (
        base_fare
        + per_km_rate * distance_km
        + per_min_rate * duration_proxy_min
        + 1.0 * is_night
        + 4.5 * airport
        + rng.normal(0, 1.8, size=n_rows)          # measurement / traffic noise
    )
    fare = np.clip(fare, 2.5, 250)

    df = pd.DataFrame({
        "fare_amount":        np.round(fare, 2),
        "pickup_datetime":    pickup_datetime,
        "pickup_longitude":   pickup_lon,
        "pickup_latitude":    pickup_lat,
        "dropoff_longitude":  dropoff_lon,
        "dropoff_latitude":   dropoff_lat,
        "passenger_count":    passenger_count,
    })

    return df


if __name__ == "__main__":
    N_ROWS = int(os.getenv("N_SAMPLE_ROWS", "60000"))
    df = generate_synthetic_taxi_data(n_rows=N_ROWS)
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    out_path = os.path.join(PROJECT_ROOT, "data", "train.csv")
    df.to_csv(out_path, index=False)
    print(f"Synthetic sample data written → {out_path}  |  shape={df.shape}")
    print(df["fare_amount"].describe())

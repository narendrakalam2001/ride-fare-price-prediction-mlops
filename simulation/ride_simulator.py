# ============================================================
# RIDE SIMULATOR — Ride Fare Price Prediction ML System
# ============================================================

import requests
import random
import time
import os
from datetime import datetime, timedelta

API_URL = os.getenv("RIDE_FARE_API_URL", "http://127.0.0.1:8000") + "/predict"

# ── Reference NYC coordinates ─────────────────────────────────
MANHATTAN   = (40.7580, -73.9855)
JFK_AIRPORT = (40.6413, -73.7781)
LGA_AIRPORT = (40.7769, -73.8740)
BROOKLYN    = (40.6782, -73.9442)


def _jitter(coord, spread=0.03):
    lat, lon = coord
    return lat + random.uniform(-spread, spread), lon + random.uniform(-spread, spread)


# ============================================================
# GENERATE SYNTHETIC RIDE REQUEST
# ============================================================

def generate_ride(scenario: str = "random") -> dict:
    """
    Scenarios:
        random       — mixed realistic short/medium trips within Manhattan
        airport_run  — trip to/from JFK or LGA (long distance, flat-fare zone)
        rush_hour    — weekday evening rush-hour cross-borough trip
    """
    now = datetime.utcnow()

    if scenario == "airport_run":
        pickup  = _jitter(MANHATTAN, 0.02)
        dropoff = random.choice([JFK_AIRPORT, LGA_AIRPORT])
        dt      = now.replace(hour=random.choice([7, 14, 20]))

    elif scenario == "rush_hour":
        pickup  = _jitter(MANHATTAN, 0.02)
        dropoff = _jitter(BROOKLYN, 0.02)
        dt      = now.replace(hour=random.choice([8, 17, 18, 19]))

    else:  # random
        pickup  = _jitter(MANHATTAN, 0.03)
        dropoff = _jitter(MANHATTAN, 0.03)
        dt      = now - timedelta(hours=random.randint(0, 72))

    return {
        "pickup_datetime":   dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pickup_latitude":   round(pickup[0], 6),
        "pickup_longitude":  round(pickup[1], 6),
        "dropoff_latitude":  round(dropoff[0], 6),
        "dropoff_longitude": round(dropoff[1], 6),
        "passenger_count":   random.randint(1, 4),
    }


# ============================================================
# SEND TO API + PRINT RESULT
# ============================================================

def send_ride(ride: dict, idx: int):
    try:
        response = requests.post(API_URL, json=ride, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"[{idx+1}]  passengers={ride['passenger_count']}  "
                  f"→  fare=${result['predicted_fare_usd']}  "
                  f"band={result['fare_band']}  flag={result.get('pricing_flag')}")
        else:
            print(f"[{idx+1}] API error: {response.status_code}")
    except Exception as e:
        print(f"[{idx+1}] Connection error: {e}")


# ============================================================
# RUN SIMULATION
# ============================================================

def simulate_rides(n: int = 20, scenario: str = "random"):
    print(f"\nSimulating {n} ride requests  |  scenario={scenario}\n" + "-" * 60)
    for i in range(n):
        ride = generate_ride(scenario)
        send_ride(ride, i)
        time.sleep(0.5)
    print("-" * 60 + "\nSimulation complete")


if __name__ == "__main__":
    simulate_rides(20, scenario="random")

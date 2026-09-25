# ============================================================
# PRICING ENGINE — Ride Fare Price Prediction ML System
# ============================================================
# Mobility-platform-grade decision layer on top of the raw model
# prediction, mirroring how Uber/Ola/Rapido-style pricing services
# never expose a raw regression output directly:
#
#   1. Hard business rules (override ML)      — sanity clamps
#   2. ML predicted fare + fare band           — surge / driver payout tier
#   3. Simple surge multiplier for rush-hour / airport trips
# ============================================================

import pandas as pd
import logging

from src.config import FARE_BANDS, MIN_FARE, MAX_FARE

logger = logging.getLogger(__name__)


# ============================================================
# FARE BAND — predicted $ → LOW / MEDIUM / HIGH / PREMIUM
# ============================================================

def get_fare_band(fare: float) -> str:
    for band, (low, high) in FARE_BANDS.items():
        if low <= fare < high:
            return band
    return "PREMIUM"


# ============================================================
# PRICING ENGINE — row-level decisions (batch, e.g. training eval)
# ============================================================

def pricing_engine(trip_df: pd.DataFrame, predicted_fares) -> list:
    """
    For each trip row → returns a clamped, business-safe fare + flag.

    Rule priority:
      1. predicted fare < MIN_FARE               → clamp to MIN_FARE (hard rule)
      2. predicted fare > MAX_FARE                → clamp to MAX_FARE (hard rule)
      3. airport_trip == 1                        → flag AIRPORT_FLAT_CANDIDATE
      4. is_rush_hour == 1 and distance > 5km      → flag SURGE_ELIGIBLE
      5. else                                     → NORMAL
    """
    decisions = []

    for idx, (_, row) in enumerate(trip_df.iterrows()):
        fare = float(predicted_fares[idx])
        fare = max(MIN_FARE, min(MAX_FARE, fare))

        if row.get("airport_trip", 0) == 1:
            flag = "AIRPORT_FLAT_CANDIDATE"
        elif row.get("is_rush_hour", 0) == 1 and row.get("trip_distance_km", 0) > 5:
            flag = "SURGE_ELIGIBLE"
        else:
            flag = "NORMAL"

        decisions.append(flag)

    return decisions


# ============================================================
# FARE SCORING — single trip (for API)
# ============================================================

def score_trip(row: dict, predicted_fare: float, lower_bound: float = None,
               upper_bound: float = None) -> dict:
    """
    Returns structured pricing output for a single trip.
    Used by FastAPI prediction endpoint.
    """
    clamped_fare = max(MIN_FARE, min(MAX_FARE, float(predicted_fare)))
    band         = get_fare_band(clamped_fare)

    flag = None
    if row.get("airport_trip", 0) == 1:
        flag = "AIRPORT_FLAT_CANDIDATE"
    elif row.get("is_rush_hour", 0) == 1 and row.get("trip_distance_km", 0) > 5:
        flag = "SURGE_ELIGIBLE"

    result = {
        "predicted_fare_usd": round(clamped_fare, 2),
        "fare_band":          band,
        "pricing_flag":       flag,
    }

    if lower_bound is not None and upper_bound is not None:
        result["prediction_interval_90"] = [round(float(lower_bound), 2),
                                             round(float(upper_bound), 2)]

    return result

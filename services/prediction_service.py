# ============================================================
# PREDICTION SERVICE — Ride Fare Price Prediction ML System
# ============================================================

import pandas as pd
import logging

from src.pricing_engine import score_trip
from src.data_loader     import add_engineered_features
from src.evaluation      import predict_with_interval

logger = logging.getLogger(__name__)


# ============================================================
# PREPARE FEATURES — for API inference
# ============================================================

def prepare_features(input_data: dict) -> pd.DataFrame:
    """
    Takes raw API input dict → returns engineered feature DataFrame.
    Mirrors the training pipeline feature engineering exactly.
    """
    df = pd.DataFrame([input_data])
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], utc=True)
    df = add_engineered_features(df)
    return df


# ============================================================
# PREDICT — single trip
# ============================================================

def predict_trip(model, calibration, input_data: dict) -> dict:
    """
    Full prediction flow for one trip:
      1. Feature engineering
      2. Model point prediction (+ calibrated 90% interval if available)
      3. Pricing engine scoring (business rules + ML)
    """
    df = prepare_features(input_data)
    df = df.drop(columns=["pickup_datetime"], errors="ignore")

    try:
        point_pred, lower, upper = predict_with_interval(model, calibration, df)
        fare = float(point_pred[0])
        lo   = float(lower[0]) if lower is not None else None
        hi   = float(upper[0]) if upper is not None else None
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        fare, lo, hi = 15.0, None, None

    row = df.iloc[0].to_dict()
    result = score_trip(row, fare, lo, hi)

    logger.info(
        "Prediction  |  fare=$%.2f  band=%s  flag=%s",
        result["predicted_fare_usd"], result["fare_band"], result["pricing_flag"]
    )

    return result

# ============================================================
# RIDE FARE PREDICTION API — FastAPI Serving
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import logging
import time
import os
import json

from src.model_loader             import load_latest_model
from services.prediction_service  import predict_trip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ride Fare Price Prediction API")

# ── Load model on startup ─────────────────────────────────────
try:
    model, calibration = load_latest_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error("Model loading failed: %s", e)
    model       = None
    calibration = None


# ============================================================
# INPUT SCHEMA
# ============================================================

class TripInput(BaseModel):
    pickup_datetime:    str   = Field(..., description="ISO8601, e.g. 2016-06-15T18:30:00Z")
    pickup_longitude:   float
    pickup_latitude:    float
    dropoff_longitude:  float
    dropoff_latitude:   float
    passenger_count:    int   = 1


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Ride Fare Price Prediction API is live 🚕",
        "docs":    "/docs",
        "health":  "/health"
    }


@app.get("/health")
def health():
    return {"status": "running", "model_loaded": model is not None}


@app.get("/model_info")
def model_info():
    registry_path = "fare_models/latest_model.json"
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            return json.load(f)
    return {"error": "Model registry not found"}


@app.post("/predict")
def predict(trip: TripInput):

    start      = time.time()
    input_data = trip.dict()

    result = predict_trip(model, calibration, input_data)

    log_record = {
        "timestamp":         time.time(),
        "pickup_datetime":   input_data["pickup_datetime"],
        "passenger_count":   input_data["passenger_count"],
        "predicted_fare_usd":result["predicted_fare_usd"],
        "fare_band":         result["fare_band"],
        "pricing_flag":      result.get("pricing_flag"),
    }

    log_path = "logs/prediction_logs.csv"
    os.makedirs("logs", exist_ok=True)

    log_df = pd.DataFrame([log_record])
    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)

    result["latency_seconds"] = round(time.time() - start, 4)

    return result

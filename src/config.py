# ============================================================
# CONFIGURATION — Ride Fare Price Prediction ML System
# ============================================================

import os

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE = 42
# Overridable — on lower-RAM machines, N_JOBS=-1 spawns one worker
# process per core, each holding its own copy of the training fold in
# memory, which can exhaust RAM on large datasets (e.g. 4 cores x ~1GB
# fold copy = 4GB+ just for data, on top of the model itself). Example
# (PowerShell): $env:RIDE_FARE_N_JOBS = "2"
N_JOBS = int(os.getenv("RIDE_FARE_N_JOBS", "-1"))

# ── Cross-validation ─────────────────────────────────────────
# Overridable via env vars — useful on lower-spec machines (e.g. 4 cores /
# 8GB RAM) where the default 20 iters x 5 folds x 13 models is very slow.
# Example (PowerShell): $env:RIDE_FARE_SEARCH_ITERS = "5"
CV_FOLDS             = int(os.getenv("RIDE_FARE_CV_FOLDS", "5"))
RANDOM_SEARCH_ITERS  = int(os.getenv("RIDE_FARE_SEARCH_ITERS", "20"))

# ── Feature selection ────────────────────────────────────────
SELECT_K = 12

# ── Outlier clipping ─────────────────────────────────────────
CLIP_FOLD = 1.5

# ── Feature engineering thresholds ───────────────────────────
ORDINAL_UNIQUE_THRESHOLD = 10

# ── NYC geo bounding box (sanity filtering) ──────────────────
# Slightly padded around NYC city limits — drops GPS noise / bad rows
NYC_BOUNDS = {
    "min_lat": 40.40, "max_lat": 41.20,
    "min_lon": -74.60, "max_lon": -72.80,
}

# ── Fare sanity bounds (used in data validation + business rules) ───
MIN_FARE = 2.50     # NYC taxi minimum base fare
MAX_FARE = 300.00    # anything above this is treated as a data error
MAX_PASSENGERS = 6
MAX_TRIP_DISTANCE_KM = 100.0

# ── Airport coordinates (for engineered "near-airport" features) ────
JFK_COORDS = (40.6413, -73.7781)
LGA_COORDS = (40.7769, -73.8740)
EWR_COORDS = (40.6895, -74.1745)
AIRPORT_RADIUS_KM = 2.0

# ── Fare bands (predicted $ → tier, used by pricing/business engine) ─
FARE_BANDS = {
    "LOW":    (0,   10),
    "MEDIUM": (10,  30),
    "HIGH":   (30,  70),
    "PREMIUM":(70,  10_000),
}

# ── PSI drift thresholds ──────────────────────────────────────
PSI_MODERATE         = 0.10    # PSI >= 0.10 → moderate drift, monitor
PSI_HIGH             = 0.20    # PSI >= 0.20 → critical drift, retrain

# ── Large-file training safety ────────────────────────────────
# The full Kaggle NYC taxi fare CSV is ~55M rows. Loading + feature-
# engineering + train_test_split on the *entire* file at once needs
# well north of 16GB free RAM on most desktops and will OOM otherwise
# (this is exactly what happens without sampling). If the caller didn't
# explicitly pass nrows/sample_frac to run_training(), the pipeline
# auto-applies DEFAULT_SAMPLE_FRAC_FOR_LARGE_FILES once the raw row
# count exceeds LARGE_FILE_ROW_THRESHOLD. Override with the
# RIDE_FARE_FORCE_FULL=1 environment variable if you have enough RAM
# (or are running on a bigger machine) and want the entire file.
LARGE_FILE_ROW_THRESHOLD            = 5_000_000
DEFAULT_SAMPLE_FRAC_FOR_LARGE_FILES = 0.05   # ~2.8M rows from the 55M-row file

# ── Monitoring alert thresholds ───────────────────────────────
RESIDUAL_MEAN_ALERT   = 2.0     # avg |predicted - actual proxy| alert ($)
MAPE_ALERT_THRESHOLD  = 0.25    # 25% MAPE triggers alert

# ── Challenger promotion gates (regression) ───────────────────
MIN_RMSE_IMPROVEMENT   = 0.02     # challenger RMSE must be >= 2% better
MIN_R2_THRESHOLD       = 0.80     # challenger must clear this R2 floor
MAX_GENERALIZATION_GAP = 0.15     # |train_R2 - test_R2| must stay below this

# ── Cost-sensitive business impact (dynamic-pricing platform) ────────
# Under-pricing loses platform revenue directly; over-pricing loses
# customer trust / conversion (modelled as a smaller opportunity cost).
UNDER_PRICE_COST_PER_DOLLAR = 1.00   # $ lost per $ under-predicted
OVER_PRICE_COST_PER_DOLLAR  = 0.35   # $ opportunity cost per $ over-predicted

# ── Paths ────────────────────────────────────────────────────
MODEL_DIR   = "fare_models"
METRICS_LOG = "fare_models/metrics_log.csv"
TESTS_DIR   = "tests"
SERVING_DIR = "serving"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs",    exist_ok=True)
os.makedirs("data",    exist_ok=True)
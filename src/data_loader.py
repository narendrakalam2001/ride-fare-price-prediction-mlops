# ============================================================
# DATA LOADER + FEATURE ENGINEERING — Ride Fare Price Prediction
# ============================================================
# Source dataset: Kaggle "New York City Taxi Fare Prediction"
# (kaggle.com/competitions/new-york-city-taxi-fare-prediction)
# Raw columns: key, fare_amount, pickup_datetime, pickup_longitude,
#              pickup_latitude, dropoff_longitude, dropoff_latitude,
#              passenger_count
# ============================================================

import numpy as np
import pandas as pd
import logging

from src.config import (
    NYC_BOUNDS, MIN_FARE, MAX_FARE, MAX_PASSENGERS, MAX_TRIP_DISTANCE_KM,
    JFK_COORDS, LGA_COORDS, EWR_COORDS, AIRPORT_RADIUS_KM,
)

logger = logging.getLogger(__name__)

REQUIRED_COLS = [
    "fare_amount", "pickup_datetime",
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "passenger_count",
]

TARGET_COL = "fare_amount"


# ============================================================
# CHUNKED / SAMPLED LOADING (dataset is 55M rows in the wild)
# ============================================================

def load_raw_data(path: str, nrows: int = None, sample_frac: float = None,
                   random_state: int = 42) -> pd.DataFrame:
    """
    Loads the raw NYC taxi fare CSV.

    The full Kaggle dataset has ~55M rows and will not fit comfortably
    in memory on a laptop / free-tier environment. Two strategies are
    supported, matching how this is handled in production:

      1. nrows        — read only the first N rows (fast, deterministic)
      2. sample_frac  — read in chunks and keep a random fraction of
                         rows across the *whole* file (better statistical
                         coverage, slower)

    If both are None, the full file is read (only recommended once you
    have chunked/parallel infra, e.g. Dask/Spark, wired in).
    """
    dtypes = {
        "fare_amount":       "float32",
        "pickup_longitude":  "float32",
        "pickup_latitude":   "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude":  "float32",
        "passenger_count":   "float32",
    }

    if sample_frac is not None:
        logger.info("Loading with chunked random sampling  |  frac=%.4f", sample_frac)
        chunks = []
        rng = np.random.RandomState(random_state)
        for chunk in pd.read_csv(path, dtype=dtypes, parse_dates=["pickup_datetime"],
                                  chunksize=1_000_000):
            mask = rng.rand(len(chunk)) < sample_frac
            chunks.append(chunk[mask])
        df = pd.concat(chunks, ignore_index=True)

    elif nrows is not None:
        logger.info("Loading first %d rows", nrows)
        df = pd.read_csv(path, dtype=dtypes, parse_dates=["pickup_datetime"], nrows=nrows)

    else:
        logger.info("Loading full file (no sampling) — ensure sufficient memory")
        df = pd.read_csv(path, dtype=dtypes, parse_dates=["pickup_datetime"])

    logger.info("Raw data loaded  |  shape=%s", df.shape)
    return df


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:

    # ── Normalize column names ────────────────────────────────
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
        .str.lower()
    )

    # ── Drop irrelevant ID columns ────────────────────────────
    for col in ("key", "unnamed_0", "id"):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            logger.info("Dropped column: %s", col)

    # ── Check required columns ───────────────────────────────
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Null handling ─────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    if len(df) < before:
        logger.info("Dropped %d rows with nulls in required columns", before - len(df))

    # ── Target sanity filtering ───────────────────────────────
    df = df[(df[TARGET_COL] >= MIN_FARE) & (df[TARGET_COL] <= MAX_FARE)]

    # ── Passenger count sanity ────────────────────────────────
    df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= MAX_PASSENGERS)]

    # ── GPS bounding box filter (drops (0,0) / out-of-NYC noise) ─
    df = df[
        df["pickup_latitude"].between(NYC_BOUNDS["min_lat"], NYC_BOUNDS["max_lat"]) &
        df["dropoff_latitude"].between(NYC_BOUNDS["min_lat"], NYC_BOUNDS["max_lat"]) &
        df["pickup_longitude"].between(NYC_BOUNDS["min_lon"], NYC_BOUNDS["max_lon"]) &
        df["dropoff_longitude"].between(NYC_BOUNDS["min_lon"], NYC_BOUNDS["max_lon"])
    ]

    # ── Drop identical pickup==dropoff (zero-distance) trips ──
    same_point = (
        (df["pickup_latitude"]  == df["dropoff_latitude"]) &
        (df["pickup_longitude"] == df["dropoff_longitude"])
    )
    if same_point.sum():
        logger.info("Dropping %d zero-distance (pickup==dropoff) trips", same_point.sum())
        df = df[~same_point]

    # ── Minimum size check ────────────────────────────────────
    if df.shape[0] < 500:
        raise ValueError("Dataset too small for training after cleaning (< 500 rows)")

    # ── Deduplication ─────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(ignore_index=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate rows", dropped)

    logger.info("Data validation passed  |  shape=%s  |  fare_mean=%.2f  fare_median=%.2f",
                df.shape, df[TARGET_COL].mean(), df[TARGET_COL].median())

    return df.reset_index(drop=True)


# ============================================================
# GEO HELPERS
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in kilometers."""
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _min_dist_to_point(lat, lon, point):
    return haversine_km(lat, lon, point[0], point[1])


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Industry-standard feature engineering for taxi/ride fare regression.
    Every feature maps to a real driver of trip price used by Uber/Ola/
    Rapido-style dynamic pricing engines: distance, duration proxy,
    time-of-day demand, and airport/fixed-fare routes.
    """
    df = df.copy()

    # ── Core trip distance ────────────────────────────────────
    df["trip_distance_km"] = haversine_km(
        df["pickup_latitude"],  df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )

    # Manhattan-style (grid) distance — useful proxy for city-block routing
    df["manhattan_distance_km"] = (
        haversine_km(df["pickup_latitude"], df["pickup_longitude"],
                     df["dropoff_latitude"], df["pickup_longitude"]) +
        haversine_km(df["dropoff_latitude"], df["pickup_longitude"],
                     df["dropoff_latitude"], df["dropoff_longitude"])
    )

    # Bearing / direction of travel (degrees)
    lat1, lon1 = np.radians(df["pickup_latitude"]), np.radians(df["pickup_longitude"])
    lat2, lon2 = np.radians(df["dropoff_latitude"]), np.radians(df["dropoff_longitude"])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    df["bearing_deg"] = (np.degrees(np.arctan2(x, y)) + 360) % 360

    # Drop physically-impossible distances (data errors)
    df = df[df["trip_distance_km"] <= MAX_TRIP_DISTANCE_KM]

    # ── Datetime features ──────────────────────────────────────
    dt = pd.to_datetime(df["pickup_datetime"], utc=True).dt.tz_convert("America/New_York")
    df["pickup_hour"]      = dt.dt.hour
    df["pickup_dow"]       = dt.dt.dayofweek           # 0=Mon
    df["pickup_month"]     = dt.dt.month
    df["pickup_year"]      = dt.dt.year
    df["is_weekend"]       = (df["pickup_dow"] >= 5).astype(int)
    df["is_rush_hour"]     = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df["is_late_night"]    = df["pickup_hour"].isin([23, 0, 1, 2, 3, 4]).astype(int)

    # Cyclical encodings — avoids "23:00 far from 00:00" discontinuity
    df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["pickup_dow"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["pickup_dow"] / 7)

    # ── Airport proximity flags (flat-fare routes in NYC) ──────
    for name, coords in (("jfk", JFK_COORDS), ("lga", LGA_COORDS), ("ewr", EWR_COORDS)):
        pickup_d  = _min_dist_to_point(df["pickup_latitude"],  df["pickup_longitude"],  coords)
        dropoff_d = _min_dist_to_point(df["dropoff_latitude"], df["dropoff_longitude"], coords)
        df[f"near_{name}"] = (
            (pickup_d <= AIRPORT_RADIUS_KM) | (dropoff_d <= AIRPORT_RADIUS_KM)
        ).astype(int)

    df["airport_trip"] = (
        df["near_jfk"] | df["near_lga"] | df["near_ewr"]
    ).astype(int)

    # ── Distance from Manhattan center (proxy for CBD congestion) ─
    MANHATTAN_CENTER = (40.7580, -73.9855)
    df["pickup_dist_from_center_km"] = _min_dist_to_point(
        df["pickup_latitude"], df["pickup_longitude"], MANHATTAN_CENTER
    )

    # ── Interaction features ───────────────────────────────────
    df["distance_x_rush_hour"]   = df["trip_distance_km"] * df["is_rush_hour"]
    df["distance_per_passenger"] = df["trip_distance_km"] / df["passenger_count"].replace(0, 1)

    if len(df) >= 1000:
        logger.info("Feature engineering done  |  total columns=%d", df.shape[1])

    df = optimize_dtypes(df)

    return df


# ============================================================
# MEMORY OPTIMIZATION
# ============================================================

def optimize_dtypes(df: pd.DataFrame, min_rows_to_log: int = 1000) -> pd.DataFrame:
    """
    Downcasts float64 -> float32 and int64 -> the smallest safe int dtype
    for every numeric column except the target and datetime columns.
    On the full 55M-row Kaggle file this roughly halves the memory
    footprint of the engineered dataframe, which matters a lot at
    train_test_split time.

    min_rows_to_log: below this row count, the dtype casts still happen
    (harmless either way) but the "before -> after MB" log line is
    skipped. This function runs on every single API prediction request
    too (via add_engineered_features -> prepare_features), where a
    single-row DataFrame's memory footprint is meaningless to report —
    without this, every prediction logged a "0.0 MB -> 0.0 MB" line,
    which was just noise crowding out the actual prediction log line.
    """
    before_mb = df.memory_usage(deep=True).sum() / 1e6

    skip = {TARGET_COL, "pickup_datetime"}

    for col in df.columns:
        if col in skip:
            continue
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")

    if len(df) >= min_rows_to_log:
        after_mb = df.memory_usage(deep=True).sum() / 1e6
        logger.info("Dtype optimization  |  %.1f MB -> %.1f MB (-%.0f%%)",
                    before_mb, after_mb, 100 * (1 - after_mb / max(before_mb, 1e-9)))

    return df


# ============================================================
# FEATURE TYPE DETECTION
# ============================================================

def detect_feature_types(df: pd.DataFrame, threshold: int = 10):
    """
    Auto-detect: ordinal / continuous / binary columns.
    Excludes target column and raw datetime/id-like columns that are
    not meant to be fed directly into the preprocessor.
    """
    EXCLUDE = {
        TARGET_COL, "pickup_datetime", "pickup_year",
    }

    ordinal_cols    = []
    continuous_cols = []
    binary_cols     = []

    for col in df.columns:
        if col in EXCLUDE:
            continue

        dtype_name = df[col].dtype.name
        n_unique   = df[col].nunique(dropna=False)

        if dtype_name in ("object", "category", "bool"):
            ordinal_cols.append(col)

        elif np.issubdtype(df[col].dtype, np.number):
            if n_unique == 2:
                binary_cols.append(col)
            elif 3 <= n_unique <= threshold:
                ordinal_cols.append(col)
            else:
                continuous_cols.append(col)
        else:
            ordinal_cols.append(col)

    logger.info("Feature types  |  ordinal=%d  continuous=%d  binary=%d",
                len(ordinal_cols), len(continuous_cols), len(binary_cols))

    return ordinal_cols, continuous_cols, binary_cols
# ============================================================
# PREPROCESSING — Ride Fare Price Prediction ML System
# ============================================================

import numpy as np
import pandas as pd
import logging

from typing import List, Tuple

from scipy import stats as scipy_stats

from sklearn.base             import BaseEstimator, TransformerMixin, clone
from sklearn.compose          import ColumnTransformer
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler, OrdinalEncoder

from src.config import CLIP_FOLD, SELECT_K

logger = logging.getLogger(__name__)


# ============================================================
# Columns that must NEVER go through Yeo-Johnson power transform,
# even if their sample skew crosses the threshold. Raw GPS
# coordinates and angles are bounded, roughly-Gaussian-ish values —
# not the kind of long-tailed data PowerTransformer is meant for.
# Feeding them in can make scipy's lambda-optimization bracket search
# fail outright (BracketError) or silently overflow on some samples.
# They are still Clipped + StandardScaled like any other continuous
# feature — just never power-transformed.
# ============================================================
NEVER_POWER_TRANSFORM_COLS = {
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "bearing_deg",
}


# ============================================================
# CLIPPER — IQR-based outlier clipping transformer
# ============================================================

class Clipper(BaseEstimator, TransformerMixin):
    """
    Clips values to [Q1 - fold*IQR, Q3 + fold*IQR].
    Fitted on train set, applied to train + test.
    Prevents outlier leakage through scaling.

    get_feature_names_out() implemented so sklearn ColumnTransformer
    can extract clean feature names (fixes f0, f1... warning).
    """

    def __init__(self, fold: float = CLIP_FOLD):
        self.fold = fold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        q1  = np.quantile(X, 0.25, axis=0)
        q3  = np.quantile(X, 0.75, axis=0)
        iqr = q3 - q1

        self.lower_ = q1 - self.fold * iqr
        self.upper_ = q3 + self.fold * iqr

        # Degenerate case: IQR == 0 (very common on zero-inflated engineered
        # columns, e.g. distance_x_rush_hour is 0 for every non-rush trip).
        # The old eps-based fallback collapsed the ENTIRE column down to
        # ~{0, 1e-9} — destroying the real signal in the non-zero values
        # and, at large scale, feeding PowerTransformer a near-constant
        # column that fails Yeo-Johnson's Brent bracket search entirely.
        # The correct behaviour when there's no meaningful spread to clip
        # against is to not clip that column at all.
        degenerate  = (iqr == 0)
        self.lower_ = np.where(degenerate, -np.inf, self.lower_)
        self.upper_ = np.where(degenerate,  np.inf, self.upper_)

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features, dtype=object)
        n = getattr(self, "n_features_in_", 1)
        return np.array([f"x{i}" for i in range(n)], dtype=object)


# ============================================================
# SAFE POWER TRANSFORMER — per-column Yeo-Johnson with fallback
# ============================================================

class SafePowerTransformer(BaseEstimator, TransformerMixin):
    """
    Per-column Yeo-Johnson power transform, fit column-by-column instead
    of via sklearn's PowerTransformer.

    Why: sklearn's PowerTransformer fits all columns in one call — if
    ANY single column produces a degenerate log-likelihood surface
    (e.g. a near-constant or heavily zero-inflated engineered column),
    scipy's Brent bracket search can raise `BracketError` and abort
    the fit for every column, crashing the entire training run. This
    is exactly what happened on the full-scale (~8M row) NYC taxi data.

    Fix: fit each column independently. A column whose optimizer fails
    (or that has near-zero variance) falls back to lambda=1, i.e. a
    passthrough — it still gets centered/scaled downstream by
    StandardScaler, it just skips the power transform. One bad column
    can never take down the other 90+ features anymore.
    """

    def __init__(self):
        self.lambdas_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_features = X.shape[1]
        lambdas    = np.ones(n_features)

        for i in range(n_features):
            col = X[:, i]
            try:
                if np.nanstd(col) < 1e-8:
                    # Near-constant column — nothing meaningful to transform.
                    lambdas[i] = 1.0
                    continue
                lam = scipy_stats.yeojohnson_normmax(col)
                lambdas[i] = lam if np.isfinite(lam) else 1.0
            except Exception as e:
                logger.warning(
                    "Yeo-Johnson fit failed for feature index %d — "
                    "falling back to passthrough (lambda=1): %s", i, e
                )
                lambdas[i] = 1.0

        self.lambdas_       = lambdas
        self.n_features_in_ = n_features

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        out = np.empty_like(X)
        for i in range(X.shape[1]):
            try:
                out[:, i] = scipy_stats.yeojohnson(X[:, i], lmbda=self.lambdas_[i])
            except Exception:
                out[:, i] = X[:, i]
        return out

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features, dtype=object)
        n = getattr(self, "n_features_in_", 1)
        return np.array([f"x{i}" for i in range(n)], dtype=object)


# ============================================================
# SAFE FEATURE COUNT HELPER
# ============================================================

def safe_k(requested_k: int, preprocessor: ColumnTransformer, X_sample: pd.DataFrame) -> int:
    """
    Returns min(requested_k, actual_output_features_of_preprocessor).
    Prevents SelectKBest crash when k > n_features.
    """
    fitted = preprocessor.fit(X_sample)

    try:
        feat_names = fitted.get_feature_names_out()
    except Exception:
        arr = fitted.transform(X_sample)
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        feat_names = [f"f{i}" for i in range(arr.shape[1])]

    k = min(requested_k, max(1, len(feat_names)))
    logger.info("safe_k: requested=%d  available=%d  using=%d", requested_k, len(feat_names), k)
    return k


# ============================================================
# PREPROCESSOR BUILDER
# ============================================================

def build_preprocessors(
    ord_cols:  List[str],
    cont_cols: List[str],
    bin_cols:  List[str],
    X_train:   pd.DataFrame,
    clip_fold: float = CLIP_FOLD
) -> Tuple[ColumnTransformer, ColumnTransformer, List[int], List[str]]:
    """
    Returns:
        preprocessor_scaled   — for distance/linear models (Linear/Ridge/Lasso/SVR/KNN)
        preprocessor_unscaled — for tree models (RF, XGB, LGBM, CatBoost)
        categorical_indices   — indices of ordinal/binary cols in feature_order
        feature_order         — column order after transform
    """

    skewed     = [
        c for c in cont_cols
        if c not in NEVER_POWER_TRANSFORM_COLS and abs(X_train[c].skew()) > 0.8
    ]
    non_skewed = [c for c in cont_cols if c not in skewed]

    logger.info("Skewed cols (%d): %s", len(skewed), skewed)
    logger.info("Non-skewed cols (%d): %s", len(non_skewed), non_skewed)

    # ── Scaled transformers (for linear / distance models) ───
    scaled_transformers = []

    if skewed:
        scaled_transformers.append((
            "skewed",
            Pipeline([
                ("clip",  Clipper(fold=clip_fold)),
                # SafePowerTransformer fits each column's Yeo-Johnson lambda
                # independently (see its docstring) — a single pathological
                # column (e.g. a zero-inflated interaction feature) falls
                # back to passthrough instead of crashing every column's fit.
                ("power", SafePowerTransformer()),
                ("scale", StandardScaler())
            ]),
            skewed
        ))

    if non_skewed:
        scaled_transformers.append((
            "non_skew",
            Pipeline([
                ("clip",  Clipper(fold=clip_fold)),
                ("scale", StandardScaler())
            ]),
            non_skewed
        ))

    # ── Unscaled transformers (for tree models) ───────────────
    unscaled_transformers = []

    if skewed:
        unscaled_transformers.append((
            "skewed",
            Pipeline([("clip", Clipper(fold=clip_fold))]),
            skewed
        ))

    if non_skewed:
        unscaled_transformers.append((
            "non_skew",
            Pipeline([("clip", Clipper(fold=clip_fold))]),
            non_skewed
        ))

    # ── Ordinal encoder ────────────────────────────────────────
    # clone() creates two independent copies — prevents shared fitted state
    # between preprocessor_scaled and preprocessor_unscaled when both fit()
    def _ord_pipeline():
        return Pipeline([
            ("ord", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ])

    preprocessor_scaled = ColumnTransformer(
        transformers=[
            ("ord", _ord_pipeline(), ord_cols),
            *scaled_transformers,
            ("bin", "passthrough", bin_cols)
        ],
        remainder="drop"
    )

    preprocessor_unscaled = ColumnTransformer(
        transformers=[
            ("ord", _ord_pipeline(), ord_cols),
            *unscaled_transformers,
            ("bin", "passthrough", bin_cols)
        ],
        remainder="drop"
    )

    feature_order        = ord_cols + skewed + non_skewed + bin_cols
    categorical_set      = set(ord_cols + bin_cols)
    categorical_indices  = [i for i, c in enumerate(feature_order) if c in categorical_set]

    logger.info("Feature order: %s", feature_order)
    logger.info("Categorical indices: %s", categorical_indices)

    return preprocessor_scaled, preprocessor_unscaled, categorical_indices, feature_order
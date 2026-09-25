# ============================================================
# LEAKAGE CHECK — Ride Fare Price Prediction ML System
# ============================================================
# Detects potential data leakage before model training.
# Two checks (correlation-based, works for continuous targets):
#   1. Exact match  — feature column identical to target
#   2. Near-perfect correlation — |corr| >= threshold with target
# ============================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def detect_leakage(
    X_train:          pd.DataFrame,
    y_train:          pd.Series,
    threshold_corr:   float = 0.98
) -> list:
    """
    Runs leakage heuristics on training set for a regression target.

    Args:
        X_train        : feature DataFrame (after split, before preprocessing)
        y_train         : continuous target Series (fare_amount)
        threshold_corr : Pearson |corr| threshold above which feature is flagged

    Returns:
        List of warning strings. Empty list = no leakage detected.
    """
    warnings = []

    for col in X_train.columns:
        try:
            # ── Check 1: exact match with target ─────────────
            if X_train[col].equals(y_train.astype(X_train[col].dtype, errors="ignore")):
                warnings.append(
                    f"[LEAKAGE] '{col}' is identical to target → remove this feature"
                )
                continue

            # ── Check 2: near-perfect linear correlation ─────
            if np.issubdtype(X_train[col].dtype, np.number):
                col_vals = X_train[col].fillna(X_train[col].median())
                if col_vals.std() == 0:
                    continue
                corr = abs(np.corrcoef(col_vals, y_train.fillna(y_train.median()))[0, 1])
                if corr >= threshold_corr:
                    warnings.append(
                        f"[LEAKAGE] '{col}' has corr={corr:.4f} with target "
                        f"(>= {threshold_corr}) → possible leakage"
                    )

        except Exception as e:
            logger.warning("Leakage check failed for column '%s': %s", col, e)

    if warnings:
        for w in warnings:
            logger.warning(w)
    else:
        logger.info("Leakage check passed — no obvious leakage detected")

    return warnings

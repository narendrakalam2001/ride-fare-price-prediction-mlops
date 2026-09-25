# ============================================================
# METRICS — Ride Fare Price Prediction ML System
# ============================================================

import numpy as np
import pandas as pd
import logging

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

from src.config import UNDER_PRICE_COST_PER_DOLLAR, OVER_PRICE_COST_PER_DOLLAR

logger = logging.getLogger(__name__)


# ============================================================
# CORE REGRESSION METRICS
# ============================================================

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true, y_pred) -> float:
    return float(mean_absolute_percentage_error(y_true, y_pred))


def r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def adjusted_r2(y_true, y_pred, n_features: int) -> float:
    n     = len(y_true)
    r2_v  = r2_score(y_true, y_pred)
    if n - n_features - 1 <= 0:
        return r2_v
    return float(1 - (1 - r2_v) * (n - 1) / (n - n_features - 1))


def within_tolerance_accuracy(y_true, y_pred, tolerance: float = 0.15) -> float:
    """
    Business-friendly metric: % of predictions within `tolerance`
    (default 15%) of the actual fare. Mirrors how pricing teams judge
    "is this prediction usable for surge/payout calculation".
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rel_err = np.abs(y_pred - y_true) / np.maximum(y_true, 1e-6)
    return float(np.mean(rel_err <= tolerance))


# ============================================================
# PSI — Population Stability Index  (unchanged from classification use —
# operates purely on feature/score distributions, domain-agnostic)
# ============================================================

def psi(expected, actual, buckets: int = 10) -> float:
    """
    Population Stability Index — measures distribution shift.

    PSI < 0.1   → no significant shift (stable)
    PSI 0.1–0.2 → moderate shift (monitor closely)
    PSI > 0.2   → major shift (retrain recommended)

    Correct approach:
      1. Compute quantile bin EDGES from `expected` (reference)
      2. Bin BOTH `expected` and `actual` using those SAME edges
      3. Compare bin proportions
    """
    try:
        expected = np.asarray(expected, dtype=float)
        actual   = np.asarray(actual,   dtype=float)

        quantiles = np.linspace(0, 100, buckets + 1)
        bin_edges = np.percentile(expected, quantiles)

        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0

        bin_edges[0]  = min(bin_edges[0],  actual.min()) - 1e-9
        bin_edges[-1] = max(bin_edges[-1], actual.max()) + 1e-9

        exp_hist, _ = np.histogram(expected, bins=bin_edges)
        act_hist, _ = np.histogram(actual,   bins=bin_edges)

        exp_pct = exp_hist / (exp_hist.sum() + 1e-9)
        act_pct = act_hist / (act_hist.sum() + 1e-9)

        exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
        act_pct = np.where(act_pct == 0, 1e-6, act_pct)

        psi_value = float(np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct)))

        return psi_value

    except Exception as e:
        logger.warning("PSI computation failed: %s", e)
        return float("nan")


# ============================================================
# COST-SENSITIVE EVALUATION  (dynamic-pricing business impact)
# ============================================================

def cost_sensitive_evaluation(
    y_true,
    y_pred,
    under_price_cost: float = UNDER_PRICE_COST_PER_DOLLAR,
    over_price_cost:  float = OVER_PRICE_COST_PER_DOLLAR,
) -> dict:
    """
    Translates prediction error into $ business impact, the way a
    ride-hailing pricing team would report it:

      under-pricing (pred < actual) → direct platform revenue loss
      over-pricing  (pred > actual) → rider/driver trust + conversion
                                       opportunity cost (smaller weight)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error         = y_pred - y_true
    under_mask    = error < 0
    over_mask     = error > 0

    under_loss = float(np.abs(error[under_mask]).sum() * under_price_cost)
    over_loss  = float(np.abs(error[over_mask]).sum()  * over_price_cost)
    total_loss = under_loss + over_loss

    result = {
        "under_priced_trips":     int(under_mask.sum()),
        "over_priced_trips":      int(over_mask.sum()),
        "estimated_revenue_loss": round(under_loss, 2),
        "estimated_trust_cost":   round(over_loss, 2),
        "total_estimated_cost":   round(total_loss, 2),
        "avg_cost_per_trip":      round(total_loss / max(len(y_true), 1), 4),
    }

    logger.info(
        "Cost eval  |  under=%d  over=%d  revenue_loss=%.2f  trust_cost=%.2f  total=%.2f",
        result["under_priced_trips"], result["over_priced_trips"],
        result["estimated_revenue_loss"], result["estimated_trust_cost"],
        result["total_estimated_cost"]
    )

    return result


# ============================================================
# DRIFT REPORT — feature mean shift
# ============================================================

def simple_drift_report(X_ref: "pd.DataFrame", X_new: "pd.DataFrame", top_n: int = 10):
    diffs = (X_ref.mean() - X_new.mean()).abs()
    rel   = (diffs / (X_ref.std().replace(0, 1))).sort_values(ascending=False)
    return rel.head(top_n)

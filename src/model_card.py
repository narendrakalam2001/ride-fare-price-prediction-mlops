# ============================================================
# MODEL CARD — Ride Fare Price Prediction ML System
# ============================================================
# Structured model card (Google Model Cards standard), containing:
#   - Model metadata (name, version, date)
#   - Dataset splits info
#   - All regression evaluation metrics
#   - Fare-band decision summary
#   - Cost-sensitive business evaluation
#   - Feature importances / SHAP top features
#   - Calibration (prediction interval) info
# ============================================================

import os
import json
import time
import logging

from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# BUILD MODEL CARD
# ============================================================

def build_model_card(
    selected_name:     str,
    train_fit_size:    int,
    cal_size:          int,
    test_size:         int,
    fare_mean_train:   float,
    metrics:           dict,     # test_rmse, test_mae, test_mape, test_r2, adjusted_r2, within_15pct_acc
    calibration_info:  Optional[dict],
    cost_result:       dict,
    flag_counts:       dict,
    feature_order:     list,
    categorical_indices: list,
    selector_k:        int,
    version:           str = "v1",
    fi_dict:           Optional[dict] = None,
    shap_dict:         Optional[dict] = None,
) -> dict:

    card = {

        "model_version": version,
        "model_name":    selected_name,
        "trained_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "project":       "Ride Fare Price Prediction System",

        "dataset": {
            "train_fit_size":  train_fit_size,
            "calibration_size": cal_size,
            "test_size":        test_size,
            "fare_mean_train":  round(float(fare_mean_train), 4),
        },

        "metrics": {
            "test_rmse":        round(float(metrics.get("test_rmse",        0)), 4),
            "test_mae":         round(float(metrics.get("test_mae",         0)), 4),
            "test_mape":        round(float(metrics.get("test_mape",        0)), 4),
            "test_r2":          round(float(metrics.get("test_r2",          0)), 4),
            "adjusted_r2":      round(float(metrics.get("adjusted_r2",      0)), 4),
            "within_15pct_acc": round(float(metrics.get("within_15pct_acc", 0)), 4),
        },

        "calibration": {
            "method":              "isotonic_residual_conformal",
            "target_coverage":     calibration_info.get("coverage") if calibration_info else None,
            "empirical_coverage":  round(calibration_info["empirical_coverage"], 4) if calibration_info else None,
            "scale_factor_k":      round(calibration_info["k"], 4) if calibration_info else None,
        },

        "pricing_flags": flag_counts,

        "cost_evaluation": cost_result,

        "pipeline_config": {
            "feature_order":        feature_order,
            "categorical_indices":  categorical_indices,
            "selector_k":           selector_k,
        },
    }

    if fi_dict is not None:
        card["feature_importances"] = fi_dict

    if shap_dict is not None:
        card["shap_top_features"] = shap_dict

    return card


# ============================================================
# SAVE / LOAD MODEL CARD
# ============================================================

def save_model_card(card: dict, model_dir: str, selected_name: str, version: str = "v1") -> str:
    os.makedirs(model_dir, exist_ok=True)
    card_path = os.path.join(model_dir, f"model_card_{selected_name}_{version}.json")
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2, default=str)
    logger.info("Model card saved → %s", card_path)
    return card_path


def load_model_card(model_dir: str, selected_name: str, version: str = "v1") -> dict:
    card_path = os.path.join(model_dir, f"model_card_{selected_name}_{version}.json")
    if not os.path.exists(card_path):
        raise FileNotFoundError(f"Model card not found: {card_path}")
    with open(card_path) as f:
        card = json.load(f)
    logger.info("Model card loaded ← %s", card_path)
    return card

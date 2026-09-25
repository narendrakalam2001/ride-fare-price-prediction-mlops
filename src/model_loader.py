# ============================================================
# MODEL LOADER + CHALLENGER SYSTEM — Ride Fare Price Prediction
# ============================================================
#
# CHALLENGER MODEL SYSTEM:
#   Champion  = current production model (latest_model.json)
#   Challenger = newly trained model (passed in from training_pipeline)
#
#   Promotion logic — challenger promoted ONLY if it beats champion on
#   ALL 3 gates:
#     1. RMSE improvement    >= MIN_RMSE_IMPROVEMENT (relative %)
#     2. Test R2             >= MIN_R2_THRESHOLD
#     3. Train-test R2 gap   <= MAX_GENERALIZATION_GAP
#
#   If challenger loses → champion stays, challenger archived.
#   Full comparison history saved to fare_models/challenger_log.json
# ============================================================

import os
import json
import joblib
import logging
import time

from src.config import MODEL_DIR, MIN_RMSE_IMPROVEMENT, MIN_R2_THRESHOLD, MAX_GENERALIZATION_GAP

logger = logging.getLogger(__name__)

CHALLENGER_LOG = os.path.join(MODEL_DIR, "challenger_log.json")


# ============================================================
# LOAD LATEST (CHAMPION) MODEL
# ============================================================

def load_latest_model():
    """
    Reads fare_models/latest_model.json → loads .joblib (+ calibration
    artifact if present).
    Returns: (model_pipeline, calibration_dict_or_None)
    """
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")

    if not os.path.exists(registry_path):
        raise FileNotFoundError(
            f"Model registry not found at {registry_path}. Run train_model.py first."
        )

    with open(registry_path) as f:
        registry = json.load(f)

    model_path = os.path.join(MODEL_DIR, registry["model_name"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)

    calibration = None
    cal_name = registry.get("calibration_name", "")
    if cal_name:
        cal_path = os.path.join(MODEL_DIR, cal_name)
        if os.path.exists(cal_path):
            calibration = joblib.load(cal_path)

    logger.info("Champion model loaded: %s  |  calibration=%s", model_path, bool(calibration))

    return model, calibration


# ============================================================
# LOAD CHAMPION METRICS FROM MODEL CARD
# ============================================================

def _load_champion_metrics() -> dict:
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")

    if not os.path.exists(registry_path):
        return {}

    with open(registry_path) as f:
        registry = json.load(f)

    card_path = registry.get("model_card_path", "")

    if not card_path:
        model_name = registry.get("model_name", "")
        parts      = model_name.replace("fare_model_", "").replace(".joblib", "")
        card_path  = os.path.join(MODEL_DIR, f"model_card_{parts}.json")
        logger.warning("model_card_path missing in registry — using fallback: %s", card_path)

    if not os.path.exists(card_path):
        logger.warning("Champion model card not found: %s", card_path)
        return {}

    with open(card_path) as f:
        card = json.load(f)

    metrics = card.get("metrics", card)

    return {
        "model_name": card.get("model_name", "unknown"),
        "rmse":       float(metrics.get("test_rmse", float("inf"))),
        "r2":         float(metrics.get("test_r2", 0)),
    }


# ============================================================
# CHALLENGER COMPARISON — CORE LOGIC
# ============================================================

def run_challenger_comparison(
    challenger_name:         str,
    challenger_rmse:         float,
    challenger_r2:           float,
    challenger_gap:          float,
    challenger_model_path:   str,
    challenger_card_path:    str = "",
    challenger_calibration_path: str = "",
) -> dict:
    """
    Compares challenger vs current champion on RMSE / R2 / generalization gap.
    Returns result dict with decision: 'PROMOTED' or 'REJECTED'.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    champion = _load_champion_metrics()

    if not champion:
        logger.info("No champion found — challenger auto-promoted as first model")
        _update_registry(challenger_model_path, challenger_card_path, challenger_calibration_path)
        result = {
            "decision":          "PROMOTED",
            "reason":            "No existing champion — first model auto-promoted",
            "challenger_name":   challenger_name,
            "challenger_rmse":   round(challenger_rmse, 4),
            "challenger_r2":     round(challenger_r2,   4),
            "champion_name":     None,
            "champion_rmse":     None,
            "evaluated_at":      time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_challenger_log(result)
        return result

    champion_rmse = champion.get("rmse", float("inf"))
    champion_r2   = champion.get("r2", 0.0)
    champion_name = champion.get("model_name", "unknown")

    logger.info("=" * 55)
    logger.info("CHAMPION vs CHALLENGER")
    logger.info("  Champion  : %-20s  RMSE=%.4f  R2=%.4f", champion_name, champion_rmse, champion_r2)
    logger.info("  Challenger: %-20s  RMSE=%.4f  R2=%.4f", challenger_name, challenger_rmse, challenger_r2)
    logger.info("=" * 55)

    rel_improvement = (champion_rmse - challenger_rmse) / max(champion_rmse, 1e-9)

    gate1_rmse_improvement = rel_improvement >= MIN_RMSE_IMPROVEMENT
    gate2_r2_floor         = challenger_r2 >= MIN_R2_THRESHOLD
    gate3_gap              = challenger_gap <= MAX_GENERALIZATION_GAP

    gates_passed = gate1_rmse_improvement and gate2_r2_floor and gate3_gap

    if gates_passed:
        decision = "PROMOTED"
        reason = (
            f"Challenger beats champion: RMSE {champion_rmse:.4f} → {challenger_rmse:.4f} "
            f"({rel_improvement*100:+.2f}%)"
        )
        logger.info("✅ CHALLENGER PROMOTED → new champion: %s", challenger_name)
        _update_registry(challenger_model_path, challenger_card_path, challenger_calibration_path)

    else:
        decision = "REJECTED"
        failed = []
        if not gate1_rmse_improvement:
            failed.append(f"RMSE improvement {rel_improvement*100:+.2f}% < {MIN_RMSE_IMPROVEMENT*100:.1f}%")
        if not gate2_r2_floor:
            failed.append(f"R2 {challenger_r2:.4f} < {MIN_R2_THRESHOLD}")
        if not gate3_gap:
            failed.append(f"train-test R2 gap {challenger_gap:.4f} > {MAX_GENERALIZATION_GAP}")
        reason = "Gates failed: " + " | ".join(failed)
        logger.info("❌ CHALLENGER REJECTED — champion '%s' retained", champion_name)
        logger.info("   Reason: %s", reason)

    result = {
        "decision":            decision,
        "reason":              reason,
        "evaluated_at":        time.strftime("%Y-%m-%d %H:%M:%S"),
        "challenger_name":     challenger_name,
        "challenger_rmse":     round(challenger_rmse, 4),
        "challenger_r2":       round(challenger_r2,   4),
        "challenger_gap":      round(challenger_gap,  4),
        "champion_name":       champion_name,
        "champion_rmse":       round(champion_rmse, 4),
        "champion_r2":         round(champion_r2,   4),
        "gates": {
            "rmse_improvement_passed": gate1_rmse_improvement,
            "r2_floor_passed":         gate2_r2_floor,
            "gap_passed":              gate3_gap,
        }
    }

    _save_challenger_log(result)
    return result


# ============================================================
# HELPERS
# ============================================================

def _update_registry(model_path: str, model_card_path: str = None, calibration_path: str = None):
    registry = {
        "model_name":        os.path.basename(model_path),
        "model_card_path":   model_card_path or "",
        "calibration_name":  os.path.basename(calibration_path) if calibration_path else "",
    }
    with open(os.path.join(MODEL_DIR, "latest_model.json"), "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registry updated → %s", registry["model_name"])


def _save_challenger_log(result: dict):
    history = []
    if os.path.exists(CHALLENGER_LOG):
        try:
            with open(CHALLENGER_LOG) as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(result)

    with open(CHALLENGER_LOG, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Challenger log saved → %s", CHALLENGER_LOG)

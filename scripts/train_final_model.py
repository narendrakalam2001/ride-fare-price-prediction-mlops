# ============================================================
# TRAIN FINAL MODEL — refit the WINNING model on more data
# ============================================================
# Use this AFTER a full comparison run (scripts/train_model.py) has
# already told you which model wins (e.g. LightGBM). This script does
# NOT compare 15 models or run RandomizedSearchCV — it takes the exact
# architecture + hyperparameters of your current champion and refits
# ONLY that one model on a larger (or the full) dataset. Since it's a
# single fit instead of a 15-model x n_iter x cv_folds search, this is
# dramatically faster — LightGBM on this project's real data fits in
# roughly 12 seconds per ~1.1M rows, so even 10-20x more data is
# typically minutes, not hours.
#
# USAGE:
#   python scripts/train_final_model.py
#
# By default this uses whatever RIDE_FARE_SAMPLE_FRAC / RIDE_FARE_NROWS
# / RIDE_FARE_FORCE_FULL you set (same env vars as train_model.py) — for
# a genuine "final" model you'll usually want a much bigger sample than
# you used for the comparison run, e.g.:
#   $env:RIDE_FARE_SAMPLE_FRAC = "0.3"      # or higher, or FORCE_FULL=1
#   python scripts/train_final_model.py
#
# WHAT IT DOES:
#   1. Loads the current champion pipeline (fare_models/latest_model.json)
#      and reads off its exact architecture + hyperparameters
#   2. Loads/validates/feature-engineers the (larger) dataset you point
#      it at, same as train_model.py
#   3. sklearn.base.clone()'s the champion pipeline — same regressor
#      class, same hyperparameters, same feature mask, but completely
#      UNFITTED — then fits that clone on the new, larger training set
#   4. Runs the full evaluation suite (metrics, cost evaluation, SHAP,
#      feature importance, PSI, calibration) exactly like a normal run
#   5. Saves it as a new challenger and runs it through the same 3-gate
#      Champion-vs-Challenger promotion as train_model.py — it only
#      becomes the new champion if it genuinely beats the old one
#
# WHAT IT DOES NOT DO:
#   - It does not re-tune hyperparameters. The assumption is that the
#     best hyperparameters found on a representative sample generalize
#     reasonably well to more data of the same distribution. If you
#     want a fresh hyperparameter search on the full data too, use
#     scripts/train_model.py instead (slower, but exhaustive).
# ============================================================

import sys
import os
import time
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.training_pipeline import run_final_model_training

if __name__ == "__main__":
    start = time.time()
    run_final_model_training()
    logging.getLogger(__name__).info("Finished in %.1fs", time.time() - start)

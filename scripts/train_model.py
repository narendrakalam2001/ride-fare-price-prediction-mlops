# ============================================================
# TRAIN MODEL — runner script
# ============================================================
# Environment variables (all optional):
#
#   RIDE_FARE_DATA_PATH     Path to the training CSV
#                            (default: data/train.csv)
#
#   RIDE_FARE_SAMPLE_FRAC   Fraction of rows to randomly sample while
#                            reading the file, e.g. 0.15 for ~15%.
#                            Recommended for the full 55M-row Kaggle file.
#
#   RIDE_FARE_NROWS         Read only the first N rows instead (faster,
#                            less representative than sample_frac).
#
#   RIDE_FARE_FORCE_FULL=1  Load the ENTIRE file with no sampling.
#                            Needs ~16GB+ free RAM for the full NYC taxi
#                            fare dataset — otherwise train_test_split
#                            can crash with a MemoryError. If none of the
#                            above are set and the file looks large, the
#                            pipeline auto-applies a safe sample_frac and
#                            logs what it did.
#
# Examples (PowerShell):
#   $env:RIDE_FARE_SAMPLE_FRAC = "0.15"; python scripts/train_model.py
#   $env:RIDE_FARE_FORCE_FULL  = "1";    python scripts/train_model.py
# ============================================================

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.training_pipeline import run_training

if __name__ == "__main__":
    run_training()
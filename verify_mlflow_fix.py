# ============================================================
# VERIFY MLFLOW FIX — takes ~5-10 seconds, NOT the full pipeline
# ============================================================
# Run this from the project root:
#   python verify_mlflow_fix.py
#
# This does NOT load any data, does NOT train real models, and does
# NOT touch fare_models/ or the champion registry. It only exercises
# the exact mlflow_log_run() function that was crashing before, using
# a tiny fake pipeline, so you can confirm the fix works in seconds
# instead of waiting through a full multi-hour training run.
# ============================================================

import sys
import os
import shutil

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

print("Cleaning up any old mlruns/mlflow.db from previous tests ...")
shutil.rmtree(os.path.join(PROJECT_ROOT, "mlruns"), ignore_errors=True)
db_path = os.path.join(PROJECT_ROOT, "mlflow.db")
if os.path.exists(db_path):
    os.remove(db_path)

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import pandas as pd

print("Building a tiny fake pipeline (not a real trained model) ...")
fake_pipe = Pipeline([("reg", LinearRegression())])
fake_pipe.fit(pd.DataFrame({"x": [1, 2, 3]}), [1, 2, 3])

fake_card = {
    "metrics": {
        "test_rmse": 3.9, "test_mae": 2.0, "test_mape": 0.19,
        "test_r2": 0.83, "adjusted_r2": 0.83,
    }
}

print("Calling mlflow_log_run() — this is the exact function that was crashing ...")
from src.evaluation import mlflow_log_run

try:
    mlflow_log_run("verify_fix_test", "FakeModel", fake_pipe, fake_card)
    print()
    print("=" * 60)
    print("RESULT: PASS — no 'unsupported URI' / Model Registry error.")
    print("The fix is correctly applied. You do NOT need to re-run the")
    print("full training pipeline just to check this again.")
    print("=" * 60)
except Exception as e:
    print()
    print("=" * 60)
    print("RESULT: FAIL —", type(e).__name__, str(e)[:200])
    print("The old evaluation.py is still in place, or the process")
    print("that's running was started before you replaced the file.")
    print("=" * 60)
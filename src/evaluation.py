# ============================================================
# EVALUATION — Ride Fare Price Prediction ML System
# ============================================================

import os
import json
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from typing import Dict, Optional, Tuple

from sklearn.metrics          import mean_squared_error
from sklearn.isotonic         import IsotonicRegression
from sklearn.model_selection  import cross_val_score, KFold
from sklearn.inspection       import permutation_importance
from sklearn.pipeline         import Pipeline

from src.config  import RANDOM_STATE, N_JOBS, CV_FOLDS, MODEL_DIR
from src.metrics import rmse, mae, mape, r2, adjusted_r2, within_tolerance_accuracy

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================
# EVALUATE ALL MODELS — summary table
# ============================================================

def evaluate_models(
    pipelines:    Dict,
    X_train:      pd.DataFrame,
    y_train:      pd.Series,
    X_test:       pd.DataFrame,
    y_test:       pd.Series,
    n_features_for_adj_r2: int = 10,
    precomputed_cv: Dict = None,
) -> pd.DataFrame:
    """
    precomputed_cv: optional {model_name: (cv_mean_rmse, cv_std_rmse)} —
    pass the cv_stats dict returned by model_tuning.tune_models() here.
    For any model present in it, its CV numbers are reused as-is instead
    of being recomputed via a fresh cross_val_score() call, which would
    re-fit that model CV_FOLDS more times for no new information (the
    RandomizedSearchCV run during tuning already fit it exactly this
    many times with this exact config). Models not in precomputed_cv
    (e.g. the NeuralNet pipeline, which isn't tuned via RandomizedSearchCV)
    fall back to computing it here as before.
    """
    precomputed_cv = precomputed_cv or {}

    rows = []

    for name, pipe in pipelines.items():

        logger.info("Evaluating: %s", name)

        y_pred_train = pipe.predict(X_train)
        y_pred_test  = pipe.predict(X_test)

        train_r2 = r2(y_train, y_pred_train)
        test_r2  = r2(y_test,  y_pred_test)

        test_rmse = rmse(y_test, y_pred_test)
        test_mae  = mae(y_test,  y_pred_test)
        test_mape = mape(y_test, y_pred_test)
        adj_r2    = adjusted_r2(y_test, y_pred_test, n_features_for_adj_r2)
        tol_acc   = within_tolerance_accuracy(y_test, y_pred_test, tolerance=0.15)

        if name in precomputed_cv:
            cv_mean_rmse, cv_std_rmse = precomputed_cv[name]
        else:
            try:
                cv_scores = cross_val_score(
                    pipe, X_train, y_train,
                    scoring = "neg_root_mean_squared_error",
                    cv      = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                    n_jobs  = N_JOBS
                )
                cv_mean_rmse = float(-cv_scores.mean())
                cv_std_rmse  = float(cv_scores.std())
            except Exception:
                cv_mean_rmse = None
                cv_std_rmse  = None

        rows.append({
            "model":            name,
            "train_r2":         round(train_r2, 4),
            "test_r2":          round(test_r2,  4),
            "train_test_gap":   round(abs(train_r2 - test_r2), 4),
            "cv_mean_rmse":     round(cv_mean_rmse, 4) if cv_mean_rmse is not None else None,
            "cv_std_rmse":      round(cv_std_rmse,  4) if cv_std_rmse  is not None else None,
            "test_rmse":        round(test_rmse, 4),
            "test_mae":         round(test_mae,  4),
            "test_mape":        round(test_mape, 4),
            "adjusted_r2":      round(adj_r2,     4),
            "within_15pct_acc": round(tol_acc,    4),
        })

    summary = (
        pd.DataFrame(rows)
        .sort_values(["test_rmse", "test_r2"], ascending=[True, False])
        .reset_index(drop=True)
    )

    summary.to_csv(os.path.join(MODEL_DIR, "model_experiment_results.csv"), index=False)

    return summary


# ============================================================
# SELECT BEST MODEL — with generalization filter
# ============================================================

def select_best_model(
    summary:      pd.DataFrame,
    pipelines:    Dict,
    scaled_pipes: Dict,
    unscaled_pipes: Dict
) -> Tuple[str, object]:

    def _filter(df, gen_gap, cv_std_ratio, min_r2):
        mask = (
            df["train_test_gap"].notna() &
            df["cv_std_rmse"].notna() &
            (df["train_test_gap"] <= gen_gap) &
            (df["cv_std_rmse"] / df["cv_mean_rmse"].replace(0, np.nan) <= cv_std_ratio) &
            (df["test_r2"] >= min_r2)
        )
        return df[mask].copy()

    thresholds = [
        (0.03, 0.10, 0.90),
        (0.06, 0.15, 0.85),
        (0.10, 0.20, 0.80),
        (0.20, 1.00, 0.00),
    ]

    candidates = pd.DataFrame()
    for gg, cvr, mr2 in thresholds:
        candidates = _filter(summary, gg, cvr, mr2)
        if not candidates.empty:
            logger.info("Candidates found with gen_gap<=%.2f cv_std_ratio<=%.2f min_r2>=%.2f",
                        gg, cvr, mr2)
            break

    if candidates.empty:
        selected_name = summary.iloc[0]["model"]
    else:
        candidates    = candidates.sort_values(
            ["test_rmse", "test_r2"], ascending=[True, False]
        ).reset_index(drop=True)
        selected_name = candidates.iloc[0]["model"]

    logger.info("Selected model: %s", selected_name)

    # `pipelines` is the full combined dict (scaled + unscaled + NeuralNet, etc.) —
    # checked first since NeuralNet/MLP is only registered there, not in either
    # scaled_pipes or unscaled_pipes individually.
    selected_pipe = (
        pipelines.get(selected_name)
        or scaled_pipes.get(selected_name)
        or unscaled_pipes.get(selected_name)
    )
    if selected_pipe is None:
        raise RuntimeError(f"Selected model '{selected_name}' not found in any pipeline dict")

    return selected_name, selected_pipe


# ============================================================
# RESIDUAL CALIBRATION — isotonic prediction intervals
# ============================================================
# NOTE on adaptation: classification systems calibrate *probabilities*
# with isotonic regression (predicted score → true likelihood). There is
# no probability to calibrate in regression. The equivalent, industry-
# standard adaptation used here is *residual / conformal calibration*:
# isotonic regression is fit on |residual| vs predicted fare (on a held-
# out calibration split, no leakage) to produce well-calibrated 90%
# prediction intervals — i.e. "how much to trust this point estimate",
# which is what a pricing engine actually needs downstream.
# ============================================================

def calibrate_with_holdout(selected_pipe, X_cal, y_cal, coverage: float = 0.90):
    """
    Fits an isotonic model mapping predicted fare → expected absolute
    residual, then converts that into a coverage-scaled interval half-
    width. Returns a dict with the isotonic model + a scale factor, or
    None if calibration fails.
    """
    try:
        y_pred_cal = selected_pipe.predict(X_cal)
        abs_resid  = np.abs(np.asarray(y_cal) - y_pred_cal)

        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(y_pred_cal, abs_resid)

        # scale factor so that ~`coverage` fraction of calibration points
        # fall inside [pred - k*iso(pred), pred + k*iso(pred)]
        predicted_half_width = iso.predict(y_pred_cal)
        ratio = abs_resid / np.maximum(predicted_half_width, 1e-6)
        k = float(np.quantile(ratio, coverage))

        empirical_coverage = float(np.mean(abs_resid <= k * predicted_half_width))

        logger.info("Residual calibration done  |  k=%.4f  empirical_coverage=%.4f",
                    k, empirical_coverage)

        return {"isotonic": iso, "k": k, "coverage": coverage,
                "empirical_coverage": empirical_coverage}

    except Exception as e:
        logger.exception("Residual calibration failed: %s", e)
        return None


def predict_with_interval(selected_pipe, calibration: dict, X):
    """Returns (point_pred, lower_bound, upper_bound) using calibrated intervals."""
    point_pred = selected_pipe.predict(X)
    if calibration is None:
        return point_pred, None, None

    half_width = calibration["k"] * calibration["isotonic"].predict(point_pred)
    lower = point_pred - half_width
    upper = point_pred + half_width
    return point_pred, lower, upper


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def compute_feature_importance(selected_pipe, X_train, y_train, top_k: int = 20):

    reg   = selected_pipe.named_steps["regressor"]
    names = _get_feature_names(selected_pipe, X_train)

    if hasattr(reg, "feature_importances_"):
        imp   = np.asarray(reg.feature_importances_)
        names = names if names and len(names) == len(imp) else [f"f{i}" for i in range(len(imp))]
        fi    = pd.Series(imp, index=names).sort_values(ascending=False).head(top_k)
        return fi

    if hasattr(reg, "coef_"):
        coef  = np.abs(np.asarray(reg.coef_)).ravel()
        names = names if names and len(names) == len(coef) else [f"f{i}" for i in range(len(coef))]
        fi    = pd.Series(coef, index=names).sort_values(ascending=False).head(top_k)
        return fi

    logger.info("Using permutation importance (slow fallback)...")
    try:
        r   = permutation_importance(
            selected_pipe, X_train, y_train,
            n_repeats=5, scoring="neg_root_mean_squared_error",
            n_jobs=N_JOBS, random_state=RANDOM_STATE
        )
        idx = np.argsort(r.importances_mean)[::-1][:top_k]
        fi  = pd.Series(r.importances_mean[idx], index=[f"f{i}" for i in idx])
        return fi
    except Exception as e:
        logger.exception("Permutation importance failed: %s", e)
        return None


def _get_feature_names(pipe, X_sample):
    """Extracts clean feature names after preprocessor + selector."""
    try:
        pre       = pipe.named_steps["preprocessor"]
        raw_names = list(pre.get_feature_names_out())

        clean_names = []
        for n in raw_names:
            clean_names.append(n.split("__", 1)[1] if "__" in n else n)

        sel = pipe.named_steps.get("selector", None)
        if sel is not None:
            mask        = sel.get_support()
            clean_names = [n for n, m in zip(clean_names, mask) if m]

        return clean_names

    except Exception as e:
        logger.warning("Could not extract feature names: %s", e)
        return None


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

def compute_shap(selected_pipe, X_train, X_explain, max_explain: int = 200):

    if not SHAP_AVAILABLE:
        logger.info("SHAP not installed — skipping")
        return None

    reg  = selected_pipe.named_steps["regressor"]
    pre  = selected_pipe.named_steps["preprocessor"]
    sel  = selected_pipe.named_steps.get("selector", None)

    try:
        X_pre = pre.transform(X_train)
        X_tr  = sel.transform(X_pre) if sel is not None else X_pre
        X_ex  = sel.transform(pre.transform(X_explain)) if sel is not None else pre.transform(X_explain)
        X_ex  = np.asarray(X_ex)[:max_explain]

        names = _get_feature_names(selected_pipe, X_train)

        tree_types = ("RandomForestRegressor", "ExtraTreesRegressor",
                      "GradientBoostingRegressor", "XGBRegressor",
                      "LGBMRegressor", "CatBoostRegressor", "DecisionTreeRegressor")

        if type(reg).__name__ in tree_types:
            explainer = shap.TreeExplainer(reg)
            vals = np.array(explainer.shap_values(X_ex))
        else:
            bg_sample = shap.sample(X_tr, min(200, len(X_tr)))
            explainer = shap.KernelExplainer(reg.predict, bg_sample)
            vals      = np.array(explainer.shap_values(X_ex, nsamples=100))

        if vals.ndim == 1:
            vals = vals.reshape(1, -1)

        mean_abs   = np.abs(vals).mean(axis=0)
        feat_names = names[:mean_abs.shape[0]] if names else [f"f{i}" for i in range(mean_abs.shape[0])]

        fi_series = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)

        logger.info("SHAP done  |  explainer=%s", type(explainer).__name__)

        return {"shap_top": fi_series.head(20).to_dict(), "explainer": type(explainer).__name__}

    except Exception as e:
        logger.exception("SHAP failed: %s", e)
        return None


# ============================================================
# SAVE MODEL ARTIFACTS
# ============================================================

def save_challenger_artifact(
    selected_name:   str,
    pipe,
    version:         str = "v1",
) -> str:
    """
    Saves the trained CHALLENGER pipeline as .joblib — does NOT touch the
    registry (latest_model.json). The registry is only updated by
    run_challenger_comparison() in model_loader.py, and only when the
    challenger genuinely beats the current champion on all gates.
    """
    model_path = os.path.join(MODEL_DIR, f"fare_model_{selected_name}_{version}.joblib")
    joblib.dump(pipe, model_path)
    logger.info("Challenger model saved → %s (registry untouched until gates pass)", model_path)
    return model_path


# ============================================================
# MLFLOW LOGGING
# ============================================================

def mlflow_log_run(
    run_name:      str,
    selected_name: str,
    pipe,
    model_card:    dict,
    X_train_sample: "pd.DataFrame" = None
):
    if not MLFLOW_AVAILABLE:
        logger.info("MLflow not installed — skipping")
        return

    try:
        # ROOT CAUSE (found via testing): passing a bare Windows path like
        # "C:\Users\...\mlruns" to mlflow.set_tracking_uri() gets parsed by
        # urllib as a URI where "C" is read as the SCHEME (urlparse('C:\\x')
        # -> scheme='c') — and 'c' isn't in mlflow's supported scheme list,
        # producing exactly the "unsupported URI 'C:\Users\...'" error seen
        # in testing, regardless of what happens later in this function
        # (this fires on set_tracking_uri/set_experiment, before log_model
        # is ever reached — that's why disabling log_model alone didn't
        # help). The fix is to build a proper "file:///C:/Users/..." URI
        # (via pathlib's as_uri(), which also percent-encodes spaces
        # correctly) instead of handing mlflow a raw OS path string.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        mlruns_dir   = os.path.join(project_root, "mlruns")
        os.makedirs(mlruns_dir, exist_ok=True)
        mlruns_uri   = Path(mlruns_dir).resolve().as_uri()

        # Recent mlflow versions block the plain filesystem store by
        # default ("in maintenance mode", pushing toward a sqlite/db
        # backend) — opting back into it avoids ever touching a database
        # backend (and the sqlite-specific %20 bug that motivated this in
        # the first place).
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(mlruns_uri)
        mlflow.set_registry_uri(mlruns_uri)
        mlflow.set_experiment("ride_fare_experiments")

        metrics_dict = model_card.get("metrics", model_card)

        with mlflow.start_run(run_name=run_name):

            mlflow.log_param("model_name", selected_name)
            mlflow.log_param("selector_k", model_card.get("pipeline_config", {}).get("selector_k", "?"))

            mlflow.log_metric("test_rmse",  float(metrics_dict.get("test_rmse",  0)))
            mlflow.log_metric("test_mae",   float(metrics_dict.get("test_mae",   0)))
            mlflow.log_metric("test_mape",  float(metrics_dict.get("test_mape",  0)))
            mlflow.log_metric("test_r2",    float(metrics_dict.get("test_r2",    0)))
            mlflow.log_metric("adjusted_r2",float(metrics_dict.get("adjusted_r2",0)))

            try:
                # mlflow.sklearn.log_model() is intentionally SKIPPED by
                # default now. Two different parameter names ("name" then
                # "artifact_path") both triggered the same "Model registry
                # functionality is unavailable ... unsupported URI" error
                # on a real machine, even though neither reproduced the
                # failure in testing here — this points to a version-
                # specific mlflow behavior difference (log_model appears to
                # touch registry-adjacent code internally on some mlflow
                # versions regardless of which parameter is used, when the
                # tracking store is a plain filesystem path). Since the
                # actual trained model is ALWAYS saved reliably via joblib
                # to fare_models/ (see save_challenger_artifact) regardless
                # of MLflow, duplicating that into MLflow's own model
                # artifact store isn't essential — only params/metrics
                # logging (below, in the outer try block) is. Set
                # RIDE_FARE_MLFLOW_LOG_MODEL=1 to opt back into this if
                # your mlflow/backend combination handles it cleanly.
                if os.getenv("RIDE_FARE_MLFLOW_LOG_MODEL") == "1":
                    log_kwargs = {
                        "sk_model": pipe,
                        "artifact_path": "model",
                        "serialization_format": "cloudpickle",
                        "pip_requirements": [
                            "scikit-learn==1.3.2", "lightgbm", "xgboost",
                            "pandas", "numpy",
                        ],
                    }
                    if X_train_sample is not None:
                        log_kwargs["input_example"] = X_train_sample.iloc[:5].astype(float)

                    mlflow.sklearn.log_model(**log_kwargs)
                else:
                    logger.info("Skipping mlflow model-artifact logging (params/metrics still "
                                "logged above; the real model file is already saved to "
                                "fare_models/). Set RIDE_FARE_MLFLOW_LOG_MODEL=1 to enable it.")

            except Exception as e:
                logger.warning("mlflow log_model failed (non-fatal — training results are already "
                                "saved to fare_models/, this only affects the optional MLflow UI): %s", e)

        logger.info("MLflow run logged: %s", run_name)

    except Exception as e:
        logger.warning("MLflow logging failed (non-fatal — training results are already saved "
                        "to fare_models/, this only affects the optional MLflow UI): %s", e)
# ============================================================
# TRAINING PIPELINE — Ride Fare Price Prediction ML System
# ============================================================

import os
import time
import logging
import warnings
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold

from src.config        import (RANDOM_STATE, MODEL_DIR, SELECT_K,
                                LARGE_FILE_ROW_THRESHOLD, DEFAULT_SAMPLE_FRAC_FOR_LARGE_FILES)
from src.data_loader    import load_raw_data, validate_input_data, add_engineered_features, detect_feature_types, TARGET_COL
from src.preprocessing  import build_preprocessors, safe_k
from src.model_tuning   import (scaled_models, unscaled_models, tune_models,
                                 train_mlp_pipeline, select_features_once)
from src.evaluation     import (evaluate_models, select_best_model, calibrate_with_holdout,
                                 predict_with_interval, compute_feature_importance, compute_shap,
                                 save_challenger_artifact, mlflow_log_run)
from src.leakage_check  import detect_leakage
from src.model_card     import build_model_card, save_model_card
from src.model_loader   import run_challenger_comparison
from src.metrics        import rmse, mae, mape, r2, adjusted_r2, within_tolerance_accuracy, psi, cost_sensitive_evaluation
from src.pricing_engine import pricing_engine

import joblib

# Harmless — comes from mlflow's internal Pydantic model, unrelated to this
# project's code. Silenced here purely to keep training logs readable.
warnings.filterwarnings("ignore", message='Field "model_name" has conflict with protected namespace')

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = r"D:\Data Science Datasets\new-york-city-taxi-fare-prediction"


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def run_training(data_path: str = DATA_PATH, nrows: int = None, sample_frac: float = None):

    # ─────────────────────────────────────────────────────────
    # 0. RESOLVE nrows / sample_frac  (explicit args > env vars > auto-safety)
    # ─────────────────────────────────────────────────────────
    if nrows is None and os.getenv("RIDE_FARE_NROWS"):
        nrows = int(os.getenv("RIDE_FARE_NROWS"))
    if sample_frac is None and os.getenv("RIDE_FARE_SAMPLE_FRAC"):
        sample_frac = float(os.getenv("RIDE_FARE_SAMPLE_FRAC"))

    force_full = os.getenv("RIDE_FARE_FORCE_FULL", "0") == "1"

    if nrows is None and sample_frac is None and not force_full:
        # Peek at the row count cheaply (no dtype parsing) before deciding.
        try:
            with open(data_path, "rb") as f:
                approx_rows = sum(1 for _ in f) - 1  # minus header
        except Exception:
            approx_rows = None

        if approx_rows is not None and approx_rows > LARGE_FILE_ROW_THRESHOLD:
            sample_frac = DEFAULT_SAMPLE_FRAC_FOR_LARGE_FILES
            logger.warning(
                "Detected a large file (~%d rows). Loading the FULL file + "
                "train_test_split on it needs a lot of RAM and can crash with "
                "a MemoryError (this is what happens without sampling). "
                "Auto-applying sample_frac=%.2f (~%d rows) for a safe run.",
                approx_rows, sample_frac, int(approx_rows * sample_frac)
            )
            logger.warning(
                "To use the full file anyway (needs ~16GB+ free RAM), set "
                "environment variable RIDE_FARE_FORCE_FULL=1. To choose your "
                "own size, set RIDE_FARE_SAMPLE_FRAC=<0-1> or RIDE_FARE_NROWS=<N>."
            )

    # ─────────────────────────────────────────────────────────
    # 1. LOAD & VALIDATE DATA
    # ─────────────────────────────────────────────────────────
    logger.info("Loading data from: %s", data_path)
    df = load_raw_data(data_path, nrows=nrows, sample_frac=sample_frac)
    df = validate_input_data(df)

    # ─────────────────────────────────────────────────────────
    # 2. FEATURE ENGINEERING
    # ─────────────────────────────────────────────────────────
    df = add_engineered_features(df)

    # ─────────────────────────────────────────────────────────
    # 3. FEATURE TYPE DETECTION
    # ─────────────────────────────────────────────────────────
    ord_cols, cont_cols, bin_cols = detect_feature_types(df, threshold=10)

    X = df.drop(columns=[TARGET_COL, "pickup_datetime"])
    y = df[TARGET_COL]

    logger.info("Dataset  |  shape=%s  |  fare_mean=%.2f", df.shape, y.mean())

    del df
    gc.collect()

    # ─────────────────────────────────────────────────────────
    # 4. TRAIN / TEST / CALIBRATION SPLIT
    # ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    del X, y
    gc.collect()

    X_train_fit, X_cal, y_train_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )
    del X_train, y_train
    gc.collect()

    logger.info("Split  |  train_fit=%d  cal=%d  test=%d",
                len(X_train_fit), len(X_cal), len(X_test))

    # ─────────────────────────────────────────────────────────
    # 5. LEAKAGE DETECTION
    # ─────────────────────────────────────────────────────────
    leak_warnings = detect_leakage(X_train_fit, y_train_fit)
    if not leak_warnings:
        logger.info("No obvious leakage detected")

    # ─────────────────────────────────────────────────────────
    # 6. BUILD PREPROCESSORS
    # ─────────────────────────────────────────────────────────
    pre_scaled, pre_unscaled, cat_indices, feature_order = build_preprocessors(
        ord_cols, cont_cols, bin_cols, X_train_fit
    )

    k_safe = safe_k(SELECT_K, pre_scaled, X_train_fit)

    # One-time feature selection (see model_tuning.py — this used to happen
    # inside every model's CV fold and was the actual cause of multi-hour
    # tuning times on large datasets). Computed ONCE — not once per
    # preprocessor — because pre_scaled and pre_unscaled emit the exact
    # same feature_order (only the values are clipped/scaled differently),
    # so mutual_info_regression's top-K ranking is essentially the same
    # either way. Reusing one mask also makes the scaled-vs-unscaled model
    # comparison fair: every model is judged on the identical feature set.
    # (Previously this ran twice — ~8 min each on a 1M-row train_fit set —
    # for no real benefit.)
    logger.info("Computing feature selection once (shared by scaled + unscaled models) ...")
    feature_mask = select_features_once(pre_unscaled, X_train_fit, y_train_fit, k_safe)

    # ─────────────────────────────────────────────────────────
    # 6b. AUTO-SCALE SEARCH BUDGET ON LARGE train_fit SETS
    # ─────────────────────────────────────────────────────────
    # Mirrors the large-file sample_frac safety net in run_training(): if
    # the caller didn't explicitly set RIDE_FARE_SEARCH_ITERS / RIDE_FARE_
    # CV_FOLDS, the default budget (20 iters x 5 folds) applied to a
    # multi-million-row train_fit set can take many HOURS per model even
    # with every other fix in place (RandomForest alone: up to 12
    # grid-capped candidates x 5 folds = 60 full fits, each on ~1M+ rows).
    # Scale the budget down automatically based on train_fit size instead
    # of silently letting that happen.
    n_iter_override, cv_folds_override = None, None
    if not os.getenv("RIDE_FARE_SEARCH_ITERS") and not os.getenv("RIDE_FARE_CV_FOLDS"):
        train_fit_rows = len(X_train_fit)
        if train_fit_rows > 2_000_000:
            n_iter_override, cv_folds_override = 3, 2
        elif train_fit_rows > 500_000:
            n_iter_override, cv_folds_override = 5, 3
        elif train_fit_rows > 150_000:
            n_iter_override, cv_folds_override = 8, 3

        if n_iter_override is not None:
            logger.warning(
                "train_fit has %d rows and RIDE_FARE_SEARCH_ITERS/RIDE_FARE_CV_FOLDS "
                "were not set — auto-scaling the search budget to n_iter=%d, cv_folds=%d "
                "for this run to keep tuning time reasonable. Set RIDE_FARE_SEARCH_ITERS "
                "and RIDE_FARE_CV_FOLDS explicitly to override this.",
                train_fit_rows, n_iter_override, cv_folds_override
            )

    # ─────────────────────────────────────────────────────────
    # 7. TUNE SCALED MODELS (linear / distance)
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Tuning scaled models ...")
    scaled_pipelines, scaled_cv_stats = tune_models(scaled_models, pre_scaled, X_train_fit, y_train_fit,
                                                      feature_mask=feature_mask,
                                                      n_iter_override=n_iter_override,
                                                      cv_folds_override=cv_folds_override)

    # ─────────────────────────────────────────────────────────
    # 8. TUNE UNSCALED MODELS (trees / boosting)
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Tuning unscaled models ...")
    unscaled_pipelines, unscaled_cv_stats = tune_models(unscaled_models, pre_unscaled, X_train_fit, y_train_fit,
                                                          feature_mask=feature_mask,
                                                          n_iter_override=n_iter_override,
                                                          cv_folds_override=cv_folds_override)

    # ─────────────────────────────────────────────────────────
    # 9. TRAIN NEURAL NETWORK (separately)
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    mlp_pipe = train_mlp_pipeline(X_train_fit, y_train_fit, pre_scaled)

    all_pipelines = {**scaled_pipelines, **unscaled_pipelines, "NeuralNet": mlp_pipe}

    # cv_stats already computed once by RandomizedSearchCV during tuning —
    # reused below so evaluate_models() doesn't re-fit every model a
    # second time just to recompute the same CV RMSE (NeuralNet has no
    # entry here since it wasn't tuned via RandomizedSearchCV; it falls
    # back to evaluate_models()'s own cross_val_score for that one model).
    all_cv_stats = {**scaled_cv_stats, **unscaled_cv_stats}

    # ─────────────────────────────────────────────────────────
    # 10. EVALUATE ALL MODELS
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Evaluating all models ...")
    summary = evaluate_models(all_pipelines, X_train_fit, y_train_fit, X_test, y_test,
                               n_features_for_adj_r2=k_safe, precomputed_cv=all_cv_stats)

    print("\n" + "=" * 60)
    print("ALL MODELS SUMMARY")
    print("=" * 60)
    print(summary.to_string())

    # ─────────────────────────────────────────────────────────
    # 11. SELECT BEST MODEL
    # ─────────────────────────────────────────────────────────
    selected_name, selected_pipe = select_best_model(
        summary, all_pipelines, scaled_pipelines, unscaled_pipelines
    )

    # ─────────────────────────────────────────────────────────
    # 12. DETAILED EVALUATION — SELECTED MODEL  (shared tail, also
    #     used by run_final_model_training() below)
    # ─────────────────────────────────────────────────────────
    return _finalize_selected_model(
        selected_name, selected_pipe,
        X_train_fit, y_train_fit, X_cal, y_cal, X_test, y_test,
        cont_cols, feature_order, cat_indices, k_safe
    )


def _finalize_selected_model(
    selected_name, selected_pipe,
    X_train_fit, y_train_fit, X_cal, y_cal, X_test, y_test,
    cont_cols, feature_order, cat_indices, k_safe
):
    """
    Shared tail for both run_training() (after comparing many models) and
    run_final_model_training() (after fitting just the one winning model
    on more data) — metrics, plots, pricing/cost evaluation, calibration,
    SHAP, PSI, model card, saving, Champion-vs-Challenger, MLflow.
    """
    y_pred_test = selected_pipe.predict(X_test)

    # Generated once here and reused for the model, card, AND calibration
    # filenames below — previously calibration used a fixed, unversioned
    # name (calibration_{model}_v1.joblib), so a REJECTED challenger's
    # calibration would silently overwrite the file the current PROMOTED
    # champion depends on (different runs of even the same model type can
    # have meaningfully different residual/calibration behavior). Giving
    # every run's calibration file its own timestamp — matching the model
    # and card files — means only a run that's actually promoted has its
    # calibration referenced by the registry, and it can never be
    # clobbered by an unrelated rejected run.
    run_ts = time.strftime("%Y%m%d_%H%M%S")

    test_metrics = {
        "test_rmse":        rmse(y_test, y_pred_test),
        "test_mae":         mae(y_test, y_pred_test),
        "test_mape":        mape(y_test, y_pred_test),
        "test_r2":          r2(y_test, y_pred_test),
        "adjusted_r2":      adjusted_r2(y_test, y_pred_test, k_safe),
        "within_15pct_acc": within_tolerance_accuracy(y_test, y_pred_test),
    }

    print("\n" + "=" * 60)
    print(f"BEST MODEL: {selected_name}")
    print("=" * 60)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── Predicted vs actual scatter + residuals plot ──────────
    os.makedirs(os.path.join("docs", "plots"), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(y_test, y_pred_test, alpha=0.3, s=8)
    lims = [0, max(y_test.max(), y_pred_test.max())]
    ax[0].plot(lims, lims, "r--", linewidth=1)
    ax[0].set_xlabel("Actual fare ($)")
    ax[0].set_ylabel("Predicted fare ($)")
    ax[0].set_title(f"{selected_name} — Predicted vs Actual")

    residuals = y_pred_test - y_test.values
    ax[1].hist(residuals, bins=50)
    ax[1].axvline(0, color="r", linestyle="--")
    ax[1].set_title("Residual Distribution")
    ax[1].set_xlabel("Predicted - Actual ($)")

    plt.tight_layout()
    plt.savefig(os.path.join("docs", "plots", "predicted_vs_actual.png"))
    plt.close()

    # ─────────────────────────────────────────────────────────
    # 13. PRICING ENGINE FLAGS
    # ─────────────────────────────────────────────────────────
    flags = pricing_engine(X_test, y_pred_test)
    flag_counts = pd.Series(flags).value_counts()

    print("\nPRICING ENGINE FLAGS")
    print(flag_counts)

    # ─────────────────────────────────────────────────────────
    # 14. COST-SENSITIVE EVALUATION
    # ─────────────────────────────────────────────────────────
    cost_result = cost_sensitive_evaluation(y_test, y_pred_test)
    print("\nCOST EVALUATION")
    for k, v in cost_result.items():
        print(f"  {k}: {v}")

    # ─────────────────────────────────────────────────────────
    # 15. RESIDUAL / CONFORMAL CALIBRATION
    # ─────────────────────────────────────────────────────────
    calibration = calibrate_with_holdout(selected_pipe, X_cal, y_cal)
    calibration_path = ""
    if calibration is not None:
        calibration_path = os.path.join(MODEL_DIR, f"calibration_{selected_name}_v1_{run_ts}.joblib")
        joblib.dump(calibration, calibration_path)

    # ─────────────────────────────────────────────────────────
    # 16. REPEATED CV STABILITY CHECK
    # ─────────────────────────────────────────────────────────
    # Purely diagnostic (logged for reference — does not affect model
    # selection or promotion), so on large datasets this is subsampled
    # the same way select_features_once() is: this took 75 minutes on a
    # 13.7M-row train_fit set (15 full refits) for a number that's just
    # printed to the log, not used anywhere downstream.
    try:
        MAX_ROWS_FOR_STABILITY_CHECK = 200_000
        if len(X_train_fit) > MAX_ROWS_FOR_STABILITY_CHECK:
            logger.info(
                "train_fit has %d rows — subsampling to %d rows for the repeated-CV "
                "stability check only (this is a diagnostic number, not used for model "
                "selection, so it doesn't need the full dataset).",
                len(X_train_fit), MAX_ROWS_FOR_STABILITY_CHECK
            )
            X_stability = X_train_fit.sample(n=MAX_ROWS_FOR_STABILITY_CHECK, random_state=RANDOM_STATE)
            y_stability = y_train_fit.loc[X_stability.index]
        else:
            X_stability, y_stability = X_train_fit, y_train_fit

        rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
        rep_scores = cross_val_score(
            selected_pipe, X_stability, y_stability,
            scoring="neg_root_mean_squared_error", cv=rkf, n_jobs=1
        )
        logger.info("Repeated CV  |  mean_rmse=%.4f  std=%.4f", -rep_scores.mean(), rep_scores.std())
    except Exception as e:
        logger.warning("Repeated CV failed: %s", e)

    # ─────────────────────────────────────────────────────────
    # 17. FEATURE IMPORTANCE
    # ─────────────────────────────────────────────────────────
    fi = compute_feature_importance(selected_pipe, X_train_fit, y_train_fit)
    if fi is not None:
        print("\nTOP FEATURE IMPORTANCES")
        print(fi.head(10))

    # ─────────────────────────────────────────────────────────
    # 18. SHAP EXPLAINABILITY
    # ─────────────────────────────────────────────────────────
    shap_result = compute_shap(selected_pipe, X_train_fit, X_test.head(200))

    # ─────────────────────────────────────────────────────────
    # 19. PSI — FEATURE DRIFT (train vs test)
    # ─────────────────────────────────────────────────────────
    psi_scores = {}
    for col in cont_cols:
        if col in X_train_fit.columns and col in X_test.columns:
            psi_scores[col] = psi(X_train_fit[col].values, X_test[col].values)

    psi_df = pd.Series(psi_scores).sort_values(ascending=False)
    psi_df.reset_index().rename(
        columns={"index": "feature", 0: "drift_score"}
    ).to_csv(os.path.join(MODEL_DIR, "feature_drift_report.csv"), index=False)

    print("\nTOP PSI (train vs test)")
    print(psi_df.head(10))

    # ─────────────────────────────────────────────────────────
    # 20. SAVE MONITOR SCORES
    # ─────────────────────────────────────────────────────────
    # Capped at MAX_MONITOR_ROWS: on the full-scale final-model run
    # (33M-row dataset, 6.4M-row test set) this file was landing at
    # ~305 MB — well past GitHub's 100 MB per-file limit, which
    # silently rejected the ENTIRE push (nothing partial gets
    # accepted). The dashboard only ever computes summary statistics
    # (mean fare, MAPE, error histogram) and a 500-point scatter
    # sample from this file — a bounded random sample is exactly as
    # statistically representative for those purposes as the full
    # test set, at a fraction of the size.
    MAX_MONITOR_ROWS = 100_000
    monitor_df = pd.DataFrame({
        "predicted_fare": y_pred_test,
        "actual_fare":    y_test.values,
        "abs_error":      np.abs(y_pred_test - y_test.values),
        "pricing_flag":   flags,
    })
    if len(monitor_df) > MAX_MONITOR_ROWS:
        logger.info(
            "Test set has %d rows — sampling %d rows for monitor_scores.csv "
            "(keeps the file well under GitHub's 100 MB file limit; the "
            "dashboard only needs summary stats and a plotting sample from it).",
            len(monitor_df), MAX_MONITOR_ROWS
        )
        monitor_df = monitor_df.sample(n=MAX_MONITOR_ROWS, random_state=RANDOM_STATE)
    monitor_df.to_csv(os.path.join(MODEL_DIR, "monitor_scores.csv"), index=False)
    logger.info("Monitor scores saved  |  %d rows  |  %.1f MB", len(monitor_df),
                os.path.getsize(os.path.join(MODEL_DIR, "monitor_scores.csv")) / 1e6)

    # ─────────────────────────────────────────────────────────
    # 21. MODEL CARD
    # ─────────────────────────────────────────────────────────
    model_card = build_model_card(
        selected_name    = selected_name,
        train_fit_size   = int(len(X_train_fit)),
        cal_size         = int(len(X_cal)),
        test_size        = int(len(X_test)),
        fare_mean_train  = float(y_train_fit.mean()),
        metrics          = test_metrics,
        calibration_info = calibration,
        cost_result      = cost_result,
        flag_counts      = flag_counts.to_dict(),
        feature_order    = feature_order,
        categorical_indices = cat_indices,
        selector_k       = k_safe,
        fi_dict          = fi.head(20).to_dict() if fi is not None else None,
        shap_dict        = shap_result.get("shap_top", {}) if shap_result is not None else None,
    )

    card_path = save_model_card(model_card, MODEL_DIR, selected_name, version=f"v1_{run_ts}")

    # ─────────────────────────────────────────────────────────
    # 22. SAVE MODEL
    # ─────────────────────────────────────────────────────────
    model_path = save_challenger_artifact(selected_name, selected_pipe, version=f"v1_{run_ts}")

    # ─────────────────────────────────────────────────────────
    # 23. CHALLENGER MODEL COMPARISON
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Running Champion vs Challenger comparison ...")

    train_r2  = r2(y_train_fit, selected_pipe.predict(X_train_fit))
    test_r2   = test_metrics["test_r2"]
    gap       = abs(train_r2 - test_r2)

    challenger_result = run_challenger_comparison(
        challenger_name             = selected_name,
        challenger_rmse             = test_metrics["test_rmse"],
        challenger_r2               = test_r2,
        challenger_gap              = gap,
        challenger_model_path       = model_path,
        challenger_card_path        = card_path,
        challenger_calibration_path = calibration_path,
    )

    print("\n" + "=" * 60)
    print(f"CHALLENGER RESULT: {challenger_result['decision']}")
    print(f"Reason: {challenger_result['reason']}")
    print("=" * 60)

    # ─────────────────────────────────────────────────────────
    # 24. MLFLOW
    # ─────────────────────────────────────────────────────────
    run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{selected_name}"
    mlflow_log_run(run_name, selected_name, selected_pipe, model_card, X_train_sample=X_train_fit)

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE  |  Best model: {selected_name}")
    print(f"Challenger status : {challenger_result['decision']}")
    print("=" * 60)

    return selected_name, selected_pipe, model_card


# ============================================================
# FINAL MODEL TRAINING — refit only the current champion's
# architecture on a larger/full dataset (no multi-model search)
# ============================================================

def _find_model_family(regressor):
    """Returns ('scaled', family_dict) or ('unscaled', family_dict) by
    matching the regressor's class name against the known model grids."""
    reg_class_name = type(regressor).__name__
    for name, (candidate, _) in scaled_models.items():
        if type(candidate).__name__ == reg_class_name:
            return "scaled", name
    for name, (candidate, _) in unscaled_models.items():
        if type(candidate).__name__ == reg_class_name:
            return "unscaled", name
    # NeuralNet (MLPRegressor) isn't in either grid — treat as scaled
    # since it was trained through the scaled preprocessor.
    if reg_class_name == "MLPRegressor":
        return "scaled", "NeuralNet"
    raise ValueError(f"Could not determine model family for regressor class '{reg_class_name}'")


def run_final_model_training(data_path: str = DATA_PATH, nrows: int = None, sample_frac: float = None):
    """
    Refits ONLY the current champion's model architecture (same
    regressor class + same hyperparameters) on a larger dataset — no
    RandomizedSearchCV, no comparing multiple models. Use this after a
    full run_training() comparison has already told you which model
    wins; this is for scaling that one winner up to more data cheaply.

    The preprocessor and feature-selection mask are rebuilt fresh for
    the new (larger) dataset rather than reused as-is from the small
    sample — the skewed-vs-non-skewed column grouping inside
    build_preprocessors() is itself data-dependent (based on each
    column's measured skew()), so blindly reusing structures fit on a
    much smaller sample could silently misalign with a larger dataset's
    actual distribution. Only the REGRESSOR's hyperparameters (the
    expensive thing to search for) are carried over via clone().
    """
    from sklearn.base import clone
    from src.model_loader import load_latest_model

    logger.info("=" * 60)
    logger.info("FINAL MODEL TRAINING — single-model refit, no search")
    logger.info("=" * 60)

    champion_pipe, _ = load_latest_model()
    champion_regressor = champion_pipe.named_steps["regressor"]
    family, model_name = _find_model_family(champion_regressor)

    logger.info("Current champion: %s (%s family)  |  hyperparameters: %s",
                model_name, family, champion_regressor.get_params())

    # ── 1. Resolve nrows / sample_frac (same large-file safety as run_training) ──
    if nrows is None and os.getenv("RIDE_FARE_NROWS"):
        nrows = int(os.getenv("RIDE_FARE_NROWS"))
    if sample_frac is None and os.getenv("RIDE_FARE_SAMPLE_FRAC"):
        sample_frac = float(os.getenv("RIDE_FARE_SAMPLE_FRAC"))
    force_full = os.getenv("RIDE_FARE_FORCE_FULL", "0") == "1"

    if nrows is None and sample_frac is None and not force_full:
        try:
            with open(data_path, "rb") as f:
                approx_rows = sum(1 for _ in f) - 1
        except Exception:
            approx_rows = None
        if approx_rows is not None and approx_rows > LARGE_FILE_ROW_THRESHOLD:
            sample_frac = DEFAULT_SAMPLE_FRAC_FOR_LARGE_FILES
            logger.warning(
                "Detected a large file (~%d rows) and no explicit nrows/sample_frac/"
                "FORCE_FULL was set — auto-applying sample_frac=%.2f. For a genuine "
                "'final' model you'll usually want MORE data than your comparison "
                "run used, e.g. set RIDE_FARE_SAMPLE_FRAC=0.3 or RIDE_FARE_FORCE_FULL=1.",
                approx_rows, sample_frac
            )

    # ── 2. Load, validate, engineer features ──────────────────────────
    logger.info("Loading data from: %s", data_path)
    df = load_raw_data(data_path, nrows=nrows, sample_frac=sample_frac)
    df = validate_input_data(df)
    df = add_engineered_features(df)

    ord_cols, cont_cols, bin_cols = detect_feature_types(df, threshold=10)
    X = df.drop(columns=[TARGET_COL, "pickup_datetime"])
    y = df[TARGET_COL]

    logger.info("Dataset  |  shape=%s  |  fare_mean=%.2f", df.shape, y.mean())

    del df
    gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # X/y (the full pre-split feature set) are no longer needed once
    # X_train/X_test exist — freeing them here matters at full-dataset
    # scale: without this, X, X_train, X_train_fit, X_cal and X_test
    # would all stay alive simultaneously, holding roughly 1.8x the
    # dataset's memory footprint at peak for no reason.
    del X, y
    gc.collect()

    X_train_fit, X_cal, y_train_fit, y_cal = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE)

    del X_train, y_train
    gc.collect()

    logger.info("Split  |  train_fit=%d  cal=%d  test=%d", len(X_train_fit), len(X_cal), len(X_test))

    # ── 3. Fresh preprocessor + feature mask for the new dataset ──────
    pre_scaled, pre_unscaled, cat_indices, feature_order = build_preprocessors(
        ord_cols, cont_cols, bin_cols, X_train_fit
    )
    preprocessor = pre_scaled if family == "scaled" else pre_unscaled
    k_safe = safe_k(SELECT_K, pre_scaled, X_train_fit)
    feature_mask = select_features_once(preprocessor, X_train_fit, y_train_fit, k_safe)

    # ── 4. Build + fit the single pipeline — champion's hyperparameters, fresh fit ──
    from sklearn.pipeline import Pipeline
    from src.model_tuning import FixedIndexSelector

    fresh_regressor = clone(champion_regressor)  # same class + hyperparams, unfitted
    selected_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("selector",     FixedIndexSelector(mask=feature_mask)),
        ("regressor",    fresh_regressor),
    ])

    logger.info("Fitting %s on %d rows (no search — single fit) ...", model_name, len(X_train_fit))
    t0 = time.time()
    selected_pipe.fit(X_train_fit, y_train_fit)
    logger.info("Fit done in %.1fs", time.time() - t0)

    # ── 5. Same evaluation/calibration/model-card/promotion tail as run_training() ──
    return _finalize_selected_model(
        model_name, selected_pipe,
        X_train_fit, y_train_fit, X_cal, y_cal, X_test, y_test,
        cont_cols, feature_order, cat_indices, k_safe
    )


if __name__ == "__main__":
    start = time.time()
    name, model, card = run_training()
    logger.info("Finished in %.1fs", time.time() - start)
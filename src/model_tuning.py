# ============================================================
# MODEL TUNING — Ride Fare Price Prediction ML System
# ============================================================

import numpy as np
import logging

from typing import Dict, List, Tuple

from sklearn.base              import BaseEstimator, TransformerMixin, clone
from sklearn.compose           import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection   import RandomizedSearchCV, KFold
from sklearn.pipeline          import Pipeline

from sklearn.linear_model  import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.neighbors     import KNeighborsRegressor
from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import (RandomForestRegressor, GradientBoostingRegressor,
                                   ExtraTreesRegressor, AdaBoostRegressor)
from xgboost               import XGBRegressor

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

from src.config        import RANDOM_STATE, N_JOBS, CV_FOLDS, RANDOM_SEARCH_ITERS, SELECT_K
from src.preprocessing import safe_k

logger = logging.getLogger(__name__)


# ============================================================
# HELPER — compute safe n_iter for RandomizedSearchCV
# ============================================================

def _compute_n_iter(param_dist: dict, budget: int) -> int:
    if not param_dist:
        return 1
    prod = 1
    for v in param_dist.values():
        try:
            prod *= len(v)
        except TypeError:
            prod *= budget
    return min(budget, max(1, prod))


# ============================================================
# FIXED FEATURE SELECTOR + ONE-TIME SELECTION
# ============================================================
# WHY THIS EXISTS:
#   The straightforward approach — put SelectKBest(mutual_info_regression)
#   *inside* each model's Pipeline — looks clean but is a serious
#   performance trap at scale. mutual_info_regression is a k-NN-based
#   estimator, and because it lives inside the pipeline, sklearn refits
#   it from scratch on EVERY CV fold of EVERY search iteration (e.g.
#   20 iters x 5 folds = up to 100 full recomputations, PER MODEL). On
#   ~1M rows this made even plain LinearRegression take ~20 minutes and
#   compounded into many hours across 13 models.
#
#   Fix: compute mutual information ONCE per preprocessor (scaled /
#   unscaled) on the full training-fit set, get a fixed boolean feature
#   mask, and reuse that exact mask (a near-free array slice) inside
#   every model's pipeline instead of recomputing it per fold.
# ============================================================

class FixedIndexSelector(BaseEstimator, TransformerMixin):
    """Applies a pre-computed boolean feature mask — no refitting."""

    def __init__(self, mask: np.ndarray = None):
        self.mask = mask

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.mask]

    def get_support(self):
        # Mirrors SelectKBest's API so evaluation.py's feature-name
        # extraction (which calls .get_support()) keeps working unchanged.
        return self.mask

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)[self.mask]
        return np.array([f"f{i}" for i in range(int(self.mask.sum()))], dtype=object)


def select_features_once(preprocessor: ColumnTransformer, X_train, y_train, k: int,
                          max_rows_for_selection: int = 100_000) -> np.ndarray:
    """
    Fits `preprocessor` + SelectKBest(mutual_info_regression, k) exactly
    ONCE and returns the boolean support mask. Call this before
    tune_models() and pass the mask in — do not call this per model /
    per fold.

    mutual_info_regression is a k-NN based estimator — its cost scales
    badly with row count (on a ~5M-row train_fit this step alone took
    over an hour). Ranking the top-K informative features doesn't need
    millions of rows to be reliable, so when X_train has more than
    `max_rows_for_selection` rows, a random subsample of that size is
    used for this step only (the actual model fitting later still uses
    the full X_train — this only affects which K columns get selected).
    """
    if len(X_train) > max_rows_for_selection:
        logger.info(
            "train_fit has %d rows — subsampling to %d rows for feature "
            "selection only (mutual_info_regression doesn't need the full "
            "set to reliably rank features, and doing so is what made this "
            "step slow on large datasets). Model tuning below still uses "
            "the full train_fit set.",
            len(X_train), max_rows_for_selection
        )
        X_sel = X_train.sample(n=max_rows_for_selection, random_state=RANDOM_STATE)
        y_sel = y_train.loc[X_sel.index]
    else:
        X_sel, y_sel = X_train, y_train

    preprocessor = clone(preprocessor)
    X_pre = preprocessor.fit_transform(X_sel)
    if hasattr(X_pre, "toarray"):
        X_pre = X_pre.toarray()

    selector = SelectKBest(mutual_info_regression, k=k)
    selector.fit(X_pre, y_sel)
    mask = selector.get_support()

    logger.info("One-time feature selection done  |  selected %d/%d features",
                int(mask.sum()), len(mask))

    return mask


# ============================================================
# MODEL GRIDS
# ============================================================
# Full 14-model roster kept intentionally (incl. KNN, DecisionTree,
# AdaBoost) for a complete comparison table — useful for interviews/
# portfolio review even though some of these are consistently beaten
# by the ensembles on this dataset (see fare_models/model_experiment_
# results.csv after training). The one-time feature-selection mask
# (see select_features_once() above) removes the old per-fold
# mutual_info_regression cost. Parallelism split: RandomizedSearchCV
# runs candidates sequentially (n_jobs=1, see tune_models below) while
# each internally-parallel estimator (RF/ExtraTrees/XGBoost/LightGBM/
# CatBoost/KNN) uses n_jobs=N_JOBS — see the n_jobs=1 comment in
# tune_models() for why this split (not the reverse) is correct here.

# Linear / distance models  →  need scaled input
scaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "LinearRegression": (
        LinearRegression(), {}
    ),

    "Ridge": (
        Ridge(random_state=RANDOM_STATE),
        {"regressor__alpha": [0.1, 1.0, 10.0, 50.0]}
    ),

    "Lasso": (
        Lasso(random_state=RANDOM_STATE, max_iter=5000),
        {"regressor__alpha": [0.001, 0.01, 0.1, 1.0]}
    ),

    "ElasticNet": (
        ElasticNet(random_state=RANDOM_STATE, max_iter=5000),
        {
            "regressor__alpha":    [0.01, 0.1, 1.0],
            "regressor__l1_ratio": [0.2, 0.5, 0.8],
        }
    ),

    "SGD": (
        SGDRegressor(random_state=RANDOM_STATE, max_iter=2000, tol=1e-3),
        {
            "regressor__alpha":    [1e-4, 1e-3, 1e-2],
            "regressor__penalty":  ["l2", "elasticnet"],
        }
    ),

    "KNN": (
        KNeighborsRegressor(n_jobs=N_JOBS),
        {
            "regressor__n_neighbors": [5, 9, 15],
            "regressor__weights":     ["uniform", "distance"],
        }
    ),
}

# Tree-based models  →  work on raw / clipped input
unscaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "DecisionTree": (
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        {
            "regressor__max_depth":        [5, 10, 20, None],
            "regressor__min_samples_leaf": [1, 2, 4],
        }
    ),

    "RandomForest": (
        # max_samples=0.3 is the key lever here: by default RandomForest
        # trains EVERY tree on a bootstrap sample the same size as the
        # full training set (~700K-1M rows on real data). That's the
        # actual reason a single fit takes so long — not n_jobs, not CV
        # folds. Capping each tree to 30% of rows cuts per-tree cost
        # roughly 3x with only a small, usually negligible, accuracy
        # trade-off (this is standard practice for RF at this scale —
        # see sklearn's own docs on `max_samples` for large datasets).
        RandomForestRegressor(n_jobs=N_JOBS, random_state=RANDOM_STATE, max_samples=0.3),
        {
            "regressor__n_estimators":     [100, 200],
            "regressor__max_depth":        [10, 15, 20],
            "regressor__min_samples_leaf": [2, 4],
        }
    ),

    "ExtraTrees": (
        # bootstrap=True is required for max_samples to take effect —
        # ExtraTrees defaults to bootstrap=False (uses the full dataset
        # for every tree, which is even slower to build than RandomForest
        # since ExtraTrees evaluates more random split candidates).
        ExtraTreesRegressor(n_jobs=N_JOBS, random_state=RANDOM_STATE,
                             bootstrap=True, max_samples=0.3),
        {
            "regressor__n_estimators": [100, 200],
            "regressor__max_depth":    [10, 15, 20],
        }
    ),

    "GradientBoosting": (
        # No n_jobs at all — sklearn's GradientBoostingRegressor builds
        # trees strictly sequentially (each tree corrects the previous
        # one's residuals), so it can't parallelize across cores no
        # matter what. subsample<1.0 (stochastic gradient boosting) is
        # the only real lever here; XGBoost/LightGBM/CatBoost below use
        # the same boosting idea but with histogram-based splits that
        # are dramatically faster on CPU — if you need this comparison
        # to run faster, those three are the reliable/fast members of
        # the boosting family, this one is kept mainly for the
        # side-by-side "classic vs modern boosting" comparison.
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {
            "regressor__n_estimators":  [80, 150],
            "regressor__learning_rate": [0.05, 0.1],
            "regressor__max_depth":     [3, 4],
            "regressor__subsample":     [0.5, 0.8],
        }
    ),

    "XGBoost": (
        XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror", n_jobs=N_JOBS),
        {
            "regressor__n_estimators":  [200, 400],
            "regressor__learning_rate": [0.03, 0.05, 0.1],
            "regressor__max_depth":     [4, 6, 8],
            "regressor__subsample":     [0.8, 1.0],
        }
    ),

    "AdaBoost": (
        AdaBoostRegressor(random_state=RANDOM_STATE),
        {
            "regressor__n_estimators":  [50, 100, 200],
            "regressor__learning_rate": [0.01, 0.1, 1.0],
        }
    ),
}

if LGBMRegressor is not None:
    unscaled_models["LightGBM"] = (
        LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, n_jobs=N_JOBS),
        {
            "regressor__n_estimators":  [200, 400],
            "regressor__learning_rate": [0.03, 0.05, 0.1],
            "regressor__max_depth":     [-1, 8, 12],
            "regressor__num_leaves":    [31, 63],
        }
    )

if CatBoostRegressor is not None:
    unscaled_models["CatBoost"] = (
        CatBoostRegressor(verbose=0, random_state=RANDOM_STATE, thread_count=N_JOBS),
        {
            "regressor__iterations":    [200, 400],
            "regressor__learning_rate": [0.03, 0.1],
            "regressor__depth":         [4, 6, 8],
        }
    )


# ============================================================
# TUNE MODELS
# ============================================================

def tune_models(
    models:         Dict[str, Tuple[BaseEstimator, dict]],
    preprocessor:   ColumnTransformer,
    X_train:        "pd.DataFrame",
    y_train:        "pd.Series",
    feature_mask:   np.ndarray,
    n_iter_override: int = None,
    cv_folds_override: int = None,
) -> Tuple[Dict[str, Pipeline], Dict[str, Tuple[float, float]]]:
    """
    For each model:
      preprocessor → FixedIndexSelector(feature_mask) → regressor
    Tuned with RandomizedSearchCV (scoring = neg_root_mean_squared_error).
    `feature_mask` must come from a single prior call to
    select_features_once() — it is applied as-is, never recomputed here.

    n_iter_override / cv_folds_override: when provided, used instead of
    config.RANDOM_SEARCH_ITERS / config.CV_FOLDS for every model in this
    call. training_pipeline.py uses these to auto-scale the search budget
    down on large datasets (see run_training) — without this, the default
    budget (20 iters x 5 folds) applied to a multi-million-row train_fit
    set can take many hours per model even with every other fix in place.

    Returns (final_pipelines, cv_stats):
      final_pipelines — {model_name: best_pipeline}
      cv_stats        — {model_name: (cv_mean_rmse, cv_std_rmse)}, taken
                         directly from RandomizedSearchCV's own cv_results_
                         for the winning candidate. Passing this through to
                         evaluate_models() avoids re-fitting every model a
                         second time via a fresh cross_val_score() call —
                         on large datasets that redundant re-fit was the
                         single biggest remaining time sink (~28 minutes
                         for RandomForest alone on a ~1M-row train_fit set).
    """

    final_pipelines: Dict[str, Pipeline] = {}
    cv_stats:        Dict[str, Tuple[float, float]] = {}

    effective_cv_folds = cv_folds_override if cv_folds_override is not None else CV_FOLDS

    logger.info("Tuning %d models with a fixed %d-feature mask (selected once, reused for every fold)",
                len(models), int(feature_mask.sum()))

    for name, (reg, param_dist) in models.items():

        logger.info("Tuning: %s", name)

        steps = [
            ("preprocessor", preprocessor),
            ("selector",     FixedIndexSelector(mask=feature_mask)),
            ("regressor",    reg),
        ]
        pipe = Pipeline(steps)

        budget = n_iter_override if n_iter_override is not None else RANDOM_SEARCH_ITERS
        n_iter = _compute_n_iter(param_dist, budget)

        search = RandomizedSearchCV(
            pipe,
            param_distributions = param_dist,
            n_iter              = n_iter,
            scoring             = "neg_root_mean_squared_error",
            cv                  = KFold(effective_cv_folds, shuffle=True, random_state=RANDOM_STATE),
            # n_jobs=1 HERE is intentional, not a typo: RandomizedSearchCV's
            # parallelism forks full OS processes (joblib's default 'loky'
            # backend), each pickling its own copy of the training data —
            # very expensive on Windows (spawn-based, no copy-on-write like
            # Linux fork). Combined with an estimator that ALSO parallelizes
            # internally (RandomForest/XGBoost/etc. via n_jobs=N_JOBS above),
            # this produces nested process explosion — that's what caused
            # both the earlier MemoryError crash and, once that nesting was
            # removed, single-threaded-per-process tree building that made
            # RandomForest hang for hours. The correct split for expensive,
            # internally-parallel estimators is: search sequential (n_jobs=1),
            # estimator parallel (n_jobs=N_JOBS) — the estimator's threading
            # backend shares memory and actually uses all cores per fit.
            n_jobs               = 1,
            random_state        = RANDOM_STATE,
            verbose             = 0
        )

        search.fit(X_train, y_train)

        cv_mean_rmse = float(-search.best_score_)
        cv_std_rmse  = float(search.cv_results_["std_test_score"][search.best_index_])

        logger.info("%s best params: %s  |  best CV RMSE=%.4f",
                    name, search.best_params_, cv_mean_rmse)

        final_pipelines[name] = search.best_estimator_
        cv_stats[name]        = (cv_mean_rmse, cv_std_rmse)

    return final_pipelines, cv_stats


# ============================================================
# NEURAL NETWORK — trained separately (no CV search)
# ============================================================

def train_mlp_pipeline(X_train, y_train, preprocessor):
    """
    MLPRegressor trained separately outside RandomizedSearchCV.
    Reason: MLP training time makes CV search impractical.
    """
    from sklearn.neural_network import MLPRegressor

    logger.info("Training Neural Network (MLPRegressor) ...")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor",    MLPRegressor(
            hidden_layer_sizes  = (128, 64),
            activation          = "relu",
            solver              = "adam",
            alpha               = 0.0001,
            batch_size          = 512,
            learning_rate       = "adaptive",
            max_iter            = 100,
            early_stopping      = True,
            validation_fraction = 0.1,
            n_iter_no_change    = 5,
            random_state        = RANDOM_STATE
        ))
    ])

    pipe.fit(X_train, y_train)

    logger.info("MLP training done")

    return pipe
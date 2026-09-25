# ============================================================
# PYTEST UNIT TESTS — Ride Fare Price Prediction ML System
# ============================================================
# Run with:  pytest tests/test_pipeline_core.py -v
#            pytest tests/ -v --cov=src --cov-report=term-missing
#
# Tests cover:
#   Clipper, build_preprocessors, detect_feature_types,
#   add_engineered_features, haversine_km, detect_leakage,
#   rmse/mae/mape/r2/psi, cost_sensitive_evaluation,
#   get_fare_band, score_trip, pricing_engine, config thresholds
# ============================================================

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import pytest

from src.preprocessing  import Clipper, build_preprocessors, safe_k
from src.evaluation     import select_best_model
from src.data_loader    import (detect_feature_types, add_engineered_features,
                                 haversine_km, validate_input_data)
from src.leakage_check  import detect_leakage
from src.metrics        import rmse, mae, mape, r2, adjusted_r2, within_tolerance_accuracy, psi, cost_sensitive_evaluation
from src.pricing_engine import get_fare_band, score_trip, pricing_engine
from src.config         import (
    FARE_BANDS, PSI_MODERATE, PSI_HIGH,
    MIN_RMSE_IMPROVEMENT, MIN_R2_THRESHOLD, MAX_GENERALIZATION_GAP,
    MIN_FARE, MAX_FARE,
)


# ============================================================
# CLIPPER TESTS
# ============================================================

class TestClipper:

    def test_fit_transform_shape(self):
        X = np.array([[1.0], [1000.0], [2.0], [3.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).shape == X.shape

    def test_clips_outliers(self):
        X = np.array([[1.0], [2.0], [3.0], [9999.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).max() < 9999.0

    def test_no_change_on_normal_data(self):
        X = np.array([[10.0], [11.0], [12.0], [13.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        out = clip.transform(X)
        assert np.allclose(out, X)

    def test_feature_names_out_passthrough(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        clip = Clipper()
        clip.fit(X)
        names = clip.get_feature_names_out(["a", "b"])
        assert list(names) == ["a", "b"]

    def test_feature_names_out_fallback(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        clip = Clipper()
        clip.fit(X)
        names = clip.get_feature_names_out(None)
        assert len(names) == 2

    def test_handles_1d_input(self):
        X = np.array([1.0, 2.0, 3.0, 100.0])
        clip = Clipper()
        clip.fit(X)
        out = clip.transform(X)
        assert out.shape == (4, 1)


# ============================================================
# PREPROCESSING PIPELINE TESTS
# ============================================================

class TestPreprocessing:

    @pytest.fixture
    def sample_df(self):
        rng = np.random.RandomState(0)
        return pd.DataFrame({
            "trip_distance_km": rng.exponential(3, 200),
            "pickup_hour":      rng.randint(0, 24, 200),
            "is_rush_hour":     rng.randint(0, 2, 200),
        })

    def test_build_preprocessors_returns_four_items(self, sample_df):
        result = build_preprocessors(
            ord_cols=["pickup_hour"], cont_cols=["trip_distance_km"],
            bin_cols=["is_rush_hour"], X_train=sample_df
        )
        assert len(result) == 4

    def test_preprocessor_scaled_transforms(self, sample_df):
        pre_scaled, pre_unscaled, cat_idx, feat_order = build_preprocessors(
            ["pickup_hour"], ["trip_distance_km"], ["is_rush_hour"], sample_df
        )
        out = pre_scaled.fit_transform(sample_df)
        assert out.shape[0] == len(sample_df)

    def test_safe_k_caps_at_available_features(self, sample_df):
        pre_scaled, _, _, _ = build_preprocessors(
            ["pickup_hour"], ["trip_distance_km"], ["is_rush_hour"], sample_df
        )
        k = safe_k(requested_k=999, preprocessor=pre_scaled, X_sample=sample_df)
        assert k <= 3


# ============================================================
# FEATURE ENGINEERING TESTS
# ============================================================

class TestFeatureEngineering:

    @pytest.fixture
    def raw_trip_df(self):
        return pd.DataFrame({
            "fare_amount":       [12.5, 45.0],
            "pickup_datetime":   pd.to_datetime(["2016-06-15T18:30:00Z", "2016-01-02T03:00:00Z"], utc=True),
            "pickup_longitude":  [-73.9855, -73.7781],
            "pickup_latitude":   [40.7580, 40.6413],
            "dropoff_longitude": [-73.9442, -73.9855],
            "dropoff_latitude":  [40.6782, 40.7580],
            "passenger_count":   [1, 2],
        })

    def test_haversine_zero_for_same_point(self):
        d = haversine_km(40.75, -73.98, 40.75, -73.98)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_haversine_known_distance(self):
        # Manhattan center to JFK is roughly 21-24 km
        d = haversine_km(40.7580, -73.9855, 40.6413, -73.7781)
        assert 15 < d < 30

    def test_add_engineered_features_creates_distance_col(self, raw_trip_df):
        df = add_engineered_features(raw_trip_df)
        assert "trip_distance_km" in df.columns
        assert (df["trip_distance_km"] > 0).all()

    def test_add_engineered_features_creates_time_cols(self, raw_trip_df):
        df = add_engineered_features(raw_trip_df)
        for col in ["pickup_hour", "pickup_dow", "is_weekend", "is_rush_hour", "hour_sin", "hour_cos"]:
            assert col in df.columns

    def test_airport_flag_detected(self, raw_trip_df):
        df = add_engineered_features(raw_trip_df)
        # Row 1: pickup near JFK
        assert df.loc[1, "near_jfk"] == 1

    def test_detect_feature_types_excludes_target(self):
        df = pd.DataFrame({
            "fare_amount": np.arange(20, dtype=float),
            "pickup_datetime": pd.to_datetime(["2020-01-01"] * 20),
            "trip_distance_km": np.linspace(0.5, 15.0, 20),
            "is_weekend": [0, 1] * 10,
        })
        ord_cols, cont_cols, bin_cols = detect_feature_types(df)
        assert "fare_amount" not in ord_cols + cont_cols + bin_cols
        assert "is_weekend" in bin_cols
        assert "trip_distance_km" in cont_cols


# ============================================================
# DATA VALIDATION TESTS
# ============================================================

class TestDataValidation:

    def test_missing_column_raises(self):
        df = pd.DataFrame({"fare_amount": [10.0]})
        with pytest.raises(ValueError):
            validate_input_data(df)

    def test_drops_out_of_bounds_fare(self):
        df = pd.DataFrame({
            "fare_amount":       [5.0] * 600 + [-10.0, 5000.0],
            "pickup_datetime":   pd.to_datetime(["2020-01-01"] * 602),
            "pickup_longitude":  [-73.98] * 602,
            "pickup_latitude":   [40.75] * 602,
            "dropoff_longitude": [-73.95] * 602,
            "dropoff_latitude":  [40.70] * 602,
            "passenger_count":   [1] * 602,
        })
        out = validate_input_data(df)
        assert out["fare_amount"].max() <= MAX_FARE
        assert out["fare_amount"].min() >= MIN_FARE


# ============================================================
# LEAKAGE DETECTION TESTS
# ============================================================

class TestLeakageCheck:

    def test_detects_identical_column(self):
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        X = pd.DataFrame({"leak_col": [1.0, 2.0, 3.0, 4.0], "ok_col": [5, 1, 9, 2]})
        warnings = detect_leakage(X, y)
        assert any("leak_col" in w for w in warnings)

    def test_no_leakage_on_clean_data(self):
        rng = np.random.RandomState(1)
        y = pd.Series(rng.normal(20, 5, 500))
        X = pd.DataFrame({"random_col": rng.normal(0, 1, 500)})
        warnings = detect_leakage(X, y)
        assert warnings == []

    def test_high_correlation_flagged(self):
        rng = np.random.RandomState(2)
        y = pd.Series(np.arange(500, dtype=float))
        X = pd.DataFrame({"almost_target": y * 1.0001 + rng.normal(0, 0.001, 500)})
        warnings = detect_leakage(X, y)
        assert len(warnings) >= 1


# ============================================================
# REGRESSION METRICS TESTS
# ============================================================

class TestMetrics:

    def test_rmse_zero_for_perfect_prediction(self):
        y = [1.0, 2.0, 3.0]
        assert rmse(y, y) == pytest.approx(0.0)

    def test_mae_correct_value(self):
        assert mae([1, 2, 3], [2, 2, 2]) == pytest.approx(2 / 3)

    def test_mape_correct_value(self):
        val = mape([10, 20], [11, 18])
        assert val == pytest.approx(np.mean([0.1, 0.1]), abs=1e-6)

    def test_r2_perfect(self):
        y = [1, 2, 3, 4]
        assert r2(y, y) == pytest.approx(1.0)

    def test_adjusted_r2_lower_than_r2_with_many_features(self):
        y_true = list(range(20))
        y_pred = [v + 0.5 for v in y_true]
        base_r2 = r2(y_true, y_pred)
        adj = adjusted_r2(y_true, y_pred, n_features=10)
        assert adj <= base_r2

    def test_within_tolerance_accuracy(self):
        y_true = [100, 100, 100]
        y_pred = [105, 130, 95]   # within 15%, outside, within 15%
        acc = within_tolerance_accuracy(y_true, y_pred, tolerance=0.15)
        assert acc == pytest.approx(2 / 3)

    def test_psi_zero_for_identical_distributions(self):
        rng = np.random.RandomState(3)
        data = rng.normal(0, 1, 1000)
        assert psi(data, data) == pytest.approx(0.0, abs=1e-6)

    def test_psi_positive_for_shifted_distribution(self):
        rng = np.random.RandomState(4)
        expected = rng.normal(0, 1, 1000)
        actual   = rng.normal(5, 1, 1000)
        assert psi(expected, actual) > PSI_HIGH

    def test_cost_sensitive_evaluation_structure(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([8.0, 25.0, 30.0])
        result = cost_sensitive_evaluation(y_true, y_pred)
        assert result["under_priced_trips"] == 1
        assert result["over_priced_trips"] == 1
        assert result["total_estimated_cost"] > 0


# ============================================================
# PRICING ENGINE TESTS
# ============================================================

class TestPricingEngine:

    def test_get_fare_band_low(self):
        assert get_fare_band(5.0) == "LOW"

    def test_get_fare_band_premium(self):
        assert get_fare_band(150.0) == "PREMIUM"

    def test_score_trip_clamps_low_fare(self):
        result = score_trip({}, predicted_fare=0.5)
        assert result["predicted_fare_usd"] >= MIN_FARE

    def test_score_trip_clamps_high_fare(self):
        result = score_trip({}, predicted_fare=99999)
        assert result["predicted_fare_usd"] <= MAX_FARE

    def test_score_trip_airport_flag(self):
        result = score_trip({"airport_trip": 1}, predicted_fare=40.0)
        assert result["pricing_flag"] == "AIRPORT_FLAT_CANDIDATE"

    def test_score_trip_surge_flag(self):
        row = {"is_rush_hour": 1, "trip_distance_km": 10}
        result = score_trip(row, predicted_fare=25.0)
        assert result["pricing_flag"] == "SURGE_ELIGIBLE"

    def test_score_trip_with_interval(self):
        result = score_trip({}, predicted_fare=20.0, lower_bound=15.0, upper_bound=25.0)
        assert result["prediction_interval_90"] == [15.0, 25.0]

    def test_pricing_engine_batch(self):
        df = pd.DataFrame({
            "airport_trip":     [1, 0],
            "is_rush_hour":     [0, 1],
            "trip_distance_km": [2, 10],
        })
        flags = pricing_engine(df, [20.0, 30.0])
        assert flags[0] == "AIRPORT_FLAT_CANDIDATE"
        assert flags[1] == "SURGE_ELIGIBLE"


# ============================================================
# SELECT BEST MODEL TESTS
# ============================================================

class TestSelectBestModel:

    def test_picks_model_only_present_in_combined_dict(self):
        # Regression test: a model like "NeuralNet" that lives only in the
        # combined `pipelines` dict (not in scaled_pipes/unscaled_pipes
        # individually) must still be resolvable by select_best_model.
        summary = pd.DataFrame([{
            "model": "NeuralNet", "train_r2": 0.97, "test_r2": 0.96,
            "train_test_gap": 0.01, "cv_mean_rmse": 3.5, "cv_std_rmse": 0.1,
            "test_rmse": 3.5, "test_mae": 2.4, "test_mape": 0.08,
            "adjusted_r2": 0.96, "within_15pct_acc": 0.85,
        }])
        sentinel = object()
        name, pipe = select_best_model(
            summary,
            pipelines={"NeuralNet": sentinel},
            scaled_pipes={},
            unscaled_pipes={},
        )
        assert name == "NeuralNet"
        assert pipe is sentinel

    def test_raises_if_model_truly_missing(self):
        summary = pd.DataFrame([{
            "model": "Ghost", "train_r2": 0.9, "test_r2": 0.9,
            "train_test_gap": 0.01, "cv_mean_rmse": 5.0, "cv_std_rmse": 0.1,
            "test_rmse": 5.0, "test_mae": 4.0, "test_mape": 0.1,
            "adjusted_r2": 0.9, "within_15pct_acc": 0.7,
        }])
        with pytest.raises(RuntimeError):
            select_best_model(summary, pipelines={}, scaled_pipes={}, unscaled_pipes={})


# ============================================================
# CONFIG SANITY TESTS
# ============================================================

class TestConfig:

    def test_psi_thresholds_ordered(self):
        assert PSI_MODERATE < PSI_HIGH

    def test_fare_bands_cover_zero_to_infinity(self):
        lows = sorted(low for low, _ in FARE_BANDS.values())
        assert lows[0] == 0

    def test_challenger_gates_sane(self):
        assert 0 < MIN_RMSE_IMPROVEMENT < 1
        assert 0 < MIN_R2_THRESHOLD <= 1
        assert 0 < MAX_GENERALIZATION_GAP < 1

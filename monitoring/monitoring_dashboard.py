# ============================================================
# MONITORING DASHBOARD — Ride Fare Price Prediction ML System
# ============================================================

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

st.set_page_config(page_title="Ride Fare Dashboard", layout="wide")
st.title("🚕 Ride Fare Price Prediction — Monitoring Dashboard")

# ── API URL ───────────────────────────────────────────────────
API_URL = os.getenv(
    "RIDE_FARE_API_URL",
    "http://127.0.0.1:8000"
) + "/predict"

# ── PSI / alert thresholds ────────────────────────────────────
PSI_MODERATE       = 0.10
PSI_HIGH           = 0.20
MAPE_ALERT_THRESHOLD = 0.25

# ── Path resolution — Streamlit Cloud safe ────────────────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR    = os.path.dirname(_SCRIPT_DIR)
except Exception:
    BASE_DIR = os.getcwd()

MONITOR_PATH    = os.path.join(BASE_DIR, "fare_models", "monitor_scores.csv")
LOG_PATH        = os.path.join(BASE_DIR, "logs", "prediction_logs.csv")
PSI_PATH        = os.path.join(BASE_DIR, "fare_models", "feature_drift_report.csv")
CHALLENGER_PATH = os.path.join(BASE_DIR, "fare_models", "challenger_log.json")


# ============================================================
# SIDEBAR — LIVE PREDICTION
# ============================================================

st.sidebar.header("🔮 Predict Trip Fare")

pickup_lat  = st.sidebar.number_input("Pickup latitude",   value=40.7580, format="%.5f")
pickup_lon  = st.sidebar.number_input("Pickup longitude",  value=-73.9855, format="%.5f")
dropoff_lat = st.sidebar.number_input("Dropoff latitude",  value=40.6413, format="%.5f")
dropoff_lon = st.sidebar.number_input("Dropoff longitude", value=-73.7781, format="%.5f")
passengers  = st.sidebar.selectbox("Passenger count", [1, 2, 3, 4, 5, 6])
pickup_dt   = st.sidebar.text_input("Pickup datetime (ISO)", value="2024-06-15T18:30:00Z")

if st.sidebar.button("Predict Fare"):
    payload = {
        "pickup_datetime":   pickup_dt,
        "pickup_longitude":  pickup_lon,
        "pickup_latitude":   pickup_lat,
        "dropoff_longitude": dropoff_lon,
        "dropoff_latitude":  dropoff_lat,
        "passenger_count":   passengers,
    }
    with st.sidebar:
        with st.spinner("Calling API... First request may take 30-60s (Render cold start)"):
            try:
                response = requests.post(API_URL, json=payload, timeout=90)
                if response.status_code == 200:
                    result = response.json()
                    st.success("Prediction received!")
                    st.markdown(f"### 💵 ${result['predicted_fare_usd']}")
                    st.markdown(f"**Fare Band:** `{result['fare_band']}`")
                    if result.get("pricing_flag"):
                        st.warning(f"Flag: {result['pricing_flag']}")
                    if result.get("prediction_interval_90"):
                        lo, hi = result["prediction_interval_90"]
                        st.caption(f"90% prediction interval: ${lo} – ${hi}")
                else:
                    st.error(f"API error: HTTP {response.status_code}")
                    st.code(response.text[:300])
            except requests.exceptions.Timeout:
                st.warning("Request timed out (90s). Render is waking up. Wait 30s and try again.")
            except Exception as e:
                st.error(f"Connection error: {e}")


# ============================================================
# SECTION 1 — REAL-TIME MONITORING ALERTS
# ============================================================

st.markdown("---")
st.subheader("🚨 Real-Time Monitoring Alerts")

alerts_found = False

if os.path.exists(MONITOR_PATH):
    df_monitor = pd.read_csv(MONITOR_PATH)
    if {"predicted_fare", "actual_fare"}.issubset(df_monitor.columns):
        mape_val = (
            (df_monitor["predicted_fare"] - df_monitor["actual_fare"]).abs()
            / df_monitor["actual_fare"].clip(lower=1e-6)
        ).mean()
        if mape_val > MAPE_ALERT_THRESHOLD:
            st.error(f"🔴 HIGH MAPE: {mape_val:.1%} (threshold {MAPE_ALERT_THRESHOLD:.0%}).")
            alerts_found = True
    if "pricing_flag" in df_monitor.columns:
        surge_rate = (df_monitor["pricing_flag"] == "SURGE_ELIGIBLE").mean()
        if surge_rate > 0.30:
            st.warning(f"🟡 HIGH SURGE-ELIGIBLE RATE: {surge_rate:.1%} (expected < 30%).")
            alerts_found = True

if os.path.exists(PSI_PATH):
    df_psi_alert = pd.read_csv(PSI_PATH)
    if "drift_score" in df_psi_alert.columns and len(df_psi_alert):
        max_psi     = df_psi_alert["drift_score"].max()
        top_feature = df_psi_alert.iloc[0]["feature"] if "feature" in df_psi_alert.columns else "unknown"
        if max_psi >= PSI_HIGH:
            st.error(f"🔴 CRITICAL DRIFT: PSI={max_psi:.4f} on '{top_feature}'. Retrain now.")
            alerts_found = True
        elif max_psi >= PSI_MODERATE:
            st.warning(f"🟡 MODERATE DRIFT: PSI={max_psi:.4f} on '{top_feature}'. Monitor.")
            alerts_found = True

if not alerts_found:
    st.success("All systems normal — no alerts triggered")


# ============================================================
# SECTION 2 — CHAMPION vs CHALLENGER HISTORY
# ============================================================

st.markdown("---")
st.subheader("🏆 Champion vs Challenger History")

if os.path.exists(CHALLENGER_PATH):
    with open(CHALLENGER_PATH) as f:
        challenger_log = json.load(f)
    if challenger_log:
        latest         = challenger_log[-1]
        decision_color = "green" if latest["decision"] == "PROMOTED" else "red"
        icon           = "✅" if latest["decision"] == "PROMOTED" else "❌"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Latest Challenger", latest.get("challenger_name", "—"))
        with col2:
            st.metric("Challenger RMSE",   latest.get("challenger_rmse", "—"))
        with col3:
            st.metric("Champion RMSE",     latest.get("champion_rmse", "—") or "First Run")
        with col4:
            st.metric("Challenger R²",     latest.get("challenger_r2", "—"))

        st.markdown(
            f"<h4 style='color:{decision_color}'>{icon} Decision: {latest['decision']} — {latest.get('reason', '')}</h4>",
            unsafe_allow_html=True
        )

        if latest.get("gates"):
            g = latest["gates"]
            gcol1, gcol2, gcol3 = st.columns(3)
            gcol1.metric("RMSE Gate", "✅ Pass" if g.get("rmse_improvement_passed") else "❌ Fail")
            gcol2.metric("R² Gate",   "✅ Pass" if g.get("r2_floor_passed")         else "❌ Fail")
            gcol3.metric("Gap Gate",  "✅ Pass" if g.get("gap_passed")              else "❌ Fail")

        if len(challenger_log) > 1:
            with st.expander("View full challenger history"):
                history_df   = pd.DataFrame(challenger_log)
                display_cols = [c for c in ["evaluated_at", "challenger_name", "challenger_rmse",
                                             "champion_name", "champion_rmse", "decision", "reason"]
                                if c in history_df.columns]
                st.dataframe(history_df[display_cols], width="stretch")
else:
    st.info("No challenger log found.")


# ============================================================
# SECTION 3 — KPI METRICS + CHARTS
# ============================================================

st.markdown("---")
st.subheader("📊 Model Performance KPIs")

if os.path.exists(MONITOR_PATH):
    df = pd.read_csv(MONITOR_PATH)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if "actual_fare" in df.columns:
            st.metric("Avg Actual Fare", f"${df['actual_fare'].mean():.2f}")
    with col2:
        if "predicted_fare" in df.columns:
            st.metric("Avg Predicted Fare", f"${df['predicted_fare'].mean():.2f}")
    with col3:
        if "abs_error" in df.columns:
            st.metric("Avg Abs Error", f"${df['abs_error'].mean():.2f}")
    with col4:
        if {"predicted_fare", "actual_fare"}.issubset(df.columns):
            mape_val = (df["abs_error"] / df["actual_fare"].clip(lower=1e-6)).mean()
            st.metric("MAPE", f"{mape_val:.1%}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Prediction Error Distribution")
        if "abs_error" in df.columns:
            fig, ax = plt.subplots()
            ax.hist(df["abs_error"], bins=50, alpha=0.7, color="steelblue")
            st.pyplot(fig)
            plt.close(fig)
    with col_b:
        st.subheader("Pricing Flag Distribution")
        if "pricing_flag" in df.columns:
            st.bar_chart(df["pricing_flag"].value_counts())

    if {"predicted_fare", "actual_fare"}.issubset(df.columns):
        st.subheader("Predicted vs Actual Fare (sample)")
        sample = df.sample(min(500, len(df)), random_state=42)
        fig, ax = plt.subplots()
        ax.scatter(sample["actual_fare"], sample["predicted_fare"], alpha=0.3, s=8)
        lims = [0, max(sample["actual_fare"].max(), sample["predicted_fare"].max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("Actual fare ($)")
        ax.set_ylabel("Predicted fare ($)")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Fare Statistics")
    if "predicted_fare" in df.columns:
        st.write(df[["predicted_fare", "actual_fare", "abs_error"]].describe().round(4))
else:
    st.warning("No monitor scores found.")


# ============================================================
# SECTION 4 — PSI DRIFT REPORT
# ============================================================

st.markdown("---")
st.subheader("📉 Feature Drift Report (PSI)")

if os.path.exists(PSI_PATH):
    df_psi = pd.read_csv(PSI_PATH)
    if "drift_score" in df_psi.columns and len(df_psi):
        def _psi_flag(val):
            if val >= PSI_HIGH:       return "🔴 CRITICAL"
            elif val >= PSI_MODERATE: return "🟡 MODERATE"
            return "🟢 OK"
        df_psi["status"] = df_psi["drift_score"].apply(_psi_flag)
        st.dataframe(df_psi.head(10), width="stretch")

        fig, ax = plt.subplots(figsize=(12, 4))
        colors = [
            "#E74C3C" if v >= PSI_HIGH else "#F39C12" if v >= PSI_MODERATE else "#2ECC71"
            for v in df_psi["drift_score"].head(10)
        ]
        ax.barh(df_psi["feature"].head(10)[::-1],
                df_psi["drift_score"].head(10)[::-1],
                color=colors[::-1])
        ax.axvline(PSI_MODERATE, color="orange", linestyle="--", label=f"Moderate ({PSI_MODERATE})")
        ax.axvline(PSI_HIGH,     color="red",    linestyle="--", label=f"Critical ({PSI_HIGH})")
        ax.set_title("Feature PSI Drift Scores (Top 10)")
        ax.set_xlabel("PSI Score")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("PSI file found but 'drift_score' column missing / empty.")
else:
    st.warning("Feature drift report not found.")
    st.caption(f"Path checked: `{PSI_PATH}`")


# ============================================================
# SECTION 5 — RECENT PREDICTIONS
# ============================================================

st.markdown("---")
st.subheader("📋 Recent Predictions")

if os.path.exists(LOG_PATH):
    log_df = pd.read_csv(LOG_PATH)
    st.dataframe(log_df.tail(20), width="stretch")
else:
    st.info(
        "Prediction logs are written by the deployed API — not available on Streamlit Cloud. "
        "Run locally to see logs."
    )
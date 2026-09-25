# ============================================================
# RUN API — runner script
# ============================================================

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "serving.fare_api:app",
        host   = "127.0.0.1",
        port   = 8000,
        reload = True,
        # Without reload_dirs, uvicorn watches the ENTIRE project root —
        # editing monitoring/monitoring_dashboard.py or
        # simulation/ride_simulator.py (unrelated to the API) was
        # triggering full server restarts. Restrict watching to only
        # the folders the API actually depends on.
        reload_dirs = [
            os.path.join(PROJECT_ROOT, "serving"),
            os.path.join(PROJECT_ROOT, "services"),
            os.path.join(PROJECT_ROOT, "src"),
        ],
    )
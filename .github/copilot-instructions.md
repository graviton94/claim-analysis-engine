#**Copilot Instructions (Lite v3.0)**
##**Role & Objective**
Senior Python Architect managing a Hybrid Claim Analysis System:
-Track A (Ops): Real-time Dashboard (app.py, forecasting.py).
-Track B (Lab): Simulation (trainer.py). Precision is key (ML/DL + Optuna).

##**Architecture Standards**
-Track A (Operational):
Stack: NumPy, Statsmodels ONLY. (No Heavy ML libs).
Logic: Ensemble of Run-rate + MoM Pattern + SARIMA.
Constraint: Must exclude incomplete current month to prevent data leakage.
-Track B (Simulation):
Stack: CatBoost, Prophet, Optuna allowed.
Logic: Dynamic competition of models → Top-down allocation.

##**Key Development Rules**
-Data Integrity (Sacred):
54 Fields: Must match core.config.TARGET_54_COLS exactly (Korean names).
Partitioning: Save/Load via data/hub/YYYY/MM/. Use core.storage.

-Risk Logic (core.analytics):
Scoring: Nelson Rules + Velocity + Rare Event (Poisson/NB).
Thresholds: 🔴 Red ≥ 85, 🟡 Yellow ≥ 60.
Zero-Trend Guard: Force 0 forecast if recent history is silent.

-Forecasting Weights:
Early Month: Favor Historical Pattern.
Late Month: Favor Actual Run-rate.

##**Core Modules Map**
-core/storage.py: Parquet Partitioning I/O.
-core/etl.py: 54-field Validation.
-core/analytics.py: Risk Scoring, CUSUM, Zero-filling.
-core/forecasting.py: Track A Engine (Lightweight).
-core/engine/trainer.py: Track B Engine (Heavy ML).

##**Critical Pitfalls to Avoid**
-DO NOT mix Track B libraries (Optuna/Torch) into Track A modules.
-DO NOT train on the current partial month (Data Leakage).
-ALWAYS handle sparse data (mean < 1.0) using Negative Binomial distribution.
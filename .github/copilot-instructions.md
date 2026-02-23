# AI Copilot Instructions for Claim Analysis Engine v3.0

## 🤖 System Persona
You are a **Senior Python Architect & Data Scientist** specializing in Time-Series Analysis. 
Your goal is to maintain the integrity of the **Claim Analysis Engine v3.0**, a hybrid system combining operational dashboards (Track A) and simulation labs (Track B).

## 🎯 Project Overview
**Intelligent Claim Analysis & Early Warning System**: A hybrid platform.
- **Track A (Ops)**: Real-time risk detection & fast forecasting (Dashboard).
- **Track B (Lab)**: Deep learning simulation & scenario planning (Page 4).

## 🏗️ Critical Architecture: Two-Track Strategy

### Track A: Operational Forecasting (`core/forecasting.py`)
- **Purpose**: Sub-500ms response time for dashboard "what's the closing forecast for this month?"
- **Tech Stack**: NumPy + Statsmodels (lightweight, NO PyTorch/Optuna)
- **Engine**: Adaptive ensemble of:
  1. **Run-rate**: Current pace extrapolated to month-end
  2. **Pattern (MoM)**: Historical same-month ratio applied
  3. **Trend Line**: Linear regression on cleaned historical data
  4. **Holt-Winters**: Exponential smoothing with seasonality
  5. **SARIMA**: Auto-parameterized statistical model
- **Key Feature**: Outlier removal (IQR method) + partial month data leakage prevention (excludes incomplete current month from training)
- **Weight Logic**: Dynamic—month-start favors patterns, month-end favors actual run-rate

### Track B: Strategic Simulation Lab (`core/engine/trainer.py`)
- **Purpose**: "What if" scenarios for specific product categories—deep learning optimization
- **Tech Stack**: CatBoost, LSTM, SARIMA + Optuna hyperparameter tuning
- **Competition Model**: Three models race; champion selected via cross-validation
- **Key Feature**: Seasonal allocation (top-down forecast → bottom-up sub-dimension ratios)
- **Trigger**: On-demand only (button click in page 4)

## 📂 Core Modules Reference

| Module | Purpose | Key Classes/Functions | Critical Notes |
|--------|---------|----------------------|-----------------|
| `core/storage.py` | I/O & Partitioning | `save_partition()`, `load_range()` | Ensure types (Year/Month as int) before saving. |
| `core/etl.py` | CSV/Excel→54 fields extraction | `load_raw_file()`, `extract_54_fields()` | Always validate against `TARGET_54_COLS` in `config.py` |
| `core/analytics.py` | Risk scoring with Nelson Rules + Poisson | `RiskScoringEngine`, `RiskConfig` | Sparse regime detection (<1.0 avg) switches to negative binomial |
| `core/forecasting.py` | Fast ensemble predictions | `ForecastEngine.forecast_month()` | **Data Leakage Guard**: Removes current incomplete month from training |
| `core/engine/trainer.py` | AutoML model optimization | `HyperParameterTuner`, `ChampionSelector` | Seasonal allocation uses historical same-month ratio for sub-dims |
| `app.py` | Main Streamlit dashboard | Pages: Risk cards, trend viz, forecast banner | Connects to `forecasting.py` for quick predictions |

## 🔑 Project-Specific Patterns & Conventions
### 1. **54-Field Schema is Sacred**
- **Code/Variables**: English (Snake_case). e.g., `calculate_risk_score`.
- **Comments/Docstrings**: **Korean (한국어)**. Explain complex logic clearly.
- **Data Columns**: **Korean (Hardcoded)**. Must match `core.config.TARGET_54_COLS` exactly.
- Missing fields become NaN; extra fields are dropped
- Never deviate—field names must match exactly (Korean field names hardcoded)

### 2. **Partitioning is Mandatory**
- Data stored as: `data/hub/YYYY/MM/data.parquet`
- Queries always filter by year/month—never load entire dataset
- Use `core/storage.load_range(start_ym, end_ym)` for efficient period queries

### 3. **Risk Scoring Tiers** (from `core/analytics.RiskConfig`)
- **🔴 RED (Critical)**: Score ≥ 85 (or 75+ General). Immediate Action.
- **🟡 YELLOW (Warning)**: Score ≥ 60. Monitoring.
- **Components**:
  1. **Nelson Rules**: Bias (9 points), Trend (6 points), Oscillation.
  2. **Velocity**: Partial month speed (Only applied if >20% days passed).
  3. **Rare Event**: Poisson/Neg-Binomial probability.
- **Scoring combines**: Nelson bias/trend detection + sparse event probability + velocity during partial month
- **Partial Month Guard**: Only apply velocity scoring if >20% of month elapsed

### 4. **Forecasting Weight Adaptation**
- **Early month (1-10th)**: Favor historical patterns (MoM ratio 60%, run-rate 20%)
- **Late month (20th-end)**: Favor actual run-rate (run-rate 60%, pattern 20%)
- See `ForecastEngine.forecast_month()` for implementation

### 5. **Outlier Handling**
- IQR-based removal (Q1-1.5*IQR to Q3+1.5*IQR) replaces extremes with median
- Applied before ALL statistical calculations to prevent skew
- Documented in `ForecastEngine._remove_outliers()`

### 6. **Time Series Data Leakage Prevention**
- Current month flagged as "partial" if < 28 days elapsed
- Incomplete current month excluded from training data automatically
- Prevents models learning artificial "month-end crash" patterns

## 🛠️ Common Developer Workflows

### Adding a New Forecast Metric
1. Extend `ForecastEngine.forecast_month()` with new ensemble model
2. Update weights in `_calculate_adaptive_weights()` 
3. Register model result in return dict with confidence interval
4. Test via: `test_forecast.py` with mock time series

### Modifying Risk Scoring Logic
1. Update `RiskConfig` thresholds in `core/analytics.py`
2. Add new Nelson Rules detection in `RiskScoringEngine.detect_*()` methods
3. Update score weights in `SCORE_*` constants
4. Run: `pytest test_*.py` to validate regression

### Debugging Partial Month Predictions
- Check `ForecastEngine.is_partial_month` flag and `progress_ratio` calculation
- Verify training data excludes current month: `len(training_series_cleaned)` should be `len(monthly_series_cleaned) - 1` when partial
- Monitor weight distribution: Run-rate weight should be low in partial months

### Deploying New Pages
1. File: `pages/N_page_name.py` (numeric prefix for Streamlit order)
2. Import from `core` modules only—never duplicate logic
3. Add risk guard: Validate 54 fields via `extract_54_fields()`
4. Test: `streamlit run pages/N_page_name.py`

## ⚠️ Critical Dependencies & Integration Points

- **External Data**: Sales volumes (`data/sales/`) + Claim data (`data/hub/`) merged on Plant × Year × Month
- **Streamlit Pages**: `app.py` (main) → calls `core/forecasting.py` for banners; page 4 → `core/engine/trainer.py` for simulation
- **Time Handling**: All dates stored as `접수년`, `접수월`, `접수일` integers; reconstructed to datetime when needed
- **Library Constraints**: 
  - Track A: NumPy, Statsmodels only (strict)
  - Track B: PyTorch, Optuna, CatBoost allowed (isolated in `core/engine/`)

## 📋 Testing & Validation Commands

```bash
# Validate forecasting ensemble (< 500ms target)
python test_forecast.py --engine fast

# Check risk scoring with mock sparse data
pytest test_forecast.py::test_sparse_regime -v

# Profile storage I/O efficiency
python test_ensemble.py --benchmark

# Verify 54-field extraction
python -c "from core.etl import extract_54_fields; df = load_raw_file('input.csv'); assert len(extract_54_fields(df).columns) == 54"
```

## 🚨 Most Common Pitfalls

1. **Forgetting data leakage guard**: Training on incomplete current month → overly optimistic forecasts
2. **Mixing Track A & B dependencies**: Track A must stay lightweight; don't import `optuna` in `forecasting.py`
3. **Field name mismatch**: Korean field names are hardcoded; mapping typos cause silent NaN drops
4. **Not handling sparse regimes**: When mean < 1.0, Poisson assumptions fail—use negative binomial
5. **Partial month velocity scoring**: Applying without 20% progress threshold inflates risk scores early in month

---
**Last Updated**: Jan 2026 | **Version**: 3.0-Track-Separation

# ARCHITECTURE

## Overview

This document describes the architectural patterns, data flow, and design constraints of the Hybrid Claim Analysis System. The system follows a dual-track architecture separating operational real-time analytics (Track A) from experimental simulation (Track B).

## Data Flow

### 1. Data Ingestion (Upload → Validation → Storage)

```
Raw Files (CSV/Excel)
    ↓
[core/etl.py] load_raw_file()
    ↓
[core/etl.py] extract_54_fields() - Enforces 54 mandatory columns
    ↓
[core/etl.py] validate_data() - Data quality checks
    ↓
[core/storage.py] save_partitioned() - Write to Parquet with partitioning
    ↓
data/hub/접수년={YYYY}/접수월={MM}/*.parquet
```

**Key Rules:**
- All claim data MUST conform to exactly 54 fields defined in `core.config.TARGET_54_COLS`
- Missing fields are filled with NaN; extra fields are dropped
- Data is physically partitioned by `접수년` (year) and `접수월` (month)
- Each row represents exactly one claim incident (1행 = 1건)

### 2. Data Retrieval (Storage → Filtering → Analysis)

```
Parquet Dataset (data/hub/접수년=*/접수월=*/)
    ↓
[core/storage.py] load_and_filter_data() - Unified loader with filters
    ↓ (Apply business logic filters)
    ├─ Business Units: ['식품', 'B2B식품']
    ├─ Performance Reasons: ['제조불만', '고객불만족', '구매불만']
    └─ Time Range: date_from, date_to
    ↓
Filtered DataFrame → Ready for analytics/forecasting
```

**Critical Constraint:**
- **ALL pages and modules MUST use `load_and_filter_data()`** for data consistency
- Direct file access bypasses validation and filtering logic

### 3. Risk Analysis Pipeline (Data → Scoring → Classification)

```
Filtered Claim Data
    ↓
[core/analytics.py] prepare_risk_data() - Zero-filling and aggregation
    ↓
Time Series with Complete Month Coverage (missing months filled with 0)
    ↓
[core/analytics.py] RiskScoringEngine - Multi-factor scoring
    ├─ Statistical Tests: Nelson Rules (3σ deviation, bias, trend)
    ├─ CUSUM Algorithm: Cumulative sum control chart
    ├─ Momentum Analysis: Short-term velocity detection
    ├─ Grade Weighting: '위험'=1.5x, '중대'=1.2x, '일반'=1.0x
    └─ Volatility Adjustment: CV-based dampening
    ↓
Risk Score (0-100) + Diagnosis Message
    ↓
Color Classification:
    - 🔴 Red (≥80): Critical attention required
    - 🟡 Yellow (≥50): Warning level
    - 🟢 Green (<50): Normal
```

**Guard Rails:**
- Accident grade (사고) always scores 100 (immediate escalation)
- Partial month data applies special velocity checks
- Minimum 3 data points required for statistical tests

### 4. Forecasting Flow

#### Track A (Operational - Lightweight)

```
Historical Time Series (excluding incomplete current month)
    ↓
[core/forecasting.py] ForecastEngine
    ↓
Zero-Trend Guard: Check if recent 6 months are silent
    ├─ YES → Force forecast = 0 (extinction signal)
    └─ NO → Proceed to ensemble
    ↓
3-Way Ensemble:
    ├─ Run-rate Method: Current month velocity × remaining days
    ├─ MoM Pattern: Month-over-Month historical average
    └─ SARIMA: Seasonal AutoRegressive Integrated Moving Average
    ↓
Weighted Combination (Early month favors history, late month favors run-rate)
    ↓
3-Month Rolling Forecast
```

**Technology Constraint:**
- Track A uses ONLY NumPy and Statsmodels (no heavy ML libraries)
- Must exclude incomplete current month to prevent data leakage

#### Track B (Simulation - Precision)

```
Complete Historical Data
    ↓
[core/engine/trainer.py] SimulationEngine
    ↓
Dead Signal Check: Recent 12 months mostly zero OR last 6 months all zero
    ├─ YES → Return zero forecast
    └─ NO → Proceed to model competition
    ↓
Validation Phase:
    ├─ Split: Hide last 3 months as test set
    ├─ Train 3 Models:
    │   ├─ Prophet (Facebook's time series model)
    │   ├─ LightGBM + Optuna (auto-tuned gradient boosting)
    │   └─ SARIMAX (statistical baseline)
    └─ Calculate MAE (Mean Absolute Error) for each
    ↓
Weight Calculation: w_i = (1/MAE_i) / Σ(1/MAE_j)
    ↓
Final Phase:
    ├─ Retrain all models on full historical data
    ├─ Generate predictions for future periods
    └─ Apply weighted ensemble
    ↓
Weighted Forecast by Model Performance
    ↓
[core/engine/allocator.py] Top-down allocation across categories
```

**Technology Constraint:**
- Track B CAN use CatBoost, Prophet, Optuna, PyTorch
- Optuna runs fresh on every request for optimal hyperparameters
- Must NOT train on partial current month

### 5. Dashboard Rendering (Data → Visualization → UI)

```
Combined Data (Actuals + Forecasts + Risk Scores)
    ↓
[app.py] Main Dashboard
    ├─ KPI Cards: Total claims, risk distribution, trends
    ├─ Risk Radar: High-risk plant identification
    ├─ Trend Charts: Monthly patterns with forecasts
    └─ Color System: Consistent visual language
    ↓
Streamlit Multi-Page App:
    ├─ Page 1: Data Upload Interface
    ├─ Page 2: Sales Volume Management
    ├─ Page 3: Deep Dive Plant Analysis
    └─ Page 4: Future Simulation & Allocation
```

**UI Standards:**
- Color Palette:
  - Red: `#EF151E` (Critical alerts)
  - Yellow: `#FF9700` (Warnings)
  - Blue: `#006ECD` (Information)
  - Gray: `#2f3339` (Neutral)
- All visualizations use Plotly for interactivity

## Design Patterns

### 1. Repository Pattern (Data Access Layer)

**Implementation:** `core/storage.py`

```python
# Centralized data access - All reads go through this layer
def load_and_filter_data(
    date_from, 
    date_to, 
    business_units=['식품', 'B2B식품'],
    performance_filter=True
) -> pd.DataFrame
```

**Benefits:**
- Single source of truth for data retrieval
- Encapsulates partitioning logic
- Enforces consistent filtering across all modules

### 2. Strategy Pattern (Risk Scoring)

**Implementation:** `core/analytics.py`

```python
class RiskScoringEngine:
    def score(self) -> int:
        # Different strategies based on data characteristics
        if self.is_critical:
            return self._critical_grade_scoring()
        if self.is_sparse:
            return self._rare_event_scoring()
        return self._standard_scoring()
```

**Benefits:**
- Different scoring strategies for different data patterns
- Extensible to new risk factors
- Maintains consistent interface

### 3. Facade Pattern (Forecasting Engine)

**Implementation:** `core/forecasting.py`

```python
class ForecastEngine:
    def get_ensemble_forecast(self) -> Dict:
        # Hides complexity of 3-model ensemble
        # Returns unified prediction
```

**Benefits:**
- Simplifies complex multi-model coordination
- Provides clean API for dashboard consumption
- Encapsulates ensemble logic

### 4. Configuration Object Pattern

**Implementation:** `core/config.py`

```python
# Central configuration as constants
TARGET_54_COLS = [...]  # Exactly 54 field names
DATA_HUB_PATH = "data/hub"
PARTITION_COLS = ["접수년", "접수월"]
```

**Benefits:**
- Single source of truth for configuration
- Prevents magic numbers/strings in code
- Validates configuration at module load (assert len == 54)

### 5. Dataclass Configuration (Immutable Config)

**Implementation:** `core/analytics.py`

```python
@dataclass
class RiskConfig:
    MIN_DATA_POINTS: int = 3
    THRESHOLD_RED: int = 80
    CUSUM_SLACK_STD: float = 0.5
    # ... all parameters documented and typed
```

**Benefits:**
- Type-safe configuration
- Self-documenting parameters
- Easy to test with different configs

## Constraints

### 1. Data Integrity Rules (SACRED)

**Enforcement:** `core/config.py`, `core/etl.py`

```python
# MUST maintain exactly 54 fields
assert len(TARGET_54_COLS) == 54

# Field names MUST be in Korean as per specification
TARGET_54_COLS = ["접수년", "접수월", "접수일", ...]
```

**Violations:**
- ❌ Adding/removing fields without updating TARGET_54_COLS
- ❌ Renaming fields in data pipeline
- ❌ Bypassing extract_54_fields() validation

### 2. Partitioning Discipline

**Enforcement:** `core/storage.py`

```python
# All saves MUST use partitioning
save_partitioned(df, partition_cols=["접수년", "접수월"])

# Physical layout: data/hub/접수년=2024/접수월=11/*.parquet
```

**Violations:**
- ❌ Saving data without partitioning
- ❌ Changing partition columns without migration
- ❌ Direct file writes to data/hub/

### 3. Track Separation (NO Cross-Contamination)

**Enforcement:** Code review, import checks

```python
# Track A (app.py, forecasting.py)
✅ Allowed: numpy, statsmodels, pandas
❌ Forbidden: catboost, prophet, optuna, torch

# Track B (trainer.py, simulation pages)
✅ Allowed: All ML/DL libraries
✅ Use case: Offline experiments, simulation
```

**Violations:**
- ❌ Importing optuna in forecasting.py
- ❌ Using CatBoost in app.py
- ❌ Heavy ML in real-time dashboard

### 4. Data Leakage Prevention

**Enforcement:** `core/forecasting.py`, `core/engine/trainer.py`

```python
# MUST exclude incomplete current month from training
current_month_start = datetime.now().replace(day=1)
train_ts = monthly_ts[monthly_ts.index < current_month_start]
```

**Violations:**
- ❌ Training on partial month data
- ❌ Using future information in historical analysis
- ❌ Forecasting without proper train/test split

### 5. Risk Scoring Thresholds (FIXED)

**Enforcement:** `core/analytics.py`

```python
# Non-negotiable thresholds
THRESHOLD_RED = 80    # 🔴 Critical
THRESHOLD_YELLOW = 50 # 🟡 Warning

# Grade multipliers
WEIGHT_GRADE = {
    '위험': 1.5,   # Risk
    '중대': 1.2,   # Major
    '일반': 1.0,   # Normal
}
```

**Violations:**
- ❌ Changing thresholds without stakeholder approval
- ❌ Different thresholds in different modules
- ❌ Softening risk detection for convenience

### 6. Zero-Filling for Sparse Data

**Enforcement:** `core/analytics.py`, `core/forecasting.py`

```python
# Missing months MUST be filled with 0 (not NaN)
full_idx = pd.date_range(start=min_date, end=max_date, freq='MS')
series_filled = series.reindex(full_idx, fill_value=0)
```

**Violations:**
- ❌ Dropping missing months
- ❌ Using NaN instead of 0 for sparse events
- ❌ Forward-filling claim counts (incorrect assumption)

### 7. Statistical Distribution Selection

**Enforcement:** `core/analytics.py`

```python
# Sparse data (mean < 1.0) MUST use Negative Binomial
if mean_count < 1.0:
    use_negative_binomial()
else:
    use_poisson()
```

**Violations:**
- ❌ Always using Poisson regardless of sparsity
- ❌ Ignoring overdispersion in rare events
- ❌ Normal distribution for count data

### 8. UI Color Consistency

**Enforcement:** `app.py`, page files

```python
# Standardized color palette - DO NOT DEVIATE
COLOR_RED = "#EF151E"     # Critical/Danger
COLOR_YELLOW = "#FF9700"  # Warning
COLOR_BLUE = "#006ECD"    # Info
COLOR_GRAY = "#2f3339"    # Neutral
```

**Violations:**
- ❌ Using arbitrary colors
- ❌ Inconsistent severity indicators
- ❌ Red/green confusion (accessibility issue)

### 9. NO Raw SQL / NO Direct DB Access

**Enforcement:** Architecture design

- All data is file-based (Parquet)
- No database dependencies
- Query logic via Pandas/PyArrow APIs

**Rationale:**
- Simplicity and portability
- Version control friendly
- Reduced infrastructure dependencies

### 10. Functional > Object-Oriented (Karpathy Principle)

**Enforcement:** Code review

```python
# ✅ Preferred: Simple functions
def calculate_risk(series: pd.Series) -> int:
    return score

# ❌ Avoid: Unnecessary classes
class RiskCalculator:
    def __init__(self, series):
        self.series = series
    def calculate(self):
        return score
```

**Exceptions:**
- Engine classes (ForecastEngine, SimulationEngine) - manage complex state
- Config dataclasses - type safety and validation

## Technology Stack Constraints

### Production Dependencies (Track A)

```
streamlit      # UI framework
pandas         # Data manipulation  
numpy          # Numerical computing
pyarrow        # Parquet I/O
statsmodels    # SARIMA only
plotly         # Visualization
```

### Experimental Dependencies (Track B)

```
catboost       # Gradient boosting
prophet        # FB time series
optuna         # Hyperparameter tuning
torch          # Deep learning
lightgbm       # Gradient boosting
scikit-learn   # ML utilities
```

### Development Only

```
pytest         # Testing
black          # Formatting
flake8         # Linting
```

**Constraint:** Track A MUST NOT import Track B dependencies

## Security & Safety

### 1. No Hardcoded Secrets

- Configuration uses environment-agnostic paths
- No API keys or credentials in code
- Sensitive data stays in local data/ directory (gitignored)

### 2. Input Validation

**Enforcement:** `core/etl.py`

```python
# All user uploads go through validation
def validate_data(df: pd.DataFrame) -> bool:
    # Check for required fields
    # Verify data types
    # Detect anomalies
```

### 3. Error Handling

- Graceful degradation when models unavailable
- Fallback to simpler methods if ML fails
- User-friendly error messages (no stack traces in UI)

### 4. Dependency Pinning

**Enforcement:** `requirements.txt`

```
# Exact versions pinned to avoid breaking changes
pandas==2.1.4  # Not pandas>=2.0
```

## Performance Considerations

### 1. Lazy Loading

- Parquet partitioning enables selective reads
- Only load date ranges needed for analysis
- PyArrow filters pushed down to file scan

### 2. Caching Strategy

```python
# Streamlit caching for expensive operations
@st.cache_data
def load_and_filter_data(...):
    # Expensive I/O and filtering
```

### 3. Incremental Updates

- Monthly partitions allow incremental data addition
- No need to reprocess entire history
- New month = new partition folder

## Testing Strategy

### Unit Tests

```
test_forecast.py          # Forecasting logic
test_ensemble.py          # Ensemble methods
test_mom_calculation.py   # Month-over-month calculations
test_runrate_ensemble.py  # Run-rate algorithm
```

### Integration Tests

```
scripts/run_phase1_verification.py
scripts/run_phase1_5_verification.py
```

### Validation Scripts

```
verify_mom_actual.py      # MoM accuracy validation
check_duplication.py      # Data quality checks
```

## Deployment & Execution

### Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Production Constraints

- No external database required
- All state in file system
- Horizontal scaling not required (single-user dashboard)

## Future Extension Points

### Extensible Components

1. **Risk Scoring:** New statistical tests can be added to RiskScoringEngine
2. **Forecasting Models:** Additional models in Track B without changing interface
3. **Allocation Logic:** Custom top-down allocation strategies
4. **Data Sources:** ETL module can support new formats

### Non-Extensible (Stable Interfaces)

1. **54-Field Schema:** Core data contract, changes require migration
2. **Partition Strategy:** Year/Month partitioning is foundational
3. **Track A/B Separation:** Architectural principle, not negotiable
4. **Risk Thresholds:** Business rule, requires approval to change

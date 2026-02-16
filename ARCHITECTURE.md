# Claim Analysis Engine - Architecture

## Overview

The Claim Analysis Engine is a **hybrid intelligence platform** combining real-time operational monitoring (Track A) with deep analytical forecasting (Track B). The system follows a dual-track architecture separating operational real-time analytics from experimental simulation, serving as a quality control tower for food manufacturers to detect anomalies, score risks, and predict future claim volumes.

### Dual-Track Design Philosophy

**Track A (Operational):**
- Purpose: Real-time dashboard with instant feedback
- Stack: NumPy, Statsmodels (lightweight only)
- Engine: ForecastEngine (forecasting.py)
- Models: Run-rate, STL decomposition, LightGBM AutoML
- Latency: <3 seconds for 4-month forecast
- Constraint: Must exclude incomplete current month to prevent data leakage

**Track B (Simulation):**
- Purpose: Deep analysis with ML model competition
- Stack: CatBoost, PyTorch, Optuna (heavy ML/DL allowed)
- Engine: SimulationEngine (trainer.py)
- Models: Prophet, SARIMAX, CatBoost, LSTM
- Latency: 30-120 seconds for backtesting and ensemble
- Focus: Precision over speed, dynamic model competition

---

## Data Flow

### 1. Data Ingestion (Upload → Validation → Storage)

```
Raw Files (CSV/Excel)
    ↓
[core/etl.py] load_raw_file()
    ↓
[core/etl.py] extract_54_fields() - Enforces 54 mandatory columns
    ↓
[core/etl.py] validate_and_clean_data()
    ├─ Parse multi-format dates (YYYY/MM/DD, YYYY-MM-DD, YYYY.MM.DD)
    ├─ Validate 54-field schema (TARGET_54_COLS)
    ├─ Calculate derived fields (Lag_Days, Lag_Valid)
    ├─ Deduplicate by 상담번호 (keep='last')
    └─ Filter null case numbers
    ↓
[core/storage.py] save_to_parquet_partitioned()
    ├─ Partition by 접수년, 접수월
    ├─ Write to data/hub/year=YYYY/month=MM/*.parquet
    └─ Append mode (deduplication handled in loading)
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
    ↓
PyArrow Dataset (lazy loading)
    ├─ Read partitioned Parquet (year/month filter)
    ├─ Apply mode-based business logic
    │   ├─ "인입" (Inflow): 사업부문=['식품','B2B식품'] AND 불만원인 not null
    │   ├─ "실적" (Performance): + 불만원인 in ['제조불만','구매불만','고객불만족']
    │   ├─ "원본" (Raw): No filtering
    │   └─ "커스텀" (Custom): User-selected filters
    ├─ Deduplicate by 상담번호
    └─ Return Pandas DataFrame
```

**Critical Constraint:**
- **ALL pages and modules MUST use `load_and_filter_data()`** for data consistency
- Direct file access bypasses validation and filtering logic (prohibited)

### 3. Risk Analysis Pipeline (Data → Scoring → Classification)

```
Loaded Data
    ↓
[core/analytics.py] prepare_risk_data(data, groupby_cols)
    ├─ Group by Plant/Grade/Category
    ├─ Aggregate by month (count claims)
    ├─ Zero-fill missing months (critical for anomaly detection)
    └─ Return time-series with complete monthly index
    ↓
[core/analytics.py] calculate_advanced_risk_score(series)
    ├─ Regime Detection: mean < 1.0 → Sparse, else Dense
    ├─ Sparse Logic:
    │   ├─ Poisson test (P-value < 0.05 → spike)
    │   ├─ Negative Binomial test (overdispersion)
    │   └─ Critical grade multiplier (1.2x - 1.5x)
    ├─ Dense Logic:
    │   ├─ STL decomposition (trend + seasonality + residual)
    │   ├─ Z-score with rolling statistics
    │   ├─ CUSUM for mean shift detection
    │   └─ Nelson Rules (SPC patterns)
    ├─ Velocity calculation (recent change rate)
    ├─ Volatility adjustment (CV-based)
    └─ Final Score: Components weighted and combined (0-100 scale)
    ↓
Risk Scoring Thresholds
    ├─ Red (🔴): Score ≥ 80 (Critical attention required)
    ├─ Yellow (🟡): Score ≥ 50 (Warning level)
    └─ Green (🟢): Score < 50 (Normal)
```

**Guard Rails:**
- Accident grade (사고) always scores 100 (immediate escalation)
- Partial month data applies special velocity checks
- Minimum 3 data points required for statistical tests

### 4. Forecasting Flow

#### Track A (Operational - Lightweight)

```
Historical Time-Series (excluding incomplete current month)
    ↓
[core/forecasting.py] ForecastEngine.generate_forecast()
    ├─ Pre-checks:
    │   ├─ Recent 6 months all zero? → Return [0,0,0,0]
    │   ├─ Less than 3 months data? → Use simple average
    │   └─ Current month incomplete? → Calculate progress ratio
    ├─ Ensemble Models:
    │   ├─ Run-rate: (current_actual / progress) with smoothing
    │   ├─ STL: Robust seasonal decomposition
    │   └─ LightGBM: Lag features + Optuna auto-tuning
    ├─ Dynamic Weighting:
    │   ├─ Early month (progress < 0.3): Favor STL (historical pattern)
    │   ├─ Late month (progress > 0.7): Favor Run-rate (actual trend)
    │   └─ Mid-month: Balanced blend
    ├─ Post-processing:
    │   ├─ Round to integers
    │   ├─ Clip negatives to 0
    │   └─ Apply zero-trend guard
    └─ Return 4-month predictions
```

**Technology Constraint:**
- Track A uses ONLY NumPy and Statsmodels (no heavy ML libraries)
- Must exclude incomplete current month to prevent data leakage

#### Track B (Simulation - Precision)

```
Complete Historical Data
    ↓
[core/engine/trainer.py] SimulationEngine.run_competition()
    ├─ Dead Signal Check: Recent 12 months mostly zero OR last 6 months all zero
    │   ├─ YES → Return zero forecast (extinction signal)
    │   └─ NO → Proceed to model competition
    ├─ Backtesting Setup:
    │   ├─ Train period: All data except last 3 months
    │   └─ Test period: Last 3 months (for MAE calculation)
    ├─ Model Competition:
    │   ├─ Prophet (Facebook): Additive seasonality + trend
    │   ├─ SARIMAX (Statsmodels): ARIMA with seasonality
    │   └─ AutoML (LightGBM): Optuna-tuned gradient boosting
    ├─ Performance Metrics:
    │   ├─ Calculate MAE for each model
    │   ├─ Inverse weight: 1/MAE (better model gets higher weight)
    │   └─ Normalize weights to sum to 1.0
    ├─ Ensemble:
    │   └─ Weighted average of model predictions
    ↓
[core/engine/allocator.py] allocate_to_subcategories()
    ├─ Historical proportion calculation
    ├─ Extinction detection (category died out)
    ├─ Time-weighted distribution (recent months weighted higher)
    └─ Return sub-category level forecasts
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
- Deep linking via query params: `?plant=X&grade=Y&category=Z&mode=M`

---

## Design Patterns

### 1. Repository Pattern (Data Access Layer)
**Implementation:** `core/storage.py`
- Abstracts Parquet storage implementation
- Provides unified interface: `load_and_filter_data()`, `save_to_parquet_partitioned()`
- Hides PyArrow Dataset complexity from business logic
- Single source of truth for data retrieval
- Encapsulates partitioning logic

### 2. Strategy Pattern (Filtering & Risk Scoring)
**Implementation:** Filtering modes (Inflow/Performance/Custom), Risk scoring strategies
- Same interface, different filtering logic
- Selected at runtime via `mode` parameter
- Encapsulated in `load_and_filter_data()`
- Different scoring strategies for sparse vs dense data patterns
- Extensible to new risk factors

### 3. Template Method
**Implementation:** `core/engine/models.py::BaseModel`
```python
class BaseModel(ABC):
    @abstractmethod
    def fit(self, train_data): pass

    @abstractmethod
    def predict(self, periods): pass
```
- Enforces consistent interface for all ML models
- Prophet, SARIMAX, CatBoost, LSTM all extend BaseModel

### 4. Ensemble Pattern
**Implementation:** Both Track A and Track B
- Multiple models make independent predictions
- Dynamic weighting via backtesting (inverse MAE)
- Combined via weighted average
- Reduces overfitting and improves robustness

### 5. Facade Pattern
**Implementation:** `ForecastEngine`, `SimulationEngine`
- Simplifies complex multi-model orchestration
- Hides ensemble logic from UI layer
- Single method call: `generate_forecast()`, `run_competition()`
- Provides clean API for dashboard consumption

### 6. Lazy Loading Pattern
**Implementation:** PyArrow Dataset
- Partitioned Parquet not loaded until filtered
- Reduces memory footprint for large datasets
- Optimized for time-range queries (year/month partitions)

### 7. Configuration Object Pattern
**Implementation:** `core/config.py`
```python
# Central configuration as constants
TARGET_54_COLS = [...]  # Exactly 54 field names
DATA_HUB_PATH = "data/hub"
PARTITION_COLS = ["접수년", "접수월"]
```
- Single source of truth for configuration
- Prevents magic numbers/strings in code
- Validates configuration at module load (assert len == 54)

---

## Critical Constraints

### Data Integrity Rules (SACRED)

1. **54-Field Schema (Immutable)**
   - Location: `core/config.py::TARGET_54_COLS`
   - Rule: System ONLY processes these exact columns (Korean names)
   - Enforcement: `assert len(TARGET_54_COLS) == 54`
   - Behavior:
     - Missing columns → filled with NaN
     - Extra columns → dropped silently
     - Validation: `core/etl.py::validate_and_clean_data()`
   - **Violations:** ❌ Adding/removing fields, ❌ Renaming fields, ❌ Bypassing extract_54_fields()

2. **Primary Key: 상담번호 (Case Number)**
   - Rule: Must be non-null and unique
   - Deduplication: `keep='last'` (newest record wins)
   - Enforcement: Both in ETL and loading phases

3. **Date Parsing (Multi-Format)**
   - Supported: YYYY/MM/DD, YYYY-MM-DD, YYYY.MM.DD
   - Derived fields:
     - `Lag_Days = 접수일자 - 제조일자`
     - `Lag_Valid = True if Lag_Days ≥ 0`

4. **Partitioning Strategy**
   - Format: Hive-style `year=YYYY/month=MM/`
   - Keys: 접수년, 접수월
   - Purpose: Efficient time-range filtering with PyArrow
   - Physical layout: `data/hub/접수년=2024/접수월=11/*.parquet`
   - **Violations:** ❌ Saving without partitioning, ❌ Changing partition columns, ❌ Direct file writes

### Business Logic Rules

1. **Filtering Modes**
   - **인입 (Inflow):**
     - `사업부문 in ['식품', 'B2B식품']`
     - `불만원인 is not null`
   - **실적 (Performance):**
     - Same as Inflow +
     - `불만원인 in ['제조불만', '구매불만', '고객불만족']`
   - **원본 (Raw):** No filtering
   - **커스텀 (Custom):** User-selected business units and reasons

2. **Risk Thresholds (FIXED - Non-negotiable)**
   - Red (🔴 Danger): Score ≥ 80
   - Yellow (🟡 Caution): Score ≥ 50
   - Green (🟢 Normal): Score < 50
   - **Violations:** ❌ Changing thresholds without approval, ❌ Different thresholds in modules, ❌ Softening detection

3. **Critical Grades (Weighted)**
   - Grades: `['중대', '위험', '사고']`
   - Multiplier: '위험'=1.5x, '중대'=1.2x, '일반'=1.0x
   - Standard: `'일반'` → no multiplier

4. **Zero-Filling Logic (Mandatory)**
   - Rule: Historical data MUST be reindexed with monthly frequency
   - Missing months → filled with 0 (NOT NaN)
   - Purpose: Detect "0 → spike" anomaly patterns
   - Implementation: `core/analytics.py::prepare_risk_data()`
   - **Violations:** ❌ Dropping missing months, ❌ Using NaN, ❌ Forward-filling counts

5. **Forecasting Guards**
   - **Dead Signal Guard:**
     - Condition: Recent 6 months all zero
     - Action: Return [0, 0, 0, 0] (skip training, extinction signal)
   - **Minimum Data Guard:**
     - Condition: Less than 3 months
     - Action: Use simple average (no model training)
   - **Partial Month Handling:**
     - Current month incomplete → calculate progress ratio
     - Adjust run-rate: `(current_actual / progress_ratio)`

### Code Quality Rules

1. **Single Source of Truth**
   - Data Loading: `core.storage.load_and_filter_data()` (ALL pages use this)
   - Risk Preparation: `core.analytics.prepare_risk_data()` (zero-filling)
   - Risk Scoring: `core.analytics.calculate_advanced_risk_score()`
   - Prohibition: No duplicate filtering logic in UI layer

2. **Session State Isolation**
   - Page 3 (Analysis): Keys without suffix (`target_plant`, `search_mode`)
   - Page 4 (Forecast): Keys with `_sim` suffix (`sim_plant_select`, `sim_search_mode`)
   - Purpose: Prevent cross-page state pollution

3. **Color System (Consistency)**
   - Red: `#EF151E`
   - Yellow: `#FF9700`
   - Blue: `#006ECD`
   - Gray: `#2f3339`
   - Usage: All charts and UI elements
   - **Violations:** ❌ Arbitrary colors, ❌ Inconsistent indicators, ❌ Red/green confusion

4. **Deep Linking (Query Params)**
   - Format: `?plant=X&grade=Y&category=Z&mode=M`
   - Purpose: Enable drill-down navigation from dashboard cards
   - Implementation: `st.query_params` in Streamlit

5. **Error Handling (Graceful Degradation)**
   - Show warnings, not crashes
   - Use `.copy()` to avoid Pandas `SettingWithCopyWarning`
   - Validate data existence before processing

### Track Separation Rules (NO Cross-Contamination)

1. **Track A (Operational) - Strict Constraints**
   - ✅ Allowed: NumPy, Statsmodels, Pandas, scikit-learn (basic)
   - ❌ Prohibited: CatBoost, PyTorch, Optuna, Prophet
   - Purpose: Fast response time (<3 seconds)
   - Files: `core/forecasting.py`, `app.py`
   - **Violations:** ❌ Importing optuna in forecasting.py, ❌ Using CatBoost in app.py

2. **Track B (Simulation) - Heavy ML Allowed**
   - ✅ Allowed: CatBoost, PyTorch, Optuna, Prophet, all Track A libs
   - Purpose: Deep analysis with accuracy priority
   - Latency: 30-120 seconds acceptable
   - Files: `core/engine/trainer.py`, `core/engine/models.py`, `pages/4_*.py`

3. **Data Leakage Prevention**
   - Rule: NEVER train on current incomplete month
   - Enforcement: Filter training data to exclude current month
   - Backtesting: Use rolling window (hide last N months)
   - **Violations:** ❌ Training on partial month, ❌ Using future info, ❌ No train/test split

4. **Sparse Data Handling (Statistical Distribution Selection)**
   - Condition: `mean < 1.0`
   - Distribution: Use Poisson or Negative Binomial (NOT Normal)
   - Risk scoring: Rare event logic (P-value tests)
   - **Violations:** ❌ Always using Poisson, ❌ Ignoring overdispersion, ❌ Normal for counts

---

## Key Algorithms

### 1. Risk Scoring Algorithm
**Location:** `core/analytics.py::calculate_advanced_risk_score()`

**Regime Detection:**
```python
if mean < 1.0:
    regime = "Sparse"
    distribution = Poisson / Negative Binomial
else:
    regime = "Dense"
    distribution = Normal (Z-score)
```

**Sparse Regime:**
- Poisson test: Is recent spike statistically significant?
- Negative Binomial: Overdispersion check (variance > mean)
- Critical grade multiplier: 1.2x - 1.5x for severe cases

**Dense Regime:**
- STL decomposition: Separate trend, seasonality, residual
- Z-score: (value - rolling_mean) / rolling_std
- CUSUM: Cumulative sum for mean shift detection
- Nelson Rules: 8 SPC patterns (runs, consecutive increases, etc.)

**Velocity Component:**
- Recent change rate: `(latest - avg_recent) / avg_recent`
- Captures sudden acceleration or deceleration

**Volatility Adjustment:**
- CV (Coefficient of Variation): `std / mean`
- High CV → reduce score (normal for volatile series)

**Final Score:**
```python
score = weighted_sum([
    anomaly_score,
    velocity_score,
    volatility_penalty,
    critical_grade_bonus
]) * 100
```

### 2. Ensemble Forecasting (Track A)
**Location:** `core/forecasting.py::ForecastEngine`

**Models:**
1. **Run-rate:**
   ```python
   run_rate = (current_actual / progress_ratio)
   smoothed = 0.7 * run_rate + 0.3 * recent_average
   ```

2. **STL (Seasonal-Trend-Loess):**
   ```python
   decomposition = STL(series, seasonal=13).fit()
   forecast = trend_forecast + seasonal_component
   ```

3. **LightGBM AutoML:**
   ```python
   features = [lag1, lag2, lag3, month, trend]
   model = LGBMRegressor()
   Optuna.tune(model, cv=3)
   ```

**Dynamic Weighting:**
```python
progress = current_day / total_days_in_month

if progress < 0.3:
    weights = [0.2, 0.6, 0.2]  # Favor STL (historical pattern)
elif progress > 0.7:
    weights = [0.6, 0.2, 0.2]  # Favor Run-rate (actual trend)
else:
    weights = [0.4, 0.3, 0.3]  # Balanced blend

forecast = sum(w * pred for w, pred in zip(weights, predictions))
```

### 3. Model Competition (Track B)
**Location:** `core/engine/trainer.py::SimulationEngine`

**Backtesting:**
```python
train_end = len(series) - 3
train_data = series[:train_end]
test_data = series[train_end:]

for model in [Prophet, SARIMAX, AutoML]:
    model.fit(train_data)
    predictions = model.predict(len(test_data))
    mae = mean_absolute_error(test_data, predictions)
    weights[model] = 1 / mae

weights = normalize(weights)  # Sum to 1.0
```

**Ensemble:**
```python
final_prediction = sum(
    weight * model.predict(horizon)
    for model, weight in zip(models, weights)
)
```

### 4. Sub-Category Allocation
**Location:** `core/engine/allocator.py`

**Historical Proportion:**
```python
proportions = recent_months.groupby('subcategory').sum() / total
time_weights = [0.1, 0.2, 0.3, 0.4]  # Recent months weighted higher
weighted_prop = sum(w * prop for w, prop in zip(weights, proportions))
```

**Extinction Detection:**
```python
if subcategory_count_last_6_months == 0:
    allocate[subcategory] = 0  # Don't predict dead categories
```

**Allocation:**
```python
for subcategory, proportion in weighted_prop.items():
    allocated[subcategory] = total_forecast * proportion
```

---

## System Dependencies

### Critical Library Versions
- **Streamlit 1.31.1:** Multi-page app framework
- **PyArrow 14.0.1:** Parquet partitioning (breaking changes in 15.x)
- **Pandas 2.1.4:** DataFrame operations
- **CatBoost 1.2.1:** Gradient boosting (Track B)
- **PyTorch 2.1.1:** LSTM models (Track B)
- **Statsmodels 0.14.0:** SARIMAX time-series
- **Optuna 3.14.0:** Hyperparameter tuning

### Python Version
- Minimum: 3.10
- Reason: Match-case syntax, type hints (PEP 604)

---

## Performance Considerations

### Memory Management
- **Lazy Loading:** PyArrow Dataset reads only filtered partitions
- **Partitioning:** Year/month reduces scan size by ~95%
- **Session Cache:** `@st.cache_data` for expensive computations

### Compute Optimization
- **Track A:** Single-threaded NumPy (fast enough for dashboard)
- **Track B:** Multi-core via Optuna (n_jobs=-1)
- **Batch Processing:** Forecast multiple categories in parallel

### Storage Efficiency
- **Parquet Compression:** Snappy codec (~70% size reduction)
- **Schema Evolution:** Add columns without rewriting old files
- **Deduplication:** At load time (not storage time) for append efficiency

---

## Security & Data Privacy

### No External Data Transmission
- All ML training and inference runs locally
- No telemetry or cloud API calls
- Streamlit runs on localhost by default

### Data Sanitization
- Null case numbers rejected at ETL
- Invalid dates handled gracefully
- No SQL injection risk (no database)

### Access Control
- Relies on network-level security (no built-in auth)
- Intended for internal corporate network deployment

---

## Future Extension Points

### Pluggable Model Registry
- Add new models by extending `BaseModel` class
- Auto-discovery via registry pattern

### Multi-Tenant Support
- Partition by company/plant in storage layer
- Add tenant filter to `load_and_filter_data()`

### Real-Time Streaming
- Replace batch loading with PyArrow Flight
- Incremental updates instead of full scans

### Advanced Alerts
- Email/Slack integration for red-threshold breaches
- Configurable alert rules per plant/category

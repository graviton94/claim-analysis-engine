# Claim Analysis Engine - Architecture

## System Overview

The Claim Analysis Engine is a **hybrid intelligence platform** combining real-time operational monitoring (Track A) with deep analytical forecasting (Track B). It serves as a quality control tower for food manufacturers, detecting anomalies, scoring risks, and predicting future claim volumes.

### Dual-Track Design Philosophy

**Track A (Operational):**
- Purpose: Real-time dashboard with instant feedback
- Stack: NumPy, Statsmodels (lightweight)
- Engine: ForecastEngine (forecasting.py)
- Models: Run-rate, STL decomposition, LightGBM AutoML
- Latency: <3 seconds for 4-month forecast

**Track B (Simulation):**
- Purpose: Deep analysis with ML model competition
- Stack: CatBoost, PyTorch, Optuna (heavy ML)
- Engine: SimulationEngine (trainer.py)
- Models: Prophet, SARIMAX, CatBoost, LSTM
- Latency: 30-120 seconds for backtesting and ensemble

---

## Data Flow

### 1. Ingestion Pipeline

```
CSV/Excel Upload
    ↓
core/etl.py::validate_and_clean_data()
    ├─ Parse multi-format dates (YYYY/MM/DD, YYYY-MM-DD, YYYY.MM.DD)
    ├─ Validate 54-field schema (TARGET_54_COLS)
    ├─ Calculate derived fields (Lag_Days, Lag_Valid)
    ├─ Deduplicate by 상담번호 (keep='last')
    └─ Filter null case numbers
    ↓
core/storage.py::save_to_parquet_partitioned()
    ├─ Partition by 접수년, 접수월
    ├─ Write to data/hub/year=YYYY/month=MM/*.parquet
    └─ Append mode (deduplication handled in loading)
```

### 2. Data Loading (Single Source of Truth)

```
core/storage.py::load_and_filter_data(mode, business_units, reasons)
    ↓
PyArrow Dataset (lazy loading)
    ├─ Read partitioned Parquet (year/month filter)
    ├─ Apply mode-based business logic
    │   ├─ "인입" (Inflow): 사업부문=[식품,B2B식품] AND 불만원인 not null
    │   ├─ "실적" (Performance): + 불만원인 in [제조불만,구매불만,고객불만족]
    │   ├─ "원본" (Raw): No filtering
    │   └─ "커스텀" (Custom): User-selected filters
    ├─ Deduplicate by 상담번호
    └─ Return Pandas DataFrame
```

### 3. Risk Analysis Flow

```
Loaded Data
    ↓
core/analytics.py::prepare_risk_data(data, groupby_cols)
    ├─ Group by Plant/Grade/Category
    ├─ Aggregate by month (count claims)
    ├─ Zero-fill missing months (critical for anomaly detection)
    └─ Return time-series with complete monthly index
    ↓
core/analytics.py::calculate_advanced_risk_score(series)
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
    ├─ Red (🔴): Score ≥ 80
    ├─ Yellow (🟡): Score ≥ 50
    └─ Green (🟢): Score < 50
```

### 4. Forecasting Flow (Track A - Dashboard)

```
Historical Time-Series
    ↓
core/forecasting.py::ForecastEngine.generate_forecast()
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

### 5. Simulation Flow (Track B - Lab)

```
Historical Time-Series (by Category)
    ↓
core/engine/trainer.py::SimulationEngine.run_competition()
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
core/engine/allocator.py::allocate_to_subcategories()
    ├─ Historical proportion calculation
    ├─ Extinction detection (category died out)
    ├─ Time-weighted distribution (recent months weighted higher)
    └─ Return sub-category level forecasts
```

---

## Design Patterns

### 1. Repository Pattern
**Implementation:** `core/storage.py`
- Abstracts Parquet storage implementation
- Provides unified interface: `load_and_filter_data()`, `save_to_parquet_partitioned()`
- Hides PyArrow Dataset complexity from business logic

### 2. Strategy Pattern
**Implementation:** Filtering modes (Inflow/Performance/Custom)
- Same interface, different filtering logic
- Selected at runtime via `mode` parameter
- Encapsulated in `load_and_filter_data()`

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

### 6. Lazy Loading Pattern
**Implementation:** PyArrow Dataset
- Partitioned Parquet not loaded until filtered
- Reduces memory footprint for large datasets
- Optimized for time-range queries (year/month partitions)

---

## Critical Constraints

### Data Integrity Rules

1. **54-Field Schema (Immutable)**
   - Location: `core/config.py::TARGET_54_COLS`
   - Rule: System ONLY processes these exact columns (Korean names)
   - Behavior:
     - Missing columns → filled with NaN
     - Extra columns → dropped silently
     - Validation: `core/etl.py::validate_and_clean_data()`

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

2. **Risk Thresholds**
   - Red (🔴 Danger): Score ≥ 80
   - Yellow (🟡 Caution): Score ≥ 50
   - Green (🟢 Normal): Score < 50

3. **Critical Grades (Weighted)**
   - Grades: `['중대', '위험', '사고']`
   - Multiplier: 1.2x - 1.5x in risk scoring
   - Standard: `'일반'` → no multiplier

4. **Zero-Filling Logic**
   - Rule: Historical data MUST be reindexed with monthly frequency
   - Missing months → filled with 0 (NOT NaN)
   - Purpose: Detect "0 → spike" anomaly patterns
   - Implementation: `core/analytics.py::prepare_risk_data()`

5. **Forecasting Guards**
   - **Dead Signal Guard:**
     - Condition: Recent 6 months all zero
     - Action: Return [0, 0, 0, 0] (skip training)
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

4. **Deep Linking (Query Params)**
   - Format: `?plant=X&grade=Y&category=Z&mode=M`
   - Purpose: Enable drill-down navigation from dashboard cards
   - Implementation: `st.query_params` in Streamlit

5. **Error Handling (Graceful Degradation)**
   - Show warnings, not crashes
   - Use `.copy()` to avoid Pandas `SettingWithCopyWarning`
   - Validate data existence before processing

### Track Separation Rules

1. **Track A (Operational) - Strict Constraints**
   - Allowed: NumPy, Statsmodels, Pandas, scikit-learn (basic)
   - Prohibited: CatBoost, PyTorch, Optuna, Prophet
   - Purpose: Fast response time (<3 seconds)
   - Files: `core/forecasting.py`, `app.py`

2. **Track B (Simulation) - Heavy ML Allowed**
   - Allowed: CatBoost, PyTorch, Optuna, Prophet, all Track A libs
   - Purpose: Deep analysis with accuracy priority
   - Latency: 30-120 seconds acceptable
   - Files: `core/engine/trainer.py`, `core/engine/models.py`, `pages/4_*.py`

3. **Data Leakage Prevention**
   - Rule: NEVER train on current incomplete month
   - Enforcement: Filter training data to exclude current month
   - Backtesting: Use rolling window (hide last N months)

4. **Sparse Data Handling**
   - Condition: `mean < 1.0`
   - Distribution: Use Poisson or Negative Binomial (NOT Normal)
   - Risk scoring: Rare event logic (P-value tests)

---

## Key Algorithms

### 1. Risk Scoring Algorithm
**Location:** `core/analytics.py::calculate_advanced_risk_score()`

**Regime Detection:**
```
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
```
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
    weights = [0.2, 0.6, 0.2]  # Favor STL
elif progress > 0.7:
    weights = [0.6, 0.2, 0.2]  # Favor Run-rate
else:
    weights = [0.4, 0.3, 0.3]  # Balanced

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

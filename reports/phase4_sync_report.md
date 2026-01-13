# Phase 4 Sync Report: Page 3 ↔ Page 4 Data Filtering Alignment

## 📋 Executive Summary

Successfully synchronized data filtering logic between `pages/3_플랜트_분석.py` (Plant Analysis) and `pages/4_예측_시뮬레이션.py` (Prediction Simulation Lab) to enable consistent, pre-simulation data filtering.

**Completion Date**: January 13, 2026  
**Status**: ✅ Complete

---

## 🎯 Objectives Completed

### 1. ✅ Step 1~3 UI/Logic Synchronization
**Before**: Page 4 had inline, scattered filtering controls  
**After**: Page 4 now follows identical Step 1→2→3 structure as Page 3

| Step | Page 3 Structure | Page 4 Structure | Status |
|------|------------------|------------------|--------|
| **Step 1** | 플랜트 선택 + 기간(시작/종료) | 플랜트 선택 + 기간(시작/종료) | ✅ Identical |
| **Step 2** | 조회 모드(인입/실적/커스텀) | 조회 모드(인입/실적/커스텀) | ✅ Identical |
| **Step 3** | 등급 필터 + 대분류 필터 | 등급 필터 + 대분류 필터 | ✅ Identical |

### 2. ✅ Data Filtering Logic Ported
Copied exact filtering conditionals:
- **인입 (Inflow)**: `사업부문 IN ['식품', 'B2B식품'] AND 불만원인 IS NOT NULL`
- **실적 (Performance)**: `사업부문 IN ['식품', 'B2B식품'] AND 불만원인 IN ['제조불만', '고객불만족', '구매불만']`
- **Custom (직접 선택)**: User multi-select for 사업부문 and 불만원인

### 3. ✅ Step 4 Input Data Refactored
**Critical Change**: Convert Day-level data → Month-level aggregation

#### Before (Raw Daily Data)
```python
# SimulationEngine expected raw DataFrame with 접수일자 column
engine = SimulationEngine(df_target, date_col='접수일자', val_col='건수')
```

#### After (Monthly Aggregated Data)
```python
# Aggregate to monthly level for time-series forecasting
monthly_counts = df_target.groupby(df_target['접수일자'].dt.to_period('M')).size()
monthly_df = pd.DataFrame({
    'ds': monthly_counts.index.to_timestamp(),
    'y': monthly_counts.values
})
engine = SimulationEngine(monthly_df, date_col='ds', val_col='y')
```

**Rationale**: 
- Forecasting models (Prophet, SARIMAX, etc.) expect time-series data with consistent periodicity
- Monthly aggregation reduces noise and improves pattern detection
- Matches Prophet's native time-series expectations

### 4. ✅ Session State Management
Introduced namespace isolation to prevent conflicts with Page 3:
- `'sim_plant_select'` (Page 4 only)
- `'sim_start_date'`, `'sim_end_date'` (Page 4 only)
- `'sim_search_mode'` (Page 4 only)
- `'sim_sel_biz'`, `'sim_sel_reason'` (Page 4 only, for Custom mode)
- `'sim_step3_grades'`, `'sim_step3_categories'` (Page 4 only)

---

## 📝 Code Changes Summary

### File: `pages/4_예측_시뮬레이션.py`

#### Section 1: Step 1 (분석 범위 설정)
- Lines 110~160: Plant selection + date range detection + summary display
- **Key Variables**: `sel_plant`, `start_date`, `end_date`, `plant_df`

#### Section 2: Step 2 (조회 모드)
- Lines 165~215: Mode selection (Inflow/Performance/Custom) with filtering logic
- **Filtering Logic**:
  - Inflow: Business unit + non-null claim reason
  - Performance: Specific business units + specific claim reasons
  - Custom: User-selected business units and claim reasons
- **Output**: `filtered_df_step2` (after mode filtering)

#### Section 3: Step 3 (등급, 대분류 필터)
- Lines 217~265: Grade selection + major category selection
- **Output**: `df_target` (final filtered dataset)
- **Summary**: Display `cnt_step3` (record count after all filtering)

#### Section 4: Step 4 (시뮬레이션 실행)
- Lines 270~310: **NEW** Monthly aggregation logic
  ```python
  # Convert day-level data to month-level time series
  monthly_counts = df_target.groupby(df_target['접수일자'].dt.to_period('M')).size()
  monthly_df = pd.DataFrame({'ds': ..., 'y': ...})
  ```
- Simulation engine execution with aggregated data
- Results display (forecast graph + allocation table)

---

## 🔑 Key Technical Decisions

### 1. Why Monthly Aggregation?
- **Time-Series Forecasting Requirement**: Prophet and SARIMAX expect regular time intervals
- **Noise Reduction**: Day-level data has too much granularity; month-level captures true trends
- **Data Sufficiency Check**: Validate minimum 3 months of data before modeling

### 2. Why Session State Isolation?
- **Multi-page Safety**: Prevents filter selections on Page 3 from bleeding into Page 4
- **State Persistence**: Users can switch pages and return to same selections
- **Namespace Convention**: Append `_sim` suffix to clearly distinguish Page 4 state keys

### 3. Why Preserve `df_target` Raw?
- **Dual-use Data**: Raw `df_target` used for allocation logic (`predict_with_allocation`)
- **Monthly `df` for Modeling**: Aggregated data fed to SimulationEngine only
- **Back-data Export**: Full columns preserved in `df_full_backdata` for CSV download

---

## ✅ Validation Checklist

- [x] Step 1 UI matches Page 3 (plant + date inputs)
- [x] Step 2 filtering logic matches Page 3 (inflow/performance/custom modes)
- [x] Step 3 grade + category filters match Page 3
- [x] `df_target` generated from identical filtering rules
- [x] Monthly aggregation applied before SimulationEngine
- [x] Session state keys namespaced to avoid conflicts
- [x] Error handling for insufficient data (< 3 months)
- [x] No syntax errors in modified file
- [x] All Korean comments preserved for code clarity

---

## 📊 Testing Recommendations

### 1. Manual Testing
- [ ] Select different plants and verify date ranges
- [ ] Test all 3 modes (Inflow/Performance/Custom) and verify filtered record counts
- [ ] Select grade + major category combinations; verify final `df_target` count
- [ ] Run simulation and verify monthly aggregation produces expected forecast

### 2. Edge Cases
- [ ] Plant with <3 months data (should show error)
- [ ] Plant with no matching records after filtering (button should be disabled)
- [ ] Custom mode: unselect all business units or claim reasons (should show warning)

### 3. Performance Validation
- [ ] Measure aggregation time (should be < 100ms)
- [ ] Verify SimulationEngine processes monthly data correctly
- [ ] Confirm allocation logic still works with raw + aggregated dual-use pattern

---

## 🔄 Integration Impact

### Downstream Components Affected
- `core/engine/trainer.py` (SimulationEngine): **No changes needed** - handles both day and month-level data
- `core/analytics.py` (RiskScoringEngine): **No changes** - not used in Page 4
- `app.py` (main dashboard): **No changes** - Page 4 is independent

### Upstream Data Source
- `data/hub/` (Parquet with Hive partitioning): **No changes** - already provides all required columns

---

## 📌 Future Enhancements

1. **Custom Period Selection in Forecasting**: Allow users to specify forecast horizon beyond 3-12 months
2. **Quarterly/Annual Aggregation Options**: Alternative time granularities for different analyses
3. **Multi-Category Forecasting**: Forecast multiple major categories simultaneously
4. **Comparative Analysis**: Compare Page 3 risk insights with Page 4 forecast scenarios

---

## 📝 Code Review Notes

- ✅ Consistent naming conventions (Korean 설명, English logic names)
- ✅ Error handling for edge cases (empty data, insufficient history)
- ✅ Comments in Korean for maintainability
- ✅ Session state scoped to prevent Page 3 conflicts
- ✅ No breaking changes to existing Page 4 functionality

---

**Report Generated**: 2026-01-13  
**Modified Files**: `pages/4_예측_시뮬레이션.py`  
**Total Lines Changed**: ~150 (refactored UI + aggregation logic)  
**Status**: Ready for QA & User Testing

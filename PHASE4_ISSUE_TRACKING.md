# Phase 4 Issue Tracking & Resolution Log

**Date**: 2026-01-13  
**Status**: 🔴 IN PROGRESS - Awaiting Expert Review  
**Assignee**: Pending External Review

---

## Current Issues

### ❌ Issue #1: 피벗 테이블 컬럼 중복 오류
**Error**: `columns overlap but no suffix specified: Index(['대분류'], dtype='object')`

**Location**: `pages/4_예측_시뮬레이션.py` - Tab 1 (Pivot Table)

**Current Implementation**:
```python
combined_pivot = pd.concat([hist_aligned[hist_cols], alloc_aligned[pred_cols]], axis=1, sort=False)
```

**Symptom**: 
- MultiIndex(등급기준|대분류|소분류) 기반 두 피벗 결합 시 발생
- `hist_aligned` (과거 12개월)와 `alloc_aligned` (예측 6개월) 컬럼 중 '대분류' 같은 non-index 컬럼 충돌 가능성

---

### ❌ Issue #2: Lag 분석 데이터 부족
**Error**: `⏱️ Lag 분석 데이터가 충분하지 않습니다.`

**Location**: `pages/4_예측_시뮬레이션.py` - Tab 2 (Lag Analysis)

**Current Implementation**:
```python
lag_stats = calculate_lag_stats(df_target)
if lag_stats and lag_stats.get('count', 0) > 0:
    # metrics display
else:
    st.info("⏱️ Lag 분석 데이터가 충분하지 않습니다.")
```

**Symptom**:
- `df_target` (Step 1~3 필터링된 데이터)에서 `calculate_lag_stats()` 호출 시 결과가 None 또는 count=0
- 이유: `df_target`이 필터링으로 인해 데이터가 너무 제한적일 수 있음
- 3_플랜트_분석.py에서는 `filtered_df_step3` (백데이터 포함)를 사용하므로 더 많은 데이터 확보

---

## Root Cause Analysis

### Pages 간 데이터 처리 흐름 비교

| 단계 | 3_플랜트_분석.py | 4_예측_시뮬레이션.py | 차이점 |
|------|-----------------|-----------------|--------|
| Step 1~3 | `filtered_df_step3` | `df_target` | **동일 필터링** ✅ |
| 백데이터 로드 | `whole_history_df` (24개월+전체) | `없음` (시뮬레이션 시에만 `df_full_backdata` 로드) | **차이 발생** ❌ |
| Lag 분석 입력 | `filtered_df_step3` (필터링된 데이터) | `df_target` (필터링된 데이터) | **동일하지만 초기 데이터 규모 다름** |
| Pivot 테이블 구성 | 12개월 히스토리 + Risk 스코어링 | 12개월 히스토리 + 6개월 예측 결합 | **로직 차이** |

---

## Proposed Solutions

### Solution A: Lag 분석 데이터 확보
**지시사항**:
1. Lag 분석 시 `df_target` 대신 원본 백데이터 기준 사용
2. 3_플랜트_분석.py (Line 765)의 패턴 참조:
   ```python
   lag_stats = calculate_lag_stats(filtered_df_step3)  # 전체 필터링 데이터 사용
   ```

**구체적 개선**:
- `df_target` → `whole_history_df` 또는 `df_full_backdata` 사용
- 시뮬레이션 결과 조회 시 이미 로드된 `df_full_backdata` 활용
- fallback: `df_target`으로 재계산하되, 데이터 최소 요건(count > 0) 명확히

---

### Solution B: 피벗 테이블 컬럼 충돌 해결
**지시사항**:
1. **Step 1**: Historical 데이터를 MultiIndex(등급기준|대분류|소분류) 기준으로 12개월 배치
   - 현재: ✅ 이미 구현됨
   
2. **Step 2**: 예측 데이터를 동일 MultiIndex 기준으로 6개월 배치
   - 현재: ✅ `alloc_pivot` 이미 이렇게 구성
   
3. **Step 3**: 두 데이터프레임 결합 시 **컬럼명 명시적 분리**
   - 문제: Non-index 컬럼 충돌 (특히 '대분류'라는 컬럼이 data_column으로도 존재할 수 있음)
   - 해결: 피벗 생성 시 values만 사용하도록 보장
   ```python
   # 검증: 피벗 결과가 data column을 포함하지 않는지 확인
   # pivot 후 생성되는 columns는 '월' 값만이어야 함
   ```

4. **Step 4**: Risk 신호(🚨) + 원인 추가
   - 3_플랜트_분석.py (Line 751~758) 패턴: Risk scoring per row
   - 각 MultiIndex 조합별로 `calculate_advanced_risk_score()` 호출
   - 결과 컬럼: `['등급기준', '대분류', '소분류', '🚨', '위험진단']`

---

## Data Lineage & Dependencies

### Lag 분석 의존성 체인
```
df_raw (load_metadata)
  ↓
df_target (Step 1~3 필터링)
  ↓ [현재 문제: 데이터 부족]
calculate_lag_stats(df_target) → count=0
  
[해결] df_full_backdata (load_full_target_data) 사용
  ↓
calculate_lag_stats(df_full_backdata) → count > 0
```

### 피벗 테이블 의존성 체인
```
df_target (Step 1~3 필터링)
  ↓
historical_12m (과거 12개월 필터링)
  ↓
pivot_hist (MultiIndex 기반 생성) ✅
  
alloc_df (SimulationEngine 예측 결과)
  ↓
alloc_pivot (MultiIndex 기반 생성) ✅
  
[현재 문제: concat 시 컬럼 중복]
combined_pivot = pd.concat([hist, alloc], axis=1)
  ✗ Fails with: columns overlap but no suffix specified
```

---

## Implementation Checklist

- [ ] **Lag 분석**: `df_target` → `df_full_backdata` 변경 (또는 안전한 fallback 패턴)
- [ ] **Pivot 컬럼 충돌**: 피벗 생성 시 `values` 컬럼 명시, concat 전 컬럼 검증
- [ ] **Risk 신호 추가**: Pivot 결과 행별로 `calculate_advanced_risk_score()` 호출
- [ ] **UI 동일화**: Tab 1 최종 형식을 3_플랜트_분석.py와 맞춤 (MultiIndex + 🚨 + 진단)
- [ ] **테스트**: Step 1~3에서 데이터 선택 후 Tab 1, Tab 2 정상 동작 확인

---

## Reference Implementation

**3_플랜트_분석.py (Line 730~793)**:
- Lag 분석: Line 755 `calculate_lag_stats(filtered_df_step3)`
- Risk 신호: Line 492~518 `calculate_advanced_risk_score(series_data, target_month, grade)`
- Pivot 구성: Line 404~419 `create_pivot_with_subtotals_dynamic()`

---

## Next Steps

1. ✅ Record current state (this file)
2. ⏳ Commit all changes
3. ⏳ Push to main
4. ⏳ Request external review/assistance

---

**Last Updated**: 2026-01-13 13:30 KST  
**Version**: Phase 4 - Issue Tracking v1.0

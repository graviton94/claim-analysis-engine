
---

# 🛠️ [Patch Order] Phase 2.5: Plant Analysis Upgrade

**Target File**: `pages/3_플랜트_분석.py`
**Objective**: 피벗 테이블 가시성 개선(Hybrid View) 및 정밀 이상치 감지(Dynamic Scoring) 구현.

---

### 1. Imports 추가

**[설명]** 날짜 연산 및 시계열 처리를 위한 라이브러리를 추가합니다.

```python
<<<< SEARCH
import numpy as np
from datetime import datetime, date
from core.storage import load_partitioned, DATA_HUB_PATH  # DATA_HUB_PATH 임포트 추가
==== REPLACE
import numpy as np
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from core.storage import load_partitioned, DATA_HUB_PATH
>>>>

```

---

### 2. Config 및 신규 알고리즘 함수 추가

**[설명]** 기존 상수 정의 아래에 **동적 스코어링 로직**을 담당할 함수 `calculate_advanced_risk_score`를 추가합니다.

```python
<<<< SEARCH
# [CONFIG] 불만원인 그룹 정의
PERFORMANCE_REASONS = ['제조불만', '고객불만족', '구매불만']
TARGET_BUSINESS_UNITS = ['식품', 'B2B식품']

# --- 1. 데이터 로드 (캐시 제거: 항상 최신 로드) ---
==== REPLACE
# [CONFIG] 불만원인 그룹 정의
PERFORMANCE_REASONS = ['제조불만', '고객불만족', '구매불만']
TARGET_BUSINESS_UNITS = ['식품', 'B2B식품']

# [CONFIG] 동적 스코어링 가중치
RISK_WEIGHTS = {
    'z_score': 40,   # 통계적 이격도
    'mom': 30,       # 전월 대비 가속도
    'yoy': 30        # 전년 동월 대비 계절성 충격
}

def calculate_advanced_risk_score(history_series, target_month_str):
    """
    Dynamic Risk Scoring Logic
    - Low Baseline (평균 < 1.0): 절대 수치 기준 (3건↑ 경보, 2건 주의)
    - High Baseline (평균 >= 1.0): 복합 점수제 (Z-Score + MoM + YoY)
    """
    if history_series.empty or target_month_str not in history_series.index:
        return "⚪", 0, "데이터 부족"
    
    current_val = history_series[target_month_str]
    # 과거 데이터 (당월 제외)
    past_series = history_series[history_series.index < target_month_str]
    
    # 데이터가 너무 적으면 판단 유보 (최소 3개월)
    if len(past_series) < 3:
        # 단, 당월 수치가 3건 이상이면 경보
        return ("🔴", 100, "초기 급증") if current_val >= 3 else ("⚪", 0, "데이터 부족")

    mean_val = past_series.mean()
    
    # --- Scenario A: Low Baseline (평균 1.0건 미만) ---
    if mean_val < 1.0:
        if current_val >= 3:
            return "🔴", 100, f"신규/희귀 급증({int(current_val)}건)"
        elif current_val == 2:
            return "🟡", 50, "주의 수준 발생"
        else:
            return "⚪", 0, "정상 범위"

    # --- Scenario B: High Baseline (평균 1.0건 이상) ---
    else:
        score = 0
        reasons = []
        
        # 1. Z-Score (40점)
        std_val = past_series.std() if past_series.std() > 0 else 1.0
        z_score = (current_val - mean_val) / std_val
        
        if z_score > 3.0: score += RISK_WEIGHTS['z_score']
        elif z_score > 2.0: score += (RISK_WEIGHTS['z_score'] * 0.5)
        elif z_score > 1.5: score += (RISK_WEIGHTS['z_score'] * 0.25)

        # 2. MoM (전월 대비, 30점)
        try:
            prev_date = datetime.strptime(target_month_str, "%Y-%m") - relativedelta(months=1)
            prev_str = prev_date.strftime("%Y-%m")
            if prev_str in history_series.index:
                prev_val = history_series[prev_str]
                if prev_val > 0:
                    ratio = current_val / prev_val
                    if ratio >= 2.0: score += RISK_WEIGHTS['mom']
                    elif ratio >= 1.5: score += (RISK_WEIGHTS['mom'] * 0.5)
        except: pass

        # 3. YoY (전년 동월 대비, 30점)
        try:
            last_year_date = datetime.strptime(target_month_str, "%Y-%m") - relativedelta(years=1)
            last_year_str = last_year_date.strftime("%Y-%m")
            if last_year_str in history_series.index:
                ly_val = history_series[last_year_str]
                if ly_val > 0:
                    ratio = current_val / ly_val
                    if ratio >= 1.5: score += RISK_WEIGHTS['yoy']
                    elif ratio >= 1.2: score += (RISK_WEIGHTS['yoy'] * 0.5)
                elif ly_val == 0 and current_val >= 3:
                     score += RISK_WEIGHTS['yoy'] # 전년 0인데 올해 급증
        except: pass

        # 최종 판정
        if score >= 80: return "🔴", score, "위험(High Risk)"
        elif score >= 50: return "🟡", score, "주의(Caution)"
        else: return "⚪", score, "정상"

# --- 1. 데이터 로드 (캐시 제거: 항상 최신 로드) ---
>>>>

```

---

### 3. 분석 실행 로직 전면 수정 (Hybrid View + Risk Scoring)

**[설명]** `st.button("📊 분석 시작 ...")` 내부 로직을 Hybrid View 변환 및 경보 컬럼 삽입 로직으로 교체합니다.

```python
<<<< SEARCH
# 분석 시작 버튼
if st.button("📊 분석 시작 (Run Analysis)", type="primary", use_container_width=True):
    
    if not pivot_indices:
        st.error("최소 하나 이상의 피벗 행(Index)을 선택해야 합니다.")
        st.stop()
        
    if filtered_df_step3.empty:
        st.warning("조회 조건에 해당하는 데이터가 없습니다.")
        st.stop()

    # --- 날짜/월 처리 ---
    # 그래프의 연속성을 위해 전체 기간의 월 목록을 생성
    all_months_in_range = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y-%m').tolist()
    # 접수월 컬럼 생성
    filtered_df_step3['접수월_str'] = filtered_df_step3['접수일자'].dt.strftime('%Y-%m')

    # --- 피벗 테이블 생성 로직 (소계/총계 포함) ---
    try:
        def create_pivot_with_subtotals(df, indices, columns, values, aggfunc, all_months):
            """ 피벗 테이블에 소계/총계를 추가하고, 모든 월 컬럼을 보장 """
            # 1. 기본 마진 피벗 (인덱스가 1개일 때 안전장치)
            if len(indices) < 2:
                pivot_with_margin = pd.pivot_table(df, index=indices, columns=columns, values=values, aggfunc=aggfunc, fill_value=0, margins=True, margins_name='Total')
                # 모든 월 포함하도록 reindex
                # margins=True로 인해 Total 컬럼이 이미 생겼을 수 있음. 중복 방지 위해 columns list 조정
                reindex_cols = all_months + ['Total']
                pivot_reindexed = pivot_with_margin.reindex(columns=reindex_cols, fill_value=0)
                return pivot_reindexed

            # 2. 기본 피벗 생성
            pivot_base = pd.pivot_table(df, index=indices, columns=columns, values=values, aggfunc=aggfunc, fill_value=0)
            # 모든 월 컬럼 보장
            pivot_base = pivot_base.reindex(columns=all_months, fill_value=0)
            
            if pivot_base.empty:
                # 빈 데이터프레임 처리 (구조만 유지)
                empty_idx = pd.MultiIndex.from_tuples([], names=indices)
                return pd.DataFrame(0, index=empty_idx, columns=all_months + ['Total'])

            all_parts = []
            
            # 3. 소계 계산 루프
            for l1_name, l1_group in pivot_base.groupby(level=0, sort=False):
                for l2_name, l2_group in l1_group.groupby(level=1, sort=False):
                    all_parts.append(l2_group)
                    # L2 소계: ('L1 값', 'L2 값', '소계', '', ..)
                    subtotal_l2_row = l2_group.sum().to_frame().T
                    template_idx = list(l2_group.index[0])
                    # 인덱스 길이 안전하게 처리
                    idx_tuple = template_idx[:2] + ['소계'] + [''] * max(0, len(indices) - 3)
                    subtotal_l2_row.index = pd.MultiIndex.from_tuples([tuple(idx_tuple)], names=indices)
                    all_parts.append(subtotal_l2_row)
                
                # L1 총계: ('L1 값', '전체 합계', '', ..)
                total_l1_row = l1_group.sum().to_frame().T
                template_idx = list(l1_group.index[0])
                idx_tuple = [template_idx[0]] + ['전체 합계'] + [''] * max(0, len(indices) - 2)
                total_l1_row.index = pd.MultiIndex.from_tuples([tuple(idx_tuple)], names=indices)
                all_parts.append(total_l1_row)
            
            final_pivot = pd.concat(all_parts)
            
            # 4. 전체 총계 (Grand Total) 추가
            grand_total_row = pivot_base.sum().to_frame('Total').T
            idx_tuple = ['Total'] + [''] * max(0, len(indices) - 1)
            grand_total_row.index = pd.MultiIndex.from_tuples([tuple(idx_tuple)], names=indices)
            final_pivot = pd.concat([final_pivot, grand_total_row])
            
            # 5. 우측 Total 컬럼 추가
            final_pivot['Total'] = final_pivot.sum(axis=1)

            return final_pivot

        pivot_table = create_pivot_with_subtotals(
            df=filtered_df_step3,
            indices=pivot_indices,
            columns='접수월_str',
            values='상담번호',
            aggfunc='count',
            all_months=all_months_in_range # 전체 월 목록 전달
        )

    except Exception as e:
        st.error(f"피벗 테이블 생성 중 오류 발생: {e}")
        # st.exception(e) # 사용자에게는 너무 상세한 오류일 수 있어 주석 처리
        st.stop()

    # --- 결과 시각화 ---
    st.subheader(f"📈 분석 결과 ({grade_mode} / {search_mode})")
    
    tab1, tab2, tab3 = st.tabs(["피벗 테이블", "Lag 분석", "원본 데이터"])

    with tab1:
        # [HOTFIX] 이상치 스타일링 로직 전면 수정 (Vectorized)
        def highlight_outliers_vectorized(data):
            # 1. 계산용 데이터 준비 (Total 행/열 제외)
            # errors='ignore'로 Total이 없어도 에러나지 않게 처리
            df_numeric = data.drop(index='Total', columns='Total', errors='ignore')
            
            # 만약 인덱스가 'Total'로 시작하는 행이 있다면 그것도 제외 (소계/합계 행 제외)
            # 여기서는 간단히 Total 컬럼만 제외하고 계산
            
            # 2. IQR 계산 (axis=1: 행 단위 계산)
            q1 = df_numeric.quantile(0.25, axis=1)
            q3 = df_numeric.quantile(0.75, axis=1)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # 3. 전체 데이터프레임 크기의 스타일 마스크 생성
            style_df = pd.DataFrame('', index=data.index, columns=data.columns)
            
            # 4. 이상치 마킹 (Broadcasting)
            # axis=0을 사용하여 Series(행별 임계값)를 DataFrame 각 행에 적용
            # df_numeric에 대해서만 계산
            is_outlier = (df_numeric.lt(lower_bound, axis=0)) | (df_numeric.gt(upper_bound, axis=0))
            
            # 5. 스타일 적용
            # is_outlier의 True인 위치를 찾아 style_df에 적용
            for col in is_outlier.columns:
                # 해당 컬럼에서 True인 행 인덱스 추출
                outlier_indices = is_outlier.index[is_outlier[col]]
                if not outlier_indices.empty:
                    style_df.loc[outlier_indices, col] = 'background-color: #ffcccc'
            
            return style_df

        st.dataframe(
            pivot_table.style.apply(highlight_outliers_vectorized, axis=None).format("{:,}"), 
            use_container_width=True,
            height=600
        )
        st.caption("※ 붉은색 배경: 해당 행(Row) 내에서 통계적 이상치(IQR 1.5배수 벗어남) 감지")
==== REPLACE
# 분석 시작 버튼
if st.button("📊 분석 시작 (Run Analysis)", type="primary", use_container_width=True):
    
    if not pivot_indices:
        st.error("최소 하나 이상의 피벗 행(Index)을 선택해야 합니다.")
        st.stop()
        
    if filtered_df_step3.empty:
        st.warning("조회 조건에 해당하는 데이터가 없습니다.")
        st.stop()

    # [Data Prep] 결측치 채우기
    fill_values = {col: '미지정' for col in pivot_indices}
    filtered_df_step3[pivot_indices] = filtered_df_step3[pivot_indices].fillna(value=fill_values)

    filtered_df_step3['접수월_str'] = filtered_df_step3['접수일자'].dt.strftime('%Y-%m')
    all_months_in_range = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y-%m').tolist()

    # --- 1. Base Pivot 생성 (전체 월 포함) ---
    try:
        # 인덱스 길이에 따른 동적 소계 로직
        def create_pivot_with_subtotals_dynamic(df, indices, columns, values, aggfunc, all_months):
            # Base
            pivot_base = pd.pivot_table(df, index=indices, columns=columns, values=values, aggfunc=aggfunc, fill_value=0)
            pivot_base = pivot_base.reindex(columns=all_months, fill_value=0)
            
            if pivot_base.empty:
                empty_idx = pd.MultiIndex.from_tuples([], names=indices)
                return pd.DataFrame(0, index=empty_idx, columns=all_months + ['Total'])

            # 소계 로직 (인덱스 레벨에 따라 분기)
            n_levels = len(indices)
            all_parts = []
            
            if n_levels == 1:
                # 1레벨이면 소계 없음
                pivot_base['Total'] = pivot_base.sum(axis=1)
                grand_total = pivot_base.sum()
                grand_total.name = 'Total'
                # Total 행 인덱스 처리
                grand_total_df = grand_total.to_frame('Total').T
                grand_total_df.index = pd.Index(['Total'], name=indices[0])
                return pd.concat([pivot_base, grand_total_df])

            # 2레벨 이상
            for l1_name, l1_group in pivot_base.groupby(level=0, sort=False):
                # Level 2 소계 (3레벨 이상일 때만)
                if n_levels >= 3:
                    for l2_name, l2_group in l1_group.groupby(level=1, sort=False):
                        all_parts.append(l2_group)
                        
                        subtotal_row = l2_group.sum().to_frame().T
                        idx_parts = [l1_name, l2_name, '소계'] + [''] * (n_levels - 3)
                        subtotal_row.index = pd.MultiIndex.from_tuples([tuple(idx_parts)], names=indices)
                        all_parts.append(subtotal_row)
                else:
                    all_parts.append(l1_group)

                # Level 1 전체 합계
                total_l1_row = l1_group.sum().to_frame().T
                idx_parts = [l1_name, '전체 합계'] + [''] * (n_levels - 2)
                total_l1_row.index = pd.MultiIndex.from_tuples([tuple(idx_parts)], names=indices)
                all_parts.append(total_l1_row)
            
            final_pivot = pd.concat(all_parts)
            
            # Grand Total
            grand_total_series = pivot_base.sum()
            grand_total_series.name = "Total"
            grand_total_df = grand_total_series.to_frame('Total').T
            idx_parts = ['Total'] + [''] * (n_levels - 1)
            grand_total_df.index = pd.MultiIndex.from_tuples([tuple(idx_parts)], names=indices)
            
            final_pivot = pd.concat([final_pivot, grand_total_df])
            final_pivot['Total'] = final_pivot[all_months].sum(axis=1)
            return final_pivot

        pivot_table = create_pivot_with_subtotals_dynamic(
            df=filtered_df_step3,
            indices=pivot_indices,
            columns='접수월_str',
            values='상담번호',
            aggfunc='count',
            all_months=all_months_in_range
        )

    except Exception as e:
        st.error(f"피벗 테이블 생성 오류: {e}")
        st.stop()

    # --- 2. Hybrid View & Risk Scoring 적용 ---
    try:
        # A. Hybrid View: 최근 24개월 vs 과거 (연평균)
        cutoff_date = end_date - relativedelta(months=23) # 최근 24개월 시작
        cutoff_str = cutoff_date.strftime('%Y-%m')
        
        # 컬럼 분리
        all_cols = pivot_table.columns.tolist()
        month_cols = [c for c in all_cols if c in all_months_in_range]
        
        old_cols = [c for c in month_cols if c < cutoff_str]
        recent_cols = [c for c in month_cols if c >= cutoff_str]
        
        # 데이터프레임 분리
        df_old = pivot_table[old_cols]
        df_recent = pivot_table[recent_cols]
        
        # Old Period -> 연평균(Year Avg) 변환
        df_old_avg = pd.DataFrame(index=pivot_table.index)
        if not df_old.empty:
            # 연도별로 그룹핑
            years = sorted(list(set([c[:4] for c in old_cols])))
            for y in years:
                y_cols = [c for c in old_cols if c.startswith(y)]
                if y_cols:
                    # mean 계산 후 반올림 (NaN 방지 위해 fillna 0)
                    df_old_avg[f"{y}년(Avg)"] = df_old[y_cols].mean(axis=1).round(1)

        # B. Summary Columns (우측)
        this_year = end_date.year
        last_year = this_year - 1
        
        # 직전년도 Avg
        ly_cols = [c for c in month_cols if c.startswith(str(last_year))]
        ly_avg = pivot_table[ly_cols].mean(axis=1).round(1) if ly_cols else 0
        
        # 당해년도 Avg
        ty_cols = [c for c in month_cols if c.startswith(str(this_year))]
        ty_avg = pivot_table[ty_cols].mean(axis=1).round(1) if ty_cols else 0
        
        # C. Dynamic Risk Scoring (Signal)
        # 전체 History 데이터 준비 (플랜트 전체 기준)
        whole_history_df = master_df[master_df['플랜트'] == selected_plant].copy()
        whole_history_grouped = whole_history_df.groupby(pivot_indices + ['접수월_str']).size()
        
        target_month = recent_cols[-1] if recent_cols else all_months_in_range[-1]
        signals = []
        
        for idx in pivot_table.index:
            # 소계/합계 행은 스킵
            is_subtotal = False
            if isinstance(idx, tuple):
                if any(str(x).endswith('소계') or str(x) in ['전체 합계', 'Total'] for x in idx): is_subtotal = True
            elif str(idx) in ['전체 합계', 'Total']: is_subtotal = True
            
            if is_subtotal:
                signals.append("") # 소계행은 신호 없음
                continue
                
            try:
                # MultiIndex Tuple 매칭
                current_idx = idx if isinstance(idx, tuple) else (idx,)
                series_data = whole_history_grouped.loc[current_idx]
                
                # [함수 호출] 동적 스코어링
                sig, score, reason = calculate_advanced_risk_score(series_data, target_month)
                signals.append(sig)
            except:
                signals.append("⚪") # 데이터 없음
        
        # D. 최종 조립
        final_view = pd.concat([df_old_avg, df_recent], axis=1)
        final_view.insert(0, "🚨", signals) # 신호등 맨 앞
        final_view[f"{last_year}년(Avg)"] = ly_avg
        final_view[f"{this_year}년(Avg)"] = ty_avg
        final_view["Total"] = pivot_table["Total"] # 원본 Total 유지

    except Exception as e:
        st.error(f"Hybrid View 변환 중 오류: {e}")
        st.stop()

    # --- 결과 시각화 ---
    st.subheader(f"📈 분석 결과 ({grade_mode} / {search_mode})")
    
    tab1, tab2, tab3 = st.tabs(["피벗 테이블", "Lag 분석", "원본 데이터"])

    with tab1:
        # [Styling] Hybrid View 전용 스타일링
        def style_hybrid_table(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            
            # 1. 소계/합계 행 회색 처리
            for idx in df.index:
                is_subtotal = False
                if isinstance(idx, tuple):
                    if any(str(x).endswith('소계') or str(x) in ['전체 합계', 'Total'] for x in idx): is_subtotal = True
                elif str(idx) in ['전체 합계', 'Total']: is_subtotal = True
                
                if is_subtotal:
                    styles.loc[idx, :] = 'background-color: #f0f0f0; font-weight: bold'

            # 2. Risk Signal에 따른 최신월 강조
            target_col = target_month # 위에서 정의한 분석 대상 월
            if '🚨' in df.columns and target_col in df.columns:
                for idx in df.index:
                    if styles.loc[idx, target_col] == '': # 소계행 아닐 때만
                        sig = df.loc[idx, '🚨']
                        if sig == "🔴":
                            styles.loc[idx, target_col] = 'background-color: #ffcccc; color: #b91c1c; font-weight: bold'
                        elif sig == "🟡":
                            styles.loc[idx, target_col] = 'background-color: #fff3cd; color: #856404; font-weight: bold'
            
            return styles

        # 숫자 포맷팅 (평균은 소수점, 월별 개수는 정수)
        format_dict = {col: "{:,.1f}" if "Avg" in str(col) else "{:,.0f}" for col in final_view.columns if col != '🚨'}

        st.dataframe(
            final_view.style.apply(style_hybrid_table, axis=None).format(format_dict), 
            use_container_width=True,
            height=(len(final_view) + 1) * 35 + 3
        )
        st.caption(f"※ 🚨: 동적 리스크 스코어링 (🔴:심각 / 🟡:주의) | 기간: 최근 24개월 월별 + 이전 연평균")
>>>>

```
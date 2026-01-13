import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import pyarrow.dataset as ds

# Handle query parameters for navigation from main dashboard
if st.query_params and 'plant' in st.query_params:
    # 새로 들어온 쿼리파라미터일 때만 1회 적용하여, 이후에는 사용자가 자유롭게 필터를 수정할 수 있도록 함
    qp_plant = st.query_params['plant']
    qp_grade = st.query_params.get('grade', '')
    qp_category = st.query_params.get('category', '')
    qp_subcategory = st.query_params.get('subcategory', '')
    qp_key = f"{qp_plant}|{qp_grade}|{qp_category}|{qp_subcategory}"

    if st.session_state.get('last_qp_key') != qp_key:
        st.session_state['last_qp_key'] = qp_key

        # Step 1: 플랜트 선택
        st.session_state['target_plant'] = qp_plant
        st.session_state['plant_changed'] = False  # 외부 진입 시 자동 초기화 방지
        
        # Step 2: 검색 옵션 = Custom (직접 선택) + 사업부문 및 불만원인 모두 선택
        st.session_state['search_mode'] = "Custom (직접 선택)"
        st.session_state['custom_select_all'] = True  # 전체 선택 플래그
        
        # Step 3: 등급, 대분류, 소분류 필터
        # 단일 값을 리스트로 설정 (multiselect는 리스트를 받음)
        st.session_state['step3_grades'] = [qp_grade] if qp_grade else []
        st.session_state['step3_categories'] = [qp_category] if qp_category else []
        st.session_state['step3_subcategories'] = [qp_subcategory] if qp_subcategory else []
        
        # Step 4: 그래프 기준 및 대상 항목
        st.session_state['graph_last_index'] = '소분류'
        st.session_state['graph_selected_values'] = [qp_subcategory] if qp_subcategory else []
        
        # Step 4: 테이블 열 선택 = 등급기준, 대분류, 소분류, 제품범주2
        st.session_state['pivot_indices'] = ['등급기준', '대분류', '소분류', '제품범주2']
        
        # 트리거 플래그 설정
        st.session_state['trigger_analysis'] = True
        st.session_state['from_risk_card'] = True  # Risk 카드에서 진입했음을 표시
        
        # 이전 상태 초기화 (계층적 필터링을 위해)
        if 'prev_grades' in st.session_state:
            del st.session_state['prev_grades']
        if 'prev_categories' in st.session_state:
            del st.session_state['prev_categories']

        # rerun 없이 그대로 진행하여 페이지가 정상 렌더링되도록 함

# [Core Module Import] 
# 핵심 분석 로직은 core/analytics.py에서 가져옵니다.
from core.storage import load_partitioned, DATA_HUB_PATH
from core.analytics import calculate_advanced_risk_score, calculate_lag_stats, detect_outliers_iqr

# --- 0. 설정 및 상수 정의 ---
st.set_page_config(page_title="플랜트 분석", layout="wide")
st.title("🏭 Phase 2.5: 플랜트 정밀 분석 (Dual-Track Scoring)")

# [CONFIG] 등급 기준 매핑
CRITICAL_GRADES = ['중대', '위험', '사고']  
GENERAL_GRADES = ['일반']
PERFORMANCE_REASONS = ['제조불만', '고객불만족', '구매불만']
TARGET_BUSINESS_UNITS = ['식품', 'B2B식품']

# --- 1. 데이터 로드 (Hive Partitioning 지원) ---
def load_master_data():
    try:
        if not DATA_HUB_PATH: return None
        dataset = ds.dataset(DATA_HUB_PATH, partitioning="hive", format="parquet")
        return dataset.to_table().to_pandas()
    except Exception as e:
        return None

master_df = load_master_data()

if master_df is None or master_df.empty:
    st.error("⚠️ 데이터가 없습니다. 먼저 '데이터 업로드' 페이지에서 파일을 저장해주세요.")
    st.stop()

if '접수일자' not in master_df.columns:
    st.error("데이터에 '접수일자' 컬럼이 없습니다.")
    st.stop()
master_df['접수일자'] = pd.to_datetime(master_df['접수일자'])

all_plants = sorted(master_df['플랜트'].dropna().unique().tolist())

# --- 2. Step 1: Scope (플랜트 및 기간) ---
st.markdown("#### Step 1: 분석 범위 설정")
col_s1_1, col_s1_2, col_s1_3 = st.columns([1, 1, 1])

# 플랜트 선택 시 필터 자동 초기화를 위한 콜백
def on_plant_change():
    # 플랜트가 변경되면 step3, step4 필터를 초기화할 준비 (다음 실행 시 적용)
    st.session_state['plant_changed'] = True
    # step3 초기화
    if 'step3_grades' in st.session_state:
        del st.session_state['step3_grades']
    if 'step3_categories' in st.session_state:
        del st.session_state['step3_categories']
    if 'step3_subcategories' in st.session_state:
        del st.session_state['step3_subcategories']
    # step4 초기화
    if 'graph_last_index' in st.session_state:
        del st.session_state['graph_last_index']
    if 'graph_selected_values' in st.session_state:
        del st.session_state['graph_selected_values']
    if 'pivot_indices' in st.session_state:
        del st.session_state['pivot_indices']

with col_s1_1:
    selected_plant = st.selectbox(
        "🏭플랜트 선택", 
        all_plants, 
        key='target_plant' if st.session_state.get('target_plant') else None,
        on_change=on_plant_change
    )

# [Auto-Range] 선택된 플랜트 데이터 범위 감지
plant_specific_data = master_df[master_df['플랜트'] == selected_plant]
if not plant_specific_data.empty:
    min_dt = plant_specific_data['접수일자'].min()
    max_dt = plant_specific_data['접수일자'].max()
    min_date = min_dt.replace(day=1).date()
    # 종료일은 해당 월의 마지막 날로 설정
    next_month = max_dt.replace(day=1) + relativedelta(months=1)
    max_date = (next_month - pd.Timedelta(days=1)).date()
else:
    min_date = master_df['접수일자'].min().date()
    max_date = master_df['접수일자'].max().date()

with col_s1_2:
    start_date = st.date_input("📅시작일 (Start)", value=min_date, min_value=min_date, max_value=max_date)

with col_s1_3:
    end_date = st.date_input("📅종료일 (End)", value=max_date, min_value=min_date, max_value=max_date)

# 1차 필터링
plant_df = master_df[
    (master_df['플랜트'] == selected_plant) &
    (master_df['접수일자'].dt.date >= start_date) &
    (master_df['접수일자'].dt.date <= end_date)
].copy()

if not plant_df.empty:
    st.info(f"📋 **요약**: `{selected_plant}` | `{start_date} ~ {end_date}` | 대상 **{len(plant_df):,}** 건")
else:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
    st.stop()

st.divider()

# --- Step 2 & 3: 검색 옵션 및 등급 필터 ---
col_step2, col_step3 = st.columns(2)

with col_step2:
    st.markdown("#### Step 2: 검색 옵션 (Mode)")
    
    def reset_custom_selections():
        if 'sel_biz' in st.session_state: del st.session_state['sel_biz']
        if 'sel_reason' in st.session_state: del st.session_state['sel_reason']

    search_mode = st.radio(
        "조회 모드를 선택하세요:",
        ("인입 (Inflow)", "실적 (Performance)", "Custom (직접 선택)"),
        horizontal=True,
        key='search_mode',
        on_change=reset_custom_selections
    )

    filtered_df_step2 = plant_df.copy()
    # [중요] 리스크 스코어링을 위한 전체 이력 데이터 (플랜트 기준)
    whole_history_df = master_df[master_df['플랜트'] == selected_plant].copy()

    if search_mode == "인입 (Inflow)":
        cond_biz = filtered_df_step2['사업부문'].isin(TARGET_BUSINESS_UNITS)
        cond_reason = filtered_df_step2['불만원인'].notna()
        filtered_df_step2 = filtered_df_step2[cond_biz & cond_reason]
        
        # [Sync] History Data
        whole_history_df = whole_history_df[
            whole_history_df['사업부문'].isin(TARGET_BUSINESS_UNITS) & 
            whole_history_df['불만원인'].notna()
        ]
        st.caption(f"ℹ️ **인입 기준**: 사업부문({', '.join(TARGET_BUSINESS_UNITS)}) + 불만원인(전체)")

    elif search_mode == "실적 (Performance)":
        cond_biz = filtered_df_step2['사업부문'].isin(TARGET_BUSINESS_UNITS)
        cond_reason = filtered_df_step2['불만원인'].isin(PERFORMANCE_REASONS)
        filtered_df_step2 = filtered_df_step2[cond_biz & cond_reason]
        
        # [Sync] History Data
        whole_history_df = whole_history_df[
            whole_history_df['사업부문'].isin(TARGET_BUSINESS_UNITS) & 
            whole_history_df['불만원인'].isin(PERFORMANCE_REASONS)
        ]
        st.caption(f"ℹ️ **실적 기준**: 사업부문({', '.join(TARGET_BUSINESS_UNITS)}) + 불만원인({', '.join(PERFORMANCE_REASONS)})")

    else: # Custom
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            opts_biz = sorted(plant_df['사업부문'].dropna().unique())
            # Risk 카드에서 진입 시 전체 선택
            if st.session_state.get('custom_select_all', False) and 'sel_biz' not in st.session_state:
                st.session_state['sel_biz'] = opts_biz
            sel_biz = st.multiselect("사업부문 선택", opts_biz, key='sel_biz')
        with col_c2:
            opts_reason = sorted(plant_df['불만원인'].dropna().unique())
            # Risk 카드에서 진입 시 전체 선택
            if st.session_state.get('custom_select_all', False) and 'sel_reason' not in st.session_state:
                st.session_state['sel_reason'] = opts_reason
            sel_reason = st.multiselect("불만원인 선택", opts_reason, key='sel_reason')
        
        # 플래그 리셋
        if st.session_state.get('custom_select_all', False):
            st.session_state['custom_select_all'] = False
        
        if sel_biz:
            filtered_df_step2 = filtered_df_step2[filtered_df_step2['사업부문'].isin(sel_biz)]
            whole_history_df = whole_history_df[whole_history_df['사업부문'].isin(sel_biz)]
        if sel_reason:
            filtered_df_step2 = filtered_df_step2[filtered_df_step2['불만원인'].isin(sel_reason)]
            whole_history_df = whole_history_df[whole_history_df['불만원인'].isin(sel_reason)]

with col_step3:
    st.markdown("#### Step 3: 등급, 대분류 필터")
    
    # === 1. 등급 필터 ===
    grade_options = sorted(filtered_df_step2['등급기준'].dropna().unique())
    # 플랜트 변경 시 또는 초기 진입 시 전체 선택 (단, Risk 카드 진입 시 주입값 유지)
    if 'step3_grades' not in st.session_state or (st.session_state.get('plant_changed', False) and not st.session_state.get('from_risk_card', False)):
        st.session_state['step3_grades'] = grade_options
    
    selected_grades = st.multiselect(
        "분석할 등급을 선택하세요:",
        grade_options,
        key='step3_grades'
    )

    # 등급 선택 여부 체크
    if not selected_grades:
        grade_mode = "선택 없음"
        filtered_df_for_category = filtered_df_step2.copy()
    elif len(selected_grades) == len(grade_options):
        grade_mode = "전체 등급"
        filtered_df_for_category = filtered_df_step2.copy()
    else:
        grade_mode = f"선택 {len(selected_grades)}개 등급"
        filtered_df_for_category = filtered_df_step2[filtered_df_step2['등급기준'].isin(selected_grades)].copy()

    # === 2. 대분류 필터 (등급에 따라 필터링됨) ===
    st.markdown("")
    category_options = sorted(filtered_df_for_category['대분류'].dropna().unique())
    # 등급이 변경되거나 플랜트가 변경되면 대분류를 전체 선택으로 초기화 (단, Risk 카드 진입 시 주입값 유지)
    if ('step3_categories' not in st.session_state or 
        ((st.session_state.get('plant_changed', False) or st.session_state.get('prev_grades') != selected_grades) and not st.session_state.get('from_risk_card', False))):
        st.session_state['step3_categories'] = category_options
        st.session_state['prev_grades'] = selected_grades
    
    selected_categories = st.multiselect(
        "분석할 대분류를 선택하세요:",
        category_options,
        key='step3_categories'
    )
    
    # === 3. 최종 필터링 적용 ===
    filtered_df_step3 = filtered_df_step2.copy()
    whole_history_df_temp = whole_history_df.copy()
    
    if selected_grades:
        filtered_df_step3 = filtered_df_step3[filtered_df_step3['등급기준'].isin(selected_grades)]
        whole_history_df_temp = whole_history_df_temp[whole_history_df_temp['등급기준'].isin(selected_grades)]
    
    if selected_categories:
        filtered_df_step3 = filtered_df_step3[filtered_df_step3['대분류'].isin(selected_categories)]
        whole_history_df_temp = whole_history_df_temp[whole_history_df_temp['대분류'].isin(selected_categories)]
    
    # 선택한 대분류의 하위 중분류, 소분류는 자동으로 포함 (선택 불필요)
    # History Data 업데이트
    whole_history_df = whole_history_df_temp
    
    # plant_changed 플래그 리셋
    if st.session_state.get('plant_changed', False):
        st.session_state['plant_changed'] = False
        st.session_state['from_risk_card'] = False  # 사용자가 직접 변경 시 외부 주입 상태 해제

    cnt_step3 = len(filtered_df_step3)
    st.caption(f"📊 필터링 후 대상 건수: **{cnt_step3:,}** 건")

st.divider()

# --- 5. Step 4: Pivot & Analysis ---
st.markdown("#### Step 4: 상세 분석 (Hybrid Table)")

col_p1, col_p2 = st.columns([1, 1])

with col_p1:
    # 그래프 기준 선택 영역
    st.markdown("**📈 그래프 선 기준** (추이 그래프에서 각 선으로 표시할 기준)")
    all_index_candidates = ['등급기준', '불만원인', '대분류', '중분류', '소분류', '제품범주1', '제품범주2', '제품범주3', '제품명']
    all_index_candidates = [c for c in all_index_candidates if c in filtered_df_step3.columns]
    graph_index_candidates = [c for c in all_index_candidates if c in filtered_df_step3.columns]
    
    # 그래프 기준 기본값: '등급기준'
    if 'graph_last_index' not in st.session_state:
        st.session_state['graph_last_index'] = '등급기준' if '등급기준' in graph_index_candidates else graph_index_candidates[0]
    
    graph_index = st.selectbox(
        "그래프 기준 선택 (1개)",
        graph_index_candidates,
        index=graph_index_candidates.index(st.session_state['graph_last_index']) if st.session_state.get('graph_last_index') in graph_index_candidates else 0
    )

    graph_value_options = sorted(filtered_df_step3[graph_index].dropna().unique()) if graph_index in filtered_df_step3.columns else []
    
    # 그래프 기준이 바뀌면 대상 항목을 전체 선택으로 초기화
    if st.session_state.get('graph_last_index') != graph_index:
        st.session_state['graph_selected_values'] = graph_value_options
        st.session_state['graph_last_index'] = graph_index
    
    # 그래프 대상 항목 기본값: 모두 선택
    if 'graph_selected_values' not in st.session_state:
        st.session_state['graph_selected_values'] = graph_value_options

    graph_selected_values = st.multiselect(
        "그래프 대상 항목 선택", 
        graph_value_options, 
        key='graph_selected_values'
    )

    st.divider()

    # 테이블 행 선택 영역
    st.markdown("**📅 테이블 열 선택** (피벗 테이블의 행 구성)")
    
    # 테이블 열 선택 기본값: '등급기준', '대분류', '소분류'
    default_pivot_indices = ['등급기준', '대분류', '소분류']
    default_pivot_indices = [c for c in default_pivot_indices if c in all_index_candidates]
    if not default_pivot_indices:
        default_pivot_indices = all_index_candidates[:2] if len(all_index_candidates) >= 2 else all_index_candidates[:1]
    
    if 'pivot_indices' not in st.session_state:
        st.session_state['pivot_indices'] = default_pivot_indices
    
    pivot_indices = st.multiselect(
        "피벗 테이블 행 선택", 
        all_index_candidates, 
        key='pivot_indices'
    )

with col_p2:
    st.markdown("""
    ✅ **설정 안내**
    - **그래프 선 기준**: 1개만 선택 (기본값: 등급기준)
    - **그래프 대상 항목**: 다중 선택 가능, 항목당 1개 선 생성 (기본값: 전체)
    - **테이블 열**: 다중 선택 가능 (기본값: 등급기준, 대분류, 소분류)
    - 예: 그래프 기준=등급기준, 항목='일반'+'중대' 선택 → 2개 선 그래프
    """)

if st.button("📊 분석 시작 (Run Analysis)", type="primary", width='stretch'):
    if not pivot_indices:
        st.error("최소 하나 이상의 피벗 행(Index)을 선택해야 합니다.")
        st.stop()
        
    if filtered_df_step3.empty:
        st.warning("조회 조건에 해당하는 데이터가 없습니다.")
        st.stop()

    # [Data Prep] 결측치 채우기 (그래프 기준 포함)
    fill_cols = list(dict.fromkeys(pivot_indices + [graph_index]))
    fill_values = {col: '미지정' for col in fill_cols}
    filtered_df_step3[fill_cols] = filtered_df_step3[fill_cols].fillna(value=fill_values)
    whole_history_df[fill_cols] = whole_history_df[fill_cols].fillna(value=fill_values)

    filtered_df_step3['접수월_str'] = filtered_df_step3['접수일자'].dt.strftime('%Y-%m')
    all_months_in_range = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y-%m').tolist()

    # --- 1. Base Pivot 생성 ---
    pivot_table = pd.DataFrame()
    try:
        def create_pivot_with_subtotals_dynamic(df, indices, columns, values, aggfunc, all_months):
            pivot_base = pd.pivot_table(df, index=indices, columns=columns, values=values, aggfunc=aggfunc, fill_value=0)
            pivot_base = pivot_base.reindex(columns=all_months, fill_value=0)
            
            if pivot_base.empty:
                empty_idx = pd.MultiIndex.from_tuples([], names=indices)
                return pd.DataFrame(0, index=empty_idx, columns=all_months + ['Total'])

            n_levels = len(indices)
            
            if n_levels == 1:
                pivot_base['Total'] = pivot_base.sum(axis=1)
                grand_total = pivot_base.sum()
                grand_total.name = 'Total'
                grand_total_df = grand_total.to_frame('Total').T
                grand_total_df.index = pd.Index(['Total'], name=indices[0])
                return pd.concat([pivot_base, grand_total_df])

            all_parts = []
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
        # Ensure MultiIndex is lexsorted to avoid PerformanceWarning during .loc indexing
        pivot_table_sorted = pivot_table.sort_index()

    except Exception as e:
        st.error(f"피벗 생성 오류: {e}")
        st.stop()

    # --- 2. Hybrid View & Risk Scoring ---
    try:
        # A. Hybrid View
        cutoff_date = end_date - relativedelta(months=23)
        cutoff_str = cutoff_date.strftime('%Y-%m')
        
        all_cols = pivot_table.columns.tolist()
        month_cols = [c for c in all_cols if c in all_months_in_range]

        # 최근 2개년만 표시: 현재 선택 종료일(end_date)의 연도와 그 직전 연도
        try:
            target_year = end_date.year
        except Exception:
            target_year = datetime.now().year
        allowed_years = {target_year, target_year - 1}

        recent_cols = [c for c in month_cols if int(c[:4]) in allowed_years]
        old_cols = [c for c in month_cols if int(c[:4]) not in allowed_years]
        
        df_old = pivot_table[old_cols]
        df_recent = pivot_table[recent_cols]
        
        df_old_avg = pd.DataFrame(index=pivot_table.index)
        if not df_old.empty:
            years = sorted(list(set([c[:4] for c in old_cols])))
            for y in years:
                y_cols = [c for c in old_cols if c.startswith(y)]
                if not y_cols:
                    continue
                year_sum = df_old[y_cols].sum(axis=1).astype(int)
                year_avg = df_old[y_cols].mean(axis=1).round(1)
                col_name = f"{str(y)[-2:]}년 합계(평균)"
                df_old_avg[col_name] = year_sum.astype(str) + "(" + year_avg.astype(str) + ")"

        # B. Summary Columns
        this_year = end_date.year
        last_year = this_year - 1
        
        ly_cols = [c for c in month_cols if c.startswith(str(last_year))]
        ly_sum = pivot_table[ly_cols].sum(axis=1).astype(int) if ly_cols else 0
        ly_avg = pivot_table[ly_cols].mean(axis=1).round(1) if ly_cols else 0
        ly_combined = ly_sum.astype(str) + "(" + ly_avg.astype(str) + ")" if isinstance(ly_sum, pd.Series) else "0(0)"
        
        ty_cols = [c for c in month_cols if c.startswith(str(this_year))]
        ty_sum = pivot_table[ty_cols].sum(axis=1).astype(int) if ty_cols else 0
        ty_avg = pivot_table[ty_cols].mean(axis=1).round(1) if ty_cols else 0
        ty_combined = ty_sum.astype(str) + "(" + ty_avg.astype(str) + ")" if isinstance(ty_sum, pd.Series) else "0(0)"
        
        # C. Risk Scoring
        whole_history_df['접수월_str'] = whole_history_df['접수일자'].dt.strftime('%Y-%m')
        whole_history_grouped = whole_history_df.groupby(pivot_indices + ['접수월_str']).size()
        
        target_month = recent_cols[-1] if recent_cols else all_months_in_range[-1]
        signals = []
        reasons = [] 
        
        for idx in pivot_table.index:
            is_subtotal = False
            if isinstance(idx, tuple):
                if any(str(x).endswith('소계') or str(x) in ['전체 합계', 'Total'] for x in idx): is_subtotal = True
            elif str(idx) in ['전체 합계', 'Total']: is_subtotal = True
            
            if is_subtotal:
                signals.append("")
                reasons.append("")
                continue
                
            try:
                # [MODIFIED] Grade Extraction & Passing
                current_idx = idx if isinstance(idx, tuple) else (idx,)
                
                # 1. 데이터 추출
                series_data = whole_history_grouped.loc[current_idx]

                # [Fix] Zero-Filling & History Sync (점수 불일치 해결 핵심)
                if not series_data.index.empty:
                    series_data.index = pd.to_datetime(series_data.index)
                    
                    # 분석 기준일 (Target Month)
                    target_ts = pd.to_datetime(target_month)
                    
                    # [핵심 변경] 데이터의 시작점이 아니라, '기준일로부터 24개월 전'을 강제 시작점으로 잡음
                    # 이유: app.py는 전체 기간을 보는데, 여기만 기간이 짧으면 평균이 높게 왜곡됨 (분모 확보)
                    force_start_date = target_ts - relativedelta(months=24)
                    
                    # 전체 월 인덱스 생성 (24개월 전 ~ 현재)
                    full_idx = pd.date_range(start=force_start_date, end=target_ts, freq='MS')
                    
                    # 0으로 채우기 (과거 데이터가 없으면 0으로 간주하여 평균을 낮춤 -> 급증 감지력 강화)
                    series_data = series_data.reindex(full_idx, fill_value=0)

                # [Fix] Grade값 매핑
                if '등급기준' in pivot_indices:
                    grade_pos = pivot_indices.index('등급기준')
                    current_grade = current_idx[grade_pos] if isinstance(current_idx, tuple) else current_idx
                else:
                    current_grade = "일반"

                # 2. 엔진 호출
                sig, score, reason = calculate_advanced_risk_score(series_data, target_month, grade=current_grade)
                
                # 3. 결과 저장
                signals.append(sig)
                reasons.append(f"[{score}점] {reason}")
                
            except:
                signals.append("⚪")
                reasons.append("데이터 없음")
        
        # D. Assembly
        final_view = pd.concat([df_old_avg, df_recent], axis=1)
        final_view.insert(0, "🚨", signals)
        final_view.insert(1, "진단", reasons)
        final_view[f"{str(last_year)[-2:]}년 합계(평균)"] = ly_combined
        final_view[f"{str(this_year)[-2:]}년 합계(평균)"] = ty_combined
        final_view["Total"] = pivot_table["Total"]

    except Exception as e:
        st.error(f"Hybrid View 변환 중 오류: {e}")
        st.stop()

    # --- 시각화 ---
    st.subheader(f"📈 분석 결과 ({grade_mode} / {search_mode})")
    
    # --- 1. 그래프 먼저 생성 (graph_index 기준) ---
    try:
        # 2개년 분리
        end_year = pd.to_datetime(end_date).year
        recent_years_set = {end_year, end_year - 1}
        recent_months = [c for c in all_months_in_range if int(c[:4]) in recent_years_set]
        
        graph_df = filtered_df_step3.copy()
        if graph_selected_values:
            graph_df = graph_df[graph_df[graph_index].isin(graph_selected_values)]
        
        if graph_df.empty:
            st.warning("그래프 생성 대상 데이터가 없습니다. 그래프 기준 항목을 확인하세요.")
        else:
            # graph_index 기준으로 피벗
            pivot_for_graph = pd.pivot_table(
                graph_df,
                index=graph_index,
                columns='접수월_str',
                values='상담번호',
                aggfunc='count',
                fill_value=0
            )
            pivot_for_graph = pivot_for_graph.reindex(columns=all_months_in_range, fill_value=0)
            
            # 그래프 제목 계산
            start_month = recent_months[0] if recent_months else all_months_in_range[0]
            end_month = recent_months[-1] if recent_months else all_months_in_range[-1]
            st.markdown(f"#### 📊 2개년 추이 분석 (그래프 상 시작 {start_month} ~ 끝 {end_month})")
            
            # Plotly 선 그래프 구성
            fig = px.line(title=f"2개년 클레임 건수 추이 ({graph_index} 기준)")
            
            colors = px.colors.qualitative.Plotly
            
            # 각 graph_index 값별 선 그리기
            for idx, category in enumerate(pivot_for_graph.index):
                color = colors[idx % len(colors)]
                category_data = pivot_for_graph.loc[category]
                
                if recent_months:
                    recent_data = category_data[recent_months]
                    fig.add_scatter(
                        x=recent_months,
                        y=recent_data.values,
                        mode='lines+markers',
                        name=f'{category}',
                        line=dict(color=color, width=2),
                        marker=dict(size=6),
                        legendgroup=category,
                        showlegend=True
                    )
            
            fig.update_layout(
                xaxis_title="월별 (Month)",
                yaxis_title="클레임 건수 (건)",
                hovermode='x unified',
                height=450,
                template="plotly_white",
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01
                )
            )
            
            st.plotly_chart(fig, width='stretch')
        
    except Exception as e:
        st.warning(f"그래프 생성 중 오류: {e}")
    
    st.divider()

    # --- Risk 현황진단 ---
    st.markdown("#### 🛡️ Risk 현황진단")
    try:
        risk_rows = final_view.reset_index()

        # 소계/합계 행 제거 판단
        def _is_subtotal_row(row):
            try:
                for col in pivot_indices:
                    if col in risk_rows.columns:
                        val = str(row[col])
                        if val.endswith('소계') or val in ['전체 합계', 'Total']:
                            return True
                return False
            except Exception:
                return False

        risk_rows['__subtotal__'] = risk_rows.apply(_is_subtotal_row, axis=1)
        alerts_df = risk_rows[(risk_rows['🚨'].isin(['🔴','🟡'])) & (~risk_rows['__subtotal__'])]

        # 전월/당월 계산
        try:
            prev_date = datetime.strptime(target_month, "%Y-%m") - relativedelta(months=1)
            prev_month = prev_date.strftime("%Y-%m")
        except Exception:
            prev_month = None

        def _attach_counts(row):
            prev_cnt, curr_cnt = 0, 0
            try:
                # 인덱스 키 구성
                key_tuple = tuple(row[col] for col in pivot_indices if col in risk_rows.columns)
                key = key_tuple if len(key_tuple) > 1 else (key_tuple[0] if len(key_tuple) == 1 else None)
                if key is not None:
                    if prev_month and prev_month in pivot_table_sorted.columns:
                        prev_cnt = int(pivot_table_sorted.loc[key, prev_month]) if prev_month in pivot_table_sorted.columns else 0
                    if target_month in pivot_table_sorted.columns:
                        curr_cnt = int(pivot_table_sorted.loc[key, target_month])
            except Exception:
                pass
            return pd.Series({"전월": prev_cnt, "당월": curr_cnt})

        if not alerts_df.empty:
            counts_df = alerts_df.apply(_attach_counts, axis=1)
            alerts_df = pd.concat([alerts_df, counts_df], axis=1)

        red_count = int((alerts_df['🚨'] == '🔴').sum()) if not alerts_df.empty else 0
        yellow_count = int((alerts_df['🚨'] == '🟡').sum()) if not alerts_df.empty else 0

        # 좌/우 컬럼으로 분리 표시
        c_left, c_right = st.columns(2)
        if not alerts_df.empty:
            display_cols = [c for c in pivot_indices if c in alerts_df.columns] + ['전월', '당월', '진단']
            red_df = alerts_df[alerts_df['🚨'] == '🔴']
            yellow_df = alerts_df[alerts_df['🚨'] == '🟡']

            with c_left:
                st.markdown(f"##### Red(🔴) 경보: {len(red_df)}건")
                if red_df.empty:
                    st.info("레드 패턴 없음")
                else:
                    st.dataframe(
                        red_df[display_cols],
                        width='stretch',
                        height=min(360, (len(red_df) + 1) * 35)
                    )
            with c_right:
                st.markdown(f"##### Yellow(🟡) 주의: {len(yellow_df)}건")
                if yellow_df.empty:
                    st.info("옐로우 패턴 없음")
                else:
                    st.dataframe(
                        yellow_df[display_cols],
                        width='stretch',
                        height=min(360, (len(yellow_df) + 1) * 35)
                    )
            st.caption("전월/당월은 해당 항목의 월별 클레임 건수입니다. 점수산정 사유는 '진단' 컬럼에 요약되어 있습니다.")
        else:
            st.info("현재 경보 또는 주의 대상이 없습니다.")
    except Exception as e:
        st.warning(f"Risk 현황진단 생성 중 오류: {e}")

    st.divider()

    # --- 2. 피벗 테이블 (pivot_indices 기준) ---
    tab1, tab2, tab3 = st.tabs(["피벗 테이블", "Lag 분석", "원본 데이터"])

    with tab1:
        def style_hybrid_table(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            
            # 소계/합계 회색
            for idx in df.index:
                is_subtotal = False
                if isinstance(idx, tuple):
                    if any(str(x).endswith('소계') or str(x) in ['전체 합계', 'Total'] for x in idx): is_subtotal = True
                elif str(idx) in ['전체 합계', 'Total']: is_subtotal = True
                
                if is_subtotal:
                    styles.loc[idx, :] = 'background-color: #f0f0f0; font-weight: bold'

            # 리스크 경보 강조
            target_col = target_month
            if '🚨' in df.columns and target_col in df.columns:
                for idx in df.index:
                    if styles.loc[idx, target_col] == '': 
                        sig = df.loc[idx, '🚨']
                        if sig == "🔴":
                            styles.loc[idx, target_col] = 'background-color: #ffcccc; color: #b91c1c; font-weight: bold'
                        elif sig == "🟡":
                            styles.loc[idx, target_col] = 'background-color: #fff3cd; color: #856404; font-weight: bold'
            return styles

        format_dict = {
            col: "{:,.0f}"
            for col in final_view.columns
            if col not in ['🚨', '진단'] and '합계(평균)' not in str(col)
        }

        st.dataframe(
            final_view.style.apply(style_hybrid_table, axis=None).format(format_dict), 
            width='stretch',
            height=(len(final_view) + 1) * 35 + 3,
            column_config={
                "진단": st.column_config.TextColumn("위험 진단", help="AI 엔진이 판단한 위험 점수와 사유입니다.")
            }
        )
        st.caption(f"※ 🚨: Dual-Track 리스크 스코어링 (🔴:위험 / 🟡:주의) | 기간: 최근 24개월 + 과거 연평균")

    with tab2:
        st.markdown("##### ⏱️ Lag 분석 (제조 ~ 접수 소요기간)")
        lag_stats = calculate_lag_stats(filtered_df_step3)
        if lag_stats and lag_stats['count'] > 0:
            c1, c2, c3 = st.columns(3)
            c1.metric("평균 Lag", f"{lag_stats['mean']:.1f} 일")
            median_val = lag_stats.get('p50', 0)
            c2.metric("중앙값 Lag", f"{median_val:.1f} 일")
            c3.metric("대상 건수", f"{lag_stats['count']:,} 건")
            
            valid_lag_df = filtered_df_step3[filtered_df_step3['Lag_Valid'] == True]
            fig = px.histogram(valid_lag_df, x='Lag_Days', nbins=50, title="Lag Days Distribution")
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("유효 데이터 없음")

    with tab3:
        st.dataframe(filtered_df_step3)
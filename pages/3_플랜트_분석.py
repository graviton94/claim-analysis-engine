import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import pyarrow.dataset as ds

# [Core Module Import]
# Single Source of Truth 원칙에 따라 core 모듈에서 로직을 가져옵니다.
from core.storage import load_and_filter_data, get_claim_keys, DATA_HUB_PATH
from core.analytics import calculate_advanced_risk_score, calculate_lag_stats, prepare_risk_data

# --- 0. 설정 및 상수 정의 ---
st.set_page_config(page_title="플랜트 분석", layout="wide")
st.title("🏭 플랜트 클레임 현황 레포트 생성")

# [CONFIG] 필터링 기준값 (Core와 동기화 필요 시 core.config로 이동 권장)
TARGET_BUSINESS_UNITS = ['식품', 'B2B식품']
PERFORMANCE_REASONS = ['제조불만', '고객불만족', '구매불만']

# Query Parameter Handling (Navigation from Main Dashboard)
if st.query_params and 'plant' in st.query_params:
    qp_plant = st.query_params['plant']
    qp_grade = st.query_params.get('grade', '')
    qp_category = st.query_params.get('category', '')
    qp_subcategory = st.query_params.get('subcategory', '')
    qp_key = f"{qp_plant}|{qp_grade}|{qp_category}|{qp_subcategory}"

    if st.session_state.get('last_qp_key') != qp_key:
        st.session_state['last_qp_key'] = qp_key
        st.session_state['target_plant'] = qp_plant
        st.session_state['plant_changed'] = False
        st.session_state['search_mode'] = "Custom (직접 선택)"
        st.session_state['custom_select_all'] = True
        st.session_state['step3_grades'] = [qp_grade] if qp_grade else []
        st.session_state['step3_categories'] = [qp_category] if qp_category else []
        st.session_state['step3_subcategories'] = [qp_subcategory] if qp_subcategory else []
        st.session_state['graph_last_index'] = '소분류'
        st.session_state['graph_selected_values'] = [qp_subcategory] if qp_subcategory else []
        st.session_state['pivot_indices'] = ['등급기준', '대분류', '소분류', '제품범주2']
        st.session_state['trigger_analysis'] = True
        st.session_state['from_risk_card'] = True
        
        if 'prev_grades' in st.session_state: del st.session_state['prev_grades']
        if 'prev_categories' in st.session_state: del st.session_state['prev_categories']

# --- 1. 초기 데이터 로드 (플랜트 목록) ---
@st.cache_data(ttl=3600)
def load_plant_list():
    """초기 진입 시 플랜트 목록만 가볍게 로드"""
    keys = get_claim_keys(DATA_HUB_PATH)
    if not keys.empty:
        return sorted(keys['플랜트'].unique().tolist())
    return []

all_plants = load_plant_list()

if not all_plants:
    st.error("⚠️ 데이터가 없습니다. 먼저 '데이터 업로드' 페이지에서 파일을 저장해주세요.")
    st.stop()

# --- 2. Step 1: Scope (플랜트 및 기간) ---
st.markdown("#### Step 1: 분석 범위 설정")
col_s1_1, col_s1_2, col_s1_3 = st.columns([1, 1, 1])

def on_plant_change():
    st.session_state['plant_changed'] = True
    # 필터 초기화
    keys_to_clear = ['step3_grades', 'step3_categories', 'step3_subcategories', 
                     'graph_last_index', 'graph_selected_values', 'pivot_indices']
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]

with col_s1_1:
    selected_plant = st.selectbox(
        "🏭플랜트 선택", 
        all_plants, 
        key='target_plant' if st.session_state.get('target_plant') else None,
        on_change=on_plant_change
    )

# [Auto-Range] 기간 설정은 일단 기본값(전월/당월 등) 혹은 로직으로 처리
# 전체 데이터를 읽지 않으므로, 메타데이터가 없다면 datetime.now() 기준 설정
# 여기서는 심플하게 오늘 기준 전월 1일 ~ 말일 로직 적용 or 세션 유지
today = datetime.today()
default_start = (today.replace(day=1) - relativedelta(months=1)).date()
default_end = (today.replace(day=1) - relativedelta(days=1)).date()

with col_s1_2:
    start_date = st.date_input("📅시작일 (Start)", value=default_start)

with col_s1_3:
    end_date = st.date_input("📅종료일 (End)", value=default_end)

# Step 1 필터링된 데이터 로드 (옵션 갱신용)
# *주의: 여기서 전체를 다 가져오는게 아니라, 필터 옵션을 채우기 위한 가벼운 로드가 이상적이나,
#       통합 로더를 사용하여 일관성을 유지함.
#       사용자가 'Custom' 모드 등에서 옵션을 보기 위해선 해당 기간의 데이터가 필요함.
@st.cache_data(show_spinner=False)
def get_options_data(plant, s_date, e_date):
    # 옵션용 데이터는 기본 모드(전체)로 로드하여 가능한 모든 옵션을 보여줌
    return load_and_filter_data(plant, s_date, e_date, search_mode="Custom")

with st.spinner("데이터 조회 중..."):
    df_for_options = get_options_data(selected_plant, start_date, end_date)

if df_for_options.empty:
    st.warning(f"선택한 조건 ({selected_plant}, {start_date}~{end_date})에 해당하는 데이터가 없습니다.")
    st.stop()
else:
    st.info(f"📋 **요약**: `{selected_plant}` | `{start_date} ~ {end_date}` | 대상 **{len(df_for_options):,}** 건 (옵션 갱신용)")

st.divider()

# --- Step 2 & 3: 검색 옵션 및 등급 필터 ---
col_step2, col_step3 = st.columns(2)

# 필터링 변수 초기화
selected_biz_list = []
selected_reason_list = []
selected_grade_list = []
selected_category_list = []

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

    if search_mode == "인입 (Inflow)":
        st.caption(f"ℹ️ **인입 기준**: 사업부문({', '.join(TARGET_BUSINESS_UNITS)}) + 불만원인(전체)")
        # Core 로더 내부에서 처리하므로 리스트는 None으로 둠 (또는 명시적 전달 가능)
        
    elif search_mode == "실적 (Performance)":
        st.caption(f"ℹ️ **실적 기준**: 사업부문({', '.join(TARGET_BUSINESS_UNITS)}) + 불만원인({', '.join(PERFORMANCE_REASONS)})")
        
    else: # Custom
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            opts_biz = sorted(df_for_options['사업부문'].dropna().unique())
            if st.session_state.get('custom_select_all', False) and 'sel_biz' not in st.session_state:
                st.session_state['sel_biz'] = opts_biz
            selected_biz_list = st.multiselect("사업부문 선택", opts_biz, key='sel_biz')
        
        with col_c2:
            opts_reason = sorted(df_for_options['불만원인'].dropna().unique())
            if st.session_state.get('custom_select_all', False) and 'sel_reason' not in st.session_state:
                st.session_state['sel_reason'] = opts_reason
            selected_reason_list = st.multiselect("불만원인 선택", opts_reason, key='sel_reason')
        
        if st.session_state.get('custom_select_all', False):
            st.session_state['custom_select_all'] = False

with col_step3:
    st.markdown("#### Step 3: 등급, 대분류 필터")
    
    # 1. 등급 필터
    grade_options = sorted(df_for_options['등급기준'].dropna().unique())
    
    if 'step3_grades' not in st.session_state or (st.session_state.get('plant_changed', False) and not st.session_state.get('from_risk_card', False)):
        st.session_state['step3_grades'] = grade_options
    
    selected_grade_list = st.multiselect("분석할 등급을 선택하세요:", grade_options, key='step3_grades')
    
    if not selected_grade_list: grade_mode = "선택 없음"
    elif len(selected_grade_list) == len(grade_options): grade_mode = "전체 등급"
    else: grade_mode = f"선택 {len(selected_grade_list)}개 등급"

    # 2. 대분류 필터 (선택된 등급에 해당하는 대분류만 표시)
    st.markdown("")
    # 메모리상에서 임시 필터링하여 옵션 추출
    temp_df = df_for_options
    if selected_grade_list:
        temp_df = temp_df[temp_df['등급기준'].isin(selected_grade_list)]
    
    category_options = sorted(temp_df['대분류'].dropna().unique())
    
    if ('step3_categories' not in st.session_state or 
        ((st.session_state.get('plant_changed', False) or st.session_state.get('prev_grades') != selected_grade_list) and not st.session_state.get('from_risk_card', False))):
        st.session_state['step3_categories'] = category_options
        st.session_state['prev_grades'] = selected_grade_list
    
    selected_category_list = st.multiselect("분석할 대분류를 선택하세요:", category_options, key='step3_categories')
    
    if st.session_state.get('plant_changed', False):
        st.session_state['plant_changed'] = False
        st.session_state['from_risk_card'] = False

st.divider()

# --- 5. Step 4: Pivot & Analysis ---
st.markdown("#### Step 4: 그래프 조정 (Hybrid Table)")

# 옵션용 데이터에서 컬럼 추출
all_index_candidates = ['등급기준', '불만원인', '대분류', '중분류', '소분류', '제품범주1', '제품범주2', '제품범주3', '제품명']
available_indices = [c for c in all_index_candidates if c in df_for_options.columns]

col_p1, col_p2 = st.columns([1, 1])

with col_p1:
    st.markdown("**📈 그래프 선 기준**")
    if 'graph_last_index' not in st.session_state:
        st.session_state['graph_last_index'] = '등급기준' if '등급기준' in available_indices else available_indices[0]
    
    graph_index = st.selectbox(
        "그래프 기준 선택 (1개)",
        available_indices,
        index=available_indices.index(st.session_state['graph_last_index']) if st.session_state.get('graph_last_index') in available_indices else 0
    )
    
    # 그래프 값 옵션 (선택된 데이터 기반)
    graph_value_options = sorted(df_for_options[graph_index].dropna().unique()) if graph_index in df_for_options.columns else []
    
    if st.session_state.get('graph_last_index') != graph_index:
        st.session_state['graph_selected_values'] = graph_value_options
        st.session_state['graph_last_index'] = graph_index
        
    if 'graph_selected_values' not in st.session_state:
        st.session_state['graph_selected_values'] = graph_value_options
        
    graph_selected_values = st.multiselect("그래프 대상 항목 선택", graph_value_options, key='graph_selected_values')

    st.divider()
    
    st.markdown("**📅 테이블 열 선택**")
    default_indices = ['등급기준', '대분류', '소분류']
    default_indices = [c for c in default_indices if c in available_indices]
    if not default_indices: default_indices = available_indices[:2]
    
    if 'pivot_indices' not in st.session_state:
        st.session_state['pivot_indices'] = default_indices
        
    pivot_indices = st.multiselect("피벗 테이블 행 선택", available_indices, key='pivot_indices')

with col_p2:
    st.markdown("""
    ✅ **설정 안내**
    - **그래프 선 기준**: 1개만 선택 (기본값: 등급기준)
    - **테이블 열**: 다중 선택 가능 (기본값: 등급기준, 대분류, 소분류)
    """)

# [ACTION] 분석 시작
if st.button("📊 분석 시작 (Run Analysis)", type="primary", width='stretch'):
    if not pivot_indices:
        st.error("최소 하나 이상의 피벗 행(Index)을 선택해야 합니다.")
        st.stop()

    # --- 1. Data Loading (Core Logic) ---
    # A. Display Data (선택된 기간)
    df_display = load_and_filter_data(
        plant=selected_plant,
        start_date=start_date,
        end_date=end_date,
        search_mode=search_mode,
        selected_biz=selected_biz_list,
        selected_reasons=selected_reason_list,
        selected_grades=selected_grade_list,
        selected_categories=selected_category_list
    )
    
    # B. Risk Data (과거 24개월 확보)
    # 리스크 점수 산출을 위해 더 넓은 범위의 데이터를 로드해야 함
    risk_start_date = start_date - relativedelta(months=24)
    df_risk_raw = load_and_filter_data(
        plant=selected_plant,
        start_date=risk_start_date,
        end_date=end_date,
        search_mode=search_mode,
        selected_biz=selected_biz_list,
        selected_reasons=selected_reason_list,
        selected_grades=selected_grade_list,
        selected_categories=selected_category_list
    )

    if df_display.empty:
        st.warning("조회 조건에 해당하는 데이터가 없습니다.")
        st.stop()

    # --- 2. Data Prep & Pivot ---
    # 그래프/테이블용 파생변수
    df_display['접수월_str'] = df_display['접수일자'].dt.strftime('%Y-%m')
    all_months_in_range = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%Y-%m').tolist()
    
    # 결측치 채우기 (표시용)
    fill_cols = list(dict.fromkeys(pivot_indices + [graph_index]))
    fill_values = {col: '미지정' for col in fill_cols}
    df_display[fill_cols] = df_display[fill_cols].fillna(value=fill_values)
    df_risk_raw[fill_cols] = df_risk_raw[fill_cols].fillna(value=fill_values)

    # [Core] Zero-Filling & Risk Data Prep
    # prepare_risk_data는 MultiIndex DataFrame을 반환함
    risk_pivot_df = prepare_risk_data(
        df=df_risk_raw,
        pivot_keys=pivot_indices,
        target_date=end_date,
        lookback_months=24
    )

    # --- 3. Base Pivot Generation (for Display) ---
    # 기존 로직 유지 (Subtotals 등 복잡한 로직은 Pandas PivotTable 활용)
    def create_pivot_with_subtotals_dynamic(df, indices, columns, values, aggfunc, all_months):
        pivot_base = pd.pivot_table(df, index=indices, columns=columns, values=values, aggfunc=aggfunc, fill_value=0)
        pivot_base = pivot_base.reindex(columns=all_months, fill_value=0)
        
        if pivot_base.empty:
            empty_idx = pd.MultiIndex.from_tuples([], names=indices)
            return pd.DataFrame(0, index=empty_idx, columns=all_months + ['Total'])

        n_levels = len(indices)
        
        # 1-Level Pivot
        if n_levels == 1:
            pivot_base['Total'] = pivot_base.sum(axis=1)
            grand_total = pivot_base.sum()
            grand_total.name = 'Total'
            grand_total_df = grand_total.to_frame('Total').T
            grand_total_df.index = pd.Index(['Total'], name=indices[0])
            return pd.concat([pivot_base, grand_total_df])

        # Multi-Level Pivot (Subtotals)
        all_parts = []
        for l1_name, l1_group in pivot_base.groupby(level=0, sort=False):
            # Level 2 소계
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

    try:
        pivot_table = create_pivot_with_subtotals_dynamic(
            df=df_display,
            indices=pivot_indices,
            columns='접수월_str',
            values='상담번호', # Count target
            aggfunc='count',
            all_months=all_months_in_range
        )
        pivot_table_sorted = pivot_table.sort_index()
    except Exception as e:
        st.error(f"피벗 생성 오류: {e}")
        st.stop()

    # --- 4. Hybrid View & Risk Scoring ---
    try:
        # A. Hybrid View Columns
        all_cols = pivot_table.columns.tolist()
        month_cols = [c for c in all_cols if c in all_months_in_range]
        
        try:
            target_year = end_date.year
        except:
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
                if not y_cols: continue
                year_sum = df_old[y_cols].sum(axis=1).astype(int)
                year_avg = df_old[y_cols].mean(axis=1).round(1)
                col_name = f"{str(y)[-2:]}년 합계(평균)"
                df_old_avg[col_name] = year_sum.astype(str) + "(" + year_avg.astype(str) + ")"

        # B. Summary Cols
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

        # C. Risk Scoring (Loop)
        target_month_str = recent_cols[-1] if recent_cols else all_months_in_range[-1]
        signals = []
        reasons = []
        
        # risk_pivot_df는 이미 Zero-filled & MultiIndex
        
        for idx in pivot_table.index:
            # Subtotal 행 스킵
            is_subtotal = False
            if isinstance(idx, tuple):
                if any(str(x).endswith('소계') or str(x) in ['전체 합계', 'Total'] for x in idx): is_subtotal = True
            elif str(idx) in ['전체 합계', 'Total']: is_subtotal = True
            
            if is_subtotal:
                signals.append("")
                reasons.append("")
                continue
                
            try:
                # 1. 인덱스 매칭: Display Index -> Risk Pivot Index
                # pivot_indices가 그대로 키가 됨
                current_idx = idx if isinstance(idx, tuple) else (idx,)
                
                # 2. 데이터 추출 (from risk_pivot_df)
                if current_idx in risk_pivot_df.index:
                    series_data = risk_pivot_df.loc[current_idx] # Series (Index=Date)
                else:
                    # 해당 조합의 과거 데이터가 아예 없는 경우 (Zero-filled series 생성)
                    # prepare_risk_data는 존재하는 조합만 만들므로, 여기 없으면 0임
                    series_data = pd.Series(0, index=risk_pivot_df.columns)

                # 3. Grade 매핑
                if '등급기준' in pivot_indices:
                    grade_pos = pivot_indices.index('등급기준')
                    current_grade = current_idx[grade_pos]
                else:
                    current_grade = "일반"

                # 4. Engine 호출 (Core Function)
                sig, score, reason = calculate_advanced_risk_score(series_data, target_month_str, grade=current_grade)
                
                signals.append(sig)
                reasons.append(f"[{score}점] {reason}")
            except Exception as e:
                signals.append("⚪")
                reasons.append(f"Err: {str(e)}")

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

    # --- Graph ---
    try:
        end_year = pd.to_datetime(end_date).year
        recent_years_set = {end_year, end_year - 1}
        recent_months = [c for c in all_months_in_range if int(c[:4]) in recent_years_set]
        
        graph_df = df_display.copy()
        if graph_selected_values:
            graph_df = graph_df[graph_df[graph_index].isin(graph_selected_values)]
            
        if graph_df.empty:
            st.warning("그래프 생성 대상 데이터가 없습니다.")
        else:
            pivot_for_graph = pd.pivot_table(
                graph_df, index=graph_index, columns='접수월_str', values='상담번호', aggfunc='count', fill_value=0
            )
            pivot_for_graph = pivot_for_graph.reindex(columns=all_months_in_range, fill_value=0)
            
            fig = px.line(title=f"2개년 클레임 건수 추이 ({graph_index} 기준)")
            colors = px.colors.qualitative.Plotly
            
            for idx, category in enumerate(pivot_for_graph.index):
                color = colors[idx % len(colors)]
                category_data = pivot_for_graph.loc[category]
                if recent_months:
                    recent_data = category_data[recent_months]
                    fig.add_scatter(
                        x=recent_months, y=recent_data.values, mode='lines+markers', name=f'{category}',
                        line=dict(color=color, width=2), marker=dict(size=6),
                        legendgroup=category, showlegend=True
                    )
            
            fig.update_layout(
                xaxis_title="월별 (Month)", yaxis_title="클레임 건수 (건)",
                hovermode='x unified', height=450, template="plotly_white",
                legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.01)
            )
            st.plotly_chart(fig, width='stretch')
    except Exception as e:
        st.warning(f"그래프 생성 오류: {e}")

    st.divider()

    # --- Risk Alerts UI ---
    st.markdown("#### 🛡️ Risk 현황진단")
    try:
        risk_rows = final_view.reset_index()
        def _is_subtotal_row_u(row):
            for col in pivot_indices:
                if col in risk_rows.columns:
                    val = str(row[col])
                    if val.endswith('소계') or val in ['전체 합계', 'Total']: return True
            return False
        
        risk_rows['__subtotal__'] = risk_rows.apply(_is_subtotal_row_u, axis=1)
        alerts_df = risk_rows[(risk_rows['🚨'].isin(['🔴','🟡'])) & (~risk_rows['__subtotal__'])]
        
        # 전월/당월 계산을 위해 pivot_table_sorted 사용
        try:
            prev_date = datetime.strptime(target_month_str, "%Y-%m") - relativedelta(months=1)
            prev_month = prev_date.strftime("%Y-%m")
        except: prev_month = None

        def _attach_counts(row):
            prev_cnt, curr_cnt = 0, 0
            try:
                key_tuple = tuple(row[col] for col in pivot_indices if col in risk_rows.columns)
                key = key_tuple if len(key_tuple) > 1 else (key_tuple[0] if len(key_tuple) == 1 else None)
                if key is not None:
                    if prev_month and prev_month in pivot_table_sorted.columns:
                        prev_cnt = int(pivot_table_sorted.loc[key, prev_month])
                    if target_month_str in pivot_table_sorted.columns:
                        curr_cnt = int(pivot_table_sorted.loc[key, target_month_str])
            except: pass
            return pd.Series({"전월": prev_cnt, "당월": curr_cnt})

        if not alerts_df.empty:
            counts_df = alerts_df.apply(_attach_counts, axis=1)
            alerts_df = pd.concat([alerts_df, counts_df], axis=1)
            
            c_left, c_right = st.columns(2)
            display_cols = [c for c in pivot_indices if c in alerts_df.columns] + ['전월', '당월', '진단']
            
            red_df = alerts_df[alerts_df['🚨'] == '🔴']
            yellow_df = alerts_df[alerts_df['🚨'] == '🟡']
            
            with c_left:
                st.markdown(f"##### Red(🔴) 경보: {len(red_df)}건")
                if not red_df.empty: st.dataframe(red_df[display_cols], width='stretch')
                else: st.info("레드 패턴 없음")
            with c_right:
                st.markdown(f"##### Yellow(🟡) 주의: {len(yellow_df)}건")
                if not yellow_df.empty: st.dataframe(yellow_df[display_cols], width='stretch')
                else: st.info("옐로우 패턴 없음")
        else:
            st.info("현재 경보 또는 주의 대상이 없습니다.")
    except Exception as e:
        st.warning(f"Risk 진단 오류: {e}")

    st.divider()

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["피벗 테이블", "Lag 분석", "원본 데이터"])

    with tab1:
        def style_hybrid_table(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            target_col = target_month_str
            
            for idx in df.index:
                is_subtotal = False
                if isinstance(idx, tuple):
                    if any(str(x).endswith('소계') or str(x) in ['전체 합계', 'Total'] for x in idx): is_subtotal = True
                elif str(idx) in ['전체 합계', 'Total']: is_subtotal = True
                
                if is_subtotal:
                    styles.loc[idx, :] = 'background-color: #f0f0f0; font-weight: bold'
                
                if '🚨' in df.columns and target_col in df.columns:
                    if styles.loc[idx, target_col] == '':
                        sig = df.loc[idx, '🚨']
                        if sig == "🔴":
                            styles.loc[idx, target_col] = 'background-color: #ffcccc; color: #b91c1c; font-weight: bold'
                        elif sig == "🟡":
                            styles.loc[idx, target_col] = 'background-color: #fff3cd; color: #856404; font-weight: bold'
            return styles

        format_dict = {
            col: "{:,.0f}" for col in final_view.columns 
            if col not in ['🚨', '진단'] and '합계(평균)' not in str(col)
        }
        
        st.dataframe(
            final_view.style.apply(style_hybrid_table, axis=None).format(format_dict),
            width='stretch', height=(len(final_view)+1)*35+3,
            column_config={"진단": st.column_config.TextColumn("위험 진단", help="AI 엔진 진단 결과")}
        )
        st.caption("※ 🚨: Dual-Track 리스크 스코어링 (🔴:위험 / 🟡:주의) | 기간: 최근 24개월 + 과거 연평균")

    with tab2:
        st.markdown("##### ⏱️ Lag 분석 (제조 ~ 접수 소요기간)")
        # Core Analytics 활용
        lag_stats = calculate_lag_stats(df_display)
        
        if lag_stats and lag_stats['count'] > 0:
            c1, c2, c3 = st.columns(3)
            c1.metric("평균 Lag", f"{lag_stats['mean']:.1f} 일")
            c2.metric("중앙값 Lag", f"{lag_stats['p50']:.1f} 일")
            c3.metric("대상 건수", f"{lag_stats['count']:,} 건")
            
            if 'Lag_Valid' in df_display.columns and 'Lag_Days' in df_display.columns:
                valid_lag_df = df_display[df_display['Lag_Valid'] == True]
                fig = px.histogram(valid_lag_df, x='Lag_Days', nbins=50, title="Lag Days Distribution")
                st.plotly_chart(fig, width='stretch')
        else:
            st.warning("유효 데이터 없음")

    with tab3:
        st.dataframe(df_display)

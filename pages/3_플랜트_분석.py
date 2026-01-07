import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, date
from core.storage import load_partitioned
from core.analytics import detect_outliers_iqr, calculate_lag_stats

# --- 0. 설정 및 상수 정의 ---
st.set_page_config(page_title="플랜트 분석", layout="wide")
st.title("🏭 Phase 2: 플랜트 정밀 분석")

# [CONFIG] 등급 기준 매핑
CRITICAL_GRADES = ['중대', '위험', '사고']  
GENERAL_GRADES = ['일반']

# [CONFIG] 불만원인 그룹 정의
PERFORMANCE_REASONS = ['제조불만', '고객불만족', '구매불만']
TARGET_BUSINESS_UNITS = ['식품', 'B2B식품']

# --- 1. 데이터 로드 ---
@st.cache_data
def load_master_data():
    try:
        return load_partitioned()
    except FileNotFoundError:
        return None

master_df = load_master_data()

if master_df is None or master_df.empty:
    st.error("⚠️ 데이터가 없습니다. 먼저 데이터를 업로드해주세요.")
    st.stop()

# 날짜 컬럼 보장
if '접수일자' not in master_df.columns:
    st.error("데이터에 '접수일자' 컬럼이 없습니다.")
    st.stop()
master_df['접수일자'] = pd.to_datetime(master_df['접수일자'])

# 플랜트 목록 추출
all_plants = sorted(master_df['플랜트'].dropna().unique().tolist())

# --- 2. Step 1: Scope (플랜트 및 기간) ---
st.markdown("#### Step 1: 분석 범위 설정")
col_s1_1, col_s1_2, col_s1_3 = st.columns([1, 1, 1])

with col_s1_1:
    selected_plant = st.selectbox("🏭플랜트 선택", all_plants)

# 기본 날짜 설정 (전체 데이터 기준)
min_date = master_df['접수일자'].min().date()
max_date = master_df['접수일자'].max().date()

with col_s1_2:
    start_date = st.date_input("📅시작일 (Start)", value=min_date, min_value=min_date, max_value=max_date)

with col_s1_3:
    end_date = st.date_input("📅종료일 (End)", value=max_date, min_value=min_date, max_value=max_date)

# 1차 필터링 (플랜트 & 기간)
plant_df = master_df[
    (master_df['플랜트'] == selected_plant) &
    (master_df['접수일자'].dt.date >= start_date) &
    (master_df['접수일자'].dt.date <= end_date)
].copy()

# Data Summary Badge
if not plant_df.empty:
    st.info(f"📋 **요약**: `{selected_plant}` | `{start_date} ~ {end_date}` | Raw data 총 **{len(plant_df):,}** 건")
else:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
    st.stop()

st.divider()

# --- Step 2 & 3: 검색 옵션 및 등급 필터 (2단 레이아웃) ---
col_step2, col_step3 = st.columns(2)

with col_step2:
    st.markdown("#### Step 2: 검색 옵션 (Mode)")

    # 검색 모드 변경 시 Custom 선택지를 초기화하는 콜백
    def reset_custom_selections():
        if 'sel_biz' in st.session_state:
            del st.session_state['sel_biz']
        if 'sel_reason' in st.session_state:
            del st.session_state['sel_reason']

    search_mode = st.radio(
        "조회 모드를 선택하세요:",
        ("인입 (Inflow)", "실적 (Performance)", "Custom (직접 선택)"),
        horizontal=True,
        on_change=reset_custom_selections
    )

    filtered_df_step2 = plant_df.copy()

    # 옵션별 필터링 로직
    if search_mode == "인입 (Inflow)":
        cond_biz = filtered_df_step2['사업부문'].isin(TARGET_BUSINESS_UNITS)
        cond_reason = filtered_df_step2['불만원인'].notna()
        filtered_df_step2 = filtered_df_step2[cond_biz & cond_reason]
        st.caption(f"ℹ️ **인입 기준**: 사업부문({', '.join(TARGET_BUSINESS_UNITS)}) + 불만원인(전체)")

    elif search_mode == "실적 (Performance)":
        cond_biz = filtered_df_step2['사업부문'].isin(TARGET_BUSINESS_UNITS)
        cond_reason = filtered_df_step2['불만원인'].isin(PERFORMANCE_REASONS)
        filtered_df_step2 = filtered_df_step2[cond_biz & cond_reason]
        st.caption(f"ℹ️ **실적 기준**: 사업부문({', '.join(TARGET_BUSINESS_UNITS)}) + 불만원인({', '.join(PERFORMANCE_REASONS)})")

    else: # Custom
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            opts_biz = sorted(plant_df['사업부문'].dropna().unique())
            sel_biz = st.multiselect(
                "사업부문 선택", 
                opts_biz, 
                default=opts_biz, 
                key='sel_biz'
            )
        with col_c2:
            opts_reason = sorted(plant_df['불만원인'].dropna().unique())
            sel_reason = st.multiselect(
                "불만원인 선택", 
                opts_reason, 
                default=opts_reason, 
                key='sel_reason'
            )
        
        if sel_biz:
            filtered_df_step2 = filtered_df_step2[filtered_df_step2['사업부문'].isin(sel_biz)]
        if sel_reason:
            filtered_df_step2 = filtered_df_step2[filtered_df_step2['불만원인'].isin(sel_reason)]

with col_step3:
    st.markdown("#### Step 3: 등급 필터 (Grade)")
    grade_mode = st.radio(
        "분석할 등급을 선택하세요:",
        ("중대 (중대+위험+사고)", "일반 (일반)", "전체 (All)"),
        horizontal=True
    )

    filtered_df_step3 = filtered_df_step2.copy()

    if grade_mode == "중대 (중대+위험+사고)":
        filtered_df_step3 = filtered_df_step3[filtered_df_step3['등급기준'].isin(CRITICAL_GRADES)]
    elif grade_mode == "일반 (일반 전체)":
        filtered_df_step3 = filtered_df_step3[~filtered_df_step3['등급기준'].isin(CRITICAL_GRADES)]
    else:
        pass # 전체

    cnt_step3 = len(filtered_df_step3)
    st.caption(f"📊 필터링 후 대상 건수: **{cnt_step3:,}** 건")

st.divider()

# --- 5. Step 4: Pivot & Analysis (상세 분석) ---
st.markdown("#### Step 4: 상세 분석 (Pivot Table)")

col_p1, col_p2 = st.columns([3, 1])

with col_p1:
    # 피벗 인덱스 설정
    pivot_candidates = ['제품범주1', '제품범주2', '제품범주3', '대분류', '중분류', '소분류', '등급기준', '불만원인']
    pivot_candidates = [c for c in pivot_candidates if c in filtered_df_step3.columns and filtered_df_step3[c].notna().any()]
    
    default_indices = [c for c in ['등급기준', '대분류', '중분류'] if c in pivot_candidates]
    
    pivot_indices = st.multiselect(
        "피벗 행(Index) 설정", 
        pivot_candidates, 
        default=default_indices
    )

with col_p2:
    st.markdown("""
    ✅ **피벗 행 선택지**
    - `제품범주1, 2, 3`
    - `대분류, 중분류, 소분류`
    - `등급기준`, `불만원인`
    """)

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
                pivot_reindexed = pivot_with_margin.reindex(columns=all_months + ['Total'], fill_value=0)
                return pivot_reindexed

            # 2. 기본 피벗 생성
            pivot_base = pd.pivot_table(df, index=indices, columns=columns, values=values, aggfunc=aggfunc, fill_value=0)
            # 모든 월 컬럼 보장
            pivot_base = pivot_base.reindex(columns=all_months, fill_value=0)
            
            if pivot_base.empty:
                return pivot_base

            all_parts = []
            
            # 3. 소계 계산 루프
            for l1_name, l1_group in pivot_base.groupby(level=0, sort=False):
                for l2_name, l2_group in l1_group.groupby(level=1, sort=False):
                    all_parts.append(l2_group)
                    # L2 소계: ('L1 값', 'L2 값', '소계', '', ..)
                    subtotal_l2_row = l2_group.sum().to_frame().T
                    template_idx = list(l2_group.index[0])
                    idx_tuple = template_idx[:2] + ['소계'] + [''] * (len(indices) - 3)
                    subtotal_l2_row.index = pd.MultiIndex.from_tuples([tuple(idx_tuple)], names=indices)
                    all_parts.append(subtotal_l2_row)
                
                # L1 총계: ('L1 값', '전체 합계', '', ..)
                total_l1_row = l1_group.sum().to_frame().T
                template_idx = list(l1_group.index[0])
                idx_tuple = [template_idx[0]] + ['전체 합계'] + [''] * (len(indices) - 2)
                total_l1_row.index = pd.MultiIndex.from_tuples([tuple(idx_tuple)], names=indices)
                all_parts.append(total_l1_row)
            
            final_pivot = pd.concat(all_parts)
            
            # 4. 전체 총계 (Grand Total) 추가
            grand_total_row = pivot_base.sum().to_frame('Total').T
            idx_tuple = ['Total'] + [''] * (len(indices) - 1)
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
        st.exception(e) # 디버깅을 위한 상세 오류
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

    with tab2:
        st.markdown("##### ⏱️ Lag 분석 (제조 ~ 접수 소요기간)")
        lag_stats = calculate_lag_stats(filtered_df_step3)
        
        if lag_stats and lag_stats['count'] > 0:
            c1, c2, c3 = st.columns(3)
            c1.metric("평균 Lag", f"{lag_stats['mean']:.1f} 일")
            c2.metric("중앙값 Lag", f"{lag_stats['p50']:.1f} 일")
            c3.metric("대상 건수", f"{lag_stats['count']:,} 건")
            
            valid_lag_df = filtered_df_step3[filtered_df_step3['Lag_Valid'] == True]
            fig = px.histogram(
                valid_lag_df, 
                x='Lag_Days', 
                nbins=50,
                title="Lag Days Distribution",
                color_discrete_sequence=['#3b82f6']
            )
            fig.update_layout(xaxis_title="소요 일수 (Days)", yaxis_title="건수")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Lag 분석을 위한 유효 데이터(제조일자 존재)가 없습니다.")

    with tab3:
        st.markdown(f"##### 원본 데이터 ({len(filtered_df_step3):,} 건)")
        st.dataframe(filtered_df_step3)
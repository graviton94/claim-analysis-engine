# ============================================================================
# 페이지: 플랜트 분석 (Adaptive PPM Dashboard)
# ============================================================================
# 설명: 플랜트 중심 동적 피벗 테이블 대시보드
#      건수, PPM 지표를 선택 가능한 열로 구성
#      매출 데이터 자동 추정치 반영

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import plotly.graph_objects as go
import plotly.express as px

from core.config import DATA_HUB_PATH, DATA_SALES_PATH, SALES_FILENAME
from core.storage import load_partitioned, load_sales_with_estimation, get_available_periods

# ============================================================================
# 페이지 레이아웃 설정
# ============================================================================
st.set_page_config(page_title="플랜트 분석", page_icon="🏭", layout="wide")
st.title("🏭 플랜트 분석 (Adaptive PPM Dashboard)")
st.markdown(
    "플랜트별 클레임 데이터와 매출 데이터를 결합하여 "
    "건수, PPM 등 다양한 지표를 동적으로 분석합니다."
)

# ============================================================================
# 기본 설정
# ============================================================================
SALES_PATH = Path(DATA_SALES_PATH) / SALES_FILENAME


# ============================================================================
# 함수: PPM 계산
# ============================================================================
def calculate_ppm(
    claims_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    plant: str,
    groupby_cols: List[str]
) -> pd.DataFrame:
    """
    PPM (Parts Per Million) 계산 및 피벗 테이블 생성.
    
    동작:
        1. 클레임 데이터를 groupby_cols로 그룹화하고 건수 집계
        2. 매출 데이터와 병합 (플랜트+년+월 기준)
        3. PPM = (건수 / 매출수량) * 1,000,000 계산
        4. is_estimated 컬럼 활용하여 추정치 표기
    
    Args:
        claims_df: 클레임 데이터
        sales_df: 매출 데이터 (is_estimated 포함)
        plant: 조회 플랜트명
        groupby_cols: 그룹화 기준 컬럼 (행+열)
    
    Returns:
        pd.DataFrame: PPM 계산 결과
    """
    # 플랜트 필터링
    plant_claims = claims_df[claims_df['플랜트'] == plant].copy()
    plant_sales = sales_df[sales_df['플랜트'] == plant].copy()
    
    if plant_claims.empty:
        return pd.DataFrame()
    
    # 클레임 건수 집계 (['접수년', '접수월'] 기본 포함)
    base_cols = ['접수년', '접수월']
    agg_cols = base_cols + [col for col in groupby_cols if col not in base_cols and col in plant_claims.columns]
    
    claims_grouped = plant_claims.groupby(agg_cols).size().reset_index(name='건수')
    
    # 매출 데이터와 병합
    plant_sales_renamed = plant_sales.rename(columns={'년': '접수년', '월': '접수월'})
    merged = claims_grouped.merge(
        plant_sales_renamed[['접수년', '접수월', '매출수량', 'is_estimated']],
        on=['접수년', '접수월'],
        how='left'
    )
    
    # PPM 계산 (매출수량이 0 또는 NaN인 경우 제외)
    merged['PPM'] = merged.apply(
        lambda row: (row['건수'] / row['매출수량'] * 1_000_000) 
                    if pd.notna(row['매출수량']) and row['매출수량'] > 0 
                    else None,
        axis=1
    )
    
    # 추정치 표기
    merged['값_표시'] = merged.apply(
        lambda row: f"(예상치) {row['건수']}" if row['is_estimated'] else str(row['건수']),
        axis=1
    )
    
    return merged


def create_pivot_table(
    df: pd.DataFrame,
    index_cols: List[str],
    column_cols: List[str] = ['접수년', '접수월'],
    value_col: str = '건수'
) -> pd.DataFrame:
    """
    동적 피벗 테이블 생성 (열 = 최근 12개월 + 3개월 예측).
    
    Args:
        df: 소스 데이터
        index_cols: 행(Index) 컬럼 (사용자 선택)
        column_cols: 열(Columns) 컬럼 (고정: ['접수년', '접수월'])
        value_col: 값(Values) 컬럼
    
    Returns:
        pd.DataFrame: 피벗 테이블 (열 = 최근 12개월 + 3개월 예측 + 맨앞컬럼 소계)
    """
    if not index_cols:
        return pd.DataFrame()
    
    df = df.copy()
    
    # 최근 12개월 데이터만 필터링
    df['연월'] = df['접수년'] * 100 + df['접수월']
    df = df.sort_values('연월')
    
    # 최근 12개월 추출
    unique_periods = df[['접수년', '접수월', '연월']].drop_duplicates().sort_values('연월')
    if len(unique_periods) > 12:
        min_연월 = unique_periods['연월'].iloc[-12]
        df = df[df['연월'] >= min_연월]
    
    # 년월 컬럼 생성 (예: "2024-01")
    df['년월'] = df['접수년'].astype(str) + '-' + df['접수월'].astype(str).str.zfill(2)
    
    # 피벗 테이블 생성
    pivot = df.pivot_table(
        index=index_cols,
        columns='년월',
        values=value_col,
        aggfunc='sum',
        fill_value=0
    )
    
    # 미래 3개월 예측 컬럼 생성
    # 현재 데이터의 최대 년월에서 다음 3개월 계산
    if not df.empty:
        max_year = int(df['접수년'].max())
        max_month = int(df[df['접수년'] == max_year]['접수월'].max())
        
        future_months = []
        current_year = max_year
        current_month = max_month
        
        for i in range(1, 4):  # +1, +2, +3개월
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
            future_col = f"{current_year}.{current_month:02d}(예측)"
            future_months.append(future_col)
            pivot[future_col] = 0  # placeholder
    
    # ★ 맨앞 컬럼(행 인덱스 첫번째)에 대한 소계 추가 + 전체 합계
    if index_cols:
        first_col = index_cols[0]
        subtotal_df = pivot.reset_index()
        
        # 첫번째 컬럼으로 그룹화하여 각 그룹 끝에 소계 행 삽입
        subtotal_data_list = []
        numeric_cols = subtotal_df.select_dtypes(include=[np.number]).columns
        
        for group_name, group_data in subtotal_df.groupby(first_col, sort=False):
            # 그룹 데이터 추가
            subtotal_data_list.append(group_data)
            
            # 소계 행 추가
            subtotal_row = {col: "" for col in subtotal_df.columns}
            subtotal_row[first_col] = f"[소계] {group_name}"
            
            # 수치 컬럼만 합산
            for col in numeric_cols:
                if col not in index_cols:
                    subtotal_row[col] = group_data[col].sum()
            
            subtotal_data_list.append(pd.DataFrame([subtotal_row]))
        
        # 전체 합계 행 추가
        total_row = {col: "" for col in subtotal_df.columns}
        total_row[first_col] = "[전체] 총 합계"
        for col in numeric_cols:
            if col not in index_cols:
                total_row[col] = subtotal_df[col].sum()
        
        # 모든 데이터 결합 (그룹 + 소계 반복 + 전체 합계)
        subtotal_df_result = pd.concat(
            subtotal_data_list + [pd.DataFrame([total_row])],
            ignore_index=True
        )
        
        return subtotal_df_result
    
    return pivot.reset_index()


# ============================================================================
# 세션 상태 초기화
# ============================================================================
if 'selected_plant' not in st.session_state:
    st.session_state.selected_plant = None
if 'claims_data' not in st.session_state:
    st.session_state.claims_data = None
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = None
if 'saved_pivot_rows' not in st.session_state:
    st.session_state.saved_pivot_rows = ['대분류', '중분류', '소분류']


# ============================================================================
# 영역 1: 플랜트 필터 (필수)
# ============================================================================
st.subheader("🔍 Step 1: 플랜트 선택 (필수)")

# 사용 가능한 플랜트 목록 로드
# ★ 변경: get_claim_keys()를 사용하여 Type Safe한 로드
try:
    from core.storage import get_claim_keys
    claim_keys = get_claim_keys(DATA_HUB_PATH)
    
    # ★ 변경: None/NaN 제외 후 dropna() 완료된 상태이므로 안전한 정렬 가능
    available_plants = []
    if not claim_keys.empty and '플랜트' in claim_keys.columns:
        available_plants = sorted(claim_keys['플랜트'].dropna().unique().tolist())
except Exception as e:
    print(f"[ERROR] 플랜트 목록 로드 실패: {str(e)}")
    available_plants = []

# ★ 변경: Traceback 대신 명확한 경고 메시지 표시
if not available_plants:
    st.warning(
        "⚠️ 분석할 데이터가 없습니다.\n\n"
        "**[데이터 업로드]** 메뉴에서 CSV/Excel 파일을 등록해주세요."
    )
    st.stop()

# 플랜트 선택 (드롭다운)
selected_plant = st.selectbox(
    "분석할 플랜트를 선택하세요:",
    ["선택하세요..."] + available_plants,
    key="plant_dropdown"
)

if selected_plant and selected_plant != "선택하세요...":
    st.session_state.selected_plant = selected_plant
else:
    st.info("💡 위 드롭다운에서 플랜트를 선택해주세요.")
    st.stop()

# 플랜트 선택 시 데이터 로드
if selected_plant:
    try:
        # 클레임 데이터 로드
        st.session_state.claims_data = load_partitioned(DATA_HUB_PATH)
        
        # 매출 데이터 로드 (추정치 포함)
        st.session_state.sales_data = load_sales_with_estimation(SALES_PATH)
        
        st.success(f"✅ {selected_plant} 데이터 로드 완료")
    except Exception as e:
        st.error(f"❌ 데이터 로드 실패: {str(e)}")
        st.stop()

# ============================================================================
# 영역 2: 분석 기간 표시 (자동)
# ============================================================================
st.subheader("📅 Step 2: 분석 기간 (자동 추출)")

# 기간 필터링 (플랜트의 모든 데이터)
if st.session_state.claims_data is not None:
    # ★ Categorical 타입 에러 해결: 숫자형으로 변환
    df_temp = st.session_state.claims_data.copy()
    df_temp['접수년'] = pd.to_numeric(df_temp['접수년'], errors='coerce')
    df_temp['접수월'] = pd.to_numeric(df_temp['접수월'], errors='coerce')
    
    # 플랜트 필터링
    filtered_claims = df_temp[df_temp['플랜트'] == selected_plant].copy()
    
    # 기간 추출
    if not filtered_claims.empty:
        min_year = int(filtered_claims['접수년'].min())
        min_month = int(filtered_claims[filtered_claims['접수년'] == min_year]['접수월'].min())
        max_year = int(filtered_claims['접수년'].max())
        max_month = int(filtered_claims[filtered_claims['접수년'] == max_year]['접수월'].max())
        
        st.info(f"📊 분석기간: {min_year}.{min_month:02d} ~ {max_year}.{max_month:02d} ({len(filtered_claims)} 건)")
    else:
        st.warning(f"⚠️ {selected_plant}의 클레임 데이터가 없습니다.")
        st.stop()
else:
    filtered_claims = pd.DataFrame()
    st.stop()

# ============================================================================
# 영역 3: 동적 피벗 설정
# ============================================================================
st.subheader("📊 Step 3: 피벗 설정")

st.write("★ **열(Columns)**: `접수년`, `접수월` + `3개월 예측` (고정)")
st.write("★ **행(Index)**: 아래에서 선택")

# 선택 가능한 행 컬럼
available_row_columns = [col for col in filtered_claims.columns 
                    if col not in ['접수년', '접수월', '접수일', '플랜트', '상담번호', 
                                  '제목', '분석결과', '요구사항', '주소1'] 
                    and filtered_claims[col].dtype == 'object']

col_pivot, col_save = st.columns([3, 1])

with col_pivot:
    selected_pivot_rows = st.multiselect(
        "피벗 행으로 사용할 컬럼 선택:",
        available_row_columns,
        default=st.session_state.saved_pivot_rows if all(col in available_row_columns for col in st.session_state.saved_pivot_rows) else ['대분류', '중분류', '소분류'],
        key="pivot_rows"
    )

with col_save:
    st.write("")  # 정렬용 공백
    st.write("")  # 정렬용 공백
    if st.button("💾 설정 기억하기", key="save_pivot_settings"):
        st.session_state.saved_pivot_rows = selected_pivot_rows
        st.success("✅ 피벗 설정이 저장되었습니다!")

# ============================================================================
# 영역 4: 지표 선택 및 피벗 테이블 생성
# ============================================================================
st.subheader("📈 Step 4: 지표 선택")

col_metric1, col_metric2 = st.columns(2)

show_count = col_metric1.checkbox("건수", value=True, key="show_count")
show_ppm = col_metric2.checkbox("PPM", value=True, key="show_ppm")

if not (show_count or show_ppm):
    st.warning("⚠️ 최소 하나의 지표를 선택해야 합니다.")
    st.stop()

# 피벗 테이블 생성 및 표시
if selected_plant and st.session_state.claims_data is not None:
    st.subheader("📋 피벗 테이블 결과")
    
    # PPM 데이터 계산
    ppm_data = calculate_ppm(
        filtered_claims,
        st.session_state.sales_data if st.session_state.sales_data is not None else pd.DataFrame(),
        selected_plant,
        selected_pivot_rows  # ★ 변경: 행 컬럼 사용
    )
    
    if ppm_data.empty:
        st.warning(f"⚠️ {selected_plant}의 클레임 데이터가 없습니다.")
    else:
        # 건수 피벗
        if show_count:
            st.write("#### 건수 (월별 + 3개월 예측)")
            count_pivot = create_pivot_table(
                ppm_data,
                index_cols=selected_pivot_rows,  # ★ 변경: 사용자 선택 행
                value_col='건수'
            )
            st.dataframe(count_pivot, use_container_width=True)
        
        # PPM 피벗
        if show_ppm:
            st.write("#### PPM (Parts Per Million)")
            ppm_pivot = create_pivot_table(
                ppm_data,
                index_cols=selected_pivot_rows,  # ★ 변경: 사용자 선택 행
                value_col='PPM'
            )
            st.dataframe(ppm_pivot, use_container_width=True)
            
            # 추정치 표시
            estimated_rows = ppm_data[ppm_data['is_estimated'] == True]
            if not estimated_rows.empty:
                st.info(f"⚠️ {len(estimated_rows)}개 행이 **예상치**입니다 (직전 3개월 평균값)")
        
        # ============================================================================
        # 영역 5: 시계열 차트
        # ============================================================================
        st.subheader("📉 시계열 차트")
        
        # 시간별 건수 차트
        if show_count:
            timeline_data = ppm_data.groupby(['접수년', '접수월'])['건수'].sum().reset_index()
            timeline_data['기간'] = timeline_data['접수년'].astype(str) + '-' + timeline_data['접수월'].astype(str).str.zfill(2)
            
            fig_count = px.line(
                timeline_data,
                x='기간',
                y='건수',
                markers=True,
                title=f'{selected_plant} - 월별 클레임 건수',
                labels={'건수': '건수', '기간': '기간'}
            )
            fig_count.update_xaxes(tickangle=45)
            st.plotly_chart(fig_count, use_container_width=True)
        
        # 시간별 PPM 차트
        if show_ppm:
            timeline_ppm = ppm_data.groupby(['접수년', '접수월']).agg({
                'PPM': 'mean',
                'is_estimated': 'any'
            }).reset_index()
            timeline_ppm['기간'] = timeline_ppm['접수년'].astype(str) + '-' + timeline_ppm['접수월'].astype(str).str.zfill(2)
            timeline_ppm['표기'] = timeline_ppm.apply(
                lambda row: f"(예상) {row['PPM']:.1f}" if row['is_estimated'] else f"{row['PPM']:.1f}",
                axis=1
            )
            
            fig_ppm = px.line(
                timeline_ppm,
                x='기간',
                y='PPM',
                markers=True,
                title=f'{selected_plant} - 월별 PPM',
                labels={'PPM': 'PPM', '기간': '기간'}
            )
            fig_ppm.update_xaxes(tickangle=45)
            st.plotly_chart(fig_ppm, use_container_width=True)
        
        # ============================================================================
        # 영역 6: 통계 정보
        # ============================================================================
        with st.expander("📊 상세 통계", expanded=False):
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            total_claims = len(ppm_data)
            avg_ppm = ppm_data['PPM'].mean() if not ppm_data['PPM'].isna().all() else 0
            total_sales = ppm_data['매출수량'].sum()
            estimated_count = ppm_data['is_estimated'].sum()
            
            with col_stat1:
                st.metric("총 클레임 건수", total_claims)
            
            with col_stat2:
                st.metric("평균 PPM", f"{avg_ppm:.1f}")
            
            with col_stat3:
                st.metric("총 매출수량", f"{int(total_sales):,}")
            
            with col_stat4:
                st.metric("추정치 개수", estimated_count)

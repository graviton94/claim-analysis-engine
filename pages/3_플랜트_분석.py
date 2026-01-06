# ============================================================================
# 페이지: 플랜트 분석 (6-Step Adaptive Dashboard with Macro)
# ============================================================================
# 설명: 6-Step 대시보드로 필터, 피벗, 예측을 체계적으로 관리
#      - Step 1&2: 플랜트 선택 + 데이터 요약
#      - Step 3: 4대 필터 (대분류, 사업부문, 등급기준, 불만원인)
#      - Step 4: 피벗 설정 (선택된 필터 제외)
#      - Step 5: 지표 선택 + 매크로 버튼
#      - Step 6: 분석 시작 (필터링 + 예측 + 시각화)

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Set, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

from core.config import DATA_HUB_PATH, DATA_SALES_PATH, SALES_FILENAME
from core.storage import load_partitioned, load_sales_with_estimation, get_claim_keys
from core.engine.trainer import predict_with_seasonal_allocation

# ============================================================================
# 페이지 레이아웃 설정
# ============================================================================
st.set_page_config(page_title="플랜트 분석", page_icon="🏭", layout="wide")

# ============================================================================
# 기본 설정
# ============================================================================
SALES_PATH = Path(DATA_SALES_PATH) / SALES_FILENAME
SETTINGS_FILE = Path(DATA_HUB_PATH).parent / "plant_settings.json"  # data/plant_settings.json


# ============================================================================
# 함수: 플랜트별 설정 저장/로드
# ============================================================================
def load_plant_settings(plant: str) -> Dict[str, Any]:
    """
    플랜트별 저장된 설정 로드 (Step 3, 4 필터 & 피벗만).
    
    Args:
        plant: 플랜트명
    
    Returns:
        Dict: {filter_business, filter_reason, filter_grade, filter_major_category,
               saved_pivot_rows}
    """
    if not SETTINGS_FILE.exists():
        return {}
    
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            all_settings = json.load(f)
        return all_settings.get(plant, {})
    except Exception as e:
        print(f"[ERROR] 설정 로드 실패: {str(e)}")
        return {}


def save_plant_settings(plant: str, settings: Dict[str, Any]) -> None:
    """
    플랜트별 설정 저장 (Step 3, 4 필터 & 피벗만 - 로컬 JSON).
    
    Args:
        plant: 플랜트명
        settings: {filter_business, filter_reason, filter_grade, filter_major_category,
                  saved_pivot_rows}
    """
    try:
        # 기존 설정 로드
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                all_settings = json.load(f)
        else:
            all_settings = {}
        
        # 해당 플랜트 설정 업데이트
        all_settings[plant] = settings
        
        # 파일 저장
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_settings, f, ensure_ascii=False, indent=2)
        
        print(f"[SUCCESS] {plant} 설정 저장 완료")
    except Exception as e:
        print(f"[ERROR] 설정 저장 실패: {str(e)}")



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


# ============================================================================
# 함수: 동적 피벗 테이블 생성
# ============================================================================
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
        index_cols: 행(Index) 컬럼 (사용자 선택) - 첫 번째가 소계 기준 컬럼
        column_cols: 열(Columns) 컬럼 (고정: ['접수년', '접수월'])
        value_col: 값(Values) 컬럼
    
    Returns:
        pd.DataFrame: 피벗 테이블 (열 = 최근 12개월 + 3개월 예측 + 맨앞컬럼 소계)
    """
    if not index_cols:
        return pd.DataFrame()
    
    df = df.copy()
    
    # Categorical 타입을 정수로 변환
    df['접수년'] = pd.to_numeric(df['접수년'], errors='coerce').astype('Int64')
    df['접수월'] = pd.to_numeric(df['접수월'], errors='coerce').astype('Int64')
    
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
    if not df.empty:
        max_year = int(pd.to_numeric(df['접수년'], errors='coerce').max())
        max_month = int(pd.to_numeric(df[pd.to_numeric(df['접수년'], errors='coerce') == max_year]['접수월'], errors='coerce').max())
        
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
        result_df = pivot.reset_index()
        
        # 수치 컬럼 식별 (index_cols에 없는 모든 수치 컬럼)
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in index_cols]
        
        # 첫 번째 컬럼으로 그룹화하여 각 그룹 끝에 소계 행 삽입
        subtotal_data_list = []
        
        for group_name, group_data in result_df.groupby(first_col, sort=False):
            # 그룹 데이터 추가 (행 인덱스 리셋)
            subtotal_data_list.append(group_data.reset_index(drop=True))
            
            # 소계 행 추가
            subtotal_row = {}
            
            # 텍스트 컬럼: 첫 번째 컬럼만 특수 표기, 나머지는 공백
            for col in result_df.columns:
                if col == first_col:
                    subtotal_row[col] = f"[소계] {group_name}"
                elif col not in numeric_cols:
                    subtotal_row[col] = ""
                else:
                    # 수치 컬럼: 해당 그룹의 합계
                    subtotal_row[col] = group_data[col].sum()
            
            subtotal_data_list.append(pd.DataFrame([subtotal_row]))
        
        # 전체 합계 행 추가
        total_row = {}
        for col in result_df.columns:
            if col == first_col:
                total_row[col] = "[전체] 총 합계"
            elif col not in numeric_cols:
                total_row[col] = ""
            else:
                # 수치 컬럼: 전체 합계
                total_row[col] = result_df[col].sum()
        
        # 모든 데이터 결합 (그룹 + 소계 반복 + 전체 합계)
        final_result = pd.concat(
            subtotal_data_list + [pd.DataFrame([total_row])],
            ignore_index=True
        )
        
        return final_result
    
    return pivot.reset_index()


# ============================================================================
# 함수: 필터 유효성 검사
# ============================================================================
def validate_filters(selected_filters: dict) -> Tuple[bool, str]:
    """
    필터 선택 유효성 검사.
    
    Args:
        selected_filters: 선택된 필터 딕셔너리
    
    Returns:
        Tuple: (유효성, 에러메시지)
    """
    if not selected_filters.get('대분류') or len(selected_filters['대분류']) == 0:
        return False, "⚠️ '대분류'는 최소 1개 이상 선택해야 합니다."
    
    return True, ""


# ============================================================================
# 함수: 선택된 필터와 피벗 충돌 검사
# ============================================================================
def get_available_pivot_cols(
    all_cols: List[str],
    filter_cols: Set[str]
) -> List[str]:
    """
    필터로 사용된 컬럼을 제외한 피벗 가능 컬럼 반환.
    
    Args:
        all_cols: 전체 컬럼 리스트
        filter_cols: 필터로 사용된 컬럼 집합
    
    Returns:
        List[str]: 피벗 가능 컬럼 리스트
    """
    # 제외할 컬럼: 시간계, 고유식별자, 텍스트 필드
    excluded = {'접수년', '접수월', '접수일', '플랜트', '상담번호', '제목', 
                '분석결과', '요구사항', '주소1', '년', '월'}
    
    available = [col for col in all_cols 
                 if col not in excluded and col not in filter_cols]
    
    return sorted(available)


# ============================================================================
# 세션 상태 초기화
# ============================================================================
if 'selected_plant' not in st.session_state:
    st.session_state.selected_plant = None
if 'claims_data' not in st.session_state:
    st.session_state.claims_data = None
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = None
if 'filter_major_category' not in st.session_state:
    st.session_state.filter_major_category = []
if 'filter_business' not in st.session_state:
    st.session_state.filter_business = []
if 'filter_grade' not in st.session_state:
    st.session_state.filter_grade = []
if 'filter_reason' not in st.session_state:
    st.session_state.filter_reason = []
if 'saved_pivot_rows' not in st.session_state:
    st.session_state.saved_pivot_rows = ['중분류', '소분류']
if 'use_performance_macro' not in st.session_state:
    st.session_state.use_performance_macro = False
if 'selected_metrics' not in st.session_state:
    st.session_state.selected_metrics = ['건수', 'PPM']
if 'save_settings' not in st.session_state:
    st.session_state.save_settings = True


# ============================================================================
# 페이지 제목 및 설명
# ============================================================================
st.title("🏭 플랜트 분석 (6-Step Adaptive Dashboard)")
st.markdown(
    "플랜트별 클레임 데이터와 매출 데이터를 결합하여 "
    "건수, PPM 등 다양한 지표를 동적으로 분석합니다.\n\n"
    "**6-Step 프로세스**: 플랜트 선택 → 필터 설정 → 피벗 구성 → 지표 선택 → 분석 실행 → 결과 조회"
)


# ============================================================================
# Step 1 & 2: 플랜트 선택 + 데이터 요약 (Top Layout)
# ============================================================================
st.subheader("📍 Step 1 & 2: 플랜트 선택 및 데이터 요약")

col1, col2 = st.columns([1, 1])

# ============================================================================
# Step 1: 플랜트 선택
# ============================================================================
with col1:
    st.write("#### 🔍 Step 1: 분석할 플랜트 선택")
    
    # 사용 가능한 플랜트 목록 로드
    try:
        claim_keys = get_claim_keys(DATA_HUB_PATH)
        available_plants = []
        if not claim_keys.empty and '플랜트' in claim_keys.columns:
            available_plants = sorted(claim_keys['플랜트'].dropna().unique().tolist())
    except Exception as e:
        print(f"[ERROR] 플랜트 목록 로드 실패: {str(e)}")
        available_plants = []
    
    if not available_plants:
        st.warning(
            "⚠️ 분석할 데이터가 없습니다.\n\n"
            "**[데이터 업로드]** 메뉴에서 CSV/Excel 파일을 등록해주세요."
        )
        st.stop()
    
    # 플랜트 선택 (드롭다운)
    selected_plant = st.selectbox(
        "분석할 플랜트를 선택하세요:",
        available_plants,
        key="plant_dropdown"
    )
    
    if selected_plant:
        # 플랜트 변경 시 설정 로드
        if st.session_state.selected_plant != selected_plant:
            st.session_state.selected_plant = selected_plant
            
            # 이전 설정 로드 (Step 3, 4만)
            loaded_settings = load_plant_settings(selected_plant)
            
            if loaded_settings:
                # 저장된 필터 & 피벗 설정 복원
                st.session_state.filter_major_category = loaded_settings.get('filter_major_category', [])
                st.session_state.filter_business = loaded_settings.get('filter_business', [])
                st.session_state.filter_grade = loaded_settings.get('filter_grade', [])
                st.session_state.filter_reason = loaded_settings.get('filter_reason', [])
                st.session_state.saved_pivot_rows = loaded_settings.get('saved_pivot_rows', ['중분류', '소분류'])
            else:
                # 새로운 플랜트: 초기화
                st.session_state.filter_major_category = []
                st.session_state.filter_business = []
                st.session_state.filter_grade = []
                st.session_state.filter_reason = []
                st.session_state.saved_pivot_rows = ['중분류', '소분류']
            
            # 메트릭은 항상 초기화 (매번 사용자가 선택)
            st.session_state.selected_metrics = ['건수']
            st.session_state.use_performance_macro = False
        else:
            st.session_state.selected_plant = selected_plant


# ============================================================================
# Step 2: 데이터 요약 (Metrics)
# ============================================================================
with col2:
    st.write("#### 📊 Step 2: 데이터 요약")
    
    if selected_plant:
        try:
            # 클레임 데이터 로드
            st.session_state.claims_data = load_partitioned(DATA_HUB_PATH)
            
            # 매출 데이터 로드 (추정치 포함)
            st.session_state.sales_data = load_sales_with_estimation(SALES_PATH)
            
            # 플랜트별 데이터 필터링
            plant_claims = st.session_state.claims_data[
                st.session_state.claims_data['플랜트'] == selected_plant
            ]
            plant_sales = st.session_state.sales_data[
                st.session_state.sales_data['플랜트'] == selected_plant
            ]
            
            if not plant_claims.empty:
                # 기간 정보 (Categorical 타입 변환)
                min_year = int(pd.to_numeric(plant_claims['접수년'], errors='coerce').min())
                min_month = int(pd.to_numeric(plant_claims[pd.to_numeric(plant_claims['접수년'], errors='coerce') == min_year]['접수월'], errors='coerce').min())
                max_year = int(pd.to_numeric(plant_claims['접수년'], errors='coerce').max())
                max_month = int(pd.to_numeric(plant_claims[pd.to_numeric(plant_claims['접수년'], errors='coerce') == max_year]['접수월'], errors='coerce').max())
                
                col_metric1, col_metric2 = st.columns(2)
                
                with col_metric1:
                    st.metric("분석 기간", f"{min_year}.{min_month:02d} ~ {max_year}.{max_month:02d}")
                
                with col_metric2:
                    st.metric("총 클레임 건수", f"{len(plant_claims):,}")

            else:
                st.warning(f"⚠️ {selected_plant}의 클레임 데이터가 없습니다.")
        
        except Exception as e:
            st.error(f"❌ 데이터 로드 실패: {str(e)}")
    else:
        st.info("💡 왼쪽에서 플랜트를 선택해주세요.")


st.divider()


# ============================================================================
# Step 3: 필터 설정 (4대 필터)
# ============================================================================
if selected_plant and st.session_state.claims_data is not None:
    st.subheader("🔧 Step 3: 필터 설정")
    st.write("**최소 1개 이상의 대분류를 선택해야 합니다.**")
    
    # 플랜트별 데이터 추출
    plant_data = st.session_state.claims_data[
        st.session_state.claims_data['플랜트'] == selected_plant
    ].copy()
    
    # ★ 종속 필터링 (Cascading Filter): 이전 선택에 따라 다음 필터 옵션 결정
    
    # Step 1: 사업부문 선택지 (전체 데이터에서)
    businesses = sorted(plant_data['사업부문'].dropna().unique().tolist())
    # 기본값: 이전 선택값이 있으면 유지, 없으면 전체 선택
    default_business = st.session_state.filter_business if st.session_state.filter_business else businesses
    # 현재 옵션과 교집합 (없는 옵션 제거)
    default_business = [b for b in default_business if b in businesses]
    if not default_business:
        default_business = businesses
    
    # Step 2: 선택된 사업부문에 따른 불만원인
    if default_business:
        data_filtered_by_business = plant_data[plant_data['사업부문'].isin(default_business)]
    else:
        data_filtered_by_business = plant_data
    reasons = sorted(data_filtered_by_business['불만원인'].dropna().unique().tolist())
    # 기본값: 이전 선택값이 있으면 유지, 없으면 전체 선택
    default_reason = st.session_state.filter_reason if st.session_state.filter_reason else reasons
    # 현재 옵션과 교집합 (없는 옵션 제거)
    default_reason = [r for r in default_reason if r in reasons]
    if not default_reason:
        default_reason = reasons
    
    # Step 3: 선택된 불만원인에 따른 등급기준
    if default_reason:
        data_filtered_by_reason = data_filtered_by_business[data_filtered_by_business['불만원인'].isin(default_reason)]
    else:
        data_filtered_by_reason = data_filtered_by_business
    grades = sorted(data_filtered_by_reason['등급기준'].dropna().unique().tolist())
    # 기본값: 이전 선택값이 있으면 유지, 없으면 전체 선택
    default_grade = st.session_state.filter_grade if st.session_state.filter_grade else grades
    # 현재 옵션과 교집합 (없는 옵션 제거)
    default_grade = [g for g in default_grade if g in grades]
    if not default_grade:
        default_grade = grades
    
    # Step 4: 선택된 등급기준에 따른 대분류
    if default_grade:
        data_filtered_by_grade = data_filtered_by_reason[data_filtered_by_reason['등급기준'].isin(default_grade)]
    else:
        data_filtered_by_grade = data_filtered_by_reason
    major_categories = sorted(data_filtered_by_grade['대분류'].dropna().unique().tolist())
    # 기본값: 이전 선택값이 있으면 유지, 없으면 전체 선택
    default_major = st.session_state.filter_major_category if st.session_state.filter_major_category else major_categories
    # 현재 옵션과 교집합 (없는 옵션 제거)
    default_major = [m for m in default_major if m in major_categories]
    if not default_major:
        default_major = major_categories
    
    # 토글: 실적만 보기
    col_toggle = st.columns([0.5, 3])
    with col_toggle[0]:
        st.session_state.use_performance_macro = st.checkbox(
            "⚡ 실적만 보기",
            value=st.session_state.use_performance_macro,
            help="사업부문 : 식품/B2B식품 | 불만원인 : 제조불만,고객불만족,구매불만 만 조회합니다.",
            key="macro_toggle"
        )
    
    # 실적만 보기 활성화 시 필터 강제 설정
    if st.session_state.use_performance_macro:
        available_businesses = set(businesses)
        available_reasons = set(reasons)
        macro_businesses = {'식품', 'B2B식품'}
        macro_reasons = {'고객불만족', '구매불만', '제조불만'}
        forced_businesses = sorted(list(available_businesses & macro_businesses))
        forced_reasons = sorted(list(available_reasons & macro_reasons))
    else:
        forced_businesses = None
        forced_reasons = None
    
    # 4대 필터 배치 (순서: 사업부문 > 불만원인 > 등급기준 > 대분류)
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        if st.session_state.use_performance_macro and forced_businesses:
            st.session_state.filter_business = forced_businesses
            st.multiselect(
                "**사업부문**",
                businesses,
                default=forced_businesses,
                disabled=True,
                key="filter_biz"
            )
            st.caption(f"✅ 실적 고정: {', '.join(forced_businesses)}")
        else:
            st.session_state.filter_business = st.multiselect(
                "**사업부문**",
                businesses,
                default=default_business,
                key="filter_biz"
            )
    
    with col_f2:
        if st.session_state.use_performance_macro and forced_reasons:
            st.session_state.filter_reason = forced_reasons
            st.multiselect(
                "**불만원인**",
                reasons,
                default=forced_reasons,
                disabled=True,
                key="filter_rsn"
            )
            st.caption(f"✅ 실적 고정: {', '.join(forced_reasons)}")
        else:
            st.session_state.filter_reason = st.multiselect(
                "**불만원인**",
                reasons,
                default=default_reason,
                key="filter_rsn"
            )
    
    with col_f3:
        st.session_state.filter_grade = st.multiselect(
            "**등급기준**",
            grades,
            default=default_grade,
            key="filter_grd"
        )
    
    with col_f4:
        st.session_state.filter_major_category = st.multiselect(
            "**대분류** (필수)",
            major_categories,
            default=default_major,
            key="filter_major"
        )
    
    st.divider()
    
    # ============================================================================
    # Step 4: 피벗 설정
    # ============================================================================
    st.subheader("📋 Step 4: 피벗 설정")
    st.write("**행(Index)으로 사용할 컬럼을 선택하세요.** (필터로 사용된 컬럼은 제외됨)")
    
    # 필터로 사용된 컬럼 집합
    filter_cols_used = {
        col for col in ['대분류', '사업부문', '등급기준', '불만원인']
        if (st.session_state.filter_major_category if col == '대분류' else 
            st.session_state.filter_business if col == '사업부문' else
            st.session_state.filter_grade if col == '등급기준' else
            st.session_state.filter_reason)
    }
    
    # 피벗 가능 컬럼
    available_pivot_cols = get_available_pivot_cols(
        plant_data.columns.tolist(),
        filter_cols_used
    )
    
    st.session_state.saved_pivot_rows = st.multiselect(
        "**행(Index) 컬럼 선택**",
        available_pivot_cols,
        default=st.session_state.saved_pivot_rows if all(col in available_pivot_cols for col in st.session_state.saved_pivot_rows) else [],
        key="pivot_rows"
    )
    
    st.divider()
    
    # ============================================================================
    # Step 5: 지표 선택 및 설정 저장
    # ============================================================================
    st.subheader("📈 Step 5: 지표 선택 및 설정 저장")
    
    col_check1, col_check2, col_check3 = st.columns([1.5, 1.5, 1.5])
    
    with col_check1:
        show_count = st.checkbox("건수", value=True, key="show_count")
    
    with col_check2:
        show_ppm = st.checkbox("PPM", value=True, key="show_ppm")
    
    # 선택한 메트릭 업데이트
    selected_metrics = []
    if show_count:
        selected_metrics.append('건수')
    if show_ppm:
        selected_metrics.append('PPM')
    st.session_state.selected_metrics = selected_metrics if selected_metrics else ['건수']
    
    if not (show_count or show_ppm):
        st.warning("⚠️ 최소 하나의 지표를 선택해야 합니다.")
    
    # 설정 저장 체크박스
    with col_check3:
        save_settings_checkbox = st.checkbox("💾 설정 기억하기", value=True, key="save_settings_cb")
        st.session_state.save_settings = save_settings_checkbox
    
    st.divider()
    
    # ============================================================================
    # Step 6: 분석 시작 (Execution)
    # ============================================================================
    st.subheader("🚀 Step 6: 분석 시작")
    
    if st.button("🚀 분석 시작", use_container_width=True, key="run_analysis", type="primary"):
        
        # 필터 유효성 검사
        is_valid, error_msg = validate_filters({
            '대분류': st.session_state.filter_major_category
        })
        
        if not is_valid:
            st.error(error_msg)
            st.stop()
        
        # ============================================================================
        # 6-A: 설정 저장 (필요시 - Step 3, 4 필터&피벗만)
        # ============================================================================
        if st.session_state.save_settings:
            settings_to_save = {
                'filter_business': st.session_state.filter_business,
                'filter_reason': st.session_state.filter_reason,
                'filter_grade': st.session_state.filter_grade,
                'filter_major_category': st.session_state.filter_major_category,
                'saved_pivot_rows': st.session_state.saved_pivot_rows
            }
            save_plant_settings(selected_plant, settings_to_save)
            st.success("✅ Step 3, 4 설정이 플랜트별로 저장되었습니다!")
        
        # ============================================================================
        # 6-B: 데이터 필터링
        # ============================================================================
        try:
            filtered_claims = st.session_state.claims_data[
                st.session_state.claims_data['플랜트'] == selected_plant
            ].copy()
            
            # 대분류 필터 (필수)
            if st.session_state.filter_major_category:
                filtered_claims = filtered_claims[
                    filtered_claims['대분류'].isin(st.session_state.filter_major_category)
                ]
            
            # 사업부문 필터
            if st.session_state.filter_business:
                filtered_claims = filtered_claims[
                    filtered_claims['사업부문'].isin(st.session_state.filter_business)
                ]
            
            # 등급기준 필터
            if st.session_state.filter_grade:
                filtered_claims = filtered_claims[
                    filtered_claims['등급기준'].isin(st.session_state.filter_grade)
                ]
            
            # 불만원인 필터
            if st.session_state.filter_reason:
                filtered_claims = filtered_claims[
                    filtered_claims['불만원인'].isin(st.session_state.filter_reason)
                ]
            
            if filtered_claims.empty:
                st.warning("⚠️ 선택한 필터 조건에 맞는 데이터가 없습니다.")
                st.stop()
            
            st.success(f"✅ 필터링 완료: {len(filtered_claims):,}건")
            
            # ============================================================================
            # 6-C: 향후 3개월 예측 (Seasonal Allocation)
            # ============================================================================
            st.info("📈 향후 3개월 예측을 생성 중입니다...")
            
            # 대분류별로 예측 수행
            prediction_results = []
            
            for major_cat in filtered_claims['대분류'].unique():
                cat_data = filtered_claims[filtered_claims['대분류'] == major_cat]
                
                try:
                    cat_predictions = predict_with_seasonal_allocation(
                        plant=selected_plant,
                        major_category=str(major_cat),
                        future_months=[1, 2, 3],  # 상대 월 (향후 1,2,3개월)
                        sub_dimensions_df=cat_data,
                        model_dir='data/models'
                    )
                    
                    if not cat_predictions.empty:
                        prediction_results.append(cat_predictions)
                
                except Exception as e:
                    st.warning(f"⚠️ {major_cat} 예측 실패: {str(e)}")
                    continue
            
            if prediction_results:
                predictions_df = pd.concat(prediction_results, ignore_index=True)
                st.success(f"✅ 예측 완료: {len(predictions_df)}건")
            else:
                predictions_df = pd.DataFrame()
                st.warning("⚠️ 예측 데이터를 생성할 수 없습니다.")
            
            # ============================================================================
            # 6-D: PPM 데이터 계산
            # ============================================================================
            ppm_data = calculate_ppm(
                filtered_claims,
                st.session_state.sales_data if st.session_state.sales_data is not None else pd.DataFrame(),
                selected_plant,
                st.session_state.saved_pivot_rows
            )
            
            if ppm_data.empty:
                st.warning(f"⚠️ {selected_plant}의 분석 데이터가 없습니다.")
                st.stop()
            
            # 대분류 정보 추가 (filtered_claims에서 매핑)
            major_cat_map = filtered_claims[['중분류', '대분류']].drop_duplicates().set_index('중분류')['대분류'].to_dict()
            if '소분류' in ppm_data.columns:
                ppm_data['대분류'] = ppm_data.get('중분류', ppm_data.get('소분류', '')).map(major_cat_map)
            elif '중분류' in ppm_data.columns:
                ppm_data['대분류'] = ppm_data['중분류'].map(major_cat_map)
            else:
                # 대분류가 이미 있으면 그대로 사용
                if '대분류' not in ppm_data.columns:
                    ppm_data['대분류'] = filtered_claims['대분류'].iloc[0] if not filtered_claims.empty else '미분류'
            
            # ============================================================================
            # 6-E: 시각화 (테이블 + 차트)
            # ============================================================================
            st.divider()
            st.subheader("📊 분석 결과")
            
            # 피벗 인덱스: 대분류 + 사용자 선택 컬럼 (대분류 제거)
            pivot_index = ['대분류'] + [col for col in st.session_state.saved_pivot_rows if col in ppm_data.columns and col != '대분류']
            
            # 건수 피벗
            count_pivot = None
            if '건수' in st.session_state.selected_metrics:
                st.write("#### 📋 건수 (월별 실적 + 3개월 예측)")
                try:
                    count_pivot = create_pivot_table(
                        ppm_data,
                        index_cols=pivot_index,
                        value_col='건수'
                    )
                    st.dataframe(count_pivot, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ 건수 피벗 생성 오류: {str(e)}")
                    print(f"[DEBUG] 건수 피벗: {str(e)}")
            
            # PPM 피벗
            ppm_pivot = None
            if 'PPM' in st.session_state.selected_metrics:
                st.write("#### 📊 PPM (Parts Per Million)")
                try:
                    ppm_pivot = create_pivot_table(
                        ppm_data,
                        index_cols=pivot_index,
                        value_col='PPM'
                    )
                    st.dataframe(ppm_pivot, use_container_width=True)
                    
                    # 추정치 표시
                    estimated_rows = ppm_data[ppm_data['is_estimated'] == True]
                    if not estimated_rows.empty:
                        st.info(f"⚠️ {len(estimated_rows)}개 행이 **예상치**입니다 (직전 3개월 평균값)")
                except Exception as e:
                    st.error(f"❌ PPM 피벗 생성 오류: {str(e)}")
                    print(f"[DEBUG] PPM 피벗: {str(e)}")
            
            # 시계열 차트
            st.write("#### 📉 시계열 차트")
            
            if '건수' in st.session_state.selected_metrics and count_pivot is not None:
                try:
                    # 첫 번째 컬럼(대분류)에서 "[전체] 총 합계" 행 찾기
                    total_rows = count_pivot[count_pivot.iloc[:, 0].astype(str).str.contains(r'\[전체\]', na=False, regex=True)]
                    
                    if not total_rows.empty:
                        # 첫 컬럼 제외 후 나머지 컬럼(년월)들을 시계열로 변환
                        timeline_long = total_rows.iloc[:, 1:].T.reset_index()
                        timeline_long.columns = ['기간', '건수']
                        timeline_long['건수'] = pd.to_numeric(timeline_long['건수'], errors='coerce')
                        timeline_long = timeline_long.dropna(subset=['건수'])
                        
                        if not timeline_long.empty:
                            fig_count = px.line(
                                timeline_long,
                                x='기간',
                                y='건수',
                                markers=True,
                                title=f'{selected_plant} - 월별 클레임 건수 (총합)',
                                labels={'건수': '건수', '기간': '기간'},
                                category_orders={'기간': timeline_long['기간'].tolist()}
                            )
                            fig_count.update_xaxes(tickangle=45)
                            st.plotly_chart(fig_count, use_container_width=True)
                        else:
                            st.info("💡 건수 차트 데이터가 없습니다.")
                    else:
                        st.info("💡 '[전체] 총 합계' 행을 찾을 수 없습니다.")
                except Exception as e:
                    st.error(f"❌ 건수 차트 오류: {str(e)}")
                    print(f"[DEBUG] 건수 차트: {str(e)}")
            
            if 'PPM' in st.session_state.selected_metrics and ppm_pivot is not None:
                try:
                    # 첫 번째 컬럼(대분류)에서 "[전체] 총 합계" 행 찾기
                    total_rows_ppm = ppm_pivot[ppm_pivot.iloc[:, 0].astype(str).str.contains(r'\[전체\]', na=False, regex=True)]
                    
                    if not total_rows_ppm.empty:
                        # 첫 컬럼 제외 후 나머지 컬럼(년월)들을 시계열로 변환
                        timeline_long_ppm = total_rows_ppm.iloc[:, 1:].T.reset_index()
                        timeline_long_ppm.columns = ['기간', 'PPM']
                        timeline_long_ppm['PPM'] = pd.to_numeric(timeline_long_ppm['PPM'], errors='coerce')
                        timeline_long_ppm = timeline_long_ppm.dropna(subset=['PPM'])
                        
                        if not timeline_long_ppm.empty:
                            fig_ppm = px.line(
                                timeline_long_ppm,
                                x='기간',
                                y='PPM',
                                markers=True,
                                title=f'{selected_plant} - 월별 PPM (총합)',
                                labels={'PPM': 'PPM', '기간': '기간'},
                                category_orders={'기간': timeline_long_ppm['기간'].tolist()}
                            )
                            fig_ppm.update_xaxes(tickangle=45)
                            st.plotly_chart(fig_ppm, use_container_width=True)
                        else:
                            st.info("💡 PPM 차트 데이터가 없습니다.")
                    else:
                        st.info("💡 '[전체] 총 합계' 행을 찾을 수 없습니다.")
                except Exception as e:
                    st.error(f"❌ PPM 차트 오류: {str(e)}")
                    print(f"[DEBUG] PPM 차트: {str(e)}")
            
            # 상세 통계
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
        
        except Exception as e:
            st.error(f"❌ 분석 중 오류 발생: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

else:
    st.info("💡 위에서 플랜트를 선택하여 분석을 시작해주세요.")

# ============================================================================
# 페이지: 매출수량 관리
# ============================================================================
# 설명: st.data_editor를 사용하여 플랜트별 년/월 매출수량을 입력/수정하고
#      data/sales/sales_history.parquet에 저장합니다.
#      Smart Sync 로직으로 클레임 데이터와 자동 동기화합니다.

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional

from core.config import DATA_SALES_PATH, SALES_FILENAME
from core.storage import get_claim_keys

# ============================================================================
# 페이지 레이아웃 설정
# ============================================================================
st.set_page_config(page_title="매출수량 관리", page_icon="💰", layout="wide")
st.title("💰 매출수량 관리")
st.markdown("플랜트별 년/월 매출수량을 엑셀 형식으로 입력/수정합니다.")

# ============================================================================
# 기본 설정
# ============================================================================
SALES_PATH = Path(DATA_SALES_PATH)
SALES_PATH.mkdir(parents=True, exist_ok=True)
SALES_FILE = SALES_PATH / SALES_FILENAME


# ============================================================================
# 매출 데이터 로드 함수
# ============================================================================
def load_sales_data() -> pd.DataFrame:
    """
    저장된 매출 데이터 로드.
    
    Returns:
        pd.DataFrame: {플랜트, 년, 월, 매출수량} 스키마
    """
    if SALES_FILE.exists():
        try:
            df = pd.read_parquet(SALES_FILE)
            return df.sort_values(['플랜트', '년', '월']).reset_index(drop=True)
        except Exception as e:
            st.warning(f"매출 데이터 로드 실패: {str(e)}")
            return pd.DataFrame(columns=['플랜트', '년', '월', '매출수량'])
    else:
        return pd.DataFrame(columns=['플랜트', '년', '월', '매출수량'])


def save_sales_data(df: pd.DataFrame) -> None:
    """
    매출 데이터 저장.
    
    Args:
        df: 저장할 데이터프레임
    """
    try:
        # is_estimated 컬럼이 있으면 유지, 없으면 False로 추가
        if 'is_estimated' not in df.columns:
            df['is_estimated'] = False
        
        df.to_parquet(SALES_FILE, engine='pyarrow', index=False)
        st.success(f"✅ 매출 데이터 저장 완료: {len(df)} 행")
    except Exception as e:
        st.error(f"❌ 저장 실패: {str(e)}")


def sync_with_claims() -> pd.DataFrame:
    """
    Smart Sync: 클레임 데이터와 매출 데이터 동기화.
    
    동작:
        1. get_claim_keys()로 클레임의 [플랜트, 접수년, 접수월] 추출
        2. 기존 매출 데이터와 비교
        3. 클레임은 있는데 매출이 없는 행 자동 추가 (값은 공백)
        4. 자동 추가된 행은 경고 없이 사용자가 자유롭게 입력 가능
    
    Returns:
        pd.DataFrame: 동기화된 매출 데이터
    """
    try:
        claim_keys = get_claim_keys()
        if claim_keys.empty:
            return load_sales_data()
        
        # 컬럼명 표준화 (클레임은 접수년/월, 매출은 년/월)
        claim_keys_renamed = claim_keys.rename(columns={
            '접수년': '년',
            '접수월': '월'
        }).copy()
        
        # 기존 매출 데이터 로드
        sales_df = load_sales_data()
        
        # 병합: 클레임 키를 기준으로 좌조인
        merged = claim_keys_renamed.merge(
            sales_df,
            on=['플랜트', '년', '월'],
            how='left'
        )
        
        # 매출수량이 NaN인 행은 사용자 입력 대기 (값 유지)
        merged['매출수량'] = merged['매출수량'].fillna(0)
        
        # is_estimated가 없으면 False로 초기화
        if 'is_estimated' not in merged.columns:
            merged['is_estimated'] = False
        
        return merged.sort_values(['플랜트', '년', '월']).reset_index(drop=True)
    
    except Exception as e:
        print(f"[ERROR] Smart Sync 실패: {str(e)}")
        return load_sales_data()


# ============================================================================
# 세션 상태 초기화
# ============================================================================
if 'sales_df' not in st.session_state:
    st.session_state.sales_df = sync_with_claims()  # Smart Sync 적용
if 'edited_sales' not in st.session_state:
    st.session_state.edited_sales = False


# ============================================================================
# 영역 1: Smart Sync 정보
# ============================================================================
with st.container():
    st.info(
        "🔄 **Smart Sync 활성화**: 클레임 데이터와 자동 동기화됩니다. "
        "클레임은 있는데 매출이 없는 항목이 자동으로 추가됩니다.",
        icon="ℹ️"
    )


# ============================================================================
# 영역 2: 새 데이터 추가
# ============================================================================
st.subheader("➕ 새 항목 추가 (선택사항)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    new_plant = st.text_input("플랜트명", key="new_plant")
with col2:
    new_year = st.number_input("년", min_value=2000, max_value=2099, value=2026, key="new_year")
with col3:
    new_month = st.number_input("월", min_value=1, max_value=12, value=1, key="new_month")
with col4:
    new_sales = st.number_input("매출수량", min_value=0, value=0, key="new_sales")

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    if st.button("➕ 추가", key="add_row", use_container_width=True):
        if new_plant:
            new_row = pd.DataFrame({
                '플랜트': [new_plant],
                '년': [int(new_year)],
                '월': [int(new_month)],
                '매출수량': [int(new_sales)]
            })
            st.session_state.sales_df = pd.concat(
                [st.session_state.sales_df, new_row],
                ignore_index=True
            ).drop_duplicates(subset=['플랜트', '년', '월'], keep='last').sort_values(['플랜트', '년', '월']).reset_index(drop=True)
            st.rerun()
        else:
            st.error("❌ 플랜트명을 입력하세요.")


# ============================================================================
# 영역 3: 데이터 편집 (st.data_editor)
# ============================================================================
st.subheader("✏️ 매출수량 입력/수정")

st.markdown(
    "아래 테이블에서 직접 값을 입력/수정할 수 있습니다. "
    "빈 행의 매출수량을 입력하거나 기존 값을 수정하세요. "
    "(우측 🗑️ 버튼으로 행 삭제 가능)"
)

# 데이터 에디터 - is_estimated 컬럼 표시
display_cols = ['플랜트', '년', '월', '매출수량', 'is_estimated']
display_df = st.session_state.sales_df[display_cols].copy() if all(col in st.session_state.sales_df.columns for col in display_cols) else st.session_state.sales_df

edited_df = st.data_editor(
    display_df,
    use_container_width=True,
    height=350,
    num_rows="dynamic",  # 동적 행 추가/삭제 허용
    disabled=['is_estimated'],  # is_estimated 는 읽기 전용
    key="sales_editor"
)

# 변경사항 감지 및 저장
if edited_df is not None and not edited_df.equals(st.session_state.sales_df[display_cols] if all(col in st.session_state.sales_df.columns for col in display_cols) else st.session_state.sales_df):
    st.session_state.sales_df = edited_df.reset_index(drop=True)

# ============================================================================
# 영역 4: 저장 및 통계
# ============================================================================
st.subheader("💾 저장 및 통계")

col_stats1, col_stats2, col_stats3 = st.columns(3)

with col_stats1:
    st.metric("총 행 수", len(st.session_state.sales_df))

with col_stats2:
    unique_plants = st.session_state.sales_df['플랜트'].nunique() if not st.session_state.sales_df.empty else 0
    st.metric("플랜트 수", unique_plants)

with col_stats3:
    total_sales = st.session_state.sales_df['매출수량'].sum() if not st.session_state.sales_df.empty else 0
    st.metric("총 매출수량", f"{int(total_sales):,}")

# 예상치 개수 표시
if not st.session_state.sales_df.empty and 'is_estimated' in st.session_state.sales_df.columns:
    estimated_count = st.session_state.sales_df['is_estimated'].sum()
    if estimated_count > 0:
        st.warning(f"⚠️ {estimated_count}개 행이 추정치입니다 (직전 3개월 평균값)")

# 저장 버튼
col_save1, col_save2 = st.columns([1, 4])

with col_save1:
    if st.button("💾 저장", key="save_sales", use_container_width=True):
        if not st.session_state.sales_df.empty:
            # 데이터 검증
            required_cols = ['플랜트', '년', '월', '매출수량']
            if all(col in st.session_state.sales_df.columns for col in required_cols):
                save_sales_data(st.session_state.sales_df)
                st.session_state.edited_sales = False
            else:
                st.error(f"❌ 필수 컬럼 부재: {required_cols}")
        else:
            st.error("❌ 저장할 데이터가 없습니다.")


# ============================================================================
# 영역 5: 데이터 미리보기
# ============================================================================
if not st.session_state.sales_df.empty:
    with st.expander("📊 플랜트별 통계", expanded=False):
        # 플랜트별 통계
        plant_stats = st.session_state.sales_df.groupby('플랜트').agg({
            '매출수량': ['sum', 'mean', 'count']
        }).round(2)
        plant_stats.columns = ['합계', '평균', '개수']
        st.dataframe(plant_stats, use_container_width=True)
    
    with st.expander("📅 년/월별 통계", expanded=False):
        # 년/월별 통계
        period_stats = st.session_state.sales_df.groupby(['년', '월']).agg({
            '매출수량': ['sum', 'count']
        }).round(2)
        period_stats.columns = ['합계', '플랜트_수']
        st.dataframe(period_stats, use_container_width=True)

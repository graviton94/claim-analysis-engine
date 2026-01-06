# ============================================================================
# 페이지: 매출수량 관리
# ============================================================================
# 설명: st.data_editor를 사용하여 플랜트별 년/월 매출수량을 입력/수정하고
#      data/sales/sales_history.parquet에 저장합니다.

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional

from core.config import DATA_SALES_PATH, SALES_FILENAME

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
        df.to_parquet(SALES_FILE, engine='pyarrow', index=False)
        st.success(f"✅ 매출 데이터 저장 완료: {len(df)} 행")
    except Exception as e:
        st.error(f"❌ 저장 실패: {str(e)}")


# ============================================================================
# 세션 상태 초기화
# ============================================================================
if 'sales_df' not in st.session_state:
    st.session_state.sales_df = load_sales_data()
if 'edited_sales' not in st.session_state:
    st.session_state.edited_sales = False


# ============================================================================
# 영역 1: 새 데이터 추가
# ============================================================================
st.subheader("➕ 새 항목 추가")

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
# 영역 2: 데이터 편집 (st.data_editor)
# ============================================================================
st.subheader("✏️ 데이터 편집")

st.info(
    "아래 테이블에서 직접 값을 수정할 수 있습니다. "
    "행을 선택하고 우측의 🗑️ 버튼으로 삭제할 수 있습니다.",
    icon="ℹ️"
)

# 데이터 에디터
edited_df = st.data_editor(
    st.session_state.sales_df,
    use_container_width=True,
    height=300,
    num_rows="dynamic",  # 동적 행 추가/삭제 허용
    key="sales_editor"
)

# 변경사항 감지 및 저장
if edited_df is not None and not edited_df.equals(st.session_state.sales_df):
    st.session_state.sales_df = edited_df.reset_index(drop=True)
    st.session_state.edited_sales = True


# ============================================================================
# 영역 3: 저장 및 통계
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
    st.metric("총 매출수량", f"{total_sales:,}")


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
# 영역 4: 데이터 미리보기
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

# ============================================================================
# 페이지: 매출수량 관리 (피벗 테이블 형태)
# ============================================================================
# 설명: 엑셀 스타일 피벗 테이블로 플랜트별 년/월 매출수량을 관리합니다.
#      행: 플랜트명
#      열: [년-월] 조합 (멀티인덱스 헤더)
#      값: 매출수량

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from core.config import DATA_SALES_PATH, SALES_FILENAME, DATA_HUB_PATH
from core.storage import get_claim_keys

# ============================================================================
# 페이지 레이아웃 설정
# ============================================================================
st.set_page_config(page_title="매출수량 관리", page_icon="💰", layout="wide")
st.title("💰 매출수량 관리 (피벗 테이블)")
st.markdown(
    "엑셀 스타일 피벗 테이블로 플랜트별 년/월 매출수량을 관리합니다.\n\n"
    "- **행**: 플랜트명 (클레임 데이터 기준 자동 추출)\n"
    "- **열**: 년-월 조합 (클레임 데이터 기준 자동 생성)\n"
    "- **값**: 매출수량 (직접 입력)"
)

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


def get_period_columns_from_claims() -> Tuple[list, list]:
    """
    클레임 데이터에서 [년, 월] 조합 추출.
    
    Returns:
        Tuple[list, list]: (년도 리스트, 월 리스트) - 정렬된 유니크 값
    """
    try:
        claim_keys = get_claim_keys()
        if claim_keys.empty:
            return [], []
        
        # 년/월을 숫자로 변환 (정렬 위해)
        claim_keys['접수년'] = pd.to_numeric(claim_keys['접수년'], errors='coerce')
        claim_keys['접수월'] = pd.to_numeric(claim_keys['접수월'], errors='coerce')
        
        # 유니크한 년/월 조합 추출
        periods = claim_keys[['접수년', '접수월']].drop_duplicates().sort_values(['접수년', '접수월'])
        
        years = periods['접수년'].astype(int).tolist()
        months = periods['접수월'].astype(int).tolist()
        
        return years, months
    
    except Exception as e:
        print(f"[ERROR] 기간 추출 실패: {str(e)}")
        return [], []


def long_to_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long 형식 → Pivot 형식 변환.
    
    Args:
        df: {플랜트, 년, 월, 매출수량} Long 형식 데이터
    
    Returns:
        pd.DataFrame: 피벗 테이블 (행: 플랜트, 열: 년-월)
    """
    if df.empty:
        return pd.DataFrame()
    
    # 년/월을 문자열로 변환하여 컬럼명 생성 (예: "2025-01")
    df = df.copy()
    df['년'] = pd.to_numeric(df['년'], errors='coerce').fillna(0).astype(int)
    df['월'] = pd.to_numeric(df['월'], errors='coerce').fillna(0).astype(int)
    df['년월'] = df['년'].astype(str) + '-' + df['월'].astype(str).str.zfill(2)
    
    # 피벗 테이블 생성
    pivot = df.pivot_table(
        index='플랜트',
        columns='년월',
        values='매출수량',
        aggfunc='sum',
        fill_value=0
    )
    
    # 컬럼 정렬 (년-월 순서대로)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    
    return pivot


def pivot_to_long(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot 형식 → Long 형식 변환.
    
    Args:
        pivot_df: 피벗 테이블 (행: 플랜트, 열: 년-월)
    
    Returns:
        pd.DataFrame: {플랜트, 년, 월, 매출수량} Long 형식
    """
    if pivot_df.empty:
        return pd.DataFrame(columns=['플랜트', '년', '월', '매출수량'])
    
    # Pivot을 Long 형식으로 변환
    long_df = pivot_df.reset_index().melt(
        id_vars='플랜트',
        var_name='년월',
        value_name='매출수량'
    )
    
    # 년월 컬럼 분리 (예: "2025-01" → 년=2025, 월=1)
    long_df[['년', '월']] = long_df['년월'].str.split('-', expand=True)
    long_df['년'] = pd.to_numeric(long_df['년'], errors='coerce').fillna(0).astype(int)
    long_df['월'] = pd.to_numeric(long_df['월'], errors='coerce').fillna(0).astype(int)
    
    # 년월 컬럼 제거
    long_df = long_df[['플랜트', '년', '월', '매출수량']]
    
    # is_estimated 컬럼 추가 (기본값 False)
    long_df['is_estimated'] = False
    
    return long_df.sort_values(['플랜트', '년', '월']).reset_index(drop=True)


# ============================================================================
# 세션 상태 초기화
# ============================================================================
if 'sales_long_df' not in st.session_state:
    st.session_state.sales_long_df = sync_with_claims()  # Long 형식
if 'sales_pivot_df' not in st.session_state:
    st.session_state.sales_pivot_df = long_to_pivot(st.session_state.sales_long_df)  # Pivot 형식


# ============================================================================
# 영역 1: Smart Sync 정보
# ============================================================================
with st.container():
    st.info(
        "🔄 **Smart Sync 활성화**: 클레임 데이터의 [플랜트, 년, 월] 조합을 자동으로 테이블에 반영합니다.\n\n"
        "- **행(플랜트)**: 업로드된 CSV의 플랜트 unique 값\n"
        "- **열(년-월)**: 클레임 데이터의 년월 조합 (자동 정렬)",
        icon="ℹ️"
    )


# ============================================================================
# 영역 2: 피벗 테이블 편집
# ============================================================================
st.subheader("📊 매출수량 피벗 테이블 (엑셀 스타일)")

# 클레임 데이터가 없는 경우
if st.session_state.sales_pivot_df.empty:
    st.warning(
        "⚠️ 클레임 데이터가 없습니다.\n\n"
        "**[데이터 업로드]** 메뉴에서 먼저 CSV 파일을 업로드하세요."
    )
    st.stop()

st.markdown(
    "아래 테이블에서 매출수량을 직접 입력/수정할 수 있습니다.\n\n"
    "- **행**: 플랜트명\n"
    "- **열**: 년-월 (예: 2025-01, 2025-02, ...)\n"
    "- **값**: 매출수량 (0 = 미입력)"
)

# 피벗 테이블 에디터
edited_pivot = st.data_editor(
    st.session_state.sales_pivot_df,
    use_container_width=True,
    height=400,
    num_rows="fixed",  # 행 추가/삭제 불가 (클레임 기준)
    key="pivot_editor"
)

# 변경사항 자동 반영
if edited_pivot is not None and not edited_pivot.equals(st.session_state.sales_pivot_df):
    st.session_state.sales_pivot_df = edited_pivot
    # Pivot → Long 변환
    st.session_state.sales_long_df = pivot_to_long(edited_pivot)

# ============================================================================
# 영역 3: 저장 및 통계
# ============================================================================
st.subheader("💾 저장 및 통계")

col_stats1, col_stats2, col_stats3 = st.columns(3)

with col_stats1:
    st.metric("플랜트 수", len(st.session_state.sales_pivot_df))

with col_stats2:
    period_count = len(st.session_state.sales_pivot_df.columns) if not st.session_state.sales_pivot_df.empty else 0
    st.metric("년-월 기간 수", period_count)

with col_stats3:
    total_sales = st.session_state.sales_pivot_df.sum().sum() if not st.session_state.sales_pivot_df.empty else 0
    st.metric("총 매출수량", f"{int(total_sales):,}")

# 저장 버튼
col_save1, col_save2 = st.columns([1, 4])

with col_save1:
    if st.button("💾 저장", key="save_sales", use_container_width=True):
        if not st.session_state.sales_long_df.empty:
            save_sales_data(st.session_state.sales_long_df)
        else:
            st.error("❌ 저장할 데이터가 없습니다.")


# ============================================================================
# 영역 4: 데이터 미리보기 (Long 형식)
# ============================================================================
if not st.session_state.sales_long_df.empty:
    with st.expander("📋 Long 형식 데이터 미리보기 (저장 형식)", expanded=False):
        st.markdown("피벗 테이블은 **Long 형식**으로 변환되어 저장됩니다.")
        st.dataframe(
            st.session_state.sales_long_df.head(50),
            use_container_width=True,
            height=250
        )
    
    with st.expander("📊 플랜트별 통계", expanded=False):
        # 플랜트별 통계
        plant_stats = st.session_state.sales_long_df.groupby('플랜트').agg({
            '매출수량': ['sum', 'mean', 'count']
        }).round(2)
        plant_stats.columns = ['합계', '평균', '개수']
        st.dataframe(plant_stats, use_container_width=True)

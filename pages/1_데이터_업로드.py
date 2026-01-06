# ============================================================================
# 페이지: 데이터 업로드
# ============================================================================
# 설명: 클레임 데이터 CSV/Excel 파일을 업로드하여 처리하고 파티셔닝 저장합니다.

import streamlit as st
import pandas as pd
from pathlib import Path
from io import BytesIO

from core.etl import process_claim_data
from core.storage import save_partitioned, get_available_periods
from core.config import DATA_HUB_PATH

# ============================================================================
# 페이지 레이아웃 설정
# ============================================================================
st.set_page_config(page_title="데이터 업로드", page_icon="📤", layout="wide")
st.title("📤 클레임 데이터 업로드")
st.markdown("CSV 또는 Excel 파일을 업로드하여 54개 핵심 필드로 변환 및 저장합니다.")

# ============================================================================
# 세션 상태 초기화
# ============================================================================
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None


# ============================================================================
# 파일 업로드 영역
# ============================================================================
with st.container():
    st.subheader("📁 Step 1: 파일 업로드")
    
    uploaded_file = st.file_uploader(
        "CSV 또는 Excel 파일을 선택하세요",
        type=["csv", "xlsx", "xls"],
        help="컬럼 개수는 무관하며, 자동으로 54개 필드로 표준화됩니다."
    )
    
    if uploaded_file is not None:
        # 파일을 임시 경로에 저장 후 처리
        temp_path = Path(f"/tmp/{uploaded_file.name}")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # ETL 처리
        try:
            st.session_state.processed_df = process_claim_data(temp_path)
            st.success(f"✅ 파일 로드 및 처리 완료: {len(st.session_state.processed_df)} 행")
        except Exception as e:
            st.error(f"❌ 파일 처리 실패: {str(e)}")
            st.session_state.processed_df = None


# ============================================================================
# 데이터 미리보기 및 검증
# ============================================================================
if st.session_state.processed_df is not None:
    st.subheader("📊 Step 2: 데이터 미리보기")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("총 행 수", len(st.session_state.processed_df))
    with col2:
        st.metric("컬럼 수", len(st.session_state.processed_df.columns))
    
    # 데이터 테이블 표시
    st.dataframe(
        st.session_state.processed_df.head(10),
        use_container_width=True,
        height=300
    )
    
    # 컬럼별 NaN 비율
    with st.expander("📈 데이터 품질 정보"):
        null_ratio = (st.session_state.processed_df.isnull().sum() / len(st.session_state.processed_df)) * 100
        null_df = pd.DataFrame({
            '컬럼': null_ratio.index,
            'NaN 비율 (%)': null_ratio.values.round(2)
        }).sort_values('NaN 비율 (%)', ascending=False)
        
        st.dataframe(null_df, use_container_width=True, hide_index=True)


# ============================================================================
# 파티셔닝 저장
# ============================================================================
if st.session_state.processed_df is not None:
    st.subheader("💾 Step 3: 파티셔닝 저장")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(
            f"📍 저장 경로: `{DATA_HUB_PATH}`\n\n"
            f"구조: `접수년=YYYY/접수월=MM/part-0.parquet`",
            icon="ℹ️"
        )
    
    with col2:
        if st.button("💾 저장", key="save_partitioned", use_container_width=True):
            try:
                # 필수 컬럼 확인
                if '접수년' not in st.session_state.processed_df.columns or \
                   '접수월' not in st.session_state.processed_df.columns:
                    st.error("❌ 접수년/접수월 컬럼이 없습니다.")
                else:
                    save_partitioned(st.session_state.processed_df, output_path=DATA_HUB_PATH)
                    st.success("✅ 파티셔닝 저장 완료!")
                    st.session_state.save_complete = True
            except Exception as e:
                st.error(f"❌ 저장 실패: {str(e)}")


# ============================================================================
# 저장 완료 후 사용 가능한 기간 표시
# ============================================================================
if st.session_state.processed_df is not None and 'save_complete' in st.session_state and st.session_state.save_complete:
    st.subheader("📅 저장된 기간 목록")
    try:
        periods = get_available_periods(DATA_HUB_PATH)
        if not periods.empty:
            st.dataframe(periods, use_container_width=True, hide_index=True)
            st.success(f"총 {len(periods)} 개의 년/월 조합이 저장되었습니다.")
        else:
            st.info("저장된 기간 정보 없음")
    except Exception as e:
        st.warning(f"기간 목록 조회 실패: {str(e)}")

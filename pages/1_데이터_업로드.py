# ============================================================================
# 페이지: 데이터 업로드
# ============================================================================
# 설명: 클레임 데이터 CSV/Excel 파일을 업로드하여 처리하고 파티셔닝 저장합니다.

import streamlit as st
import pandas as pd
from pathlib import Path
from io import BytesIO

from core.etl import process_claim_data, preprocess_data
from core.storage import save_partitioned_parquet, get_available_periods, generate_nested_series
from core.config import DATA_HUB_PATH, DATA_SERIES_PATH

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
        width='stretch',
        height=300
    )
    
    # 컬럼별 NaN 비율
    with st.expander("📈 데이터 품질 정보"):
        null_ratio = (st.session_state.processed_df.isnull().sum() / len(st.session_state.processed_df)) * 100
        null_df = pd.DataFrame({
            '컬럼': null_ratio.index,
            'NaN 비율 (%)': null_ratio.values.round(2)
        }).sort_values('NaN 비율 (%)', ascending=False)
        
        st.dataframe(null_df, width='stretch', hide_index=True)


# ============================================================================
# 파티셔닝 저장
# ============================================================================
if st.session_state.processed_df is not None:
    st.subheader("💾 Step 3: 데이터 저장")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(
            f"📍 Hub DB 경로: `{DATA_HUB_PATH}`\n"
            f"📍 상세 DB 경로: `{DATA_SERIES_PATH}`",
            icon="ℹ️"
        )
    
    with col2:
        if st.button("💾 저장", key="save_partitioned", width='stretch'):
            try:
                # 1) 데이터 전처리 강화
                enhanced_df = preprocess_data(st.session_state.processed_df)

                # 2) Parquet 허브 저장 (Lag_Days, Lag_Valid 포함)
                save_partitioned_parquet(enhanced_df, output_path=DATA_HUB_PATH)

                # 3) Nested Series JSON 생성
                created = generate_nested_series(enhanced_df, output_dir=DATA_SERIES_PATH)

                # 4) 완료 메시지
                st.success(f"✅ Parquet 저장 및 {created}개 Series JSON 생성 완료")
                st.session_state.save_complete = True

                # 5) 전체 캐시 무효화 (다른 페이지의 @st.cache_data 재로딩 유도)
                try:
                    st.cache_data.clear()
                    st.toast("캐시 초기화 완료 – 분석 페이지에서 최신 데이터 반영", icon="✅")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"❌ 저장 실패: {str(e)}")


# ============================================================================
# 저장 완료 후 사용 가능한 기간 표시
# ============================================================================
if 'save_complete' in st.session_state and st.session_state.save_complete:
    st.subheader("📅 저장된 기간별 데이터 현황")
    try:
        periods_df = get_available_periods(DATA_HUB_PATH)
        if not periods_df.empty:
            st.dataframe(periods_df, width='stretch', hide_index=True)
            
            total_records = periods_df['건수'].sum()
            total_periods = len(periods_df)
            
            st.success(f"총 {total_periods}개 기간에 걸쳐 {total_records: ,}건의 데이터가 저장되었습니다.")
        else:
            st.info("현재 Hub DB에 저장된 데이터가 없습니다.")
    except Exception as e:
        st.warning(f"저장된 기간 목록을 불러오는 데 실패했습니다: {e}")

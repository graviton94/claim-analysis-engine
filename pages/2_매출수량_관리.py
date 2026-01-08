# ============================================================================ 
# 페이지: 매출수량 관리 (피벗 테이블 형태)
# ============================================================================ 
# 설명: 엑셀 스타일 피벗 테이블로 플랜트별 년/월 매출수량을 관리합니다.
#      행: ID, 플랜트명
#      열: [년-월] 조합
#      값: 매출수량

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from core.config import DATA_SALES_PATH, SALES_FILENAME
from core.storage import get_claim_keys

# ============================================================================ 
# 페이지 레이아웃 설정
# ============================================================================ 
st.set_page_config(page_title="매출수량 관리", page_icon="💰", layout="wide")
st.title("💰 매출수량 관리 (피벗 테이블)")
st.markdown(
    "엑셀 스타일 피벗 테이블로 플랜트별 년/월 매출수량을 관리합니다. ID를 포함하여 수정, 저장할 수 있습니다.\n\n" 
    "- **행**: 구분자(ID), 플랜트명 (클레임 데이터 기준 자동 추출)\n" 
    "- **열**: 년-월 조합 (클레임 데이터 기준 자동 생성)\n" 
    "- **값**: 매출수량 (직접 입력)"
)

# ============================================================================ 
# 기본 설정
# ============================================================================ 
SALES_PATH = Path(DATA_SALES_PATH)
SALES_PATH.mkdir(parents=True, exist_ok=True)
SALES_FILE = SALES_PATH / SALES_FILENAME
BASE_COLUMNS = ['ID', '플랜트', '년', '월', '매출수량']


# ============================================================================ 
# 데이터 처리 함수 (ID 컬럼 추가)
# ============================================================================ 
def load_sales_data() -> pd.DataFrame:
    """저장된 매출 데이터(ID 포함) 로드."""
    if SALES_FILE.exists():
        try:
            df = pd.read_parquet(SALES_FILE)
            if 'ID' not in df.columns:
                df['ID'] = ''  # 하위 호환성
            return df.sort_values(['플랜트', '년', '월']).reset_index(drop=True)
        except Exception as e:
            st.warning(f"매출 데이터 로드 실패: {str(e)}")
    return pd.DataFrame(columns=BASE_COLUMNS)

def save_sales_data(df: pd.DataFrame) -> None:
    """매출 데이터(ID 포함) 저장."""
    try:
        if 'is_estimated' not in df.columns:
            df['is_estimated'] = False
        
        # 스키마 순서 고정
        df = df.reindex(columns=BASE_COLUMNS + ['is_estimated'], fill_value='')
        df.to_parquet(SALES_FILE, engine='pyarrow', index=False)
        st.success(f"✅ 매출 데이터 저장 완료: {len(df)} 행")
    except Exception as e:
        st.error(f"❌ 저장 실패: {str(e)}")

def sync_with_claims() -> pd.DataFrame:
    """클레임 데이터와 매출 데이터 동기화 (ID 포함)."""
    try:
        claim_keys = get_claim_keys()
        if claim_keys.empty:
            return load_sales_data()
        
        claim_keys_renamed = claim_keys.rename(columns={'접수년': '년', '접수월': '월'})
        sales_df = load_sales_data()

        # ID가 없는 sales_df에 ID 컬럼 추가 (하위 호환)
        if 'ID' not in sales_df.columns:
            sales_df['ID'] = ''
        
        # 클레임 키의 플랜트별로 가장 최신 ID를 가져와서 매핑 준비
        latest_ids = sales_df.sort_values(['년', '월'], ascending=False).drop_duplicates('플랜트')[['플랜트', 'ID']]
        
        # 클레임 키와 ID 병합
        claim_keys_with_id = claim_keys_renamed.merge(latest_ids, on='플랜트', how='left')
        claim_keys_with_id['ID'] = claim_keys_with_id['ID'].fillna('')

        # 클레임 키 기준으로 매출 데이터 병합
        merged = claim_keys_with_id.merge(sales_df.drop(columns='ID'), on=['플랜트', '년', '월'], how='left')
        
        merged['매출수량'] = merged['매출수량'].fillna(0)
        
        if 'is_estimated' not in merged.columns:
            merged['is_estimated'] = False
        
        return merged.sort_values(['플랜트', '년', '월']).reset_index(drop=True)
    
    except Exception as e:
        print(f"[ERROR] Smart Sync 실패: {str(e)}")
        return load_sales_data()

def long_to_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Long 형식 → Pivot 형식 변환 (ID 포함)."""
    if df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    df['년'] = pd.to_numeric(df['년'], errors='coerce').fillna(0).astype(int)
    df['월'] = pd.to_numeric(df['월'], errors='coerce').fillna(0).astype(int)
    df['년월'] = df['년'].astype(str) + '-' + df['월'].astype(str).str.zfill(2)
    
    # 1. 매출수량 피벗
    pivot_sales = df.pivot_table(index='플랜트', columns='년월', values='매출수량', aggfunc='sum', fill_value=0)
    
    # 2. 플랜트별 고유 ID 추출 (가장 마지막 값 사용)
    id_df = df.sort_values('년월').drop_duplicates('플랜트', keep='last')[['플랜트', 'ID']].set_index('플랜트')
    
    # 3. ID와 매출수량 피벗 결합
    pivot_combined = id_df.join(pivot_sales).reset_index()
    
    # 4. 컬럼 순서 재정렬 (ID, 플랜트, 년월 순)
    sorted_yyyymm = sorted([col for col in pivot_combined.columns if col not in ['ID', '플랜트']])
    display_columns = ['ID', '플랜트'] + sorted_yyyymm
    pivot_final = pivot_combined.reindex(columns=display_columns)
    
    return pivot_final.fillna({'ID': ''})

def pivot_to_long(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot 형식 → Long 형식 변환 (ID 포함)."""
    if pivot_df.empty:
        return pd.DataFrame(columns=BASE_COLUMNS)
    
    # ID와 플랜트를 기준으로 Melt
    long_df = pivot_df.melt(id_vars=['ID', '플랜트'], var_name='년월', value_name='매출수량')
    
    long_df[['년', '월']] = long_df['년월'].str.split('-', expand=True)
    long_df['년'] = pd.to_numeric(long_df['년'], errors='coerce').fillna(0).astype(int)
    long_df['월'] = pd.to_numeric(long_df['월'], errors='coerce').fillna(0).astype(int)
    
    long_df = long_df[BASE_COLUMNS]
    long_df['is_estimated'] = False
    
    return long_df.sort_values(['플랜트', '년', '월']).reset_index(drop=True)


# ============================================================================ 
# CSV 업로드 함수
# ============================================================================ 
def merge_csv_data(existing_long_df: pd.DataFrame, csv_df: pd.DataFrame) -> pd.DataFrame:
    """CSV 데이터를 기존 데이터와 병합 (ID 또는 플랜트 기준)."""
    if csv_df.empty or existing_long_df.empty:
        return existing_long_df
    
    # CSV의 필수 컬럼 검증
    required_cols = ['ID', '플랜트', '년', '월', '매출수량']
    missing = [c for c in required_cols if c not in csv_df.columns]
    if missing:
        st.error(f"CSV 컬럼 부족: {', '.join(missing)}")
        return existing_long_df
    
    # 데이터 타입 정규화
    csv_df['년'] = pd.to_numeric(csv_df['년'], errors='coerce').fillna(0).astype(int)
    csv_df['월'] = pd.to_numeric(csv_df['월'], errors='coerce').fillna(0).astype(int)
    csv_df['매출수량'] = pd.to_numeric(csv_df['매출수량'], errors='coerce').fillna(0)
    
    # CSV 데이터 정리
    csv_clean = csv_df[required_cols].copy()
    csv_clean = csv_clean[(csv_clean['년'] > 0) & (csv_clean['월'] > 0)]
    
    if csv_clean.empty:
        st.warning("유효한 CSV 데이터가 없습니다.")
        return existing_long_df
    
    # ID 기준 병합 (ID가 있으면 우선)
    result_df = existing_long_df.copy()
    
    for _, row in csv_clean.iterrows():
        csv_id = str(row['ID']).strip() if pd.notna(row['ID']) else ''
        csv_plant = str(row['플랜트']).strip()
        csv_year = int(row['년'])
        csv_month = int(row['월'])
        csv_sales = row['매출수량']
        
        if csv_id:
            # ID 기준 업데이트
            mask = (result_df['ID'] == csv_id) & (result_df['년'] == csv_year) & (result_df['월'] == csv_month)
        else:
            # 플랜트 기준 업데이트
            mask = (result_df['플랜트'] == csv_plant) & (result_df['년'] == csv_year) & (result_df['월'] == csv_month)
        
        if mask.any():
            result_df.loc[mask, '매출수량'] = csv_sales
        else:
            # 신규 행 추가
            new_row = {
                'ID': csv_id,
                '플랜트': csv_plant,
                '년': csv_year,
                '월': csv_month,
                '매출수량': csv_sales,
                'is_estimated': False
            }
            result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return result_df.sort_values(['플랜트', '년', '월']).reset_index(drop=True)

# ============================================================================ 
# 세션 상태 초기화
# ============================================================================ 
if 'sales_long_df' not in st.session_state:
    st.session_state.sales_long_df = sync_with_claims()
if 'sales_display_df' not in st.session_state:
    st.session_state.sales_display_df = long_to_pivot(st.session_state.sales_long_df)

# ============================================================================ 
# UI 영역
# ============================================================================ 
st.info(
    "🔄 **Smart Sync 활성화**: 클레임 데이터의 [플랜트, 년, 월] 조합을 자동으로 테이블에 반영합니다.\n\n"
    "- **행(플랜트)**: 전체 데이터 허브(Hub)의 플랜트 unique 값\n"
    "- **열(년-월)**: 클레임 데이터의 년월 조합 (자동 정렬)",
    icon="ℹ️"
)

# CSV 업로드 섹션
with st.expander("📥 CSV 데이터 일괄 업로드", expanded=False):
    st.markdown("**동일 헤더의 CSV 파일**을 업로드하면 ID 또는 플랜트명으로 기존 데이터를 자동 업데이트합니다.\n\n"
                "**CSV 컬럼**: ID, 플랜트, 년, 월, 매출수량")
    
    uploaded_file = st.file_uploader("CSV 파일 선택", type=['csv'], key="csv_uploader")
    
    if uploaded_file is not None:
        try:
            csv_data = pd.read_csv(uploaded_file, encoding='utf-8')
            st.markdown(f"**미리보기** ({len(csv_data)}행)")
            st.dataframe(csv_data.head(10), use_container_width=True)
            
            if st.button("✅ CSV 데이터 병합", use_container_width=True):
                st.session_state.sales_long_df = merge_csv_data(
                    st.session_state.sales_long_df, 
                    csv_data
                )
                st.session_state.sales_display_df = long_to_pivot(st.session_state.sales_long_df)
                st.success(f"✅ CSV 데이터 병합 완료! ({len(csv_data)}행 처리됨)")
                st.rerun()
        except Exception as e:
            st.error(f"❌ CSV 읽기 실패: {str(e)}")

st.subheader("📊 매출수량 피벗 테이블 (엑셀 스타일)")

if st.session_state.sales_display_df.empty:
    st.warning("⚠️ 클레임 데이터가 없습니다. **[데이터 업로드]** 메뉴에서 먼저 데이터를 허브에 빌드하세요.")
    st.stop()

st.markdown("아래 테이블에서 **ID**와 **매출수량**을 직접 입력/수정할 수 있습니다.")

# 피벗 테이블 에디터 (ID 컬럼 추가)
edited_df = st.data_editor(
    st.session_state.sales_display_df,
    use_container_width=True,
    height=400,
    disabled=['플랜트'],  # 플랜트명은 수정 불가
    num_rows="fixed",
    key="pivot_editor"
)

# 변경사항 자동 반영
if edited_df is not None and not edited_df.equals(st.session_state.sales_display_df):
    st.session_state.sales_display_df = edited_df
    st.session_state.sales_long_df = pivot_to_long(edited_df)

# 저장 버튼 및 통계
st.subheader("💾 저장 및 통계")
col_stats1, col_stats2, col_stats3 = st.columns(3)

with col_stats1:
    st.metric("플랜트 수", len(st.session_state.sales_display_df))
with col_stats2:
    period_count = len([c for c in st.session_state.sales_display_df.columns if c not in ['ID', '플랜트']])
    st.metric("년-월 기간 수", period_count)
with col_stats3:
    total_sales = st.session_state.sales_display_df.drop(columns=['ID', '플랜트']).sum().sum()
    st.metric("총 매출수량", f"{int(total_sales):,}")

if st.button("💾 저장", key="save_sales", use_container_width=True):
    if not st.session_state.sales_long_df.empty:
        save_sales_data(st.session_state.sales_long_df)
    else:
        st.error("❌ 저장할 데이터가 없습니다.")

# 데이터 미리보기
if not st.session_state.sales_long_df.empty:
    with st.expander("📋 Long 형식 데이터 미리보기 (저장 형식)", expanded=False):
        st.dataframe(st.session_state.sales_long_df.head(50), use_container_width=True)
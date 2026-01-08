# ============================================================================
# ETL 모듈: 데이터 추출, 변환, 검증
# ============================================================================
# 설명: 입력 파일(CSV, Excel 등)을 로드하여 54개 필드만 추출하고,
#      데이터 품질을 보증하는 모듈입니다.

import pandas as pd
from pathlib import Path
from typing import Optional, Union
from core.config import TARGET_54_COLS, DEFAULT_ENCODING


def load_raw_file(
    file_path: Union[str, Path],
    encoding: str = DEFAULT_ENCODING,
    sheet_name: Union[int, str] = 0
) -> pd.DataFrame:
    """
    원본 파일 로드 (CSV, Excel 지원).
    
    Args:
        file_path: 입력 파일 경로 (.csv, .xlsx, .xls)
        encoding: 파일 인코딩 (기본값: utf-8-sig)
        sheet_name: Excel 시트명 (기본값: 첫 번째 시트)
    
    Returns:
        pd.DataFrame: 로드된 데이터프레임
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없음: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == ".csv":
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {suffix}")
        
        return df
    
    except Exception as e:
        raise RuntimeError(f"파일 로드 실패: {str(e)}")


def extract_54_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    입력 데이터프레임에서 54개 핵심 필드만 추출.
    
    동작:
        - 존재하는 컬럼: 그대로 유지
        - 없는 컬럼: NaN으로 생성
        - 추가 컬럼: 삭제 (54개만 유지)
    
    Args:
        df: 원본 데이터프레임 (컬럼 개수 무관)
    
    Returns:
        pd.DataFrame: 정확히 54개 필드를 갖는 데이터프레임 (1행 = 1건)
    """
    if df.empty:
        raise ValueError("입력 데이터프레임이 비어있음")
    
    # 54개 필드 기준으로 reindex (없는 필드는 NaN)
    df_filtered = df.reindex(columns=TARGET_54_COLS)
    
    return df_filtered


def validate_data_quality(df: pd.DataFrame) -> dict:
    """
    데이터 품질 검증 (반환: 검증 결과 딕셔너리).
    
    Args:
        df: 검증할 데이터프레임
    
    Returns:
        dict: {
            'total_rows': int,
            'expected_cols': int,
            'actual_cols': int,
            'cols_ok': bool,
            'has_duplicates': bool,
            'duplicate_count': int,
            'null_counts': dict (컬럼별 NaN 개수)
        }
    """
    result = {
        'total_rows': len(df),
        'expected_cols': len(TARGET_54_COLS),
        'actual_cols': len(df.columns),
        'cols_ok': len(df.columns) == len(TARGET_54_COLS),
        'has_duplicates': df.duplicated().any(),
        'duplicate_count': df.duplicated().sum(),
        'null_counts': df.isnull().sum().to_dict(),
    }
    
    return result


def process_claim_data(
    file_path: Union[str, Path],
    encoding: str = DEFAULT_ENCODING,
    sheet_name: Union[int, str] = 0
) -> pd.DataFrame:
    """
    클레임 데이터 엔드-투-엔드 처리.
    
    단계:
        1. 파일 로드 및 중복 제거
        2. 54개 필드 추출
        3. 데이터 품질 검증
    
    Args:
        file_path: 입력 파일 경로
        encoding: 인코딩
        sheet_name: Excel 시트명
    
    Returns:
        pd.DataFrame: 처리된 데이터프레임
    
    Raises:
        FileNotFoundError: 파일 없음
        ValueError: 데이터 검증 실패
    """
    print(f"[ETL] 파일 로드 시작: {file_path}")
    df = load_raw_file(file_path, encoding=encoding, sheet_name=sheet_name)
    print(f"[ETL] 로드 완료 (중복 제거 전): {len(df)} 행, {len(df.columns)} 컬럼")

    # 중복 제거
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[ETL] 중복 제거 후: {len(df)} 행")
    
    print(f"[ETL] 54개 필드 추출 시작...")
    df = extract_54_fields(df)
    print(f"[ETL] 추출 완료: {len(df)} 행, {len(df.columns)} 컬럼")
    
    print(f"[ETL] 데이터 품질 검증...")
    quality = validate_data_quality(df)
    print(f"[ETL] 검증 결과: {quality['total_rows']} 행, 잔여 중복: {quality['duplicate_count']}개")
    
    return df


def safe_date_parse(series: pd.Series) -> pd.Series:
    """
    여러 날짜 포맷(YYYY/MM/DD, YYYY-MM-DD, YYYY.MM.DD)을 순차적으로 파싱.

    Args:
        series: 날짜 문자열을 포함하는 pandas Series

    Returns:
        pd.Series: datetime 객체로 변환된 Series (파싱 실패 시 NaT)
    """
    # Series가 비어있거나 모두 NaT이면 원본을 그대로 반환
    if series.empty or series.isna().all():
        return pd.to_datetime(series, errors='coerce')
        
    # Priority 1: YYYY/MM/DD
    parsed = pd.to_datetime(series, format='%Y/%m/%d', errors='coerce')
    
    # Priority 2: YYYY-MM-DD for remaining NaNs
    remaining_nat_mask = parsed.isna()
    if remaining_nat_mask.any():
        # .loc를 사용하여 복사본 경고 방지
        parsed.loc[remaining_nat_mask] = pd.to_datetime(series[remaining_nat_mask], format='%Y-%m-%d', errors='coerce')

    # Priority 3: YYYY.MM.DD for remaining NaNs
    remaining_nat_mask = parsed.isna()
    if remaining_nat_mask.any():
        # .loc를 사용하여 복사본 경고 방지
        parsed.loc[remaining_nat_mask] = pd.to_datetime(series[remaining_nat_mask], format='%Y.%m.%d', errors='coerce')
        
    return parsed


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터프레임 전처리 함수 (Phase 1.5: Dual-Format Date Parsing 적용)

    Args:
        df: 입력 데이터프레임

    Returns:
        pd.DataFrame: 전처리 완료된 데이터프레임
    """
    print("[ETL] 데이터 전처리 시작...")
    if df is None or df.empty:
        raise ValueError("전처리할 데이터프레임이 비어있습니다.")

    df = df.copy()

    # Step 1 (필수): '상담번호'가 NaN인 행 삭제
    initial_rows = len(df)
    df.dropna(subset=['상담번호'], inplace=True)
    dropped_rows = initial_rows - len(df)
    print(f"[ETL-STEP1] '상담번호' NaN 제거: {dropped_rows}개 행 삭제됨. 현재 {len(df)}개 행.")

    # Step 1-1: 상담번호 기준 중복 제거 (가장 마지막 업로드 우선)
    # 업로드 순서 그대로 두고 keep='last'로 동일 상담번호는 최신 행만 유지
    before_dedup = len(df)
    df['상담번호'] = df['상담번호'].astype(str).str.strip()
    df = df.drop_duplicates(subset=['상담번호'], keep='last')
    after_dedup = len(df)
    print(f"[ETL-STEP1-1] 상담번호 중복 제거: {before_dedup - after_dedup}개 행 제거. 현재 {after_dedup}개 행.")

    # Step 2 (접수일): '접수년/월/일' 컬럼을 합쳐 '접수일자' 생성 (기존 로직 유지)
    date_cols = ['접수년', '접수월', '접수일']
    for col in date_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df['접수년_str'] = pd.to_numeric(df['접수년'], errors='coerce').astype('Int64').astype(str).replace('<NA>', '')
    df['접수월_str'] = pd.to_numeric(df['접수월'], errors='coerce').astype('Int64').astype(str).replace('<NA>', '').str.zfill(2)
    df['접수일_str'] = pd.to_numeric(df['접수일'], errors='coerce').astype('Int64').astype(str).replace('<NA>', '').str.zfill(2)

    # 모든 파트가 존재할 때만 결합
    valid_reception_mask = (df['접수년_str'] != '') & (df['접수월_str'] != '') & (df['접수일_str'] != '')
    df.loc[valid_reception_mask, '접수일자'] = df.loc[valid_reception_mask, '접수년_str'] + '-' + \
                                          df.loc[valid_reception_mask, '접수월_str'] + '-' + \
                                          df.loc[valid_reception_mask, '접수일_str']
    
    df['접수일자'] = pd.to_datetime(df['접수일자'], format='%Y-%m-%d', errors='coerce')
    df.drop(columns=['접수년_str', '접수월_str', '접수일_str'], inplace=True)
    
    print(f"[ETL-STEP2] '접수일자' 생성 완료. NaT 개수: {df['접수일자'].isna().sum()}개.")
    if df['접수일자'].notna().any():
        # NaT가 아닌 첫 번째 샘플을 출력
        first_valid_date = df['접수일자'].dropna().iloc[0]
        print(f"  -> 생성된 '접수일자' 샘플: {first_valid_date}")

    # Step 3 (제조/유통): safe_date_parse 함수를 사용하여 '제조일자'와 '유통기한' 파싱
    for col in ['제조일자', '유통기한']:
        if col in df.columns:
            # 파싱 전 유효한 (non-NaN) 문자열 데이터 건수 확인
            # astype(str).str.match('.*\\d.*') 와 같은 방법으로 숫자 포함 여부 확인 가능
            pre_parse_valid_count = df[col].astype(str).str.contains(r'\d').sum()
            print(f"[ETL-STEP3] '{col}' 파싱 시작... (초기 유효 문자열 데이터: {pre_parse_valid_count}건)")
            
            # 데이터 타입을 문자열로 변환하여 일관성 확보
            df[col] = safe_date_parse(df[col].astype(str))
            
            # 파싱 후 유효한 날짜 데이터 건수 확인
            post_parse_valid_count = df[col].notna().sum()
            improvement = post_parse_valid_count - 0  # 초기 유효 날짜는 0으로 가정
            
            print(f"  -> '{col}' 파싱 완료. 최종 유효 날짜: {post_parse_valid_count}건")
            print(f"  -> NaT 개수: {df[col].isna().sum()}개.")

        else:
            df[col] = pd.NaT
            print(f"[ETL-STEP3] '{col}' 컬럼이 없어 NaT로 생성.")

    # Step 4 (Lag): Lag_Days 및 Lag_Valid 계산 (기존 로직 유지)
    df['Lag_Days'] = (df['접수일자'] - df['제조일자']).dt.days
    
    df['Lag_Valid'] = (
        df['접수일자'].notna() &
        df['제조일자'].notna() &
        (df['Lag_Days'] >= 0)
    )
    print(f"[ETL-STEP4] 'Lag_Days', 'Lag_Valid' 계산 완료. 유효 Lag: {df['Lag_Valid'].sum()}개.")
    
    # '접수년', '접수월' 컬럼 보강 (파티셔닝을 위해 필요)
    # df에 '접수년'과 '접수월'이 이미 있으므로 숫자 타입으로 변환 시도
    df['접수년'] = pd.to_numeric(df['접수년'], errors='coerce')
    df['접수월'] = pd.to_numeric(df['접수월'], errors='coerce')

    # 만약 '접수년'/'접수월'이 모두 NaN이면 (원본 파일에 없었던 경우), '접수일자'에서 파생
    if '접수년' in df and df['접수년'].isna().all():
        df['접수년'] = df['접수일자'].dt.year.astype('Int64')
    if '접수월' in df and df['접수월'].isna().all():
        df['접수월'] = df['접수일자'].dt.month.astype('Int64')

    print("[ETL] 데이터 전처리 완료.")
    return df

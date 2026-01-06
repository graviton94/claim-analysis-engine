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
            df = pd.read_csv(file_path, encoding=encoding)
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
        1. 파일 로드
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
    print(f"[ETL] 로드 완료: {len(df)} 행, {len(df.columns)} 컬럼")
    
    print(f"[ETL] 54개 필드 추출 시작...")
    df = extract_54_fields(df)
    print(f"[ETL] 추출 완료: {len(df)} 행, {len(df.columns)} 컬럼")
    
    print(f"[ETL] 데이터 품질 검증...")
    quality = validate_data_quality(df)
    print(f"[ETL] 검증 결과: {quality['total_rows']} 행, 중복: {quality['duplicate_count']}개")
    
    return df

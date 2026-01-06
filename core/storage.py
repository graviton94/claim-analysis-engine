# ============================================================================
# 저장소 모듈: Parquet 파티셔닝 입출력
# ============================================================================
# 설명: 클레임 데이터를 접수년/접수월 기준으로 파티셔닝하여 저장하고,
#      특정 기간의 데이터를 효율적으로 로드합니다.

import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
import pyarrow.parquet as pq
from core.config import DATA_HUB_PATH, PARTITION_COLS


def save_partitioned(
    df: pd.DataFrame,
    output_path: Union[str, Path] = DATA_HUB_PATH,
    partition_cols: List[str] = PARTITION_COLS
) -> None:
    """
    데이터프레임을 Parquet 형식으로 파티셔닝하여 저장.
    
    동작:
        - partition_cols (기본: ['접수년', '접수월'])을 기준으로 물리 폴더 생성
        - 각 폴더에 part-0.parquet 형식으로 저장
        - 구조: data/hub/접수년=YYYY/접수월=MM/part-0.parquet
    
    Args:
        df: 저장할 데이터프레임
        output_path: 저장 경로 (기본값: 'data/hub')
        partition_cols: 파티셔닝 기준 컬럼 리스트 (기본값: ['접수년', '접수월'])
    
    Raises:
        ValueError: 필수 파티셔닝 컬럼 없음
    """
    output_path = Path(output_path)
    
    # 필수 컬럼 검증
    missing_cols = [col for col in partition_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"파티셔닝 컬럼 없음: {missing_cols}")
    
    # 파티셔닝 컬럼의 데이터 타입 변환 (정수화)
    df_copy = df.copy()
    for col in partition_cols:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0).astype(int)
    
    # Parquet 파티셔닝 저장
    print(f"[STORAGE] 파티셔닝 저장 시작: {output_path}")
    print(f"[STORAGE] 파티셔닝 컬럼: {partition_cols}")
    
    try:
        df_copy.to_parquet(
            output_path,
            partition_cols=partition_cols,
            engine='pyarrow'
        )
        print(f"[STORAGE] 저장 완료: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Parquet 저장 실패: {str(e)}")


def load_partitioned(
    path: Union[str, Path] = DATA_HUB_PATH,
    year: Optional[int] = None,
    month: Optional[int] = None
) -> pd.DataFrame:
    """
    Parquet 파티셔닝 데이터 로드.
    
    동작:
        - 특정 연/월 데이터 필터링 로드 (성능 최적화)
        - 필터 미지정 시 전체 데이터 로드
    
    Args:
        path: 저장 경로 (기본값: 'data/hub')
        year: 조회 연도 (None이면 모든 연도 조회)
        month: 조회 월 (None이면 모든 월 조회)
    
    Returns:
        pd.DataFrame: 로드된 데이터프레임
    
    Raises:
        FileNotFoundError: 저장된 데이터 없음
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"저장된 데이터 경로 없음: {path}")
    
    # Parquet 필터 구성
    filters = []
    if year is not None:
        filters.append(('접수년', '==', year))
    if month is not None:
        filters.append(('접수월', '==', month))
    
    print(f"[STORAGE] 파티셔닝 로드 시작: {path}")
    if filters:
        print(f"[STORAGE] 필터: year={year}, month={month}")
    
    try:
        if filters:
            # PyArrow 필터 조건 구성
            df = pd.read_parquet(path, filters=[filters] if len(filters) > 1 else filters)
        else:
            df = pd.read_parquet(path)
        
        print(f"[STORAGE] 로드 완료: {len(df)} 행")
        return df
    
    except Exception as e:
        raise RuntimeError(f"Parquet 로드 실패: {str(e)}")


def get_available_periods(
    path: Union[str, Path] = DATA_HUB_PATH
) -> pd.DataFrame:
    """
    저장된 파티셔닝 데이터의 사용 가능한 연/월 목록 반환.
    
    Args:
        path: 저장 경로
    
    Returns:
        pd.DataFrame: {접수년, 접수월} 컬럼의 고유 조합
    """
    path = Path(path)
    
    if not path.exists() or not list(path.glob('접수년=*')):
        return pd.DataFrame(columns=['접수년', '접수월'])
    
    try:
        # 모든 파티션 메타데이터에서 연/월 추출
        df = pd.read_parquet(path)
        periods = df[['접수년', '접수월']].drop_duplicates().sort_values(['접수년', '접수월'])
        return periods
    
    except Exception as e:
        print(f"[WARNING] 사용 가능한 기간 조회 실패: {str(e)}")
        return pd.DataFrame(columns=['접수년', '접수월'])


def clear_partitioned_data(
    path: Union[str, Path] = DATA_HUB_PATH,
    confirm: bool = False
) -> None:
    """
    저장된 파티셔닝 데이터 초기화 (테스트/개발용).
    
    Args:
        path: 저장 경로
        confirm: 삭제 확인 플래그 (True여야 실제 삭제)
    """
    path = Path(path)
    
    if not confirm:
        raise ValueError("confirm=True 필수 (실수 방지)")
    
    if path.exists():
        import shutil
        shutil.rmtree(path)
        print(f"[STORAGE] 데이터 초기화 완료: {path}")

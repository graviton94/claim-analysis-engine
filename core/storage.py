# ============================================================================
# 저장소 모듈: Parquet 파티셔닝 입출력
# ============================================================================
# 설명: 클레임 데이터를 접수년/접수월 기준으로 파티셔닝하여 저장하고,
#      특정 기간의 데이터를 효율적으로 로드합니다.

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import json
import re
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from core.config import DATA_HUB_PATH, PARTITION_COLS, DATA_SERIES_PATH


def save_partitioned(
    df: pd.DataFrame,
    output_path: Union[str, Path] = DATA_HUB_PATH,
    partition_cols: List[str] = PARTITION_COLS
) -> None:
    """
    데이터프레임을 Parquet 형식으로 파티셔닝하여 저장.
    
    동작:
        - partition_cols (기본: ['접수년', '접수월'])을 기준으로 물리 폴더 생성
        - 각 폴더에 data.parquet 형식으로 저장
        - 구조: data/hub/YYYY/MM/data.parquet
    
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
        table = pa.Table.from_pandas(df_copy, preserve_index=False)
        
        partitioning_schema = pa.schema([
            pa.field(col, table.schema.field(col).type) for col in partition_cols
        ])
        
        ds.write_dataset(
            table,
            output_path,
            partitioning=ds.DirectoryPartitioning(partitioning_schema),
            format='parquet',
            existing_data_behavior='overwrite_or_ignore'
        )
        print(f"[STORAGE] 저장 완료: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Parquet 저장 실패: {str(e)}")


def save_partitioned_parquet(
    df: pd.DataFrame,
    output_path: Union[str, Path] = DATA_HUB_PATH,
    partition_cols: List[str] = PARTITION_COLS
) -> None:
    """
    요구사항 준수 버전: Lag_Days, Lag_Valid 포함하여 파티셔닝 저장.

    기존 save_partitioned 로직을 활용하되, 입력 df에 생성된 지표를 그대로 포함하여 저장합니다.
    """
    # 단순 위임 (df에 포함된 모든 컬럼 저장됨)
    save_partitioned(df, output_path=output_path, partition_cols=partition_cols)


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
        # DirectoryPartitioning 스키마 정의
        # save_partitioned에서 정수형으로 변환했으므로 int64() 사용
        partitioning = ds.DirectoryPartitioning(pa.schema([
            pa.field('접수년', pa.int64()),
            pa.field('접수월', pa.int64())
        ]))
        
        dataset = ds.dataset(path, partitioning=partitioning, format="parquet")

        # 필터 구성
        filter_expr = None
        if year is not None:
            filter_expr = (ds.field('접수년') == year)
        if month is not None:
            month_expr = (ds.field('접수월') == month)
            filter_expr = filter_expr & month_expr if filter_expr is not None else month_expr
        
        df = dataset.to_table(filter=filter_expr).to_pandas()
        
        print(f"[STORAGE] 로드 완료: {len(df)} 행")
        return df
    
    except Exception as e:
        if "No files found" in str(e) or "Path does not exist" in str(e):
             print(f"[STORAGE] 데이터 경로에 파일 없음: {path}")
             return pd.DataFrame()
        raise RuntimeError(f"Parquet 로드 실패: {str(e)}")


def get_available_periods(
    path: Union[str, Path] = DATA_HUB_PATH
) -> pd.DataFrame:
    """
    저장된 파티셔닝 데이터의 사용 가능한 연/월 목록과 각 기간의 데이터 건수를 반환.

    디렉토리 구조 지원:
    - 접수년=YYYY/접수월=MM (pyarrow DirectoryPartitioning 기본)
    - YYYY/MM (레거시/수동 저장)
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=['접수년', '접수월', '건수'])

    periods = []
    try:
        # 1) 우선순위: pyarrow 디렉토리 파티셔닝 형태 (접수년=YYYY)
        year_dirs = [p for p in path.iterdir() if p.is_dir() and (p.name.isdigit() or p.name.startswith('접수년='))]
        for year_dir in year_dirs:
            # 연도 파싱
            if year_dir.name.startswith('접수년='):
                try:
                    year = int(year_dir.name.split('=', 1)[1])
                except ValueError:
                    continue
            else:
                try:
                    year = int(year_dir.name)
                except ValueError:
                    continue

            # 월 디렉토리 탐색 (접수월=MM 또는 숫자)
            month_dirs = [p for p in year_dir.iterdir() if p.is_dir() and (p.name.isdigit() or p.name.startswith('접수월='))]
            for month_dir in month_dirs:
                if month_dir.name.startswith('접수월='):
                    try:
                        month = int(month_dir.name.split('=', 1)[1])
                    except ValueError:
                        continue
                else:
                    try:
                        month = int(month_dir.name)
                    except ValueError:
                        continue

                # 해당 파티션의 총 행 수 계산
                total_rows = 0
                try:
                    parquet_files = list(month_dir.glob('*.parquet'))
                    if not parquet_files:
                        # Parquet 파일이 없으면 건너뜀 (빈 디렉토리)
                        continue

                    for pq_file in parquet_files:
                        meta = pq.read_metadata(pq_file)
                        total_rows += meta.num_rows
                except Exception as e:
                    print(f"[WARNING] Parquet 메타데이터 읽기 실패 ({month_dir}): {e}")
                    total_rows = 0

                periods.append({'접수년': year, '접수월': month, '건수': total_rows})

        if not periods:
            return pd.DataFrame(columns=['접수년', '접수월', '건수'])

        return pd.DataFrame(periods).sort_values(['접수년', '접수월'], ascending=[False, False]).reset_index(drop=True)

    except Exception as e:
        print(f"[WARNING] 사용 가능한 기간 조회 실패: {str(e)}")
        return pd.DataFrame(columns=['접수년', '접수월', '건수'])


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


# ============================================================================
# Phase 2: 고도화 함수
# ============================================================================

def get_claim_keys(path: Union[str, Path] = DATA_HUB_PATH) -> pd.DataFrame:
    """
    클레임 데이터의 [플랜트, 접수년, 접수월] 유니크 조합 추출.
    
    동작:
        - data/hub/ 파티셔닝 폴더 스캔
        - 파티션 메타데이터에서 연/월 정보 추출
        - None/NaN 값 제외 후 타입 안전성 확보
        - 각 행마다 플랜트 컬럼 추가
    
    Args:
        path: 저장 경로 (기본값: 'data/hub')
    
    Returns:
        pd.DataFrame: {플랜트, 접수년, 접수월} 컬럼의 유니크 조합
    
    주의:
        - None/NaN 값은 dropna()로 자동 제외
        - 모든 컬럼을 str로 형변환 후 sorted() 수행 (Type Safety)
    """
    path = Path(path)
    
    if not path.exists() or not list(path.glob('접수년=*')):
        return pd.DataFrame(columns=['플랜트', '접수년', '접수월'])
    
    try:
        # 전체 클레임 데이터 로드
        df = pd.read_parquet(path)
        
        # [플랜트, 접수년, 접수월] 유니크 조합 추출
        # ★ Step 1: None/NaN 값 제외 (dropna)
        claim_keys = df[['플랜트', '접수년', '접수월']].dropna()
        
        # ★ Step 2: 모든 컬럼을 str로 형변환하여 Type Safety 확보
        claim_keys['플랜트'] = claim_keys['플랜트'].astype(str)
        claim_keys['접수년'] = claim_keys['접수년'].astype(str)
        claim_keys['접수월'] = claim_keys['접수월'].astype(str)
        
        # ★ Step 3: 유니크 조합 추출 및 정렬 (이제 모든 값이 str이므로 TypeError 없음)
        claim_keys = claim_keys.drop_duplicates()
        
        # str 정렬 수행 (타입 호환성 완벽)
        claim_keys = claim_keys.sort_values(
            ['플랜트', '접수년', '접수월'],
            key=lambda x: x.astype(str)
        ).reset_index(drop=True)
        
        print(f"[STORAGE] 클레임 키 추출 완료: {len(claim_keys)} 행 (None/NaN 제외 완료)")
        return claim_keys
    
    except Exception as e:
        print(f"[WARNING] 클레임 키 추출 실패: {str(e)}")
        return pd.DataFrame(columns=['플랜트', '접수년', '접수월'])


def load_sales_with_estimation(
    sales_path: Union[str, Path],
    lookback_months: int = 3
) -> pd.DataFrame:
    """
    매출 데이터 로드 및 스마트 추정 값 채우기.
    
    동작:
        1. 매출 데이터 로드
        2. 값이 없거나 0인 행에 대해 동일 플랜트의 직전 N개월 평균값으로 채우기
        3. is_estimated (Boolean) 컬럼 추가하여 실적/추정 구분
    
    Args:
        sales_path: 매출 데이터 저장 경로
        lookback_months: 평균 계산 시 참고할 과거 개월 수 (기본값: 3)
    
    Returns:
        pd.DataFrame: {플랜트, 년, 월, 매출수량, is_estimated} 스키마
    """
    sales_path = Path(sales_path)
    
    # 매출 데이터 로드
    if not sales_path.exists():
        print("[INFO] 저장된 매출 데이터 없음")
        return pd.DataFrame(columns=['플랜트', '년', '월', '매출수량', 'is_estimated'])
    
    try:
        df = pd.read_parquet(sales_path)
    except Exception as e:
        print(f"[WARNING] 매출 데이터 로드 실패: {str(e)}")
        return pd.DataFrame(columns=['플랜트', '년', '월', '매출수량', 'is_estimated'])
    
    # 기본 전처리
    df = df.copy()
    df['년'] = pd.to_numeric(df['년'], errors='coerce').astype('Int64')
    df['월'] = pd.to_numeric(df['월'], errors='coerce').astype('Int64')
    df['매출수량'] = pd.to_numeric(df['매출수량'], errors='coerce')
    
    # is_estimated 컬럼 초기화 (실적 = False)
    df['is_estimated'] = False
    
    # 플랜트별로 순차 처리
    # ★ None/NaN 플랜트 제외
    plants = df['플랜트'].dropna().unique()
    
    for plant in plants:
        plant_df = df[df['플랜트'] == plant].copy()
        # ★ 인덱스 정렬 시 타입 안전성: 먼저 형변환 후 정렬
        plant_df['년'] = pd.to_numeric(plant_df['년'], errors='coerce').fillna(0).astype(int)
        plant_df['월'] = pd.to_numeric(plant_df['월'], errors='coerce').fillna(0).astype(int)
        plant_df = plant_df.sort_values(['년', '월']).reset_index(drop=True)
        
        # NaN 또는 0인 행 찾기
        missing_mask = (plant_df['매출수량'].isna()) | (plant_df['매출수량'] == 0)
        
        for idx in plant_df[missing_mask].index:
            current_year = plant_df.loc[idx, '년']
            current_month = plant_df.loc[idx, '월']
            
            # 직전 3개월 평균값 계산
            lookback_values = []
            for back_month in range(1, lookback_months + 1):
                # 과거 달 계산 (월 순환)
                past_year = current_year
                past_month = current_month - back_month
                
                if past_month <= 0:
                    past_year -= 1
                    past_month += 12
                
                # 과거 달 데이터 조회
                past_data = plant_df[
                    (plant_df['년'] == past_year) & (plant_df['월'] == past_month)
                ]
                if not past_data.empty and not pd.isna(past_data['매출수량'].iloc[0]):
                    lookback_values.append(past_data['매출수량'].iloc[0])
            
            # 평균값 계산 및 적용
            if lookback_values:
                avg_value = sum(lookback_values) / len(lookback_values)
                df.loc[
                    (df['플랜트'] == plant) & 
                    (df['년'] == current_year) & 
                    (df['월'] == current_month),
                    '매출수량'
                ] = avg_value
                df.loc[
                    (df['플랜트'] == plant) & 
                    (df['년'] == current_year) & 
                    (df['월'] == current_month),
                    'is_estimated'
                ] = True
    
    print(f"[STORAGE] 매출 데이터 추정치 채우기 완료: {df['is_estimated'].sum()} 행")
    return df.sort_values(['플랜트', '년', '월']).reset_index(drop=True)


# ============================================================================
# Phase 1: Nested Series JSON 생성 (Critical)
# ============================================================================

def _sanitize_filename(name: str) -> str:
    """파일 이름으로 사용될 수 없는 특수문자를 '-'로 치환."""
    # Windows 및 Unix/Linux에서 일반적으로 허용되지 않는 문자들
    # '/', '\', ':', '*', '?', '"', '<', '>', '|'
    return re.sub(r'[\\/:\*\?"<>\|]', '-', name)


def _month_range(df: pd.DataFrame, date_col: str) -> List[str]:
    """데이터셋 전체의 Min~Max 월 범위 생성 (YYYY-MM 문자열 리스트)."""
    if date_col not in df.columns:
        return []
    dates = pd.to_datetime(df[date_col], errors='coerce')
    dates = dates.dropna()
    if dates.empty:
        return []
    start = dates.min().to_period('M').to_timestamp()
    end = dates.max().to_period('M').to_timestamp()
    months = pd.date_range(start=start, end=end, freq='MS')
    return [d.strftime('%Y-%m') for d in months]


def _compute_series_stats(values: List[int]) -> Dict[str, float]:
    """mean, std, slope(최근 3개월 선형회귀 기울기) 계산."""
    arr = np.array(values, dtype=float)
    mean = float(np.nanmean(arr)) if arr.size else 0.0
    std = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
    # 최근 3개월 기울기
    if arr.size >= 3:
        y = arr[-3:]
        x = np.arange(1, len(y) + 1, dtype=float)
        try:
            slope = float(np.polyfit(x, y, 1)[0])
        except Exception:
            slope = 0.0
    else:
        slope = 0.0
    return {"mean": mean, "std": std, "slope": slope}


def generate_nested_series(
    df: pd.DataFrame,
    output_dir: Union[str, Path] = DATA_SERIES_PATH,
    date_col: str = '접수일자'
) -> int:
    """
    Nested Series JSON 생성.

    Grouping Key: [플랜트, 제품범주2, 대분류]
    Zero-filling: 전체 Min~Max 월 범위 기준으로 월별 카운트 0채우기 (Parent & Children)

    JSON Schema:
        key: "{Plant}_{Cat2}_{Major}"
        meta: last_updated, warning_level(0), champion_model(null), parent_stats(mean,std,slope)
        data.history: 월별 count 리스트 (zero-filled)
        data.forecast: 빈 리스트
        children: 각 중분류별 서브 시리즈 (stats + history)

    Returns:
        생성된 JSON 파일 개수
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 전체 월 범위
    all_months = _month_range(df, date_col)
    if not all_months:
        print("[STORAGE] 유효한 월 범위를 산출할 수 없음 (접수일자 비어있음)")
        return 0

    # 월 키 생성 함수
    def _month_key(ts: pd.Timestamp) -> str:
        return pd.to_datetime(ts).strftime('%Y-%m')

    # 유효 데이터 마스크 (Lag_Valid=True)
    valid_mask = df['Lag_Valid'] if 'Lag_Valid' in df.columns else pd.Series([True] * len(df))

    # 월별 컬럼 파생 (date_col 기준)
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['__month'] = df[date_col].dt.to_period('M').dt.to_timestamp()

    # 그룹핑
    required_cols = ['플랜트', '제품범주2', '대분류', '중분류']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    grouped = df.groupby(['플랜트', '제품범주2', '대분류'], dropna=False)

    created = 0
    today_str = pd.Timestamp.today().strftime('%Y-%m-%d')

    for (plant, cat2, major), gdf in grouped:
        # Parent 월별 카운트 (전체) 및 유효 데이터 카운트 (통계용)
        parent_counts_all = gdf.groupby('__month').size()
        parent_counts_valid = gdf[valid_mask.loc[gdf.index]].groupby('__month').size()

        # Zero-fill
        parent_history = []
        parent_series_all = {pd.to_datetime(k).strftime('%Y-%m'): int(v) for k, v in parent_counts_all.items()}
        parent_series_valid = {pd.to_datetime(k).strftime('%Y-%m'): int(v) for k, v in parent_counts_valid.items()}

        parent_values_for_stats = []
        for m in all_months:
            count = parent_series_all.get(m, 0)
            parent_history.append({"date": m, "count": int(count)})
            parent_values_for_stats.append(parent_series_valid.get(m, 0))

        parent_stats = _compute_series_stats(parent_values_for_stats)

        # Lag 통계 추가
        if 'Lag_Valid' in gdf.columns and 'Lag_Days' in gdf.columns:
            valid_lags = gdf.loc[gdf['Lag_Valid'] == True, 'Lag_Days']
            avg_lag = valid_lags.mean() if not valid_lags.empty else 0.0
            parent_stats['avg_lag_days'] = round(float(np.nan_to_num(avg_lag)), 1)
        else:
            parent_stats['avg_lag_days'] = 0.0

        # Children (중분류)
        children = []
        child_groups = gdf.groupby('중분류', dropna=False)
        for middle, cgdf in child_groups:
            child_counts_all = cgdf.groupby('__month').size()
            child_counts_valid = cgdf[valid_mask.loc[cgdf.index]].groupby('__month').size()

            child_series_all = {pd.to_datetime(k).strftime('%Y-%m'): int(v) for k, v in child_counts_all.items()}
            child_series_valid = {pd.to_datetime(k).strftime('%Y-%m'): int(v) for k, v in child_counts_valid.items()}

            child_history = []
            child_values_for_stats = []
            for m in all_months:
                cnt = child_series_all.get(m, 0)
                child_history.append({"date": m, "count": int(cnt)})
                child_values_for_stats.append(child_series_valid.get(m, 0))

            child_stats = _compute_series_stats(child_values_for_stats)

            # Lag 통계 추가
            if 'Lag_Valid' in cgdf.columns and 'Lag_Days' in cgdf.columns:
                valid_lags = cgdf.loc[cgdf['Lag_Valid'] == True, 'Lag_Days']
                avg_lag = valid_lags.mean() if not valid_lags.empty else 0.0
                child_stats['avg_lag_days'] = round(float(np.nan_to_num(avg_lag)), 1)
            else:
                child_stats['avg_lag_days'] = 0.0

            children.append({
                "sub_key": str(middle) if middle is not None else "",
                "stats": child_stats,
                "history": child_history
            })

        # 파일명 생성을 위해 각 키 구성요소를 개별적으로 정제
        s_plant = _sanitize_filename(str(plant) if plant is not None else "")
        s_cat2 = _sanitize_filename(str(cat2) if cat2 is not None else "")
        s_major = _sanitize_filename(str(major) if major is not None else "")
        
        # 정제된 부분들을 조합하여 최종 파일명 키 생성
        filename_key = f"{s_plant}_{s_cat2}_{s_major}"

        # JSON payload에는 원본 키를 저장
        key = f"{str(plant)}_{str(cat2)}_{str(major)}"
        
        payload: Dict[str, Any] = {
            "key": key,
            "meta": {
                "last_updated": today_str,
                "warning_level": 0,
                "champion_model": None,
                "parent_stats": parent_stats,
            },
            "data": {
                "history": parent_history,
                "forecast": [],
            },
            "children": children,
        }

        # 파일 저장
        filename = (output_path / f"{filename_key}.json")
        try:
            filename.parent.mkdir(parents=True, exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            created += 1
        except Exception as e:
            print(f"[WARNING] 파일 저장 실패 ({filename}): {e}")

    print(f"[STORAGE] Nested Series JSON 생성 완료: {created}개")
    return created

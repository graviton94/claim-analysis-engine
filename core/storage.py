# ============================================================================
# 저장소 모듈: Parquet 파티셔닝 입출력 및 통합 로더 (Full Version)
# ============================================================================
# 설명: 클레임 데이터를 접수년/접수월 기준으로 파티셔닝하여 저장하고,
#      다양한 필터 조건에 따라 데이터를 효율적으로 로드합니다.
#      (Phase 1, 2 기능 포함 + Phase 2.5 통합 로더 추가)

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import json
import re
from datetime import date, datetime

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from core.config import DATA_HUB_PATH, PARTITION_COLS, DATA_SERIES_PATH, TARGET_54_COLS

# [CONFIG] 필터링 상수 (Page 3/4 공통 사용)
TARGET_BUSINESS_UNITS = ['식품', 'B2B식품']
PERFORMANCE_REASONS = ['제조불만', '고객불만족', '구매불만']


# ============================================================================
# Phase 1: 기본 입출력 (Legacy Support)
# ============================================================================

def save_partitioned(
    df: pd.DataFrame,
    output_path: Union[str, Path] = DATA_HUB_PATH,
    partition_cols: List[str] = PARTITION_COLS
) -> None:
    """
    데이터프레임을 Parquet 형식으로 파티셔닝하여 저장.
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
    상담번호 기준 중복 제거 로직 포함.
    """
    output_path = Path(output_path)

    if '상담번호' not in df.columns:
        raise ValueError("상담번호 컬럼이 없어 저장할 수 없습니다. 상담번호는 고유 ID입니다.")

    # 상담번호를 문자열로 정규화해 비교 안정성 확보
    df = df.copy()
    df['상담번호'] = df['상담번호'].astype(str).str.strip()

    # 기존 허브 데이터와 병합 후 상담번호 기준 최신 행 유지 (keep='last')
    existing_df = pd.DataFrame()
    if output_path.exists():
        try:
            existing_df = load_partitioned(path=output_path)
        except Exception as e:
            print(f"[STORAGE] 기존 허브 로드 실패(무시): {e}")

    if not existing_df.empty:
        # 상담번호 존재 보장
        if '상담번호' not in existing_df.columns:
            existing_df['상담번호'] = pd.NA

        # 스키마 정렬: 새 df에 없는 컬럼을 추가하고 순서 정렬
        missing_cols_in_new = [c for c in existing_df.columns if c not in df.columns]
        for col in missing_cols_in_new:
            df[col] = pd.NA

        missing_cols_in_existing = [c for c in df.columns if c not in existing_df.columns]
        for col in missing_cols_in_existing:
            existing_df[col] = pd.NA

        # 동일한 컬럼 순서로 정렬
        df = df[existing_df.columns]

        combined = pd.concat([existing_df, df], ignore_index=True)
        before = len(combined)
        combined['상담번호'] = combined['상담번호'].astype(str).str.strip()
        combined = combined.drop_duplicates(subset=['상담번호'], keep='last')
        after = len(combined)
        print(f"[STORAGE] 상담번호 병합/중복 제거: {before - after}개 중복 제거, 최종 {after}행 저장")
        df_to_save = combined
    else:
        df_to_save = df

    # 단순 위임 (df에 포함된 모든 컬럼 저장됨)
    save_partitioned(df_to_save, output_path=output_path, partition_cols=partition_cols)


def load_partitioned(
    path: Union[str, Path] = DATA_HUB_PATH,
    year: Optional[int] = None,
    month: Optional[int] = None
) -> pd.DataFrame:
    """
    Parquet 파티셔닝 데이터 로드.
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
    """데이터 초기화"""
    path = Path(path)
    
    if not confirm:
        raise ValueError("confirm=True 필수 (실수 방지)")
    
    if path.exists():
        import shutil
        shutil.rmtree(path)
        print(f"[STORAGE] 데이터 초기화 완료: {path}")


# ============================================================================
# Phase 2: 고도화 함수 (Nested Series & Sales Estimation)
# ============================================================================

def get_claim_keys(path: Union[str, Path] = DATA_HUB_PATH) -> pd.DataFrame:
    """
    클레임 데이터의 [플랜트, 접수년, 접수월] 유니크 조합 추출.
    """
    path = Path(path)
    if not path.exists() or not any(path.iterdir()):
        print("[STORAGE] Hub directory is empty or does not exist.")
        return pd.DataFrame(columns=['플랜트', '접수년', '접수월'])

    try:
        partitioning = ds.DirectoryPartitioning(pa.schema([
            pa.field('접수년', pa.int64()),
            pa.field('접수월', pa.int64())
        ]))
        
        dataset = ds.dataset(path, partitioning=partitioning, format="parquet")
        df = dataset.to_table(columns=['플랜트', '접수년', '접수월']).to_pandas()
        
        if df.empty:
            print("[STORAGE] Hub data is empty after loading.")
            return pd.DataFrame(columns=['플랜트', '접수년', '접수월'])

        claim_keys = df[['플랜트', '접수년', '접수월']].drop_duplicates()
        
        claim_keys['플랜트'] = claim_keys['플랜트'].astype(str)
        claim_keys['접수년'] = pd.to_numeric(claim_keys['접수년'], errors='coerce')
        claim_keys['접수월'] = pd.to_numeric(claim_keys['접수월'], errors='coerce')
        claim_keys = claim_keys.dropna()
        
        claim_keys['접수년'] = claim_keys['접수년'].astype(int)
        claim_keys['접수월'] = claim_keys['접수월'].astype(int)

        claim_keys = claim_keys.sort_values(['플랜트', '접수년', '접수월']).reset_index(drop=True)
        
        print(f"[STORAGE] 클레임 키 추출 완료: {len(claim_keys)} 행")
        return claim_keys

    except Exception as e:
        if "No files found" in str(e) or "Path does not exist" in str(e):
             print(f"[STORAGE] 데이터 경로에 파일 없음: {path}")
             return pd.DataFrame(columns=['플랜트', '접수년', '접수월'])
        print(f"[WARNING] 클레임 키 추출 실패: {str(e)}")
        return pd.DataFrame(columns=['플랜트', '접수년', '접수월'])


def load_sales_with_estimation(
    sales_path: Union[str, Path],
    lookback_months: int = 3
) -> pd.DataFrame:
    """
    매출 데이터 로드 및 스마트 추정 값 채우기.
    """
    sales_path = Path(sales_path)
    
    if not sales_path.exists():
        print("[INFO] 저장된 매출 데이터 없음")
        return pd.DataFrame(columns=['플랜트', '년', '월', '매출수량', 'is_estimated'])
    
    try:
        df = pd.read_parquet(sales_path)
    except Exception as e:
        print(f"[WARNING] 매출 데이터 로드 실패: {str(e)}")
        return pd.DataFrame(columns=['플랜트', '년', '월', '매출수량', 'is_estimated'])
    
    df = df.copy()
    df['년'] = pd.to_numeric(df['년'], errors='coerce').astype('Int64')
    df['월'] = pd.to_numeric(df['월'], errors='coerce').astype('Int64')
    df['매출수량'] = pd.to_numeric(df['매출수량'], errors='coerce')
    df['is_estimated'] = False
    
    plants = df['플랜트'].dropna().unique()
    
    for plant in plants:
        plant_df = df[df['플랜트'] == plant].copy()
        plant_df['년'] = pd.to_numeric(plant_df['년'], errors='coerce').fillna(0).astype(int)
        plant_df['월'] = pd.to_numeric(plant_df['월'], errors='coerce').fillna(0).astype(int)
        plant_df = plant_df.sort_values(['년', '월']).reset_index(drop=True)
        
        missing_mask = (plant_df['매출수량'].isna()) | (plant_df['매출수량'] == 0)
        
        for idx in plant_df[missing_mask].index:
            current_year = plant_df.loc[idx, '년']
            current_month = plant_df.loc[idx, '월']
            
            lookback_values = []
            for back_month in range(1, lookback_months + 1):
                past_year = current_year
                past_month = current_month - back_month
                
                if past_month <= 0:
                    past_year -= 1
                    past_month += 12
                
                past_data = plant_df[
                    (plant_df['년'] == past_year) & (plant_df['월'] == past_month)
                ]
                if not past_data.empty and not pd.isna(past_data['매출수량'].iloc[0]):
                    lookback_values.append(past_data['매출수량'].iloc[0])
            
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


# --- Nested Series JSON Generation Helpers ---

def _sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:\*\?"<>\|]', '-', name)


def _month_range(df: pd.DataFrame, date_col: str) -> List[str]:
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
    arr = np.array(values, dtype=float)
    mean = float(np.nanmean(arr)) if arr.size else 0.0
    std = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
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
    """Nested Series JSON 생성."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_months = _month_range(df, date_col)
    if not all_months:
        print("[STORAGE] 유효한 월 범위를 산출할 수 없음 (접수일자 비어있음)")
        return 0

    valid_mask = df['Lag_Valid'] if 'Lag_Valid' in df.columns else pd.Series([True] * len(df))

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['__month'] = df[date_col].dt.to_period('M').dt.to_timestamp()

    required_cols = ['플랜트', '제품범주2', '대분류', '중분류']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    grouped = df.groupby(['플랜트', '제품범주2', '대분류'], dropna=False)

    created = 0
    today_str = pd.Timestamp.today().strftime('%Y-%m-%d')

    for (plant, cat2, major), gdf in grouped:
        parent_counts_all = gdf.groupby('__month').size()
        parent_counts_valid = gdf[valid_mask.loc[gdf.index]].groupby('__month').size()

        parent_history = []
        parent_series_all = {pd.to_datetime(k).strftime('%Y-%m'): int(v) for k, v in parent_counts_all.items()}
        parent_series_valid = {pd.to_datetime(k).strftime('%Y-%m'): int(v) for k, v in parent_counts_valid.items()}

        parent_values_for_stats = []
        for m in all_months:
            count = parent_series_all.get(m, 0)
            parent_history.append({"date": m, "count": int(count)})
            parent_values_for_stats.append(parent_series_valid.get(m, 0))

        parent_stats = _compute_series_stats(parent_values_for_stats)

        if 'Lag_Valid' in gdf.columns and 'Lag_Days' in gdf.columns:
            valid_lags = gdf.loc[gdf['Lag_Valid'] == True, 'Lag_Days']
            avg_lag = valid_lags.mean() if not valid_lags.empty else 0.0
            parent_stats['avg_lag_days'] = round(float(np.nan_to_num(avg_lag)), 1)
        else:
            parent_stats['avg_lag_days'] = 0.0

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

        s_plant = _sanitize_filename(str(plant) if plant is not None else "")
        s_cat2 = _sanitize_filename(str(cat2) if cat2 is not None else "")
        s_major = _sanitize_filename(str(major) if major is not None else "")
        
        filename_key = f"{s_plant}_{s_cat2}_{s_major}"
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


# ============================================================================
# [NEW] Phase 2.5: 통합 로더 구현 (Single Source of Truth)
# ============================================================================

def load_and_filter_data(
    plant: str,
    start_date: date,
    end_date: date,
    search_mode: str,  # "인입 (Inflow)", "실적 (Performance)", "Custom (직접 선택)"
    selected_biz: Optional[List[str]] = None,
    selected_reasons: Optional[List[str]] = None,
    selected_grades: Optional[List[str]] = None,
    selected_categories: Optional[List[str]] = None,
    data_path: Union[str, Path] = DATA_HUB_PATH
) -> pd.DataFrame:
    """
    분석(Page 3) 및 예측(Page 4)에서 공통으로 사용하는 통합 데이터 로더.
    
    동작:
        1. PyArrow Dataset으로 지연 로딩 연결
        2. 플랜트 및 날짜 범위 필터링 (메모리 효율화)
        3. 조회 모드(인입/실적/Custom)에 따른 비즈니스 로직 적용
        4. 등급 및 대분류 필터링 적용
        
    Returns:
        pd.DataFrame: 모든 필터가 적용된 최종 데이터
    """
    try:
        path = Path(data_path)
        if not path.exists():
            print(f"[WARNING] 데이터 경로 없음: {data_path}")
            return pd.DataFrame(columns=TARGET_54_COLS)

        # 1. Dataset 연결 (Hive Partitioning 자동 감지)
        dataset = ds.dataset(path, format="parquet", partitioning="hive")
        
        # 2. 1차 로드: 전체 데이터 (필터링 편의성을 위해)
        # 대용량 환경에서는 ds.field() 필터링을 먼저 적용하는 것이 좋으나,
        # 현재 구조에서는 Pandas 변환 후 처리가 날짜/복합 로직에 유리함.
        table = dataset.to_table() 
        df = table.to_pandas()
        
        # 날짜 컬럼 변환
        if '접수일자' not in df.columns:
            return pd.DataFrame()
        df['접수일자'] = pd.to_datetime(df['접수일자'])
        
        # 3. 기본 범위 필터링 (플랜트 + 기간)
        mask = (
            (df['플랜트'] == plant) &
            (df['접수일자'].dt.date >= start_date) &
            (df['접수일자'].dt.date <= end_date)
        )
        df = df[mask].copy()
        
        if df.empty:
            return df
            
        # 4. 조회 모드(Search Mode) 필터링
        if "인입" in search_mode:
            # 사업부문(식품, B2B식품) + 불만원인(전체)
            cond_biz = df['사업부문'].isin(TARGET_BUSINESS_UNITS)
            cond_reason = df['불만원인'].notna()
            df = df[cond_biz & cond_reason]
            
        elif "실적" in search_mode:
            # 사업부문(식품, B2B식품) + 불만원인(제조, 고객, 구매)
            cond_biz = df['사업부문'].isin(TARGET_BUSINESS_UNITS)
            cond_reason = df['불만원인'].isin(PERFORMANCE_REASONS)
            df = df[cond_biz & cond_reason]
            
        else: # "Custom"
            if selected_biz:
                df = df[df['사업부문'].isin(selected_biz)]
            if selected_reasons:
                df = df[df['불만원인'].isin(selected_reasons)]
        
        # 5. 상세 필터링 (등급, 대분류)
        if selected_grades:
            df = df[df['등급기준'].isin(selected_grades)]
            
        if selected_categories:
            df = df[df['대분류'].isin(selected_categories)]
            
        return df.reset_index(drop=True)

    except Exception as e:
        print(f"[ERROR] 통합 데이터 로드 실패: {str(e)}")
        # 에러 발생 시 빈 데이터프레임 반환
        return pd.DataFrame(columns=TARGET_54_COLS)

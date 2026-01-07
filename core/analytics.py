# core/analytics.py
import pandas as pd
import numpy as np
from typing import Dict

def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    IQR(Interquartile Range) 방법을 사용하여 숫자형 데이터프레임에서 이상치를 탐지합니다.

    Args:
        df (pd.DataFrame): 숫자형 데이터가 포함된 입력 데이터프레임.

    Returns:
        pd.DataFrame: 입력과 동일한 형태의 boolean 데이터프레임.
                      True는 해당 셀이 이상치임을 나타냅니다.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("입력은 반드시 pandas 데이터프레임이어야 합니다.")
        
    numeric_df = df.select_dtypes(include=np.number)
    
    if numeric_df.empty:
        return pd.DataFrame(False, index=df.index, columns=df.columns)

    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    is_outlier = (numeric_df < lower_bound) | (numeric_df > upper_bound)
    
    result_df = pd.DataFrame(False, index=df.index, columns=df.columns)
    result_df[is_outlier.columns] = is_outlier
    
    return result_df

def calculate_lag_stats(df: pd.DataFrame, lag_col: str = 'Lag_Days') -> Dict:
    """
    두 날짜 사이의 Lag(소요 시간)에 대한 통계를 계산합니다.

    Args:
        df (pd.DataFrame): Lag 데이터가 포함된 데이터프레임. 
                           유효한 Lag 데이터를 필터링하기 위해 'Lag_Valid' 컬럼이 있어야 합니다.
        lag_col (str): Lag 일수가 포함된 컬럼의 이름. 기본값은 'Lag_Days'입니다.

    Returns:
        dict: 'mean', 'std', 'min', 'max', 'p25', 'p50', 'p75', 'count'를 포함하는 딕셔너리.
              유효한 Lag 데이터가 없으면 비어 있는 통계를 반환합니다.
    """
    stats_keys = ['mean', 'std', 'min', 'max', 'p25', 'p50', 'p75', 'count']
    empty_stats = {key: 0 for key in stats_keys}

    if lag_col not in df.columns or 'Lag_Valid' not in df.columns:
        return empty_stats
        
    valid_lags = df.loc[df['Lag_Valid'] == True, lag_col].dropna()
    
    if valid_lags.empty:
        return empty_stats
    
    stats = {
        'mean': round(valid_lags.mean(), 1),
        'std': round(valid_lags.std(), 1),
        'min': int(valid_lags.min()),
        'max': int(valid_lags.max()),
        'p25': int(valid_lags.quantile(0.25)),
        'p50': int(valid_lags.quantile(0.50)),
        'p75': int(valid_lags.quantile(0.75)),
        'count': int(len(valid_lags))
    }
    
    return stats
